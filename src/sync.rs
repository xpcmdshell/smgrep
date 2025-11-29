//! File synchronization and indexing engine

use std::{
   collections::HashSet,
   path::{Path, PathBuf},
   sync::Arc,
};

use futures::stream::{self, StreamExt};
use indicatif::ProgressBar;

pub use crate::types::SyncProgress;
use crate::{
   Result, Str,
   chunker::{Chunker, anchor::create_anchor_chunk},
   config,
   embed::Embedder,
   file::FileSystem,
   index_lock::IndexLock,
   meta::{FileHash, MetaStore},
   store::Store,
   types::{PreparedChunk, VectorRecord},
};

/// Gets file modification time as Unix seconds
async fn get_mtime(path: &Path) -> u64 {
   let Ok(metadata) = tokio::fs::metadata(path).await else {
      return 0;
   };
   metadata
      .modified()
      .ok()
      .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
      .map_or(0, |d| d.as_secs())
}

/// Engine for synchronizing files to the index
pub struct SyncEngine<F: FileSystem, E: Embedder, S: Store> {
   file_system: F,
   chunker:     Chunker,
   embedder:    E,
   store:       S,
}

/// Result summary from a sync operation
#[derive(Debug, Clone)]
pub struct SyncResult {
   pub processed: usize,
   pub indexed:   usize,
   pub skipped:   usize,
   pub deleted:   usize,
}

/// Trait for receiving sync progress updates
pub trait SyncProgressCallback: Send {
   fn progress(&mut self, progress: SyncProgress);
}

impl<F: FnMut(SyncProgress) + Send> SyncProgressCallback for F {
   fn progress(&mut self, progress: SyncProgress) {
      self(progress);
   }
}

impl SyncProgressCallback for () {
   fn progress(&mut self, _progress: SyncProgress) {}
}

impl SyncProgressCallback for ProgressBar {
   fn progress(&mut self, progress: SyncProgress) {
      self.update(|state| {
         state.set_len(progress.total as u64);
         state.set_pos(progress.processed as u64);
      });
      if let Some(file) = &progress.current_file {
         let short = file.rsplit('/').next().unwrap_or(&**file);
         self.set_message(short.to_string());
      }
   }
}

impl<F, E, S> SyncEngine<F, E, S>
where
   F: FileSystem + Sync,
   E: Embedder + Send + Sync,
   S: Store + Send + Sync,
{
   pub const fn new(file_system: F, chunker: Chunker, embedder: E, store: S) -> Self {
      Self { file_system, chunker, embedder, store }
   }

   /// Performs an initial sync of files to the index
   pub async fn initial_sync(
      &self,
      store_id: &str,
      root: &Path,
      dry_run: bool,
      callback: &mut dyn SyncProgressCallback,
   ) -> Result<SyncResult> {
      const SAVE_INTERVAL: usize = 25;

      let _lock = IndexLock::acquire(store_id)?;

      let mut meta_store = MetaStore::load(store_id)?;
      let model_changed = meta_store.model_mismatch();
      let batch_size = config::get().batch_size();

      if model_changed && !dry_run {
         self.store.delete_store(store_id).await?;
         meta_store.reset_for_model_change();
      }

      // If lance store is empty but meta_store has entries for this root,
      // clear the stale metadata (data was deleted externally)
      if !dry_run && self.store.is_empty(store_id).await? {
         meta_store.delete_by_prefix(root);
      }

      let files = self.file_system.get_files(root)?.collect::<HashSet<_>>();

      let mut processed = 0;
      let mut indexed = 0;
      let mut skipped = 0;

      let deleted_paths: Vec<PathBuf> = meta_store
         .all_paths()
         .filter(|p| !files.contains(*p))
         .cloned()
         .collect();

      if !dry_run && !deleted_paths.is_empty() {
         self.store.delete_files(store_id, &deleted_paths).await?;
         for path in &deleted_paths {
            meta_store.remove(path);
         }
      }

      let deleted_count = deleted_paths.len();

      let scanned = stream::iter(files.into_iter().map(|file_path| async {
         let current_mtime = get_mtime(&file_path).await;

         if let Some(stored_mtime) = meta_store.get_mtime(&file_path)
            && stored_mtime == current_mtime
         {
            return None;
         }

         // TODO: blocking I/O in filter_map - could be improved with async iteration
         let content = std::fs::read(&file_path).ok()?;
         let hash = FileHash::sum(&content);

         let existing_hash = meta_store.get_hash(file_path.as_path());
         let needs_indexing = existing_hash != Some(hash);
         let has_existing_hash = existing_hash.is_some();

         Some((file_path, hash, content, current_mtime, needs_indexing, has_existing_hash))
      }))
      .buffer_unordered(64)
      .filter_map(|x| async move { x })
      .collect::<Vec<_>>()
      .await;

      if !dry_run {
         let changed_files = scanned
            .iter()
            .filter_map(|(file_path, _, _, _, needs_indexing, has_existing_hash)| {
               if *needs_indexing && *has_existing_hash {
                  Some(file_path.clone())
               } else {
                  None
               }
            })
            .collect::<Vec<_>>();
         if !changed_files.is_empty() {
            self.store.delete_files(store_id, &changed_files).await?;
         }
      }

      let files_to_index: Vec<_> = scanned
         .into_iter()
         .filter_map(|(path_str, hash, content, mtime, needs_indexing, _)| {
            processed += 1;
            if !needs_indexing {
               skipped += 1;
               None
            } else if dry_run {
               indexed += 1;
               None
            } else {
               Some((path_str, hash, content, mtime))
            }
         })
         .collect();

      let chunked_files: Vec<_> = stream::iter(files_to_index.into_iter())
         .map(|(path, hash, content, mtime)| {
            let chunker = self.chunker.clone();
            async move {
               let content_str = Str::from_utf8_lossy(&content);
               let path_arc = Arc::new(path.clone());

               let chunks = match chunker.chunk(&content_str, &path).await {
                  Ok(c) => c,
                  Err(e) => {
                     tracing::warn!("Failed to chunk {}: {}", path.display(), e);
                     return None;
                  },
               };
               let anchor_chunk = create_anchor_chunk(&content_str, &path);

               let mut prepared_chunks = Vec::with_capacity(chunks.len() + 1);

               let anchor_prepared = PreparedChunk {
                  id: format!("{}:anchor", path.display()),
                  path: Arc::clone(&path_arc),
                  hash,
                  content: anchor_chunk.content,
                  start_line: anchor_chunk.start_line as u32,
                  end_line: anchor_chunk.end_line as u32,
                  chunk_index: Some(0),
                  is_anchor: Some(true),
                  chunk_type: anchor_chunk.chunk_type,
                  context_prev: None,
                  context_next: None,
               };
               prepared_chunks.push(anchor_prepared);

               for (idx, chunk) in chunks.iter().enumerate() {
                  let context_prev: Option<Str> = if idx > 0 {
                     Some(chunks[idx - 1].content.clone())
                  } else {
                     None
                  };

                  let context_next: Option<Str> = if idx < chunks.len() - 1 {
                     Some(chunks[idx + 1].content.clone())
                  } else {
                     None
                  };

                  let prepared = PreparedChunk {
                     id: format!("{}:{}", path.display(), idx),
                     path: Arc::clone(&path_arc),
                     hash,
                     content: chunk.content.clone(),
                     start_line: chunk.start_line as u32,
                     end_line: chunk.end_line as u32,
                     chunk_index: Some(idx as u32 + 1),
                     is_anchor: Some(false),
                     chunk_type: chunk.chunk_type,
                     context_prev,
                     context_next,
                  };
                  prepared_chunks.push(prepared);
               }

               Some((path, hash, mtime, prepared_chunks))
            }
         })
         .buffer_unordered(64)
         .filter_map(|x| async move { x })
         .collect()
         .await;

      let mut embed_queue: Vec<(PathBuf, FileHash, u64, Vec<PreparedChunk>)> =
         Vec::with_capacity(batch_size);
      let mut since_save = 0;
      let total_to_embed = chunked_files.len();
      let mut embedded = 0;

      for (path, hash, mtime, prepared_chunks) in chunked_files {
         embed_queue.push((path, hash, mtime, prepared_chunks));

         if embed_queue.len() >= batch_size {
            callback.progress(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: Some(
                  format!("Embedding batch ({} files)...", embed_queue.len()).into(),
               ),
            });

            let batch = std::mem::take(&mut embed_queue);
            let batch_count = batch.len();
            let batch_indexed = self
               .process_embed_batch(store_id, batch, &mut meta_store)
               .await?;
            indexed += batch_indexed;
            embedded += batch_count;
            since_save += batch_count;

            if since_save >= SAVE_INTERVAL {
               meta_store.save()?;
               since_save = 0;
            }

            callback.progress(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: None,
            });
         }
      }

      if !dry_run && !embed_queue.is_empty() {
         callback.progress(SyncProgress {
            processed: embedded,
            indexed,
            total: total_to_embed,
            current_file: Some(
               format!("Embedding final batch ({} files)...", embed_queue.len()).into(),
            ),
         });

         let batch = std::mem::take(&mut embed_queue);
         let batch_count = batch.len();
         let batch_indexed = self
            .process_embed_batch(store_id, batch, &mut meta_store)
            .await?;
         indexed += batch_indexed;
         embedded += batch_count;
      }

      if !dry_run {
         callback.progress(SyncProgress {
            processed: embedded,
            indexed,
            total: total_to_embed,
            current_file: Some("Creating indexes...".into()),
         });

         meta_store.save()?;

         if indexed > 0 {
            self.store.create_fts_index(store_id).await?;
            self.store.create_vector_index(store_id).await?;
         }
      }

      callback.progress(SyncProgress {
         processed: total_to_embed,
         indexed,
         total: total_to_embed,
         current_file: None,
      });

      Ok(SyncResult { processed, indexed, skipped, deleted: deleted_count })
   }

   async fn process_embed_batch(
      &self,
      store_id: &str,
      batch: Vec<(PathBuf, FileHash, u64, Vec<PreparedChunk>)>,
      meta_store: &mut MetaStore,
   ) -> Result<usize> {
      let file_count = batch.len();
      let all_chunks: Vec<PreparedChunk> = batch
         .iter()
         .flat_map(|(_, _, _, chunks)| chunks.iter().cloned())
         .collect();

      if all_chunks.is_empty() {
         return Ok(0);
      }

      let texts: Vec<Str> = all_chunks.iter().map(|c| c.content.clone()).collect();

      let embeddings = self.embedder.compute_hybrid(&texts).await?;

      let records: Vec<VectorRecord> = all_chunks
         .into_iter()
         .zip(embeddings.into_iter())
         .map(|(chunk, embedding)| VectorRecord {
            id:            chunk.id,
            path:          chunk.path,
            hash:          chunk.hash,
            content:       chunk.content,
            start_line:    chunk.start_line,
            end_line:      chunk.end_line,
            chunk_index:   chunk.chunk_index,
            is_anchor:     chunk.is_anchor,
            chunk_type:    chunk.chunk_type,
            context_prev:  chunk.context_prev,
            context_next:  chunk.context_next,
            vector:        embedding.dense,
            colbert:       embedding.colbert,
            colbert_scale: embedding.colbert_scale,
         })
         .collect();

      self.store.insert_batch(store_id, records).await?;

      for (path, hash, mtime, _) in batch {
         meta_store.set_meta(path, hash, mtime);
      }

      Ok(file_count)
   }
}
