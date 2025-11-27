use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{Str, meta::FileHash};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
   Function,
   Class,
   Interface,
   Method,
   TypeAlias,
   Block,
   Other,
}

pub type ContextVec = SmallVec<[Str; 4]>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
   pub content:     Str,
   pub start_line:  usize,
   pub start_col:   usize,
   pub end_line:    usize,
   pub chunk_type:  Option<ChunkType>,
   pub context:     ContextVec,
   pub chunk_index: Option<i32>,
   pub is_anchor:   Option<bool>,
}

impl Chunk {
   pub fn new(
      content: Str,
      start_line: usize,
      end_line: usize,
      chunk_type: ChunkType,
      context: &[Str],
   ) -> Self {
      Self {
         content,
         start_line,
         start_col: 0,
         end_line,
         chunk_type: Some(chunk_type),
         context: context.iter().cloned().collect(),
         chunk_index: None,
         is_anchor: Some(false),
      }
   }

   pub const fn with_col(mut self, col: usize) -> Self {
      self.start_col = col;
      self
   }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedChunk {
   pub id:           String,
   pub path:         PathBuf,
   pub hash:         FileHash,
   pub content:      Str,
   pub start_line:   u32,
   pub end_line:     u32,
   pub chunk_index:  Option<u32>,
   pub is_anchor:    Option<bool>,
   pub chunk_type:   Option<ChunkType>,
   pub context_prev: Option<Str>,
   pub context_next: Option<Str>,
}

#[derive(Debug, Clone)]
pub struct VectorRecord {
   pub id:            String,
   pub path:          PathBuf,
   pub hash:          FileHash,
   pub content:       Str,
   pub start_line:    u32,
   pub end_line:      u32,
   pub chunk_index:   Option<u32>,
   pub is_anchor:     Option<bool>,
   pub chunk_type:    Option<ChunkType>,
   pub context_prev:  Option<Str>,
   pub context_next:  Option<Str>,
   pub vector:        Vec<f32>,
   pub colbert:       Vec<u8>,
   pub colbert_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
   pub path:       PathBuf,
   pub content:    Str,
   pub score:      f32,
   pub start_line: u32,
   pub num_lines:  u32,
   pub chunk_type: Option<ChunkType>,
   pub is_anchor:  Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchStatus {
   Ready,
   Indexing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
   pub results:  Vec<SearchResult>,
   pub status:   SearchStatus,
   pub progress: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreInfo {
   pub store_id:  String,
   pub row_count: u64,
   pub path:      PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncProgress {
   pub processed:    usize,
   pub indexed:      usize,
   pub total:        usize,
   pub current_file: Option<Str>,
}
