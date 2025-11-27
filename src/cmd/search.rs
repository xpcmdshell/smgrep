use std::{
   path::PathBuf,
   process::{Command, Stdio},
   sync::Arc,
   time::Duration,
};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tokio::time;

use crate::{
   Result,
   chunker::Chunker,
   embed::worker::EmbedWorker,
   error::Error,
   file::LocalFileSystem,
   git,
   ipc::{self, Request, Response},
   search::SearchEngine,
   store::LanceStore,
   sync::SyncEngine,
   usock,
};

#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
   path:       PathBuf,
   score:      f32,
   content:    String,
   #[serde(skip_serializing_if = "Option::is_none")]
   chunk_type: Option<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   start_line: Option<usize>,
   #[serde(skip_serializing_if = "Option::is_none")]
   end_line:   Option<usize>,
   #[serde(skip_serializing_if = "Option::is_none")]
   is_anchor:  Option<bool>,
}

#[derive(Debug, Serialize)]
struct JsonOutput {
   results: Vec<SearchResult>,
}

#[allow(clippy::too_many_arguments)]
pub async fn execute(
   query: String,
   path: Option<PathBuf>,
   max: usize,
   per_file: usize,
   content: bool,
   compact: bool,
   scores: bool,
   sync: bool,
   dry_run: bool,
   json: bool,
   no_rerank: bool,
   plain: bool,
   store_id: Option<String>,
) -> Result<()> {
   let root = std::env::current_dir()?;
   let search_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id.map_or_else(|| git::resolve_store_id(&search_path), Ok)?;

   if let Some(results) =
      try_daemon_search(&query, max, !no_rerank, &search_path, &resolved_store_id).await?
   {
      if json {
         println!("{}", serde_json::to_string(&JsonOutput { results })?);
      } else {
         format_results(&results, &query, &root, content, compact, scores, plain)?;
      }
      return Ok(());
   }

   if dry_run {
      if json {
         println!("{}", serde_json::to_string(&JsonOutput { results: vec![] })?);
      } else {
         println!("Dry run: would search for '{query}' in {search_path:?}");
         println!("Store ID: {resolved_store_id}");
         println!("Max results: {max}");
      }
      return Ok(());
   }

   if sync && !json {
      let spinner = ProgressBar::new_spinner();
      spinner.set_style(
         ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
      );
      spinner.enable_steady_tick(Duration::from_millis(100));
      spinner.set_message("Syncing files to index...");

      time::sleep(Duration::from_millis(100)).await;

      spinner.finish_with_message("Sync complete");
   }

   let results =
      perform_search(&query, &search_path, &resolved_store_id, max, per_file, !no_rerank).await?;

   if results.is_empty() {
      if json {
         println!("{}", serde_json::to_string(&JsonOutput { results: vec![] })?);
      } else {
         println!("No results found for '{query}'");
         if !sync {
            println!("\nTip: Use --sync to re-index before searching");
         }
      }
      return Ok(());
   }

   if json {
      println!("{}", serde_json::to_string(&JsonOutput { results })?);
   } else {
      format_results(&results, &query, &root, content, compact, scores, plain)?;
   }

   Ok(())
}

async fn try_daemon_search(
   query: &str,
   max: usize,
   rerank: bool,
   path: &PathBuf,
   store_id: &str,
) -> Result<Option<Vec<SearchResult>>> {
   let stream = if let Ok(s) = usock::Stream::connect(store_id).await {
      s
   } else {
      spawn_daemon(path)?;

      for _ in 0..50 {
         time::sleep(Duration::from_millis(100)).await;
         if let Ok(s) = usock::Stream::connect(store_id).await {
            return send_search_request(s, query, max, rerank, path)
               .await
               .map(Some);
         }
      }

      return Ok(None);
   };

   send_search_request(stream, query, max, rerank, path)
      .await
      .map(Some)
}

async fn send_search_request(
   mut stream: usock::Stream,
   query: &str,
   max: usize,
   rerank: bool,
   path: &PathBuf,
) -> Result<Vec<SearchResult>> {
   let request =
      Request::Search { query: query.to_string(), limit: max, path: Some(path.clone()), rerank };

   let mut buffer = ipc::SocketBuffer::new();
   buffer.send(&mut stream, &request).await?;
   let response: Response = buffer.recv(&mut stream).await?;

   match response {
      Response::Search(search_response) => {
         let results = search_response
            .results
            .into_iter()
            .map(|r| SearchResult {
               path:       r.path,
               score:      r.score,
               content:    r.content.into_string(),
               chunk_type: r.chunk_type.map(|ct| format!("{ct:?}").to_lowercase()),
               start_line: Some(r.start_line as usize),
               end_line:   Some((r.start_line + r.num_lines) as usize),
               is_anchor:  r.is_anchor,
            })
            .collect();
         Ok(results)
      },
      Response::Error { message } => Err(Error::Server { op: "search", reason: message }),
      _ => Err(Error::UnexpectedResponse("search")),
   }
}

fn spawn_daemon(path: &PathBuf) -> Result<()> {
   let exe = std::env::current_exe()?;

   Command::new(&exe)
      .arg("serve")
      .arg("--path")
      .arg(path)
      .stdin(Stdio::null())
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .spawn()
      .map_err(Error::DaemonSpawn)?;

   Ok(())
}

async fn perform_search(
   query: &str,
   path: &PathBuf,
   store_id: &str,
   max: usize,
   per_file: usize,
   rerank: bool,
) -> Result<Vec<SearchResult>> {
   let store = Arc::new(LanceStore::new()?);
   let embedder = Arc::new(EmbedWorker::new()?);

   let fs = LocalFileSystem::new();
   let chunker = Chunker::default();
   let sync_engine = SyncEngine::new(fs, chunker, embedder.clone(), store.clone());

   sync_engine
      .initial_sync(store_id, path, false, &mut ())
      .await?;

   let engine = SearchEngine::new(store, embedder);
   let response = engine
      .search(store_id, query, max, per_file, None, rerank)
      .await?;

   let root_str = path.to_string_lossy().to_string();

   let results = response
      .results
      .into_iter()
      .map(|r| {
         let rel_path = r
            .path
            .strip_prefix(&root_str)
            .unwrap_or(&r.path)
            .to_string_lossy()
            .trim_start_matches('/')
            .into();

         SearchResult {
            path:       rel_path,
            score:      r.score,
            content:    r.content.into_string(),
            chunk_type: Some(format!("{:?}", r.chunk_type).to_lowercase()),
            start_line: Some(r.start_line as usize),
            end_line:   Some((r.start_line + r.num_lines) as usize),
            is_anchor:  r.is_anchor,
         }
      })
      .collect();

   Ok(results)
}

fn format_results(
   results: &[SearchResult],
   query: &str,
   root: &PathBuf,
   content: bool,
   compact: bool,
   scores: bool,
   plain: bool,
) -> Result<()> {
   const MAX_PREVIEW_LINES: usize = 12;

   if compact {
      for result in results {
         println!("{}", result.path.display());
      }
      return Ok(());
   }

   if plain {
      println!("\nSearch results for: {query}");
      println!("Root: {}\n", root.display());
   } else {
      println!("\n{}", style(format!("Search results for: {query}")).bold());
      println!("{}", style(format!("Root: {}\n", root.display())).dim());
   }

   let display_results: Vec<_> = results
      .iter()
      .filter(|r| !r.is_anchor.unwrap_or(false))
      .collect();

   for (i, result) in display_results.iter().enumerate() {
      let start_line = result.start_line.unwrap_or(1);
      let lines: Vec<&str> = result.content.lines().collect();
      let total_lines = lines.len();
      let show_all = content || total_lines <= MAX_PREVIEW_LINES;
      let display_lines = if show_all {
         total_lines
      } else {
         MAX_PREVIEW_LINES
      };
      let line_num_width = format!("{}", start_line + display_lines).len();

      if plain {
         print!("{}) {}:{}", i + 1, result.path.display(), start_line);

         if scores {
            print!(" (score: {:.3})", result.score);
         }

         println!();

         for (j, line) in lines.iter().take(display_lines).enumerate() {
            let line_num = start_line + j;
            println!("{line_num:>line_num_width$} | {line}");
         }

         if !show_all {
            let remaining = total_lines - display_lines;
            println!("{:>width$} | ... (+{} more lines)", "", remaining, width = line_num_width);
         }
      } else {
         print!("{}", style(format!("{}) ", i + 1)).bold().cyan());
         print!("{}:{}", style(result.path.display()).green(), start_line);

         if scores {
            print!(" {}", style(format!("(score: {:.3})", result.score)).dim());
         }

         println!();

         for (j, line) in lines.iter().take(display_lines).enumerate() {
            let line_num = start_line + j;
            println!(
               "{:>width$} {} {}",
               style(line_num).dim(),
               style("|").dim(),
               line,
               width = line_num_width
            );
         }

         if !show_all {
            let remaining = total_lines - display_lines;
            println!(
               "{:>width$} {} {}",
               "",
               style("|").dim(),
               style(format!("... (+{remaining} more lines)")).dim(),
               width = line_num_width
            );
         }
      }

      println!();
   }

   Ok(())
}
