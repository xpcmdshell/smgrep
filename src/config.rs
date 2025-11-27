use std::{path::PathBuf, sync::OnceLock};

use directories::BaseDirs;

pub const DENSE_MODEL: &str = "ibm-granite/granite-embedding-30m-english";
pub const COLBERT_MODEL: &str = "answerdotai/answerai-colbert-small-v1";

pub const DENSE_DIM: usize = 384;
pub const COLBERT_DIM: usize = 96;

pub const QUERY_PREFIX: &str = "";

pub const DENSE_MAX_LENGTH: usize = 256;
pub const COLBERT_MAX_LENGTH: usize = 256;

pub const DEFAULT_BATCH_SIZE: usize = 48;
pub const MAX_BATCH_SIZE: usize = 96;

pub const MAX_THREADS: usize = 32;

static DEFAULT_THREADS: OnceLock<usize> = OnceLock::new();

pub fn default_threads() -> usize {
   *DEFAULT_THREADS.get_or_init(|| (num_cpus::get().saturating_sub(4)).clamp(1, MAX_THREADS))
}

pub fn data_dir() -> PathBuf {
   BaseDirs::new()
      .expect("failed to locate base directories")
      .home_dir()
      .join(".rsgrep")
}

pub fn model_dir() -> PathBuf {
   data_dir().join("models")
}

pub fn port() -> u16 {
   std::env::var("RSGREP_PORT")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(4444)
}

pub fn threads() -> usize {
   std::env::var("RSGREP_THREADS")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or_else(default_threads)
      .min(MAX_THREADS)
}

pub fn batch_size() -> usize {
   std::env::var("RSGREP_BATCH_SIZE")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(DEFAULT_BATCH_SIZE)
      .min(MAX_BATCH_SIZE)
}

pub fn low_impact() -> bool {
   std::env::var("RSGREP_LOW_IMPACT")
      .ok()
      .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn disable_gpu() -> bool {
   std::env::var("RSGREP_DISABLE_GPU")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn worker_timeout_ms() -> u64 {
   std::env::var("RSGREP_WORKER_TIMEOUT_MS")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(60000)
}

pub fn fast_mode() -> bool {
   std::env::var("RSGREP_FAST")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn profile_enabled() -> bool {
   std::env::var("RSGREP_PROFILE")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn skip_meta_save() -> bool {
   std::env::var("RSGREP_SKIP_META_SAVE")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn debug_models() -> bool {
   std::env::var("RSGREP_DEBUG_MODELS")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn debug_embed() -> bool {
   std::env::var("RSGREP_DEBUG_EMBED")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}
