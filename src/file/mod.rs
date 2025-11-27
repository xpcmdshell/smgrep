pub mod discovery;
pub mod ignore;
pub mod watcher;

use std::path::Path;

pub use discovery::*;
pub use ignore::*;
pub use watcher::*;

pub fn normalize_path(path: &Path) -> String {
   path.to_string_lossy().replace('\\', "/")
}
