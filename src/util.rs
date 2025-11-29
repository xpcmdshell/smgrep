//! Utility functions for filesystem operations

use std::{fs, path::Path};

use crate::Result;

/// Recursively calculates the total size of a directory in bytes
pub fn get_dir_size(path: &Path) -> Result<u64> {
   let mut total = 0u64;

   if path.is_dir() {
      for entry in fs::read_dir(path)? {
         let entry = entry?;
         let metadata = entry.metadata()?;

         if metadata.is_dir() {
            total += get_dir_size(&entry.path())?;
         } else {
            total += metadata.len();
         }
      }
   }

   Ok(total)
}

/// Formats a byte count as a human-readable size string
pub fn format_size(bytes: u64) -> String {
   const KB: u64 = 1024;
   const MB: u64 = KB * 1024;
   const GB: u64 = MB * 1024;

   if bytes < KB {
      format!("{bytes} B")
   } else if bytes < MB {
      format!("{:.1} KB", bytes as f64 / KB as f64)
   } else if bytes < GB {
      format!("{:.1} MB", bytes as f64 / MB as f64)
   } else {
      format!("{:.1} GB", bytes as f64 / GB as f64)
   }
}
