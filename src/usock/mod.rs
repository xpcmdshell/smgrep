//! Unix domain socket and TCP socket abstractions for IPC

/// Errors that can occur during socket operations
#[derive(Debug, thiserror::Error)]
pub enum SocketError {
   #[error("server already running")]
   AlreadyRunning,

   #[error("failed to connect: {0}")]
   Connect(#[source] io::Error),

   #[error("failed to bind: {0}")]
   Bind(#[source] io::Error),

   #[error("accept failed: {0}")]
   Accept(#[source] io::Error),

   #[error("failed to create socket directory: {0}")]
   CreateDir(#[source] io::Error),

   #[error("failed to remove stale socket: {0}")]
   RemoveStale(#[source] io::Error),

   #[error("failed to read port file: {0}")]
   ReadPort(#[source] io::Error),

   #[error("invalid port in port file: {0}")]
   InvalidPort(#[source] io::Error),

   #[error("failed to write port file: {0}")]
   WritePort(#[source] io::Error),
}

#[cfg(unix)]
mod unix;
use std::io;

#[cfg(unix)]
pub use unix::*;

#[cfg(not(unix))]
mod tcp;
#[cfg(not(unix))]
pub use tcp::*;
