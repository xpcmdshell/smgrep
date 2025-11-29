//! Stop all servers command.
//!
//! Gracefully shuts down all running smgrep daemon servers.

use console::style;

use crate::{
   Result,
   ipc::{self, Request, Response},
   usock,
};

/// Executes the stop-all command to shut down all running servers.
pub async fn execute() -> Result<()> {
   let servers = usock::list_running_servers();

   if servers.is_empty() {
      println!("{}", style("No servers running").yellow());
      return Ok(());
   }

   let mut stopped = 0;
   let mut failed = 0;

   for store_id in servers {
      if let Ok(mut stream) = usock::Stream::connect(&store_id).await {
         let mut buffer = ipc::SocketBuffer::new();
         if let Err(e) = buffer.send(&mut stream, &Request::Shutdown).await {
            tracing::debug!("Failed to send shutdown to {}: {}", store_id, e);
            failed += 1;
            continue;
         }

         match buffer.recv(&mut stream).await {
            Ok(Response::Shutdown { success: true }) | Err(_) => stopped += 1,
            Ok(_) => failed += 1,
         }
      } else {
         usock::remove_socket(&store_id);
         stopped += 1;
      }
   }

   println!("{}", style(format!("Stopped {stopped} servers, {failed} failed")).green());

   Ok(())
}
