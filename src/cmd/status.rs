//! Server status command.
//!
//! Displays the status of all running smgrep daemon servers.

use console::style;

use crate::{
   Result,
   ipc::{self, Request, Response},
   usock,
};

/// Executes the status command to show running servers.
pub async fn execute() -> Result<()> {
   let servers = usock::list_running_servers();

   if servers.is_empty() {
      println!("{}", style("No servers running").dim());
      return Ok(());
   }

   println!("{}", style("Running servers:").bold());
   println!();

   let mut buffer = ipc::SocketBuffer::new();
   for store_id in servers {
      match usock::Stream::connect(&store_id).await {
         Ok(mut stream) => {
            if buffer.send(&mut stream, &Request::Health).await.is_err() {
               println!("  {} {} {}", style("●").yellow(), store_id, style("(unresponsive)").dim());
               continue;
            }

            match buffer.recv(&mut stream).await {
               Ok(Response::Health { status }) => {
                  let state = if status.indexing {
                     format!("indexing {}%", status.progress)
                  } else {
                     "ready".to_string()
                  };
                  println!(
                     "  {} {} {}",
                     style("●").green(),
                     store_id,
                     style(format!("({state})")).dim()
                  );
               },
               _ => {
                  println!("  {} {} {}", style("●").yellow(), store_id, style("(unknown)").dim());
               },
            }
         },
         Err(_) => {
            println!("  {} {} {}", style("●").red(), store_id, style("(stale)").dim());
         },
      }
   }

   Ok(())
}
