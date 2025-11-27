use std::{env, path::PathBuf};

use console::style;

use crate::{
   Result, git,
   ipc::{self, Request, Response},
   usock,
};

pub async fn execute(path: Option<PathBuf>) -> Result<()> {
   let root = env::current_dir()?;
   let target_path = path.unwrap_or(root);

   let store_id = git::resolve_store_id(&target_path)?;

   if !usock::socket_path(&store_id).exists() {
      println!("{}", style("No server running for this project").yellow());
      return Ok(());
   }

   let mut buffer = ipc::SocketBuffer::new();

   if let Ok(mut stream) = usock::Stream::connect(&store_id).await {
      buffer.send(&mut stream, &Request::Shutdown).await?;

      match buffer.recv(&mut stream).await {
         Ok(Response::Shutdown { success: true }) => {
            println!("{}", style("Server stopped").green());
         },
         Ok(_) => {
            println!("{}", style("Unexpected response from server").yellow());
         },
         Err(_) => {
            println!("{}", style("Server stopped").green());
         },
      }
   } else {
      usock::remove_socket(&store_id);
      println!("{}", style("Removed stale socket").yellow());
   }

   Ok(())
}
