use std::{
   fs, io,
   path::PathBuf,
   pin::Pin,
   task::{self, Poll},
};

use tokio::{
   io::ReadBuf,
   net::{TcpListener as TokioTcpListener, TcpStream as TokioTcpStream},
};

use super::SocketError;
use crate::{Result, config};

pub fn socket_dir() -> PathBuf {
   config::data_dir().join("socks")
}

fn port_file_path(store_id: &str) -> PathBuf {
   socket_dir().join(format!("{}.port", store_id))
}

pub fn socket_path(store_id: &str) -> PathBuf {
   port_file_path(store_id)
}

pub fn list_running_servers() -> Vec<String> {
   let dir = socket_dir();
   if !dir.exists() {
      return Vec::new();
   }

   fs::read_dir(&dir)
      .into_iter()
      .flatten()
      .filter_map(|e| e.ok())
      .filter(|e| e.path().extension().is_some_and(|ext| ext == "port"))
      .filter_map(|e| {
         e.path()
            .file_stem()
            .and_then(|s| s.to_str())
            .map(String::from)
      })
      .collect()
}

pub fn remove_socket(store_id: &str) {
   let _ = fs::remove_file(port_file_path(store_id));
}

pub struct Listener {
   inner:     TokioTcpListener,
   port_file: PathBuf,
   port:      u16,
}

impl Listener {
   pub async fn bind(store_id: &str) -> Result<Self> {
      let port_file = port_file_path(store_id);

      if let Some(parent) = port_file.parent() {
         fs::create_dir_all(parent).map_err(SocketError::CreateDir)?;
      }

      if port_file.exists() {
         if Stream::connect(store_id).await.is_ok() {
            return Err(SocketError::AlreadyRunning.into());
         }
         fs::remove_file(&port_file).map_err(SocketError::RemoveStale)?;
      }

      let inner = TokioTcpListener::bind("127.0.0.1:0")
         .await
         .map_err(SocketError::Bind)?;

      let port = inner.local_addr().map_err(SocketError::Bind)?.port();

      fs::write(&port_file, port.to_string()).map_err(SocketError::WritePort)?;

      Ok(Self { inner, port_file, port })
   }

   pub async fn accept(&self) -> Result<Stream> {
      let (stream, _) = self.inner.accept().await.map_err(SocketError::Accept)?;
      Ok(Stream { inner: stream })
   }

   pub fn local_addr(&self) -> String {
      format!("127.0.0.1:{}", self.port)
   }
}

impl Drop for Listener {
   fn drop(&mut self) {
      let _ = fs::remove_file(&self.port_file);
   }
}

#[repr(transparent)]
pub struct Stream {
   inner: TokioTcpStream,
}

impl Stream {
   pub async fn connect(store_id: &str) -> Result<Self> {
      let port_file = port_file_path(store_id);

      let port_str = fs::read_to_string(&port_file).map_err(SocketError::ReadPort)?;

      let port: u16 = port_str
         .trim()
         .parse()
         .map_err(|e: std::num::ParseIntError| SocketError::InvalidPort(io::Error::other(e)))?;

      let inner = TokioTcpStream::connect(format!("127.0.0.1:{}", port))
         .await
         .map_err(SocketError::Connect)?;

      Ok(Self { inner })
   }
}

impl tokio::io::AsyncRead for Stream {
   fn poll_read(
      mut self: Pin<&mut Self>,
      cx: &mut task::Context<'_>,
      buf: &mut ReadBuf<'_>,
   ) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_read(cx, buf)
   }
}

impl tokio::io::AsyncWrite for Stream {
   fn poll_write(
      mut self: Pin<&mut Self>,
      cx: &mut task::Context<'_>,
      buf: &[u8],
   ) -> Poll<io::Result<usize>> {
      Pin::new(&mut self.inner).poll_write(cx, buf)
   }

   fn poll_flush(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_flush(cx)
   }

   fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_shutdown(cx)
   }
}
