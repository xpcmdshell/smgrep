//! Model Context Protocol (MCP) server implementation.
//!
//! Implements the MCP JSON-RPC protocol to expose smgrep's semantic search
//! capabilities as a tool that can be used by Claude and other MCP clients.

use std::{
   io::Write,
   path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{
   Result,
   cmd::daemon,
   error::Error,
   git,
   ipc::{Request, Response, SocketBuffer},
   usock,
};

/// Incoming JSON-RPC 2.0 request from an MCP client.
#[derive(Deserialize)]
struct JsonRpcRequest {
   #[allow(dead_code, reason = "jsonrpc field is required by JSON-RPC spec but not used in code")]
   jsonrpc: String,
   id:      Option<Value>,
   method:  String,
   #[serde(default)]
   params:  Value,
}

/// Outgoing JSON-RPC 2.0 response to an MCP client.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
   jsonrpc: &'static str,
   id:      Value,
   #[serde(skip_serializing_if = "Option::is_none")]
   result:  Option<Value>,
   #[serde(skip_serializing_if = "Option::is_none")]
   error:   Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Serialize)]
struct JsonRpcError {
   code:    i32,
   message: String,
}

impl JsonRpcResponse {
   const fn success(id: Value, result: Value) -> Self {
      Self { jsonrpc: "2.0", id, result: Some(result), error: None }
   }

   const fn error(id: Value, code: i32, message: String) -> Self {
      Self { jsonrpc: "2.0", id, result: None, error: Some(JsonRpcError { code, message }) }
   }
}

/// Connection to a smgrep daemon for executing searches.
struct DaemonConn {
   stream: usock::Stream,
   buffer: SocketBuffer,
   cwd:    PathBuf,
}

impl DaemonConn {
   async fn connect(cwd: PathBuf) -> Result<Self> {
      let store_id = git::resolve_store_id(&cwd)?;
      let stream = daemon::connect_matching_daemon(&cwd, &store_id).await?;

      Ok(Self { stream, buffer: SocketBuffer::new(), cwd })
   }

   async fn search(&mut self, query: &str, limit: usize) -> Result<String> {
      let request = Request::Search {
         query: query.to_string(),
         limit,
         path: Some(self.cwd.clone()),
         rerank: true,
      };

      self.buffer.send(&mut self.stream, &request).await?;
      let response: Response = self.buffer.recv(&mut self.stream).await?;

      match response {
         Response::Search(search_response) => {
            let mut output = String::new();
            for r in search_response.results {
               use std::fmt::Write;
               writeln!(output, "{}:{}", r.path.display(), r.start_line).unwrap();
               for line in r.content.lines().take(10) {
                  writeln!(output, "  {line}").unwrap();
               }
               output.push('\n');
            }
            if output.is_empty() {
               output = format!("No results found for '{query}'");
            }
            Ok(output)
         },
         Response::Error { message } => Err(Error::Server { op: "search", reason: message }),
         _ => Err(Error::UnexpectedResponse("search")),
      }
   }
}

/// Executes the MCP server, reading JSON-RPC requests from stdin and writing
/// responses to stdout.
pub async fn execute() -> Result<()> {
   let stdin = BufReader::new(tokio::io::stdin());
   let mut lines = stdin.lines();

   let cwd = std::env::current_dir()?;
   let mut conn: Option<DaemonConn> = None;

   while let Some(line) = lines.next_line().await? {
      if line.is_empty() {
         continue;
      }

      let request: JsonRpcRequest = match serde_json::from_str(&line) {
         Ok(r) => r,
         Err(e) => {
            let response = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
            write_response(&response)?;
            continue;
         },
      };

      let id = request.id.clone().unwrap_or(Value::Null);
      let response = handle_request(request, &cwd, &mut conn).await;
      let response = match response {
         Ok(result) => JsonRpcResponse::success(id, result),
         Err(e) => JsonRpcResponse::error(id, -32603, e.to_string()),
      };

      write_response(&response)?;
   }

   Ok(())
}

/// Writes a JSON-RPC response to stdout.
fn write_response(response: &JsonRpcResponse) -> Result<()> {
   let stdout = std::io::stdout();
   let mut stdout = stdout.lock();
   serde_json::to_writer(&mut stdout, response)?;
   stdout.write_all(b"\n")?;
   stdout.flush()?;
   Ok(())
}

/// Handles an incoming JSON-RPC request and returns the result value.
async fn handle_request(
   request: JsonRpcRequest,
   cwd: &Path,
   conn: &mut Option<DaemonConn>,
) -> Result<Value> {
   match request.method.as_str() {
      "initialize" => Ok(json!({
         "protocolVersion": "2024-11-05",
         "capabilities": {
            "tools": {}
         },
         "serverInfo": {
            "name": "smgrep",
            "version": env!("CARGO_PKG_VERSION")
         }
      })),

      "notifications/initialized" => Ok(Value::Null),

      "tools/list" => Ok(json!({
         "tools": [{
            "name": "sem_search",
            "description": "Semantic code search. Finds code by meaning, not just text matching. Use for questions like 'where is X implemented' or 'how does Y work'.",
            "inputSchema": {
               "type": "object",
               "properties": {
                  "query": {
                     "type": "string",
                     "description": "Natural language query describing what you're looking for"
                  },
                  "limit": {
                     "type": "integer",
                     "description": "Maximum number of results (default: 10)",
                     "default": 10
                  }
               },
               "required": ["query"]
            }
         }]
      })),

      "tools/call" => {
         let name = request
            .params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
         let args = request
            .params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

         match name {
            "sem_search" => {
               let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
               let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

               let result = do_search_with_retry(cwd.to_path_buf(), conn, query, limit).await?;
               Ok(json!({
                  "content": [{
                     "type": "text",
                     "text": result
                  }]
               }))
            },
            _ => Err(Error::McpUnknownTool(name.to_string())),
         }
      },

      _ => Err(Error::McpUnknownMethod(request.method)),
   }
}

/// Executes a search with automatic retry on connection failure.
async fn do_search_with_retry(
   cwd: PathBuf,
   conn: &mut Option<DaemonConn>,
   query: &str,
   limit: usize,
) -> Result<String> {
   let result = {
      let conn_ref = ensure_conn(&cwd, conn).await?;
      conn_ref.search(query, limit).await
   };

   if let Ok(res) = result {
      Ok(res)
   } else {
      *conn = Some(DaemonConn::connect(cwd.clone()).await?);
      let conn_ref = ensure_conn(&cwd, conn).await?;
      conn_ref.search(query, limit).await
   }
}

/// Ensures a daemon connection exists, creating one if necessary.
async fn ensure_conn<'a>(
   cwd: &Path,
   conn: &'a mut Option<DaemonConn>,
) -> Result<&'a mut DaemonConn> {
   if conn.is_none() {
      *conn = Some(DaemonConn::connect(cwd.to_path_buf()).await?);
   }
   Ok(conn.as_mut().expect("connection initialized"))
}
