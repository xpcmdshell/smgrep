pub mod anchor;

use std::{
   borrow::Cow,
   path::Path,
   slice,
   sync::{Arc, LazyLock},
};

use regex::Regex;
use tree_sitter::Language;

use crate::{
   Str,
   error::{ChunkerError, Result},
   grammar::GrammarManager,
   types::{Chunk, ChunkType},
};

pub const MAX_LINES: usize = 75;
pub const MAX_CHARS: usize = 2000;
pub const OVERLAP_LINES: usize = 10;
pub const OVERLAP_CHARS: usize = 200;
pub const STRIDE_CHARS: usize = MAX_CHARS - OVERLAP_CHARS;
pub const STRIDE_LINES: usize = MAX_LINES - OVERLAP_LINES;

static SCREAMING_CONST: LazyLock<Regex> =
   LazyLock::new(|| Regex::new(r"(?:^|\n)\s*(?:export\s+)?const\s+[A-Z0-9_]+\s*=").unwrap());

#[derive(Clone, Debug, Default)]
pub struct Chunker(Arc<GrammarManager>);

impl Chunker {
   fn get_language(&self, path: &Path) -> Option<Language> {
      match self.0.get_language_for_path(path) {
         Ok(Some(lang)) => Some(lang),
         Ok(None) => None,
         Err(e) => {
            tracing::warn!("failed to load language for {}: {}", path.display(), e);
            None
         },
      }
   }

   fn line_range_to_byte_range(
      content: &str,
      start_line: usize,
      end_line: usize,
   ) -> (usize, usize) {
      let mut start_byte = 0;
      let mut end_byte = content.len();
      let mut current_line = 0;

      for (idx, c) in content.char_indices() {
         if current_line == start_line && start_byte == 0 && current_line != 0 {
            start_byte = idx;
         }
         if c == '\n' {
            current_line += 1;
            if current_line == end_line {
               end_byte = idx;
               break;
            }
            if current_line == start_line {
               start_byte = idx + 1;
            }
         }
      }

      if start_line == 0 {
         start_byte = 0;
      }

      (start_byte, end_byte)
   }

   fn simple_chunk(content: &Str, path: &Path) -> Vec<Chunk> {
      let lines: Vec<&str> = content.lines().collect();
      let mut chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let context: Str = format!("File: {}", path.display()).into();
      let stack = slice::from_ref(&context);

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.is_empty() {
            break;
         }

         let (start_byte, end_byte) = Self::line_range_to_byte_range(content, i, end);
         let sub_content = content.slice(start_byte..end_byte);

         if sub_content.len() <= MAX_CHARS {
            chunks.push(Chunk::new(sub_content, i, end, ChunkType::Block, stack));
            i += stride;
         } else {
            let split_chunks = Self::split_content_by_chars(&sub_content, i, stack);
            chunks.extend(split_chunks);
            i += stride;
         }
      }

      chunks
   }

   fn chunk_with_tree_sitter(&self, content: &Str, path: &Path) -> Result<Option<Vec<Chunk>>> {
      let Some(language) = self.get_language(path) else {
         return Ok(None);
      };

      let (mut parser, store) = self.0.create_parser_with_store()?;
      parser
         .set_wasm_store(store)
         .map_err(ChunkerError::SetWasmStore)?;

      parser
         .set_language(&language)
         .map_err(ChunkerError::SetLanguage)?;

      let tree = parser
         .parse(content.as_str(), None)
         .ok_or(ChunkerError::ParseFailed)?;

      let root = tree.root_node();
      let file_context: Str = format!("File: {}", path.display()).into();

      let mut chunks = Vec::new();
      let mut block_chunks = Vec::new();
      let mut cursor_index = 0;
      let mut cursor_row = 0;
      let mut saw_definition = false;

      let mut cursor = root.walk();
      for child in root.named_children(&mut cursor) {
         self.visit_node(
            &child,
            content,
            slice::from_ref(&file_context),
            &mut chunks,
            &mut saw_definition,
         );

         let effective = self.unwrap_export(&child);
         let is_definition = self.is_definition_node(&effective, content.as_str());

         if is_definition {
            if child.start_byte() > cursor_index {
               let gap_text = content.slice(cursor_index..child.start_byte());
               if !gap_text.trim().is_empty() {
                  block_chunks.push(Chunk::new(
                     gap_text,
                     cursor_row,
                     child.start_position().row,
                     ChunkType::Block,
                     slice::from_ref(&file_context),
                  ));
               }
            }

            cursor_index = child.end_byte();
            cursor_row = child.end_position().row;
         }
      }

      if cursor_index < content.len() {
         let tail_text = content.slice(cursor_index..);
         if !tail_text.trim().is_empty() {
            block_chunks.push(Chunk::new(
               tail_text,
               cursor_row,
               root.end_position().row,
               ChunkType::Block,
               &[file_context],
            ));
         }
      }

      if !saw_definition {
         return Ok(Some(Vec::new()));
      }

      let mut combined = block_chunks;
      combined.extend(chunks);
      combined.sort_by(|a, b| {
         a.start_line
            .cmp(&b.start_line)
            .then(a.end_line.cmp(&b.end_line))
      });

      Ok(Some(combined))
   }

   fn visit_node(
      &self,
      node: &tree_sitter::Node,
      content: &Str,
      stack: &[Str],
      chunks: &mut Vec<Chunk>,
      saw_definition: &mut bool,
   ) {
      let effective = self.unwrap_export(node);
      let is_definition = self.is_definition_node(&effective, content.as_str());
      let mut stack = Cow::Borrowed(stack);

      if is_definition {
         *saw_definition = true;
         let label = self.label_for_node(&effective, content.as_str());
         if let Some(label) = label {
            stack.to_mut().push(label.into());
         }

         let node_text = content.slice(effective.start_byte()..effective.end_byte());
         chunks.push(Chunk::new(
            node_text,
            effective.start_position().row,
            effective.end_position().row,
            self.classify_node(&effective),
            stack.as_ref(),
         ));
      }

      let mut cursor = effective.walk();
      for child in effective.named_children(&mut cursor) {
         self.visit_node(&child, content, &stack, chunks, saw_definition);
      }
   }

   fn unwrap_export<'a>(&self, node: &'a tree_sitter::Node) -> tree_sitter::Node<'a> {
      if node.kind() == "export_statement" && node.named_child_count() > 0 {
         return node.named_child(0).unwrap();
      }
      *node
   }

   fn is_definition_node(&self, node: &tree_sitter::Node, content: &str) -> bool {
      let kind = node.kind();
      matches!(
         kind,
         "function_declaration"
            | "function_definition"
            | "method_definition"
            | "method_declaration"
            | "class_declaration"
            | "class_definition"
            | "interface_declaration"
            | "type_alias_declaration"
            | "type_declaration"
      ) || self.is_top_level_value_def(node, content)
   }

   fn is_top_level_value_def(&self, node: &tree_sitter::Node, content: &str) -> bool {
      let kind = node.kind();
      if kind != "lexical_declaration" && kind != "variable_declaration" {
         return false;
      }

      if let Some(parent) = node.parent() {
         let parent_kind = parent.kind();
         if !matches!(parent_kind, "program" | "module" | "source_file" | "class_body") {
            return false;
         }
      }

      let text = &content[node.start_byte()..node.end_byte()];

      if text.contains("=>") {
         return true;
      }
      if text.contains("function ") {
         return true;
      }
      if text.contains("class ") {
         return true;
      }

      if SCREAMING_CONST.is_match(text) {
         return true;
      }

      false
   }

   fn classify_node(&self, node: &tree_sitter::Node) -> ChunkType {
      let kind = node.kind();
      if kind.contains("class") {
         ChunkType::Class
      } else if kind.contains("interface") {
         ChunkType::Interface
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         ChunkType::TypeAlias
      } else {
         ChunkType::Other
      }
   }

   fn get_node_name(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
      if let Some(name_node) = node.child_by_field_name("name") {
         let name = &content[name_node.start_byte()..name_node.end_byte()];
         return Some(name.to_string());
      }

      if let Some(property_node) = node.child_by_field_name("property") {
         let name = &content[property_node.start_byte()..property_node.end_byte()];
         return Some(name.to_string());
      }

      if let Some(identifier_node) = node.child_by_field_name("identifier") {
         let name = &content[identifier_node.start_byte()..identifier_node.end_byte()];
         return Some(name.to_string());
      }

      let mut cursor = node.walk();
      for child in node.named_children(&mut cursor) {
         let child_kind = child.kind();
         if matches!(
            child_kind,
            "identifier" | "property_identifier" | "type_identifier" | "field_identifier"
         ) {
            let name = &content[child.start_byte()..child.end_byte()];
            return Some(name.to_string());
         }

         if child_kind == "variable_declarator"
            && let Some(name) = self.get_node_name(&child, content)
         {
            return Some(name);
         }
      }

      None
   }

   fn label_for_node(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
      let name = self.get_node_name(node, content);
      let kind = node.kind();

      if kind.contains("class") {
         Some(format!("Class: {}", name.as_deref().unwrap_or("<anonymous class>")))
      } else if kind.contains("method") {
         Some(format!("Method: {}", name.as_deref().unwrap_or("<anonymous method>")))
      } else if kind.contains("interface") {
         Some(format!("Interface: {}", name.as_deref().unwrap_or("<anonymous interface>")))
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         Some(format!("Type: {}", name.as_deref().unwrap_or("<anonymous type>")))
      } else if kind.contains("function") {
         Some(format!("Function: {}", name.as_deref().unwrap_or("<anonymous function>")))
      } else if self.is_top_level_value_def(node, content) {
         Some(format!("Function: {}", name.as_deref().unwrap_or("<anonymous function>")))
      } else {
         name.map(|n| format!("Symbol: {n}"))
      }
   }

   fn split_if_too_big(&self, chunk: Chunk) -> Vec<Chunk> {
      let char_count = chunk.content.len();
      let lines: Vec<&str> = chunk.content.lines().collect();
      let line_count = lines.len();

      if line_count <= MAX_LINES && char_count <= MAX_CHARS {
         return vec![chunk];
      }

      if char_count > MAX_CHARS && line_count <= MAX_LINES {
         return Self::split_by_chars(chunk);
      }

      let mut sub_chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let header = self.extract_header_line(&chunk.content);

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.len() < 3 && i > 0 {
            i += stride;
            continue;
         }

         let (start_byte, end_byte) = Self::line_range_to_byte_range(&chunk.content, i, end);
         let sub_content = if let Some(ref h) = header
            && i > 0
            && chunk.chunk_type != Some(ChunkType::Block)
         {
            Str::from_string(format!("{h}\n{}", chunk.content.slice(start_byte..end_byte)))
         } else {
            chunk.content.slice(start_byte..end_byte)
         };

         sub_chunks.push(Chunk::new(
            sub_content,
            chunk.start_line + i,
            chunk.start_line + end,
            chunk.chunk_type.unwrap_or(ChunkType::Other),
            &chunk.context,
         ));

         i += stride;
      }

      sub_chunks
         .into_iter()
         .flat_map(|sc| {
            if sc.content.len() > MAX_CHARS {
               Self::split_by_chars(sc)
            } else {
               vec![sc]
            }
         })
         .collect()
   }

   fn split_content_by_chars(input: &Str, start_line: usize, context: &[Str]) -> Vec<Chunk> {
      let mut chunks = Vec::new();

      let mut iter = input.as_str();
      let mut ln = start_line;
      loop {
         iter = iter.trim_start();
         if iter.is_empty() {
            break;
         }

         let lim = iter.floor_char_boundary(MAX_CHARS);
         let (pre, post) = iter.split_at(lim);
         iter = post;

         let content = pre.trim_end();
         if content.is_empty() {
            continue;
         }

         let lines = content.lines().count();

         chunks.push(Chunk::new(
            input.slice_ref(content),
            ln,
            ln + lines,
            ChunkType::Block,
            context,
         ));
         ln += lines;
      }

      chunks
   }

   fn split_by_chars(chunk: Chunk) -> Vec<Chunk> {
      let mut chunks = Vec::new();

      let mut iter = chunk.content.as_str();
      let mut ln = chunk.start_line;
      loop {
         iter = iter.trim_start();
         if iter.is_empty() {
            break;
         }
         let lim = iter.floor_char_boundary(MAX_CHARS);
         let (pre, post) = iter.split_at(lim);
         iter = post;
         let content = pre.trim_end();
         if content.is_empty() {
            continue;
         }

         let lines = content.lines().count();

         chunks.push(Chunk::new(
            chunk.content.slice_ref(content),
            ln,
            ln + lines,
            chunk.chunk_type.unwrap_or(ChunkType::Other),
            &chunk.context,
         ));
         ln += lines;
      }

      chunks
   }

   fn extract_header_line(&self, text: &str) -> Option<String> {
      for line in text.lines() {
         let trimmed = line.trim();
         if !trimmed.is_empty() {
            return Some(trimmed.to_string());
         }
      }
      None
   }

   pub fn chunk(&self, content: &Str, path: &Path) -> Result<Vec<Chunk>> {
      let raw_chunks = self
         .chunk_with_tree_sitter(content, path)?
         .unwrap_or_else(|| Self::simple_chunk(content, path));

      let chunks: Vec<Chunk> = raw_chunks
         .into_iter()
         .flat_map(|c| self.split_if_too_big(c))
         .collect();

      Ok(chunks)
   }
}
