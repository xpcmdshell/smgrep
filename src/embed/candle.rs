use std::{
   path::PathBuf,
   sync::{
      Arc,
      atomic::{AtomicUsize, Ordering},
   },
};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use parking_lot::RwLock;
use tokenizers::Tokenizer;

use crate::{
   config::{
      self, COLBERT_DIM, COLBERT_MODEL, DENSE_MODEL, QUERY_PREFIX, batch_size, debug_embed,
      debug_models,
   },
   embed::{Embedder, HybridEmbedding, QueryEmbedding},
   error::{Result, RsgrepError},
};

const MAX_SEQ_LEN_DENSE: usize = 256;
const MAX_SEQ_LEN_COLBERT: usize = 512;
const MIN_BATCH_SIZE: usize = 1;

pub struct CandleEmbedder {
   dense_model:         Arc<RwLock<Option<DenseModelState>>>,
   colbert_model:       Arc<RwLock<Option<ColbertModelState>>>,
   device:              Device,
   adaptive_batch_size: AtomicUsize,
}

struct DenseModelState {
   bert:      BertModel,
   tokenizer: Tokenizer,
}

struct ColbertModelState {
   bert:       BertModel,
   projection: Linear,
   tokenizer:  Tokenizer,
}

fn is_oom_error(err: &str) -> bool {
   err.contains("out of memory")
      || err.contains("OUT_OF_MEMORY")
      || err.contains("CUDA_ERROR_OUT_OF_MEMORY")
      || err.contains("alloc")
}

impl CandleEmbedder {
   pub fn new() -> Result<Self> {
      let device = if config::disable_gpu() {
         Device::Cpu
      } else {
         Device::cuda_if_available(0).unwrap_or(Device::Cpu)
      };
      if device.is_cuda() {
         tracing::info!("using CUDA device");
      } else {
         tracing::info!("using CPU device");
      }

      let initial_batch = batch_size();

      Ok(Self {
         dense_model: Arc::new(RwLock::new(None)),
         colbert_model: Arc::new(RwLock::new(None)),
         device,
         adaptive_batch_size: AtomicUsize::new(initial_batch),
      })
   }

   pub fn current_batch_size(&self) -> usize {
      self.adaptive_batch_size.load(Ordering::Relaxed)
   }

   fn reduce_batch_size(&self) -> usize {
      let current = self.adaptive_batch_size.load(Ordering::Relaxed);
      let new_size = (current / 2).max(MIN_BATCH_SIZE);
      self.adaptive_batch_size.store(new_size, Ordering::Relaxed);
      tracing::warn!("OOM detected, reducing batch size: {} -> {}", current, new_size);
      new_size
   }

   fn ensure_dense_loaded(&self) -> Result<()> {
      if self.dense_model.read().is_some() {
         return Ok(());
      }

      let mut guard = self.dense_model.write();
      if guard.is_some() {
         return Ok(());
      }

      let (bert, tokenizer) = Self::load_dense_model(&self.device)?;
      *guard = Some(DenseModelState { bert, tokenizer });
      Ok(())
   }

   fn ensure_colbert_loaded(&self) -> Result<()> {
      if self.colbert_model.read().is_some() {
         return Ok(());
      }

      let mut guard = self.colbert_model.write();
      if guard.is_some() {
         return Ok(());
      }

      let (bert, projection, tokenizer) = Self::load_colbert_model(&self.device)?;
      *guard = Some(ColbertModelState { bert, projection, tokenizer });
      Ok(())
   }

   fn load_dense_model(device: &Device) -> Result<(BertModel, Tokenizer)> {
      let model_path = Self::download_model(DENSE_MODEL)?;

      if debug_models() {
         tracing::info!("loading dense model from {:?}", model_path);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(|e| RsgrepError::Embedding(format!("failed to load tokenizer: {}", e)))?;

      let config: BertConfig = serde_json::from_str(
         &std::fs::read_to_string(model_path.join("config.json"))
            .map_err(|e| RsgrepError::Embedding(format!("failed to read config: {}", e)))?,
      )
      .map_err(|e| RsgrepError::Embedding(format!("failed to parse config: {}", e)))?;

      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], DTYPE, device)
            .map_err(|e| RsgrepError::Embedding(format!("failed to load weights: {}", e)))?
      };

      let bert = BertModel::load(vb, &config)
         .map_err(|e| RsgrepError::Embedding(format!("failed to load model: {}", e)))?;

      if debug_models() {
         tracing::info!("dense model loaded");
      }

      Ok((bert, tokenizer))
   }

   fn load_colbert_model(device: &Device) -> Result<(BertModel, Linear, Tokenizer)> {
      let model_path = Self::download_model(COLBERT_MODEL)?;

      if debug_models() {
         tracing::info!("loading colbert model from {:?}", model_path);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(|e| RsgrepError::Embedding(format!("failed to load tokenizer: {}", e)))?;

      let config: BertConfig = serde_json::from_str(
         &std::fs::read_to_string(model_path.join("config.json"))
            .map_err(|e| RsgrepError::Embedding(format!("failed to read config: {}", e)))?,
      )
      .map_err(|e| RsgrepError::Embedding(format!("failed to parse config: {}", e)))?;

      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], DTYPE, device)
            .map_err(|e| RsgrepError::Embedding(format!("failed to load weights: {}", e)))?
      };

      let bert = BertModel::load(vb.clone(), &config)
         .map_err(|e| RsgrepError::Embedding(format!("failed to load colbert model: {}", e)))?;

      let projection = candle_nn::linear_no_bias(config.hidden_size, COLBERT_DIM, vb.pp("linear"))
         .map_err(|e| RsgrepError::Embedding(format!("failed to load projection: {}", e)))?;

      if debug_models() {
         tracing::info!(
            "colbert model loaded (hidden={}, proj={})",
            config.hidden_size,
            COLBERT_DIM
         );
      }

      Ok((bert, projection, tokenizer))
   }

   fn download_model(model_id: &str) -> Result<PathBuf> {
      let cache_dir = crate::config::model_dir();
      std::fs::create_dir_all(&cache_dir)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create model cache: {}", e)))?;

      let api = Api::new()
         .map_err(|e| RsgrepError::Embedding(format!("failed to initialize hf_hub API: {}", e)))?;

      let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

      let model_files = ["config.json", "tokenizer.json", "model.safetensors"];
      let mut paths = Vec::new();

      for filename in &model_files {
         let path = repo.get(filename).map_err(|e| {
            RsgrepError::Embedding(format!(
               "failed to download {} from {}: {}. Run 'rsgrep setup' to download models.",
               filename, model_id, e
            ))
         })?;
         paths.push(path);
      }

      paths[0]
         .parent()
         .ok_or_else(|| RsgrepError::Embedding("invalid model path".to_string()))
         .map(|p| p.to_path_buf())
   }

   fn tokenize_dense(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let state = self.dense_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("dense model not loaded".to_string()))?;

      let encoding = state
         .tokenizer
         .encode(text, true)
         .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];

      if token_ids.len() > MAX_SEQ_LEN_DENSE {
         token_ids.truncate(MAX_SEQ_LEN_DENSE);
         attention_mask.truncate(MAX_SEQ_LEN_DENSE);
      }

      Ok((token_ids, attention_mask))
   }

   fn tokenize_dense_batch(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let state = self.dense_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("dense model not loaded".to_string()))?;

      texts
         .iter()
         .map(|text| {
            let encoding = state
               .tokenizer
               .encode(text.as_str(), true)
               .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = vec![1u32; token_ids.len()];

            if token_ids.len() > MAX_SEQ_LEN_DENSE {
               token_ids.truncate(MAX_SEQ_LEN_DENSE);
               attention_mask.truncate(MAX_SEQ_LEN_DENSE);
            }

            Ok((token_ids, attention_mask))
         })
         .collect()
   }

   fn tokenize_colbert(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let state = self.colbert_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("colbert model not loaded".to_string()))?;

      let encoding = state
         .tokenizer
         .encode(text, true)
         .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];

      if token_ids.len() > MAX_SEQ_LEN_COLBERT {
         token_ids.truncate(MAX_SEQ_LEN_COLBERT);
         attention_mask.truncate(MAX_SEQ_LEN_COLBERT);
      }

      Ok((token_ids, attention_mask))
   }

   fn tokenize_colbert_batch(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let state = self.colbert_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("colbert model not loaded".to_string()))?;

      texts
         .iter()
         .map(|text| {
            let encoding = state
               .tokenizer
               .encode(text.as_str(), true)
               .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = vec![1u32; token_ids.len()];

            if token_ids.len() > MAX_SEQ_LEN_COLBERT {
               token_ids.truncate(MAX_SEQ_LEN_COLBERT);
               attention_mask.truncate(MAX_SEQ_LEN_COLBERT);
            }

            Ok((token_ids, attention_mask))
         })
         .collect()
   }

   fn normalize_l2(embeddings: &mut [f32]) {
      let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
      if norm > 0.0 {
         for x in embeddings.iter_mut() {
            *x /= norm;
         }
      }
   }

   fn quantize_embeddings(embeddings: &[Vec<f32>]) -> (Vec<u8>, f64) {
      if embeddings.is_empty() || embeddings[0].is_empty() {
         return (Vec::new(), 1.0);
      }

      let max_val = embeddings
         .iter()
         .flatten()
         .map(|x| x.abs())
         .fold(0.0f32, f32::max);

      if max_val == 0.0 {
         return (vec![0; embeddings.len() * embeddings[0].len()], 1.0);
      }

      let scale = max_val as f64 / 127.0;
      let quantized: Vec<u8> = embeddings
         .iter()
         .flatten()
         .map(|x| ((x / max_val) * 127.0) as i8 as u8)
         .collect();

      (quantized, scale)
   }

   fn compute_dense_embedding(&self, text: &str) -> Result<Vec<f32>> {
      let (token_ids, attention_mask) = self.tokenize_dense(text)?;

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let state = self.dense_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("dense model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let cls_embedding = embeddings
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to get batch: {}", e)))?
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to extract CLS: {}", e)))?;

      let mut dense_vec: Vec<f32> = cls_embedding
         .to_vec1()
         .map_err(|e| RsgrepError::Embedding(format!("failed to convert to vec: {}", e)))?;

      Self::normalize_l2(&mut dense_vec);
      Ok(dense_vec)
   }

   fn compute_dense_embeddings_batch_inner(
      &self,
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Vec<Vec<f32>>> {
      if tokenized.is_empty() {
         return Ok(Vec::new());
      }

      let max_len = tokenized
         .iter()
         .map(|(ids, _)| ids.len())
         .max()
         .unwrap_or(0);
      let batch_size = tokenized.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for (token_ids, attention_mask) in tokenized {
         all_token_ids.extend(token_ids);
         all_token_ids.extend(vec![0u32; max_len - token_ids.len()]);
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(vec![0u32; max_len - attention_mask.len()]);
      }

      let token_ids_tensor = Tensor::new(&all_token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let state = self.dense_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("dense model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let all_embeddings: Vec<Vec<Vec<f32>>> = embeddings
         .to_vec3()
         .map_err(|e| RsgrepError::Embedding(format!("failed to convert: {}", e)))?;

      let mut results = Vec::with_capacity(batch_size);
      for batch_emb in &all_embeddings {
         let mut dense_vec = batch_emb[0].clone();
         Self::normalize_l2(&mut dense_vec);
         results.push(dense_vec);
      }

      Ok(results)
   }

   fn compute_colbert_embedding(&self, text: &str) -> Result<Vec<Vec<f32>>> {
      let (token_ids, attention_mask) = self.tokenize_colbert(text)?;
      let seq_len = token_ids.len();

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let state = self.colbert_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("colbert model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let projected = state
         .projection
         .forward(&embeddings)
         .map_err(|e| RsgrepError::Embedding(format!("projection failed: {}", e)))?;

      let batch_embeddings = projected
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to get batch: {}", e)))?;

      let mut token_embeddings = Vec::with_capacity(seq_len);
      for i in 0..seq_len {
         let token_emb = batch_embeddings
            .get(i)
            .map_err(|e| RsgrepError::Embedding(format!("failed to extract token {}: {}", i, e)))?;

         let mut vec: Vec<f32> = token_emb
            .to_vec1()
            .map_err(|e| RsgrepError::Embedding(format!("failed to convert token {}: {}", i, e)))?;

         Self::normalize_l2(&mut vec);
         token_embeddings.push(vec);
      }

      Ok(token_embeddings)
   }

   fn compute_colbert_embeddings_batch_inner(
      &self,
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Vec<Vec<Vec<f32>>>> {
      if tokenized.is_empty() {
         return Ok(Vec::new());
      }

      let max_len = tokenized
         .iter()
         .map(|(ids, _)| ids.len())
         .max()
         .unwrap_or(0);
      let batch_size = tokenized.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for (token_ids, attention_mask) in tokenized {
         all_token_ids.extend(token_ids);
         all_token_ids.extend(vec![0u32; max_len - token_ids.len()]);
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(vec![0u32; max_len - attention_mask.len()]);
      }

      let token_ids_tensor = Tensor::new(&all_token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let state = self.colbert_model.read();
      let state = state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("colbert model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let projected = state
         .projection
         .forward(&embeddings)
         .map_err(|e| RsgrepError::Embedding(format!("projection failed: {}", e)))?;

      let projected_f32 = projected
         .to_dtype(DType::F32)
         .map_err(|e| RsgrepError::Embedding(format!("dtype conversion failed: {}", e)))?;

      let all_embeddings: Vec<Vec<Vec<f32>>> = projected_f32
         .to_vec3()
         .map_err(|e| RsgrepError::Embedding(format!("failed to convert: {}", e)))?;

      let mut results = Vec::with_capacity(batch_size);
      for (i, batch_emb) in all_embeddings.into_iter().enumerate() {
         let seq_len = tokenized[i].0.len();
         let mut token_embeddings = Vec::with_capacity(seq_len);

         for mut vec in batch_emb.into_iter().take(seq_len) {
            Self::normalize_l2(&mut vec);
            token_embeddings.push(vec);
         }

         results.push(token_embeddings);
      }

      Ok(results)
   }

   fn compute_hybrid_with_adaptive_batching(
      &self,
      texts: &[String],
   ) -> Result<Vec<HybridEmbedding>> {
      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let mut current_batch_size = self.adaptive_batch_size.load(Ordering::Relaxed);
      let mut all_results = Vec::with_capacity(texts.len());
      let mut offset = 0;

      while offset < texts.len() {
         let end = (offset + current_batch_size).min(texts.len());
         let batch_texts = &texts[offset..end];

         let dense_tokenized = self.tokenize_dense_batch(batch_texts)?;
         let colbert_tokenized = self.tokenize_colbert_batch(batch_texts)?;

         match self.try_compute_batch(&dense_tokenized, &colbert_tokenized) {
            Ok((dense_embeddings, colbert_embeddings)) => {
               for i in 0..batch_texts.len() {
                  let dense = dense_embeddings[i].clone();
                  let colbert_tokens = &colbert_embeddings[i];
                  let (colbert, colbert_scale) = Self::quantize_embeddings(colbert_tokens);
                  all_results.push(HybridEmbedding { dense, colbert, colbert_scale });
               }
               offset = end;
            },
            Err(e) => {
               let err_str = e.to_string();
               if is_oom_error(&err_str) && current_batch_size > MIN_BATCH_SIZE {
                  current_batch_size = self.reduce_batch_size();
               } else {
                  return Err(e);
               }
            },
         }
      }

      Ok(all_results)
   }

   fn try_compute_batch(
      &self,
      dense_tokenized: &[(Vec<u32>, Vec<u32>)],
      colbert_tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)> {
      let dense_embeddings = self.compute_dense_embeddings_batch_inner(dense_tokenized)?;
      let colbert_embeddings = self.compute_colbert_embeddings_batch_inner(colbert_tokenized)?;
      Ok((dense_embeddings, colbert_embeddings))
   }
}

#[async_trait::async_trait]
impl Embedder for CandleEmbedder {
   async fn compute_hybrid(&self, texts: &[String]) -> Result<Vec<HybridEmbedding>> {
      self.ensure_dense_loaded()?;
      self.ensure_colbert_loaded()?;

      self.compute_hybrid_with_adaptive_batching(texts)
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      self.ensure_dense_loaded()?;
      self.ensure_colbert_loaded()?;

      let query_text = if QUERY_PREFIX.is_empty() {
         text.to_string()
      } else {
         format!("{}{}", QUERY_PREFIX, text)
      };

      if debug_embed() {
         tracing::info!("encoding query: {:?}", text);
      }

      let dense = self.compute_dense_embedding(&query_text)?;
      let colbert = self.compute_colbert_embedding(&query_text)?;

      if debug_embed() {
         tracing::info!(
            "query embedding - dense_dim: {}, colbert_tokens: {}",
            dense.len(),
            colbert.len()
         );
      }

      Ok(QueryEmbedding { dense, colbert })
   }

   fn is_ready(&self) -> bool {
      self.dense_model.read().is_some() && self.colbert_model.read().is_some()
   }
}

impl Default for CandleEmbedder {
   fn default() -> Self {
      Self::new().expect("failed to create CandleEmbedder")
   }
}
