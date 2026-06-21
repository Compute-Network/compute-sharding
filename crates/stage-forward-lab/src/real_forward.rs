use crate::prompting::{GemmaPromptMode, format_gemma_prompt};
use crate::real_math::{self, GemmaLayerConfig};
use crate::tokenizer::GemmaTokenizer;
use crate::{
    LayerExecutionProgram, LayerOperatorView, PackedTensorEntry, PayloadKind, StageForwardBackend,
    StageLayout, StageModelView, StageSample, StageTensor, StageTensorStore,
    encode_stage_tensor_bytes, quants, stage_tensor_byte_sections,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

type CachedMatrix = (usize, usize, Arc<Vec<f32>>);
type CachedTensorBytes = Arc<Vec<u8>>;
type PlePrewarmHandle = std::thread::JoinHandle<Result<PlePrewarmData>>;
const REAL_PLE_AUX_MAGIC: [u8; 4] = *b"rpl1";
const REAL_STAGE_AUX_MAGIC: [u8; 4] = *b"rsa1";
const TOKEN_ROW_CACHE_LIMIT: usize = 512;
const PLE_TOKEN_ROW_CACHE_LIMIT: usize = 256;
const PLE_COMBINED_TOKEN_CACHE_LIMIT: usize = 128;
const DECODE_SESSION_LIMIT: usize = 64;
const PREFILL_PREFIX_CACHE_LIMIT: usize = 8;
const TAIL_PREFILL_CACHE_LIMIT: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct RealForwardProfile {
    pub seq_len: usize,
    pub layers: usize,
    pub embed_micros: u128,
    pub aux_micros: u128,
    pub aux_lookup_micros: u128,
    pub aux_project_micros: u128,
    pub aux_combine_micros: u128,
    pub aux_materialize_micros: u128,
    pub attn_micros: u128,
    pub attn_qkv_micros: u128,
    pub attn_core_micros: u128,
    pub attn_out_micros: u128,
    pub ffn_micros: u128,
    pub ffn_gate_up_micros: u128,
    pub ffn_down_micros: u128,
    pub ple_micros: u128,
    pub ple_gate_micros: u128,
    pub ple_proj_micros: u128,
}

#[derive(Debug, Clone, Default)]
struct RealAuxProfile {
    lookup_micros: u128,
    project_micros: u128,
    combine_micros: u128,
    materialize_micros: u128,
}

struct PlePrewarmData {
    matrix_name: String,
    matrix_out: usize,
    matrix_in: usize,
    matrix: Vec<f32>,
    norm_name: String,
    norm: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct LayerAttentionConfig {
    rope_base_theta: f32,
    rope_rotary_dim: usize,
    proportional_rope: bool,
    shared_kv_source_layer: Option<u32>,
    sliding_window: Option<usize>,
}

struct ContinueForwardResult {
    request_id: String,
    hidden_dim: usize,
    stage_trace: Vec<String>,
    states: Vec<Vec<f32>>,
    prompt_aux_bytes: Option<Vec<u8>>,
    max_tokens: Option<u32>,
    profile: RealForwardProfile,
}

struct DecodeSession {
    seq_len: usize,
    attention_cache_by_layer: HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

#[derive(Clone)]
struct PrefillPrefixCacheEntry {
    token_ids: Vec<u32>,
    output_states: Vec<Vec<f32>>,
    attention_cache_by_layer: HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

#[derive(Clone)]
struct TailPrefillCacheEntry {
    input_bytes: Vec<u8>,
    prefix_hashes: Vec<u64>,
    output_states: Vec<Vec<f32>>,
    seq_len: usize,
    attention_cache_by_layer: HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

#[derive(Debug, Clone, Default)]
struct PromptAuxData {
    ple_all: Vec<Vec<Vec<f32>>>,
    prefix_hashes: Vec<u64>,
}

pub struct RealGemmaBackend {
    index_path: std::path::PathBuf,
    layout: Option<StageLayout>,
    store: Option<StageTensorStore>,
    model_view: Option<StageModelView>,
    config: Option<GemmaLayerConfig>,
    rope_freqs: Option<Vec<f32>>,
    token_embd: Option<Vec<u8>>,
    token_embd_type: u32,
    vocab_size: usize,
    vocab_tokens: Option<Vec<String>>,
    tokenizer: Option<GemmaTokenizer>,
    ple_token_embd: Option<Vec<u8>>,
    ple_token_embd_type: u32,
    ple_model_proj_entry: Option<PackedTensorEntry>,
    ple_proj_norm_entry: Option<PackedTensorEntry>,
    ple_dim: usize,
    ple_num_layers: usize,
    debug_layer_cap: Option<usize>,
    debug_vocab_cap: Option<usize>,
    debug_disable_ple: bool,
    raw_tensor_cache: RwLock<HashMap<String, CachedTensorBytes>>,
    vector_cache: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    matrix_cache: RwLock<HashMap<String, CachedMatrix>>,
    token_row_cache: RwLock<HashMap<u32, Arc<Vec<f32>>>>,
    ple_token_row_cache: RwLock<HashMap<u32, Arc<Vec<f32>>>>,
    ple_combined_token_cache: RwLock<HashMap<u32, Arc<Vec<f32>>>>,
    ple_prewarm: Mutex<Option<PlePrewarmHandle>>,
    decode_sessions: Mutex<HashMap<String, DecodeSession>>,
    prefill_prefix_cache: Mutex<Vec<PrefillPrefixCacheEntry>>,
    tail_prefill_cache: Mutex<Vec<TailPrefillCacheEntry>>,
    last_forward_profile: RwLock<Option<RealForwardProfile>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealTailLogitsTrace {
    pub projection_tensor: String,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub selected_token_id: u32,
    pub selected_score: f32,
    pub top_logits: Vec<(u32, f32)>,
    pub state_rms: f32,
}

impl RealGemmaBackend {
    fn should_use_packed_rope_freqs(layout: &StageLayout) -> bool {
        !layout.model_id.contains("gemma-4-e4b")
    }

    pub fn new(index_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            index_path: index_path.into(),
            layout: None,
            store: None,
            model_view: None,
            config: None,
            rope_freqs: None,
            token_embd: None,
            token_embd_type: 0,
            vocab_size: 0,
            vocab_tokens: None,
            tokenizer: None,
            ple_token_embd: None,
            ple_token_embd_type: 0,
            ple_model_proj_entry: None,
            ple_proj_norm_entry: None,
            ple_dim: 0,
            ple_num_layers: 0,
            debug_layer_cap: None,
            debug_vocab_cap: None,
            debug_disable_ple: false,
            raw_tensor_cache: RwLock::new(HashMap::new()),
            vector_cache: RwLock::new(HashMap::new()),
            matrix_cache: RwLock::new(HashMap::new()),
            token_row_cache: RwLock::new(HashMap::new()),
            ple_token_row_cache: RwLock::new(HashMap::new()),
            ple_combined_token_cache: RwLock::new(HashMap::new()),
            ple_prewarm: Mutex::new(None),
            decode_sessions: Mutex::new(HashMap::new()),
            prefill_prefix_cache: Mutex::new(Vec::new()),
            tail_prefill_cache: Mutex::new(Vec::new()),
            last_forward_profile: RwLock::new(None),
        }
    }

    pub fn set_debug_layer_cap(&mut self, cap: Option<usize>) {
        self.debug_layer_cap = cap;
    }

    pub fn set_debug_vocab_cap(&mut self, cap: Option<usize>) {
        self.debug_vocab_cap = cap;
    }

    pub fn set_debug_disable_ple(&mut self, disable: bool) {
        self.debug_disable_ple = disable;
    }

    pub fn last_forward_profile(&self) -> Option<RealForwardProfile> {
        self.last_forward_profile
            .read()
            .expect("real-forward profile cache poisoned")
            .clone()
    }

    pub fn load_vocab_json(&mut self, path: &std::path::Path) -> Result<()> {
        let data = std::fs::read(path)?;
        let tokens: Vec<String> = serde_json::from_slice(&data)?;
        self.vocab_tokens = Some(tokens);
        Ok(())
    }

    pub fn set_vocab(&mut self, tokens: Vec<String>) {
        self.vocab_tokens = Some(tokens);
    }

    pub fn load_tokenizer(
        &mut self,
        vocab_path: &std::path::Path,
        scores_path: Option<&std::path::Path>,
    ) -> Result<()> {
        let tok = GemmaTokenizer::load(vocab_path, scores_path)?;
        self.vocab_tokens = Some(tok.id_to_token().to_vec());
        self.tokenizer = Some(tok);
        Ok(())
    }

    fn tokenize_prompt(&self, prompt: &str) -> Vec<u32> {
        if let Some(tok) = &self.tokenizer {
            tok.encode_with_bos(prompt)
        } else {
            Self::simple_tokenize(prompt, self.vocab_size.max(1))
        }
    }

    pub fn tokenize_text(&self, text: &str) -> Vec<u32> {
        self.tokenize_prompt(text)
    }

    pub fn tokenize_prompt_mode(&self, prompt: &str, mode: GemmaPromptMode) -> Vec<u32> {
        self.tokenize_prompt(&format_gemma_prompt(mode, prompt))
    }

    pub fn tokenize_generation_prompt(&self, prompt: &str) -> Vec<u32> {
        self.tokenize_prompt_mode(prompt, GemmaPromptMode::GemmaInstruct)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.as_ref().map(GemmaTokenizer::eos_id)
    }

    pub fn decode_token_ids(&self, ids: &[u32]) -> String {
        if let Some(tok) = &self.tokenizer {
            tok.decode_ids(ids)
        } else {
            ids.iter()
                .map(|id| self.decode_token(*id))
                .collect::<Vec<_>>()
                .join("")
        }
    }

    pub fn decode_hidden_states_payload(bytes: &[u8], hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
        Self::decode_hidden_states(bytes, hidden_dim)
    }

    pub fn decode_last_hidden_state_payload(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        Self::decode_hidden_state(bytes, hidden_dim)
    }

    fn decode_token(&self, id: u32) -> String {
        if let Some(vocab) = &self.vocab_tokens {
            if (id as usize) < vocab.len() {
                return vocab[id as usize].clone();
            }
        }
        if id < 128 {
            return (id as u8 as char).to_string();
        }
        format!("<{}>", id)
    }

    fn take_decode_session(&self, request_id: &str) -> Option<DecodeSession> {
        self.decode_sessions
            .lock()
            .expect("real-forward decode session cache poisoned")
            .remove(request_id)
    }

    fn store_decode_session(&self, request_id: String, session: DecodeSession) {
        let mut sessions = self
            .decode_sessions
            .lock()
            .expect("real-forward decode session cache poisoned");
        if !sessions.contains_key(&request_id) && sessions.len() >= DECODE_SESSION_LIMIT {
            sessions.clear();
        }
        sessions.insert(request_id, session);
    }

    pub fn clear_decode_session(&self, request_id: &str) {
        self.decode_sessions
            .lock()
            .expect("real-forward decode session cache poisoned")
            .remove(request_id);
    }

    fn prefix_hashes(token_ids: &[u32]) -> Vec<u64> {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        let mut out = Vec::with_capacity(token_ids.len());
        for &token_id in token_ids {
            for byte in token_id.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            out.push(hash);
        }
        out
    }

    fn shared_prefix_hash_len(left: &[u64], right: &[u64]) -> usize {
        left.iter()
            .zip(right.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    fn matching_prefix_len(left: &[u32], right: &[u32]) -> usize {
        left.iter()
            .zip(right.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    fn find_prefill_prefix(&self, token_ids: &[u32]) -> Option<(PrefillPrefixCacheEntry, usize)> {
        self.prefill_prefix_cache
            .lock()
            .expect("real-forward prefill prefix cache poisoned")
            .iter()
            .filter_map(|entry| {
                let prefix_len = Self::matching_prefix_len(token_ids, &entry.token_ids);
                (prefix_len > 0).then(|| (entry.clone(), prefix_len))
            })
            .max_by_key(|(_, prefix_len)| *prefix_len)
    }

    fn store_prefill_prefix(
        &self,
        token_ids: &[u32],
        output_states: &[Vec<f32>],
        attention_cache_by_layer: &HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    ) {
        let mut cache = self
            .prefill_prefix_cache
            .lock()
            .expect("real-forward prefill prefix cache poisoned");
        if let Some(existing_idx) = cache.iter().position(|entry| entry.token_ids == token_ids) {
            cache.remove(existing_idx);
        } else if cache.len() >= PREFILL_PREFIX_CACHE_LIMIT {
            cache.remove(0);
        }
        cache.push(PrefillPrefixCacheEntry {
            token_ids: token_ids.to_vec(),
            output_states: output_states.to_vec(),
            attention_cache_by_layer: attention_cache_by_layer.clone(),
        });
    }

    fn find_tail_prefill(&self, input_bytes: &[u8]) -> Option<TailPrefillCacheEntry> {
        self.tail_prefill_cache
            .lock()
            .expect("real-forward tail prefill cache poisoned")
            .iter()
            .find(|entry| entry.input_bytes == input_bytes)
            .cloned()
    }

    fn find_tail_prefill_by_prefix_hashes(
        &self,
        prefix_hashes: &[u64],
    ) -> Option<(TailPrefillCacheEntry, usize)> {
        self.tail_prefill_cache
            .lock()
            .expect("real-forward tail prefill cache poisoned")
            .iter()
            .filter_map(|entry| {
                let prefix_len = Self::shared_prefix_hash_len(prefix_hashes, &entry.prefix_hashes);
                (prefix_len > 0).then(|| (entry.clone(), prefix_len))
            })
            .max_by_key(|(_, prefix_len)| *prefix_len)
    }

    fn store_tail_prefill(
        &self,
        input_bytes: &[u8],
        prefix_hashes: &[u64],
        output_states: &[Vec<f32>],
        seq_len: usize,
        attention_cache_by_layer: &HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    ) {
        let mut cache = self
            .tail_prefill_cache
            .lock()
            .expect("real-forward tail prefill cache poisoned");
        if let Some(existing_idx) = cache
            .iter()
            .position(|entry| entry.input_bytes == input_bytes)
        {
            cache.remove(existing_idx);
        } else if cache.len() >= TAIL_PREFILL_CACHE_LIMIT {
            cache.remove(0);
        }
        cache.push(TailPrefillCacheEntry {
            input_bytes: input_bytes.to_vec(),
            prefix_hashes: prefix_hashes.to_vec(),
            output_states: output_states.to_vec(),
            seq_len,
            attention_cache_by_layer: attention_cache_by_layer.clone(),
        });
    }

    fn truncate_attention_cache(
        attention_cache_by_layer: &HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
        prefix_len: usize,
    ) -> HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        attention_cache_by_layer
            .iter()
            .map(|(&layer_idx, (k_cache, v_cache))| {
                (
                    layer_idx,
                    (
                        k_cache.iter().take(prefix_len).cloned().collect(),
                        v_cache.iter().take(prefix_len).cloned().collect(),
                    ),
                )
            })
            .collect()
    }

    fn layout(&self) -> Result<&StageLayout> {
        self.layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No layout loaded"))
    }

    fn store(&self) -> Result<&StageTensorStore> {
        self.store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No store loaded"))
    }

    fn model_view(&self) -> Result<&StageModelView> {
        self.model_view
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model view loaded"))
    }

    fn config(&self) -> Result<&GemmaLayerConfig> {
        self.config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No config loaded"))
    }

    fn decode_f32_vector(store: &StageTensorStore, entry: &PackedTensorEntry) -> Result<Vec<f32>> {
        if entry.ggml_type != quants::GGML_TYPE_F32 {
            bail!(
                "Expected F32 tensor for {}, got type {}",
                entry.name,
                entry.ggml_type
            );
        }
        let bytes = store.read(&entry.name)?;
        quants::dequantize_f32_tensor(&bytes)
    }

    fn read_tensor_cached(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<CachedTensorBytes> {
        {
            let cache = self
                .raw_tensor_cache
                .read()
                .expect("real-forward raw tensor cache poisoned");
            if let Some(cached) = cache.get(&entry.name) {
                return Ok(Arc::clone(cached));
            }
        }
        let raw = Arc::new(store.read(&entry.name)?);
        self.raw_tensor_cache
            .write()
            .expect("real-forward raw tensor cache poisoned")
            .insert(entry.name.clone(), Arc::clone(&raw));
        Ok(raw)
    }

    fn decode_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<(usize, usize, Vec<f32>)> {
        if entry.dimensions.len() != 2 {
            bail!(
                "Expected 2D tensor for {}, got {}D",
                entry.name,
                entry.dimensions.len()
            );
        }
        let in_dim = entry.dimensions[0] as usize;
        let out_dim = entry.dimensions[1] as usize;
        let bytes = store.read(&entry.name)?;
        let matrix = quants::dequantize_tensor(entry.ggml_type, &bytes)?;
        if matrix.len() != in_dim * out_dim {
            bail!(
                "Matrix {} decoded to {} elements but expected {}",
                entry.name,
                matrix.len(),
                in_dim * out_dim
            );
        }
        Ok((out_dim, in_dim, matrix))
    }

    fn decode_f32_vector_cached(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<Arc<Vec<f32>>> {
        {
            let cache = self
                .vector_cache
                .read()
                .expect("real-forward vector cache poisoned");
            if let Some(cached) = cache.get(&entry.name) {
                return Ok(Arc::clone(cached));
            }
        }
        let raw = self.read_tensor_cached(store, entry)?;
        let decoded = Arc::new(quants::dequantize_f32_tensor(&raw)?);
        self.vector_cache
            .write()
            .expect("real-forward vector cache poisoned")
            .insert(entry.name.clone(), Arc::clone(&decoded));
        Ok(decoded)
    }

    fn start_ple_prewarm(&self) {
        let Some(ple_model_proj_entry) = self.ple_model_proj_entry.clone() else {
            return;
        };
        let Some(ple_proj_norm_entry) = self.ple_proj_norm_entry.clone() else {
            return;
        };
        let mut guard = self
            .ple_prewarm
            .lock()
            .expect("real-forward PLE prewarm mutex poisoned");
        if guard.is_some() {
            return;
        }
        let index_path = self.index_path.clone();
        *guard = Some(std::thread::spawn(move || -> Result<PlePrewarmData> {
            let store = StageTensorStore::load(&index_path)?;
            let (matrix_out, matrix_in, matrix) =
                RealGemmaBackend::decode_matrix(&store, &ple_model_proj_entry)?;
            let norm = RealGemmaBackend::decode_f32_vector(&store, &ple_proj_norm_entry)?;
            Ok(PlePrewarmData {
                matrix_name: ple_model_proj_entry.name,
                matrix_out,
                matrix_in,
                matrix,
                norm_name: ple_proj_norm_entry.name,
                norm,
            })
        }));
    }

    fn finish_ple_prewarm(&self, wait: bool) -> Result<()> {
        let handle = {
            let mut guard = self
                .ple_prewarm
                .lock()
                .expect("real-forward PLE prewarm mutex poisoned");
            let ready = guard
                .as_ref()
                .map(|handle| wait || handle.is_finished())
                .unwrap_or(false);
            if ready { guard.take() } else { None }
        };

        if let Some(handle) = handle {
            let data = handle
                .join()
                .map_err(|_| anyhow::anyhow!("PLE prewarm thread panicked"))??;
            self.matrix_cache
                .write()
                .expect("real-forward matrix cache poisoned")
                .insert(
                    data.matrix_name,
                    (data.matrix_out, data.matrix_in, Arc::new(data.matrix)),
                );
            self.vector_cache
                .write()
                .expect("real-forward vector cache poisoned")
                .insert(data.norm_name, Arc::new(data.norm));
        }
        Ok(())
    }

    fn decode_matrix_cached(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<CachedMatrix> {
        {
            let cache = self
                .matrix_cache
                .read()
                .expect("real-forward matrix cache poisoned");
            if let Some(cached) = cache.get(&entry.name) {
                return Ok((cached.0, cached.1, Arc::clone(&cached.2)));
            }
        }
        let (out_dim, in_dim, matrix) = Self::decode_matrix(store, entry)?;
        let decoded = (out_dim, in_dim, Arc::new(matrix));
        self.matrix_cache
            .write()
            .expect("real-forward matrix cache poisoned")
            .insert(entry.name.clone(), decoded.clone());
        Ok(decoded)
    }

    fn matmul_entry_many(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        self.matmul_entry_many_refs_range(
            store,
            entry,
            &input_refs,
            0,
            entry.dimensions[1] as usize,
        )
    }

    fn matmul_entry_many_refs_range(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        inputs: &[&[f32]],
        row_start: usize,
        row_count: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if entry.dimensions.len() != 2 {
            bail!(
                "Expected 2D tensor for {}, got {}D",
                entry.name,
                entry.dimensions.len()
            );
        }
        let in_dim = entry.dimensions[0] as usize;
        let out_dim = entry.dimensions[1] as usize;
        if row_start + row_count > out_dim {
            bail!(
                "Requested rows {}..{} from {} with out_dim {}",
                row_start,
                row_start + row_count,
                entry.name,
                out_dim
            );
        }

        match entry.ggml_type {
            quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => {
                let raw = self.read_tensor_cached(store, entry)?;
                real_math::matmul_quantized_many_refs_range(
                    entry.ggml_type,
                    &raw,
                    inputs,
                    row_start,
                    row_count,
                    in_dim,
                )
            }
            _ => {
                let (_, _, matrix) = self.decode_matrix_cached(store, entry)?;
                Ok(real_math::matmul_many_refs_range(
                    &matrix, inputs, row_start, row_count, in_dim,
                ))
            }
        }
    }

    fn matmul_entry_many_refs_range_token_major(
        &self,
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        inputs: &[&[f32]],
        row_start: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        if entry.dimensions.len() != 2 {
            bail!(
                "Expected 2D tensor for {}, got {}D",
                entry.name,
                entry.dimensions.len()
            );
        }
        let in_dim = entry.dimensions[0] as usize;
        let out_dim = entry.dimensions[1] as usize;
        if row_start + row_count > out_dim {
            bail!(
                "Requested rows {}..{} from {} with out_dim {}",
                row_start,
                row_start + row_count,
                entry.name,
                out_dim
            );
        }

        match entry.ggml_type {
            quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => {
                let raw = self.read_tensor_cached(store, entry)?;
                real_math::matmul_quantized_many_refs_range_token_major(
                    entry.ggml_type,
                    &raw,
                    inputs,
                    row_start,
                    row_count,
                    in_dim,
                )
            }
            _ => {
                let (_, _, matrix) = self.decode_matrix_cached(store, entry)?;
                Ok(real_math::matmul_many_refs_range_token_major(
                    &matrix, inputs, row_start, row_count, in_dim,
                ))
            }
        }
    }

    fn matmul_entry_many_pair_token_major(
        &self,
        store: &StageTensorStore,
        entry_a: &PackedTensorEntry,
        entry_b: &PackedTensorEntry,
        inputs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if entry_a.dimensions.len() != 2 || entry_b.dimensions.len() != 2 {
            bail!(
                "Expected 2D tensors for paired matmul, got {}D and {}D",
                entry_a.dimensions.len(),
                entry_b.dimensions.len()
            );
        }
        if entry_a.dimensions != entry_b.dimensions {
            bail!(
                "Paired matmul requires matching dimensions, got {:?} and {:?}",
                entry_a.dimensions,
                entry_b.dimensions
            );
        }
        let in_dim = entry_a.dimensions[0] as usize;
        let row_count = entry_a.dimensions[1] as usize;

        if entry_a.ggml_type == quants::GGML_TYPE_Q4_K
            && entry_b.ggml_type == quants::GGML_TYPE_Q4_K
        {
            let raw_a = self.read_tensor_cached(store, entry_a)?;
            let raw_b = self.read_tensor_cached(store, entry_b)?;
            real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
                &raw_a, &raw_b, inputs, 0, row_count, in_dim,
            )
        } else {
            Ok((
                self.matmul_entry_many_refs_range_token_major(
                    store, entry_a, inputs, 0, row_count,
                )?,
                self.matmul_entry_many_refs_range_token_major(
                    store, entry_b, inputs, 0, row_count,
                )?,
            ))
        }
    }

    fn logits_entry<'a>(
        store: &'a StageTensorStore,
        model_view: &'a StageModelView,
    ) -> Option<&'a PackedTensorEntry> {
        store
            .entry("output.weight")
            .or_else(|| {
                model_view
                    .tail_only
                    .iter()
                    .find(|entry| entry.name == "output.weight")
            })
            .or_else(|| store.entry("token_embd.weight"))
    }

    fn logits_vocab_size(
        entry: &PackedTensorEntry,
        hidden_dim: usize,
        vocab_rows: Option<usize>,
    ) -> Result<(usize, usize)> {
        if entry.dimensions.len() != 2 {
            bail!(
                "Expected 2D logits tensor for {}, got {}D",
                entry.name,
                entry.dimensions.len()
            );
        }
        let raw_vocab_size = if entry.dimensions[0] as usize == hidden_dim {
            entry.dimensions[1] as usize
        } else if entry.dimensions[1] as usize == hidden_dim {
            entry.dimensions[0] as usize
        } else {
            bail!(
                "Logits tensor {} dimensions {:?} do not match hidden_dim {}",
                entry.name,
                entry.dimensions,
                hidden_dim
            );
        };
        let vocab_size = vocab_rows
            .map(|cap| cap.min(raw_vocab_size))
            .unwrap_or(raw_vocab_size);
        Ok((raw_vocab_size, vocab_size))
    }

    fn embed_tokens(&self, token_ids: &[u32], hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
        let embd_raw = self
            .token_embd
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No token embedding loaded"))?;
        let mut embeddings = Vec::with_capacity(token_ids.len());
        let scale = (hidden_dim as f32).sqrt();
        for &tid in token_ids {
            let decoded = Self::decode_row_cached(
                &self.token_row_cache,
                self.token_embd_type,
                embd_raw,
                tid,
                hidden_dim,
                TOKEN_ROW_CACHE_LIMIT,
            )?;
            let mut row = decoded.as_ref().clone();
            for value in &mut row {
                *value *= scale;
            }
            embeddings.push(row);
        }
        Ok(embeddings)
    }

    pub fn trace_tail_logits(
        &self,
        input: &StageTensor,
        top_k: usize,
    ) -> Result<RealTailLogitsTrace> {
        let (_, trace) = self.sample_tail_with_trace(input.clone(), top_k)?;
        Ok(trace)
    }

    pub fn sample_tail_with_trace(
        &self,
        input: StageTensor,
        top_k: usize,
    ) -> Result<(StageSample, RealTailLogitsTrace)> {
        if !self.layout()?.is_tail {
            bail!("Only the tail stage may trace logits");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail logits trace requires hidden-state payloads");
        }

        let state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;
        self.sample_hidden_state_with_trace(
            input.request_id,
            input.hidden_dim,
            input.max_tokens,
            state,
            top_k,
        )
    }

    fn sample_hidden_state_with_trace(
        &self,
        request_id: String,
        hidden_dim: usize,
        max_tokens: Option<u32>,
        mut state: Vec<f32>,
        top_k: usize,
    ) -> Result<(StageSample, RealTailLogitsTrace)> {
        let layout = self.layout()?;
        let store = self.store()?;
        let model_view = self.model_view()?;
        let config = self.config()?;

        if let Some(entry) = model_view
            .tail_only
            .iter()
            .find(|e| e.name == "output_norm.weight")
        {
            let w = Self::decode_f32_vector(store, entry)?;
            real_math::rms_norm_inplace(&mut state, &w, config.eps);
        }

        let entry = Self::logits_entry(store, model_view)
            .ok_or_else(|| anyhow::anyhow!("Tail stage has no logits tensor"))?;
        let (_, vocab_size) = Self::logits_vocab_size(entry, hidden_dim, self.debug_vocab_cap)?;
        if vocab_size == 0 {
            bail!(
                "Tail logits tensor {} has an empty vocab dimension",
                entry.name
            );
        }

        let raw = self.read_tensor_cached(store, entry)?;
        let logits = real_math::matmul_raw_top_k(
            entry.ggml_type,
            &raw,
            &state,
            0,
            vocab_size,
            hidden_dim,
            top_k,
            config.logit_softcap,
        )?;
        let selected = logits.argmax_idx;
        let max_tokens_count = max_tokens.unwrap_or(1).max(1) as usize;
        let token_ids = vec![selected as u32; max_tokens_count];
        let text = if let Some(tok) = &self.tokenizer {
            tok.decode_ids(&token_ids)
        } else {
            token_ids
                .iter()
                .map(|&id| self.decode_token(id))
                .collect::<Vec<_>>()
                .join("")
        };
        let top_logits = logits
            .top
            .into_iter()
            .map(|(id, score)| (id as u32, score))
            .collect();
        let state_rms =
            (state.iter().map(|v| v * v).sum::<f32>() / state.len().max(1) as f32).sqrt();

        let trace = RealTailLogitsTrace {
            projection_tensor: entry.name.clone(),
            hidden_dim,
            vocab_size,
            selected_token_id: selected as u32,
            selected_score: logits.argmax_score,
            top_logits,
            state_rms,
        };
        let completion_tokens = token_ids.len() as u32;
        let sample = StageSample {
            request_id,
            model_id: layout.model_id.clone(),
            text,
            token_ids,
            completion_tokens,
        };

        Ok((sample, trace))
    }

    fn begin_token_sequence(
        &self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        let config = self.config()?;

        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }

        if token_ids.is_empty() {
            bail!("Empty prompt");
        }

        let hidden_dim = config.hidden_dim;
        let mut session = if token_ids.len() == 1 && self.ple_dim > 0 {
            self.take_decode_session(request_id)
        } else {
            self.take_decode_session(request_id);
            None
        };
        let had_decode_session = session.is_some();
        let cached_prefill = if session.is_none() && token_ids.len() > 1 {
            self.find_prefill_prefix(token_ids)
        } else {
            None
        };
        let prefill_prefix_len = cached_prefill
            .as_ref()
            .map(|(_, prefix_len)| *prefix_len)
            .unwrap_or(0);
        let position_offset = session
            .as_ref()
            .map(|session| session.seq_len as u32)
            .unwrap_or(prefill_prefix_len as u32);
        let mut profile = RealForwardProfile {
            seq_len: token_ids.len(),
            ..Default::default()
        };

        let embed_start = Instant::now();
        let mut embedded_states = self.embed_tokens(token_ids, hidden_dim)?;
        profile.embed_micros = embed_start.elapsed().as_micros();

        let prompt_aux = if !self.debug_disable_ple && self.ple_dim > 0 {
            let aux_start = Instant::now();
            let (prompt_aux, aux_profile) =
                self.compute_ple_inputs(token_ids, &embedded_states, 0, self.ple_num_layers)?;
            profile.aux_micros = aux_start.elapsed().as_micros();
            profile.aux_lookup_micros = aux_profile.lookup_micros;
            profile.aux_project_micros = aux_profile.project_micros;
            profile.aux_combine_micros = aux_profile.combine_micros;
            profile.aux_materialize_micros = aux_profile.materialize_micros;
            Some(prompt_aux)
        } else {
            None
        };
        let mut attention_cache_by_layer = session
            .take()
            .map(|session| session.attention_cache_by_layer)
            .or_else(|| {
                cached_prefill.as_ref().map(|(entry, prefix_len)| {
                    Self::truncate_attention_cache(&entry.attention_cache_by_layer, *prefix_len)
                })
            })
            .unwrap_or_default();
        let mut output_states = cached_prefill
            .as_ref()
            .map(|(entry, prefix_len)| entry.output_states[..*prefix_len].to_vec())
            .unwrap_or_default();
        let mut suffix_states = if prefill_prefix_len == 0 {
            embedded_states
        } else {
            embedded_states.split_off(prefill_prefix_len)
        };
        if !suffix_states.is_empty() {
            self.run_stage_layers(
                &mut suffix_states,
                prompt_aux.as_ref(),
                prefill_prefix_len,
                position_offset,
                &mut attention_cache_by_layer,
                &mut profile,
            )?;
        }
        if output_states.is_empty() {
            output_states = suffix_states;
        } else {
            output_states.extend(suffix_states);
        }
        self.store_decode_session(
            request_id.to_string(),
            DecodeSession {
                seq_len: if had_decode_session {
                    position_offset as usize + token_ids.len()
                } else {
                    token_ids.len()
                },
                attention_cache_by_layer: attention_cache_by_layer.clone(),
            },
        );
        if token_ids.len() > 1 {
            self.store_prefill_prefix(token_ids, &output_states, &attention_cache_by_layer);
        }

        let prefix_hashes = if token_ids.len() > 1 {
            Self::prefix_hashes(token_ids)
        } else {
            Vec::new()
        };
        let prompt_aux_bytes =
            Self::encode_prompt_aux(prompt_aux.as_deref(), self.ple_dim, &prefix_hashes)?;
        let bytes = Self::encode_hidden_states_payload(&output_states, prompt_aux_bytes.as_deref());
        *self
            .last_forward_profile
            .write()
            .expect("real-forward profile cache poisoned") = Some(profile);

        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim,
            bytes,
            prompt_text: None,
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    pub fn begin_token_ids(
        &self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        self.begin_token_sequence(request_id, token_ids, max_tokens)
    }

    fn sample_hidden_state(
        &self,
        request_id: String,
        hidden_dim: usize,
        max_tokens: Option<u32>,
        state: Vec<f32>,
    ) -> Result<StageSample> {
        let layout = self.layout()?;
        let store = self.store()?;
        let model_view = self.model_view()?;
        if Self::logits_entry(store, model_view).is_none() {
            return Ok(StageSample {
                request_id,
                model_id: layout.model_id.clone(),
                text: "?".to_string(),
                token_ids: vec!['?' as u32],
                completion_tokens: 1,
            });
        }
        let (sample, _) =
            self.sample_hidden_state_with_trace(request_id, hidden_dim, max_tokens, state, 0)?;
        Ok(sample)
    }

    fn compute_ple_inputs(
        &self,
        token_ids: &[u32],
        inputs_embeds: &[Vec<f32>],
        layer_start: usize,
        layer_count: usize,
    ) -> Result<(Vec<Vec<Vec<f32>>>, RealAuxProfile)> {
        let ple_raw = self
            .ple_token_embd
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No PLE token embedding loaded"))?;
        let ple_dim = self.ple_dim;
        let num_layers = self.ple_num_layers;
        if ple_dim == 0 || num_layers == 0 {
            bail!(
                "PLE not configured (ple_dim={}, num_layers={})",
                ple_dim,
                num_layers
            );
        }
        let layer_end = (layer_start + layer_count).min(num_layers);
        if layer_start >= layer_end {
            return Ok((vec![Vec::new(); token_ids.len()], RealAuxProfile::default()));
        }
        let active_layer_count = layer_end - layer_start;
        let total_ple_dim = num_layers * ple_dim;
        let seq_len = token_ids.len();
        let mut missing_token_ids = Vec::new();
        let mut missing_inputs = Vec::new();
        let mut seen_missing = HashSet::new();
        let lookup_start = Instant::now();
        {
            let cache = self
                .ple_combined_token_cache
                .read()
                .expect("real-forward PLE combined token cache poisoned");
            for (token_idx, &token_id) in token_ids.iter().enumerate() {
                if !cache.contains_key(&token_id) && seen_missing.insert(token_id) {
                    missing_token_ids.push(token_id);
                    missing_inputs.push(inputs_embeds[token_idx].as_slice());
                }
            }
        }
        let mut aux_profile = RealAuxProfile {
            lookup_micros: lookup_start.elapsed().as_micros(),
            ..Default::default()
        };

        if !missing_inputs.is_empty() {
            let project_start = Instant::now();
            self.finish_ple_prewarm(true)?;
            let store = self.store()?;
            let embed_scale = (ple_dim as f32).sqrt();
            let proj_scale = (inputs_embeds[0].len() as f32).powf(-0.5);
            let combine_scale = (2.0f32).powf(-0.5);
            let ple_model_proj_entry = self
                .ple_model_proj_entry
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No PLE model projection loaded"))?;
            let ple_proj_norm_entry = self
                .ple_proj_norm_entry
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No PLE projection norm loaded"))?;
            let (proj_out, proj_in, proj_mat) =
                self.decode_matrix_cached(store, ple_model_proj_entry)?;
            let proj_norm = self.decode_f32_vector_cached(store, ple_proj_norm_entry)?;
            let proj_row_start = layer_start * ple_dim;
            let proj_row_count = (layer_end - layer_start) * ple_dim;
            if proj_row_start + proj_row_count > proj_out {
                bail!(
                    "PLE projection rows {}..{} exceed projection output {}",
                    proj_row_start,
                    proj_row_start + proj_row_count,
                    proj_out
                );
            }
            let full_context_proj_all = real_math::matmul_many_refs_range(
                &proj_mat,
                &missing_inputs,
                0,
                total_ple_dim,
                proj_in,
            );
            aux_profile.project_micros = project_start.elapsed().as_micros();

            let combine_start = Instant::now();
            let mut cache = self
                .ple_combined_token_cache
                .write()
                .expect("real-forward PLE combined token cache poisoned");
            if cache.len() + missing_token_ids.len() > PLE_COMBINED_TOKEN_CACHE_LIMIT {
                cache.clear();
            }
            for (token_id, mut combined) in missing_token_ids
                .iter()
                .copied()
                .zip(full_context_proj_all.into_iter())
            {
                if cache.contains_key(&token_id) {
                    continue;
                }
                let token_row = Self::decode_row_cached(
                    &self.ple_token_row_cache,
                    self.ple_token_embd_type,
                    ple_raw,
                    token_id,
                    total_ple_dim,
                    PLE_TOKEN_ROW_CACHE_LIMIT,
                )?;
                for value in &mut combined {
                    *value *= proj_scale;
                }
                real_math::rms_norm_chunked_inplace(&mut combined, ple_dim, &proj_norm, 1e-6);
                for (out, token_value) in combined.iter_mut().zip(token_row.iter()) {
                    *out = (*out + *token_value * embed_scale) * combine_scale;
                }
                cache.insert(token_id, Arc::new(combined));
            }
            aux_profile.combine_micros = combine_start.elapsed().as_micros();
        }

        let materialize_start = Instant::now();
        let mut result: Vec<Vec<Vec<f32>>> = Vec::with_capacity(seq_len);
        let cache = self
            .ple_combined_token_cache
            .read()
            .expect("real-forward PLE combined token cache poisoned");

        for t in 0..seq_len {
            let combined = cache.get(&token_ids[t]).ok_or_else(|| {
                anyhow::anyhow!("Missing cached PLE combined token {}", token_ids[t])
            })?;
            let mut per_layer_slices = Vec::with_capacity(active_layer_count);
            for layer_i in layer_start..layer_end {
                let token_start = layer_i * ple_dim;
                let token_end = token_start + ple_dim;
                per_layer_slices.push(combined[token_start..token_end].to_vec());
            }
            result.push(per_layer_slices);
        }
        aux_profile.materialize_micros = materialize_start.elapsed().as_micros();
        Ok((result, aux_profile))
    }

    fn decode_row_cached(
        cache: &RwLock<HashMap<u32, Arc<Vec<f32>>>>,
        ggml_type: u32,
        raw: &[u8],
        token_id: u32,
        row_elements: usize,
        cache_limit: usize,
    ) -> Result<Arc<Vec<f32>>> {
        {
            let guard = cache.read().expect("real-forward token-row cache poisoned");
            if let Some(cached) = guard.get(&token_id) {
                return Ok(Arc::clone(cached));
            }
        }

        let mut decoded_row = vec![0.0f32; row_elements];
        quants::dequantize_row_into(
            ggml_type,
            raw,
            token_id as usize,
            row_elements,
            &mut decoded_row,
        )?;
        let decoded_row = Arc::new(decoded_row);

        let mut guard = cache
            .write()
            .expect("real-forward token-row cache poisoned");
        if let Some(cached) = guard.get(&token_id) {
            return Ok(Arc::clone(cached));
        }
        if cache_limit > 0 && guard.len() >= cache_limit {
            guard.clear();
        }
        guard.insert(token_id, Arc::clone(&decoded_row));
        Ok(decoded_row)
    }

    fn simple_tokenize(prompt: &str, vocab_size: usize) -> Vec<u32> {
        prompt
            .bytes()
            .map(|b| (b as u32) % vocab_size as u32)
            .collect()
    }

    fn layer_config(layer: &LayerOperatorView, fallback: &GemmaLayerConfig) -> GemmaLayerConfig {
        let hidden_dim = layer
            .attn_q
            .as_ref()
            .and_then(|e| e.dimensions.first().copied())
            .unwrap_or(fallback.hidden_dim as u64) as usize;
        let q_dim = layer
            .attn_q
            .as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or((fallback.n_heads * fallback.head_dim) as u64) as usize;
        let k_dim = layer
            .attn_k
            .as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or((fallback.n_kv_heads * fallback.head_dim) as u64)
            as usize;
        let ffn_dim = layer
            .ffn_up
            .as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(fallback.ffn_dim as u64) as usize;
        let head_dim = layer
            .attn_q_norm
            .as_ref()
            .and_then(|e| e.dimensions.first().copied())
            .or_else(|| {
                layer
                    .attn_k_norm
                    .as_ref()
                    .and_then(|e| e.dimensions.first().copied())
            })
            .map(|dim| dim as usize)
            .filter(|dim| *dim > 0 && q_dim % *dim == 0 && k_dim % *dim == 0)
            .unwrap_or_else(|| {
                GemmaLayerConfig::from_dims(hidden_dim, q_dim, k_dim, ffn_dim).head_dim
            });
        let mut config = GemmaLayerConfig {
            hidden_dim,
            n_heads: (q_dim / head_dim).max(1),
            n_kv_heads: (k_dim / head_dim).max(1),
            head_dim,
            ffn_dim,
            eps: fallback.eps,
            rope_base_theta: fallback.rope_base_theta,
            logit_softcap: fallback.logit_softcap,
        };
        config.eps = fallback.eps;
        config.rope_base_theta = fallback.rope_base_theta;
        config.logit_softcap = fallback.logit_softcap;
        config
    }

    fn layer_attention_config(
        &self,
        layer: &LayerOperatorView,
        config: &GemmaLayerConfig,
    ) -> LayerAttentionConfig {
        Self::layer_attention_config_for_layout(self.layout.as_ref(), layer, config)
    }

    fn layer_attention_config_for_layout(
        layout: Option<&StageLayout>,
        layer: &LayerOperatorView,
        config: &GemmaLayerConfig,
    ) -> LayerAttentionConfig {
        let default = LayerAttentionConfig {
            rope_base_theta: config.rope_base_theta,
            rope_rotary_dim: config.head_dim,
            proportional_rope: false,
            shared_kv_source_layer: None,
            sliding_window: None,
        };

        let Some(layout) = layout else {
            return default;
        };

        if !layout.model_id.contains("gemma-4-e4b")
            || config.hidden_dim != 2560
            || !matches!(config.head_dim, 256 | 512)
        {
            return default;
        }

        if layer.layer_index % 6 == 5 {
            LayerAttentionConfig {
                rope_base_theta: 1_000_000.0,
                rope_rotary_dim: config.head_dim / 4,
                proportional_rope: true,
                shared_kv_source_layer: if layer.layer_index >= 24 {
                    Some(23)
                } else {
                    None
                },
                sliding_window: None,
            }
        } else {
            LayerAttentionConfig {
                rope_base_theta: 10_000.0,
                rope_rotary_dim: config.head_dim,
                proportional_rope: false,
                shared_kv_source_layer: if layer.layer_index >= 24 {
                    Some(22)
                } else {
                    None
                },
                sliding_window: Some(512),
            }
        }
    }

    fn validate_stage_split_constraints(
        layout: &StageLayout,
        model_view: &StageModelView,
        config: &GemmaLayerConfig,
    ) -> Result<()> {
        for layer in &model_view.operator_layers {
            let attention_config =
                Self::layer_attention_config_for_layout(Some(layout), layer, config);
            if let Some(source_layer) = attention_config.shared_kv_source_layer {
                if source_layer < layout.start_layer || source_layer > layout.end_layer {
                    bail!(
                        "real_forward stage {} (layers {}-{}) is unsupported for {}: layer {} requires shared KV from layer {} outside this stage; current contract keeps shared-KV caches stage-local",
                        layout.stage_id,
                        layout.start_layer,
                        layout.end_layer,
                        layout.model_id,
                        layer.layer_index,
                        source_layer
                    );
                }
            }
        }
        Ok(())
    }

    fn forward_layer_seq(
        &self,
        states: &mut [Vec<f32>],
        layer: &LayerOperatorView,
        _program: &LayerExecutionProgram,
        position_offset: u32,
        ple_inputs: Option<&[&[f32]]>,
        existing_layer_cache: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
        shared_attention_cache: Option<&(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
        profile: &mut RealForwardProfile,
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
        let store = self.store()?;
        let base_config = self.config()?;
        let config = Self::layer_config(layer, base_config);
        let attention_config = self.layer_attention_config(layer, &config);
        let seq_len = states.len();

        let attn_norm_weight = layer
            .attn_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let q_matrix = layer.attn_q.as_ref();
        let k_matrix = layer.attn_k.as_ref();
        let v_matrix = layer.attn_v.as_ref();
        let q_norm_weight = layer
            .attn_q_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let k_norm_weight = layer
            .attn_k_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let o_matrix = layer.attn_output.as_ref();
        let post_attn_norm_weight = layer
            .post_attention_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let ffn_norm_weight = layer
            .ffn_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let gate_matrix = layer.ffn_gate.as_ref();
        let up_matrix = layer.ffn_up.as_ref();
        let down_matrix = layer.ffn_down.as_ref();
        let post_ffn_norm_weight = layer
            .post_ffw_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let layer_scale = layer
            .layer_output_scale
            .as_ref()
            .map(|e| {
                let bytes = self.read_tensor_cached(store, e).ok()?;
                if bytes.len() >= 4 {
                    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                } else {
                    None
                }
            })
            .flatten();

        let inp_gate_matrix = layer.inp_gate.as_ref();
        let proj_matrix = layer.proj.as_ref();
        let post_norm_weight = layer
            .post_norm
            .as_ref()
            .map(|e| self.decode_f32_vector_cached(store, e))
            .transpose()?;
        let has_ple = !self.debug_disable_ple
            && inp_gate_matrix.is_some()
            && proj_matrix.is_some()
            && ple_inputs.is_some();

        let has_attn = q_matrix.is_some()
            && (shared_attention_cache.is_some() || (k_matrix.is_some() && v_matrix.is_some()));
        let has_ffn = gate_matrix.is_some() && up_matrix.is_some() && down_matrix.is_some();

        let (mut k_cache, mut v_cache) = existing_layer_cache.unwrap_or_default();
        profile.layers += 1;

        if has_attn {
            let attn_start = Instant::now();
            let mut attn_inputs = states.to_vec();
            if let Some(ref w) = attn_norm_weight {
                for input in &mut attn_inputs {
                    real_math::rms_norm_inplace(input, w, config.eps);
                }
            }
            let qkv_start = Instant::now();
            let mut q_all = self.matmul_entry_many(store, q_matrix.unwrap(), &attn_inputs)?;
            if let Some(ref w) = q_norm_weight {
                for q in &mut q_all {
                    real_math::per_head_rms_norm(q, w, config.n_heads, config.head_dim);
                }
            }

            let mut k_all = if shared_attention_cache.is_none() {
                Some(self.matmul_entry_many(store, k_matrix.unwrap(), &attn_inputs)?)
            } else {
                None
            };
            let mut v_all = if shared_attention_cache.is_none() {
                Some(self.matmul_entry_many(store, v_matrix.unwrap(), &attn_inputs)?)
            } else {
                None
            };

            if let Some(ref w) = k_norm_weight {
                if let Some(k_all) = k_all.as_mut() {
                    for k in k_all.iter_mut() {
                        real_math::per_head_rms_norm(k, w, config.n_kv_heads, config.head_dim);
                    }
                }
            }
            if let Some(v_all) = v_all.as_mut() {
                for v in v_all.iter_mut() {
                    real_math::per_head_rms_norm_no_scale(v, config.n_kv_heads, config.head_dim);
                }
            }
            profile.attn_qkv_micros += qkv_start.elapsed().as_micros();

            let mut attn_outputs = Vec::with_capacity(seq_len);
            let attn_core_start = Instant::now();
            for t in 0..seq_len {
                let mut q = std::mem::take(&mut q_all[t]);
                let attn_out =
                    if let Some((shared_k_cache, shared_v_cache)) = shared_attention_cache {
                        if let Some(freqs) = self.rope_freqs.as_ref() {
                            let mut shared_k_scratch =
                                vec![0.0f32; config.n_kv_heads * config.head_dim];
                            real_math::rope_apply_with_base_and_rotary_dim_mode(
                                &mut q,
                                &mut shared_k_scratch,
                                freqs,
                                position_offset + t as u32,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                attention_config.rope_base_theta,
                                attention_config.rope_rotary_dim,
                                attention_config.proportional_rope,
                            );
                        }
                        real_math::gqa_attention_seq_with_window_and_limit(
                            &q,
                            shared_k_cache,
                            shared_v_cache,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            attention_config.sliding_window,
                            Some(position_offset as usize + t + 1),
                        )
                    } else {
                        let mut k = std::mem::take(&mut k_all.as_mut().unwrap()[t]);
                        let v = std::mem::take(&mut v_all.as_mut().unwrap()[t]);

                        if let Some(freqs) = self.rope_freqs.as_ref() {
                            real_math::rope_apply_with_base_and_rotary_dim_mode(
                                &mut q,
                                &mut k,
                                freqs,
                                position_offset + t as u32,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                attention_config.rope_base_theta,
                                attention_config.rope_rotary_dim,
                                attention_config.proportional_rope,
                            );
                        }

                        k_cache.push(k);
                        v_cache.push(v);

                        real_math::gqa_attention_seq_with_window_and_limit(
                            &q,
                            &k_cache,
                            &v_cache,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            attention_config.sliding_window,
                            Some(position_offset as usize + t + 1),
                        )
                    };
                attn_outputs.push(attn_out);
            }
            profile.attn_core_micros += attn_core_start.elapsed().as_micros();

            let attn_out_start = Instant::now();
            let mut attn_proj_all = if let Some(entry) = o_matrix {
                self.matmul_entry_many(store, entry, &attn_outputs)?
            } else {
                attn_outputs
            };
            profile.attn_out_micros += attn_out_start.elapsed().as_micros();

            for t in 0..seq_len {
                let mut next_state = std::mem::take(&mut attn_proj_all[t]);
                if let Some(ref w) = post_attn_norm_weight {
                    real_math::rms_norm_inplace(&mut next_state, w, config.eps);
                    real_math::vec_add_inplace(&mut next_state, &states[t]);
                } else {
                    real_math::vec_add_inplace(&mut next_state, &states[t]);
                }
                states[t] = next_state;
            }
            profile.attn_micros += attn_start.elapsed().as_micros();
        }

        if has_ffn {
            let ffn_start = Instant::now();
            let mut ffn_inputs = states.to_vec();
            if let Some(ref w) = ffn_norm_weight {
                for input in &mut ffn_inputs {
                    real_math::rms_norm_inplace(input, w, config.eps);
                }
            }
            let gate_entry = gate_matrix.unwrap();
            let up_entry = up_matrix.unwrap();
            let down_entry = down_matrix.unwrap();
            let gate_up_dim = gate_entry.dimensions[1] as usize;
            let down_dim = down_entry.dimensions[1] as usize;
            let ffn_input_refs: Vec<&[f32]> =
                ffn_inputs.iter().map(|input| input.as_slice()).collect();
            let gate_up_start = Instant::now();
            let (mut gate_all, up_all) = self.matmul_entry_many_pair_token_major(
                store,
                gate_entry,
                up_entry,
                &ffn_input_refs,
            )?;
            profile.ffn_gate_up_micros += gate_up_start.elapsed().as_micros();
            for t in 0..seq_len {
                let start = t * gate_up_dim;
                let end = start + gate_up_dim;
                real_math::gelu_pytorch_tanh_mul_inplace(
                    &mut gate_all[start..end],
                    &up_all[start..end],
                );
            }
            let down_start = Instant::now();
            let gate_refs: Vec<&[f32]> = gate_all.chunks_exact(gate_up_dim).collect();
            let mut down_all = self.matmul_entry_many_refs_range_token_major(
                store, down_entry, &gate_refs, 0, down_dim,
            )?;
            profile.ffn_down_micros += down_start.elapsed().as_micros();

            for t in 0..seq_len {
                let start = t * down_dim;
                let end = start + down_dim;
                let next_state = &mut down_all[start..end];
                if let Some(ref w) = post_ffn_norm_weight {
                    real_math::rms_norm_inplace(next_state, w, config.eps);
                    real_math::vec_add_inplace(next_state, &states[t]);
                } else {
                    real_math::vec_add_inplace(next_state, &states[t]);
                }
                states[t].copy_from_slice(next_state);
            }
            profile.ffn_micros += ffn_start.elapsed().as_micros();
        }

        if has_ple {
            let ple_start = Instant::now();
            let ple_inputs = ple_inputs.unwrap();
            let ple_gate_start = Instant::now();
            let mut gated_all = self.matmul_entry_many(store, inp_gate_matrix.unwrap(), states)?;
            profile.ple_gate_micros += ple_gate_start.elapsed().as_micros();
            for t in 0..seq_len {
                real_math::gelu_pytorch_tanh_mul_inplace(&mut gated_all[t], ple_inputs[t]);
            }
            let ple_proj_start = Instant::now();
            let mut projected_all =
                self.matmul_entry_many(store, proj_matrix.unwrap(), &gated_all)?;
            profile.ple_proj_micros += ple_proj_start.elapsed().as_micros();

            for t in 0..seq_len {
                let mut next_state = std::mem::take(&mut projected_all[t]);
                if let Some(ref w) = post_norm_weight {
                    real_math::rms_norm_inplace(&mut next_state, w, config.eps);
                    real_math::vec_add_inplace(&mut next_state, &states[t]);
                } else {
                    real_math::vec_add_inplace(&mut next_state, &states[t]);
                }
                states[t] = next_state;
            }
            profile.ple_micros += ple_start.elapsed().as_micros();
        }

        if let Some(scale) = layer_scale {
            for state in states.iter_mut() {
                for v in state.iter_mut() {
                    *v *= scale;
                }
            }
        }

        Ok(if has_attn && shared_attention_cache.is_none() {
            Some((k_cache, v_cache))
        } else {
            None
        })
    }

    #[cfg(test)]
    fn encode_hidden_state(values: &[f32]) -> Vec<u8> {
        Self::encode_hidden_states(&[values.to_vec()])
    }

    fn encode_hidden_states(states: &[Vec<f32>]) -> Vec<u8> {
        let total_values = states.iter().map(Vec::len).sum::<usize>();
        let mut bytes = Vec::with_capacity(total_values * 4);
        for state in states {
            for value in state {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn encode_hidden_states_payload(
        states: &[Vec<f32>],
        prompt_aux_bytes: Option<&[u8]>,
    ) -> Vec<u8> {
        let hidden_bytes = Self::encode_hidden_states(states);
        encode_stage_tensor_bytes(&hidden_bytes, prompt_aux_bytes)
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        Ok(Self::decode_hidden_states(bytes, hidden_dim)?
            .into_iter()
            .last()
            .unwrap_or_default())
    }

    fn decode_hidden_states(bytes: &[u8], hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
        let hidden_bytes = stage_tensor_byte_sections(bytes)
            .map(|sections| sections.hidden_bytes)
            .unwrap_or(bytes);
        if hidden_dim == 0 {
            bail!("Hidden-state hidden_dim must be nonzero");
        }
        if hidden_bytes.len() % (hidden_dim * 4) != 0 {
            bail!(
                "Hidden-state byte length {} is not a multiple of hidden_dim * 4 = {}",
                hidden_bytes.len(),
                hidden_dim * 4
            );
        }
        let seq_len = hidden_bytes.len() / (hidden_dim * 4);
        let mut states = Vec::with_capacity(seq_len);
        for state_bytes in hidden_bytes.chunks_exact(hidden_dim * 4) {
            let mut values = Vec::with_capacity(hidden_dim);
            for chunk in state_bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            states.push(values);
        }
        Ok(states)
    }

    fn encode_prompt_aux(
        ple_all: Option<&[Vec<Vec<f32>>]>,
        ple_dim: usize,
        prefix_hashes: &[u64],
    ) -> Result<Option<Vec<u8>>> {
        let seq_len = ple_all.map(|all| all.len()).unwrap_or(0);
        let layer_count = ple_all
            .and_then(|all| all.first().map(Vec::len))
            .unwrap_or(0);
        let expected_values = seq_len
            .checked_mul(layer_count)
            .and_then(|count| count.checked_mul(ple_dim))
            .ok_or_else(|| anyhow::anyhow!("PLE aux dimensions overflow"))?;
        if expected_values == 0 && prefix_hashes.is_empty() {
            return Ok(None);
        }

        let mut bytes = Vec::with_capacity(20 + expected_values * 4 + prefix_hashes.len() * 8);
        bytes.extend_from_slice(&REAL_STAGE_AUX_MAGIC);
        bytes.extend_from_slice(&(seq_len as u32).to_le_bytes());
        bytes.extend_from_slice(&(layer_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(ple_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(prefix_hashes.len() as u32).to_le_bytes());
        if let Some(ple_all) = ple_all {
            for token_layers in ple_all {
                if token_layers.len() != layer_count {
                    bail!("PLE aux layer count mismatch while encoding");
                }
                for layer_values in token_layers {
                    if layer_values.len() != ple_dim {
                        bail!("PLE aux dim mismatch while encoding");
                    }
                    for value in layer_values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        for hash in prefix_hashes {
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        Ok(Some(bytes))
    }

    fn decode_prompt_aux(bytes: &[u8]) -> Result<PromptAuxData> {
        if bytes.len() >= 20 && bytes[..4] == REAL_STAGE_AUX_MAGIC {
            let seq_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            let layer_count =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            let ple_dim = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
            let prefix_hash_count =
                u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
            let ple_bytes = seq_len
                .checked_mul(layer_count)
                .and_then(|count| count.checked_mul(ple_dim))
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| anyhow::anyhow!("PLE aux dimensions overflow"))?;
            let expected_bytes = 20
                + ple_bytes
                + prefix_hash_count
                    .checked_mul(8)
                    .ok_or_else(|| anyhow::anyhow!("Prefix-hash dimensions overflow"))?;
            if bytes.len() != expected_bytes {
                bail!(
                    "Invalid prompt-aux byte length {} (expected {})",
                    bytes.len(),
                    expected_bytes
                );
            }

            let mut offset = 20;
            let mut ple_all = Vec::with_capacity(seq_len);
            for _ in 0..seq_len {
                let mut token_layers = Vec::with_capacity(layer_count);
                for _ in 0..layer_count {
                    let mut values = Vec::with_capacity(ple_dim);
                    for _ in 0..ple_dim {
                        let chunk = &bytes[offset..offset + 4];
                        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                        offset += 4;
                    }
                    token_layers.push(values);
                }
                ple_all.push(token_layers);
            }
            let mut prefix_hashes = Vec::with_capacity(prefix_hash_count);
            for _ in 0..prefix_hash_count {
                let chunk = &bytes[offset..offset + 8];
                prefix_hashes.push(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
                offset += 8;
            }
            return Ok(PromptAuxData {
                ple_all,
                prefix_hashes,
            });
        }

        if bytes.len() < 16 || bytes[..4] != REAL_PLE_AUX_MAGIC {
            bail!("Invalid prompt-aux payload");
        }
        let seq_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let layer_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let ple_dim = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let expected_bytes = 16
            + seq_len
                .checked_mul(layer_count)
                .and_then(|count| count.checked_mul(ple_dim))
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| anyhow::anyhow!("PLE aux dimensions overflow"))?;
        if bytes.len() != expected_bytes {
            bail!(
                "Invalid prompt-aux byte length {} (expected {})",
                bytes.len(),
                expected_bytes
            );
        }

        let mut offset = 16;
        let mut ple_all = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            let mut token_layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let mut values = Vec::with_capacity(ple_dim);
                for _ in 0..ple_dim {
                    let chunk = &bytes[offset..offset + 4];
                    values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    offset += 4;
                }
                token_layers.push(values);
            }
            ple_all.push(token_layers);
        }
        Ok(PromptAuxData {
            ple_all,
            prefix_hashes: Vec::new(),
        })
    }

    fn run_stage_layers(
        &self,
        states: &mut Vec<Vec<f32>>,
        ple_all: Option<&Vec<Vec<Vec<f32>>>>,
        ple_token_offset: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, (Vec<Vec<f32>>, Vec<Vec<f32>>)>,
        profile: &mut RealForwardProfile,
    ) -> Result<()> {
        let model_view = self.model_view()?;
        let config = self.config()?;
        let layer_iter = model_view
            .operator_layers
            .iter()
            .zip(model_view.execution_programs.iter())
            .take(self.debug_layer_cap.unwrap_or(usize::MAX));

        for (layer, program) in layer_iter {
            let ple_for_layer: Option<Vec<&[f32]>> = ple_all.as_ref().map(|all| {
                let layer_index = layer.layer_index as usize;
                all.iter()
                    .skip(ple_token_offset)
                    .take(states.len())
                    .map(|token_layers| token_layers[layer_index].as_slice())
                    .collect()
            });
            let ple_ref = ple_for_layer.as_deref();
            let attention_config = self.layer_attention_config(layer, config);
            if let Some(source_layer) = attention_config.shared_kv_source_layer {
                let shared_attention_cache = attention_cache_by_layer.get(&source_layer);
                if let Some(cache) = self.forward_layer_seq(
                    states,
                    layer,
                    program,
                    position_offset,
                    ple_ref,
                    None,
                    shared_attention_cache,
                    profile,
                )? {
                    attention_cache_by_layer.insert(layer.layer_index, cache);
                }
            } else {
                let existing_layer_cache = attention_cache_by_layer.remove(&layer.layer_index);
                if let Some(cache) = self.forward_layer_seq(
                    states,
                    layer,
                    program,
                    position_offset,
                    ple_ref,
                    existing_layer_cache,
                    None,
                    profile,
                )? {
                    attention_cache_by_layer.insert(layer.layer_index, cache);
                }
            }
        }

        Ok(())
    }

    fn continue_forward_inner(&self, input: StageTensor) -> Result<ContinueForwardResult> {
        let layout = self.layout()?;

        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }

        let prompt_aux_bytes = stage_tensor_byte_sections(&input.bytes)
            .and_then(|sections| sections.aux_bytes)
            .map(|bytes| bytes.to_vec());
        let prompt_aux = prompt_aux_bytes
            .as_deref()
            .map(Self::decode_prompt_aux)
            .transpose()?;
        let state_count = Self::decode_hidden_states_payload(&input.bytes, input.hidden_dim)?.len();
        let use_cached_decode = state_count == 1 && prompt_aux.is_some();
        let session = if use_cached_decode {
            self.take_decode_session(&input.request_id)
        } else {
            self.take_decode_session(&input.request_id);
            if prompt_aux.is_some() {
                if let Some(cached) = self.find_tail_prefill(&input.bytes) {
                    self.store_decode_session(
                        input.request_id.clone(),
                        DecodeSession {
                            seq_len: cached.seq_len,
                            attention_cache_by_layer: cached.attention_cache_by_layer.clone(),
                        },
                    );
                    let mut stage_trace = input.stage_trace;
                    stage_trace.push(layout.stage_id.clone());
                    return Ok(ContinueForwardResult {
                        request_id: input.request_id,
                        hidden_dim: input.hidden_dim,
                        stage_trace,
                        states: cached.output_states,
                        prompt_aux_bytes,
                        max_tokens: input.max_tokens,
                        profile: RealForwardProfile {
                            seq_len: cached.seq_len,
                            ..Default::default()
                        },
                    });
                }
                if let Some(prompt_aux) = prompt_aux.as_ref() {
                    if !prompt_aux.prefix_hashes.is_empty() {
                        if let Some((cached, prefix_len)) =
                            self.find_tail_prefill_by_prefix_hashes(&prompt_aux.prefix_hashes)
                        {
                            let mut states =
                                Self::decode_hidden_states(&input.bytes, input.hidden_dim)?;
                            if prefix_len < states.len()
                                && prefix_len <= cached.output_states.len()
                                && prefix_len <= cached.seq_len
                            {
                                let mut attention_cache_by_layer = Self::truncate_attention_cache(
                                    &cached.attention_cache_by_layer,
                                    prefix_len,
                                );
                                let mut output_states = cached.output_states[..prefix_len].to_vec();
                                let mut suffix_states = states.split_off(prefix_len);
                                let mut profile = RealForwardProfile {
                                    seq_len: prefix_len + suffix_states.len(),
                                    ..Default::default()
                                };
                                self.run_stage_layers(
                                    &mut suffix_states,
                                    (!prompt_aux.ple_all.is_empty()).then_some(&prompt_aux.ple_all),
                                    prefix_len,
                                    prefix_len as u32,
                                    &mut attention_cache_by_layer,
                                    &mut profile,
                                )?;
                                output_states.extend(suffix_states);
                                let seq_len = output_states.len();
                                if state_count > 1 {
                                    self.store_tail_prefill(
                                        &input.bytes,
                                        &prompt_aux.prefix_hashes,
                                        &output_states,
                                        seq_len,
                                        &attention_cache_by_layer,
                                    );
                                }
                                self.store_decode_session(
                                    input.request_id.clone(),
                                    DecodeSession {
                                        seq_len,
                                        attention_cache_by_layer: attention_cache_by_layer.clone(),
                                    },
                                );

                                let mut stage_trace = input.stage_trace;
                                stage_trace.push(layout.stage_id.clone());
                                return Ok(ContinueForwardResult {
                                    request_id: input.request_id,
                                    hidden_dim: input.hidden_dim,
                                    stage_trace,
                                    states: output_states,
                                    prompt_aux_bytes,
                                    max_tokens: input.max_tokens,
                                    profile,
                                });
                            }
                        }
                    }
                }
            }
            None
        };
        let mut states = Self::decode_hidden_states(&input.bytes, input.hidden_dim)?;
        let mut profile = RealForwardProfile {
            seq_len: states.len(),
            ..Default::default()
        };
        let position_offset = input
            .prompt_text
            .as_ref()
            .map(|prompt| {
                self.tokenize_prompt(prompt)
                    .len()
                    .saturating_sub(states.len()) as u32
            })
            .unwrap_or_else(|| {
                session
                    .as_ref()
                    .map(|session| session.seq_len as u32)
                    .unwrap_or(0)
            });

        let ple_all = if !self.debug_disable_ple && self.ple_dim > 0 && self.token_embd.is_some() {
            let aux_start = Instant::now();
            if let Some(prompt_aux) = prompt_aux.as_ref() {
                profile.aux_micros = aux_start.elapsed().as_micros();
                Some(prompt_aux.ple_all.clone())
            } else if let Some(ref prompt) = input.prompt_text {
                let token_ids = self.tokenize_prompt(prompt);
                if !token_ids.is_empty() {
                    let embed_start = Instant::now();
                    let embeds = self.embed_tokens(&token_ids, input.hidden_dim)?;
                    profile.embed_micros = embed_start.elapsed().as_micros();
                    let aux_start = Instant::now();
                    let (ple, aux_profile) =
                        self.compute_ple_inputs(&token_ids, &embeds, 0, self.ple_num_layers)?;
                    profile.aux_micros = aux_start.elapsed().as_micros();
                    profile.aux_lookup_micros = aux_profile.lookup_micros;
                    profile.aux_project_micros = aux_profile.project_micros;
                    profile.aux_combine_micros = aux_profile.combine_micros;
                    profile.aux_materialize_micros = aux_profile.materialize_micros;
                    Some(ple)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let mut session = session.unwrap_or(DecodeSession {
            seq_len: position_offset as usize,
            attention_cache_by_layer: HashMap::new(),
        });
        self.run_stage_layers(
            &mut states,
            ple_all.as_ref(),
            0,
            position_offset,
            &mut session.attention_cache_by_layer,
            &mut profile,
        )?;
        session.seq_len = position_offset as usize + states.len();
        if !use_cached_decode && prompt_aux.is_some() && state_count > 1 {
            self.store_tail_prefill(
                &input.bytes,
                prompt_aux
                    .as_ref()
                    .map(|aux| aux.prefix_hashes.as_slice())
                    .unwrap_or(&[]),
                &states,
                session.seq_len,
                &session.attention_cache_by_layer,
            );
        }
        self.store_decode_session(input.request_id.clone(), session);

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());
        Ok(ContinueForwardResult {
            request_id: input.request_id,
            hidden_dim: input.hidden_dim,
            stage_trace,
            states,
            prompt_aux_bytes,
            max_tokens: input.max_tokens,
            profile,
        })
    }
}

impl StageForwardBackend for RealGemmaBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let store = StageTensorStore::load(&self.index_path)?;
        store.validate_offsets()?;

        let model_view = store.model_view();

        let first_layer = model_view.operator_layers.first();

        let hidden_dim = first_layer
            .and_then(|l| l.attn_q.as_ref())
            .and_then(|e| e.dimensions.first().copied())
            .unwrap_or(2560) as usize;
        let q_dim = first_layer
            .and_then(|l| l.attn_q.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(2048) as usize;
        let k_dim = first_layer
            .and_then(|l| l.attn_k.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(512) as usize;
        let ffn_dim = first_layer
            .and_then(|l| l.ffn_up.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(10240) as usize;

        let mut config = GemmaLayerConfig::from_dims(hidden_dim, q_dim, k_dim, ffn_dim);
        config.rope_base_theta = 1_000_000.0; // Gemma 4 uses 1M base theta
        config.logit_softcap = Some(30.0); // Gemma 4 final logit softcapping

        let loaded_rope_freqs = model_view
            .positional
            .iter()
            .find(|e| e.name == "rope_freqs.weight")
            .map(|e| Self::decode_f32_vector(&store, e))
            .transpose()?;
        let rope_freqs = if Self::should_use_packed_rope_freqs(&layout) {
            loaded_rope_freqs
        } else {
            Some(Vec::new())
        };

        let (token_embd, token_embd_type, vocab_size) = {
            let entry = model_view
                .prompt_ingress
                .iter()
                .chain(model_view.shared_auxiliary.iter())
                .chain(model_view.tail_only.iter())
                .find(|e| e.name == "token_embd.weight")
                .or_else(|| store.entry("token_embd.weight"));
            if let Some(entry) = entry {
                let raw = store.read(&entry.name)?;
                let vocab = if entry.dimensions.len() == 2 {
                    entry.dimensions[1] as usize
                } else {
                    0
                };
                (Some(raw), entry.ggml_type, vocab)
            } else {
                (None, 0, 0)
            }
        };

        let ple_token_entry = store.entry("per_layer_token_embd.weight");
        let (ple_token_embd, ple_token_embd_type) = if let Some(entry) = ple_token_entry {
            let raw = store.read(&entry.name)?;
            (Some(raw), entry.ggml_type)
        } else {
            (None, 0)
        };

        let ple_model_proj_entry = store.entry("per_layer_model_proj.weight").cloned();
        let ple_proj_norm_entry = store.entry("per_layer_proj_norm.weight").cloned();

        let ple_dim = ple_proj_norm_entry
            .as_ref()
            .and_then(|e| e.dimensions.first().copied())
            .unwrap_or(0) as usize;
        let ple_num_layers = if ple_dim > 0 {
            ple_model_proj_entry
                .as_ref()
                .and_then(|e| e.dimensions.get(1).copied())
                .map(|out| out as usize / ple_dim)
                .unwrap_or(0)
        } else {
            0
        };

        Self::validate_stage_split_constraints(&layout, &model_view, &config)?;

        self.config = Some(config);
        self.rope_freqs = rope_freqs;
        self.token_embd = token_embd;
        self.token_embd_type = token_embd_type;
        self.vocab_size = vocab_size;
        self.ple_token_embd = ple_token_embd;
        self.ple_token_embd_type = ple_token_embd_type;
        self.ple_model_proj_entry = ple_model_proj_entry;
        self.ple_proj_norm_entry = ple_proj_norm_entry;
        self.ple_dim = ple_dim;
        self.ple_num_layers = ple_num_layers;
        self.layout = Some(layout);
        self.model_view = Some(model_view);
        self.store = Some(store);
        self.start_ple_prewarm();
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let token_ids = self.tokenize_prompt(prompt);
        self.begin_token_sequence(request_id, &token_ids, max_tokens)
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let continued = self.continue_forward_inner(input)?;
        let ContinueForwardResult {
            request_id,
            hidden_dim,
            stage_trace,
            states,
            prompt_aux_bytes,
            max_tokens,
            profile,
        } = continued;
        let bytes = Self::encode_hidden_states_payload(&states, prompt_aux_bytes.as_deref());
        *self
            .last_forward_profile
            .write()
            .expect("real-forward profile cache poisoned") = Some(profile);

        Ok(StageTensor {
            request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim,
            bytes,
            prompt_text: None,
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;
        self.sample_hidden_state(input.request_id, input.hidden_dim, input.max_tokens, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PackedStageIndex, PackedTensorEntry, stage_tensor_byte_sections};
    use std::fs;
    use tempfile::tempdir;

    fn write_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn prefill_prefix_cache_prefers_longest_matching_prefix() {
        let backend = RealGemmaBackend::new("dummy.index.json");
        let empty_cache = HashMap::new();
        backend.store_prefill_prefix(&[1, 2], &[vec![1.0]], &empty_cache);
        backend.store_prefill_prefix(&[1, 2, 3], &[vec![2.0]], &empty_cache);

        let (cached, prefix_len) = backend
            .find_prefill_prefix(&[1, 2, 3, 4])
            .expect("missing prefix cache hit");
        assert_eq!(prefix_len, 3);
        assert_eq!(cached.token_ids, vec![1, 2, 3]);
        assert_eq!(cached.output_states, vec![vec![2.0]]);
    }

    #[test]
    fn prefill_prefix_cache_finds_partial_shared_prefix() {
        let backend = RealGemmaBackend::new("dummy.index.json");
        let empty_cache = HashMap::new();
        backend.store_prefill_prefix(
            &[1, 2, 3, 4],
            &[vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            &empty_cache,
        );

        let (cached, prefix_len) = backend
            .find_prefill_prefix(&[1, 2, 9])
            .expect("missing partial prefix cache hit");
        assert_eq!(cached.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(prefix_len, 2);
    }

    #[test]
    fn tail_prefill_cache_matches_exact_payload() {
        let backend = RealGemmaBackend::new("dummy.index.json");
        let empty_cache = HashMap::new();
        backend.store_tail_prefill(&[1, 2, 3], &[10, 20, 30], &[vec![4.0]], 3, &empty_cache);

        let cached = backend
            .find_tail_prefill(&[1, 2, 3])
            .expect("missing tail prefill cache hit");
        assert_eq!(cached.seq_len, 3);
        assert_eq!(cached.prefix_hashes, vec![10, 20, 30]);
        assert_eq!(cached.output_states, vec![vec![4.0]]);
        assert!(backend.find_tail_prefill(&[1, 2]).is_none());
    }

    #[test]
    fn tail_prefill_cache_finds_longest_shared_prefix_hash_chain() {
        let backend = RealGemmaBackend::new("dummy.index.json");
        let empty_cache = HashMap::new();
        backend.store_tail_prefill(&[1], &[11, 22], &[vec![1.0]], 2, &empty_cache);
        backend.store_tail_prefill(&[2], &[11, 22, 33], &[vec![2.0]], 3, &empty_cache);

        let (cached, prefix_len) = backend
            .find_tail_prefill_by_prefix_hashes(&[11, 22, 44])
            .expect("missing shared-prefix tail prefill cache hit");
        assert_eq!(prefix_len, 2);
        assert_eq!(cached.output_states, vec![vec![2.0]]);
    }

    #[test]
    fn real_backend_loads_and_runs_minimal_stage() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");

        let hidden_dim = 4;
        let vocab_size = 8;
        let n_heads = 2;
        let head_dim = 2;
        let n_kv_heads = 1;
        let ffn_dim = 8;

        let mut pack_data = Vec::new();
        let mut tensors = Vec::new();
        let mut offset = 0u64;

        let add_tensor = |name: &str,
                          dims: Vec<u64>,
                          data: &[f32],
                          pack: &mut Vec<u8>,
                          tensors: &mut Vec<PackedTensorEntry>,
                          off: &mut u64| {
            let bytes = write_f32_bytes(data);
            let entry = PackedTensorEntry {
                name: name.to_string(),
                pack_offset: *off,
                byte_len: bytes.len() as u64,
                source_file_offset: 0,
                dimensions: dims,
                ggml_type: quants::GGML_TYPE_F32,
            };
            pack.extend_from_slice(&bytes);
            *off += bytes.len() as u64;
            tensors.push(entry);
        };

        let embd: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|i| (i as f32 * 0.1) - 0.5)
            .collect();
        add_tensor(
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let rope: Vec<f32> = vec![1.0; head_dim / 2];
        add_tensor(
            "rope_freqs.weight",
            vec![rope.len() as u64],
            &rope,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let norm_w = vec![1.0f32; hidden_dim];
        add_tensor(
            "blk.0.attn_norm.weight",
            vec![hidden_dim as u64],
            &norm_w,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let identity_4x4: Vec<f32> = (0..hidden_dim)
            .flat_map(|r| (0..hidden_dim).map(move |c| if r == c { 1.0 } else { 0.0 }))
            .collect();

        add_tensor(
            "blk.0.attn_q.weight",
            vec![hidden_dim as u64, (n_heads * head_dim) as u64],
            &identity_4x4,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let k_weights: Vec<f32> = (0..hidden_dim * n_kv_heads * head_dim)
            .map(|i| if i % (hidden_dim + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        add_tensor(
            "blk.0.attn_k.weight",
            vec![hidden_dim as u64, (n_kv_heads * head_dim) as u64],
            &k_weights,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        add_tensor(
            "blk.0.attn_v.weight",
            vec![hidden_dim as u64, (n_kv_heads * head_dim) as u64],
            &k_weights,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let q_norm = vec![1.0f32; head_dim];
        add_tensor(
            "blk.0.attn_q_norm.weight",
            vec![head_dim as u64],
            &q_norm,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );
        add_tensor(
            "blk.0.attn_k_norm.weight",
            vec![head_dim as u64],
            &q_norm,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        add_tensor(
            "blk.0.attn_output.weight",
            vec![(n_heads * head_dim) as u64, hidden_dim as u64],
            &identity_4x4,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let post_attn_norm = vec![1.0f32; hidden_dim];
        add_tensor(
            "blk.0.post_attention_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );
        add_tensor(
            "blk.0.ffn_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let ffn_gate: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| if i % (hidden_dim + 1) == 0 { 0.5 } else { 0.0 })
            .collect();
        add_tensor(
            "blk.0.ffn_gate.weight",
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_gate,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );
        add_tensor(
            "blk.0.ffn_up.weight",
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_gate,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let ffn_down: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| if i % (ffn_dim + 1) == 0 { 0.5 } else { 0.0 })
            .collect();
        add_tensor(
            "blk.0.ffn_down.weight",
            vec![ffn_dim as u64, hidden_dim as u64],
            &ffn_down,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        add_tensor(
            "blk.0.post_ffw_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        let scale = vec![1.0f32];
        add_tensor(
            "blk.0.layer_output_scale.weight",
            vec![1],
            &scale,
            &mut pack_data,
            &mut tensors,
            &mut offset,
        );

        fs::write(&pack_path, &pack_data).unwrap();
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "test-gemma".into(),
                architecture: "gemma4".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: offset,
                tensor_count: tensors.len(),
                tensors,
            })
            .unwrap(),
        )
        .unwrap();

        let mut backend = RealGemmaBackend::new(&index_path);
        backend
            .load_layout(StageLayout {
                model_id: "test-gemma".into(),
                stage_id: "stage-0".into(),
                start_layer: 0,
                end_layer: 0,
                is_head: true,
                is_tail: false,
            })
            .unwrap();

        let tensor = backend.begin_prompt("test-req", "hi", Some(1), 0).unwrap();
        assert_eq!(tensor.kind, PayloadKind::HiddenState);
        assert_eq!(tensor.hidden_dim, hidden_dim);
        assert_eq!(tensor.hidden_state_len() % (hidden_dim * 4), 0);
        assert!(tensor.hidden_state_len() >= hidden_dim * 4);
        assert!(tensor.bytes.len() >= tensor.hidden_state_len());

        let state = RealGemmaBackend::decode_hidden_state(&tensor.bytes, hidden_dim).unwrap();
        assert!(
            state.iter().all(|v| v.is_finite()),
            "State contains non-finite values: {:?}",
            state
        );
    }

    #[test]
    fn begin_token_ids_matches_begin_prompt() {
        let temp = tempdir().unwrap();
        let hidden_dim = 4;
        let vocab_size = 256;
        let mut builder = TestStageBuilder::new();

        let token_embd: Vec<f32> = (0..vocab_size)
            .flat_map(|token| [token as f32 * 0.01, 1.0, 0.5, -0.25])
            .collect();
        builder.add_f32(
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &token_embd,
        );
        builder.add_f32("rope_freqs.weight", vec![1], &[1.0]);
        builder.add_f32(
            "blk.0.attn_q.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32(
            "blk.0.attn_k.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32(
            "blk.0.attn_v.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32(
            "blk.0.attn_output.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32("blk.0.attn_norm.weight", vec![hidden_dim as u64], &[1.0; 4]);
        builder.add_f32(
            "blk.0.post_ffw_norm.weight",
            vec![hidden_dim as u64],
            &[1.0; 4],
        );
        builder.add_f32(
            "blk.0.ffn_gate.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32(
            "blk.0.ffn_up.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32(
            "blk.0.ffn_down.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 16],
        );
        builder.add_f32("blk.0.layer_output_scale.weight", vec![1], &[1.0]);
        let index_path = builder.write(temp.path(), "stage-0", "head", 0);

        let mut backend = RealGemmaBackend::new(&index_path);
        backend
            .load_layout(StageLayout {
                model_id: "test-gemma".into(),
                stage_id: "stage-0".into(),
                start_layer: 0,
                end_layer: 0,
                is_head: true,
                is_tail: false,
            })
            .unwrap();

        let prompt = "hi";
        let prompt_tensor = backend.begin_prompt("prompt", prompt, Some(1), 0).unwrap();
        let token_ids = backend.tokenize_text(prompt);
        let token_tensor = backend
            .begin_token_ids("tokens", &token_ids, Some(1), 0)
            .unwrap();

        assert_eq!(prompt_tensor.hidden_dim, token_tensor.hidden_dim);
        assert_eq!(prompt_tensor.stage_trace, token_tensor.stage_trace);
        assert_eq!(prompt_tensor.bytes, token_tensor.bytes);
    }

    struct TestStageBuilder {
        pack_data: Vec<u8>,
        tensors: Vec<PackedTensorEntry>,
        offset: u64,
    }

    impl TestStageBuilder {
        fn new() -> Self {
            Self {
                pack_data: Vec::new(),
                tensors: Vec::new(),
                offset: 0,
            }
        }

        fn add_f32(&mut self, name: &str, dims: Vec<u64>, data: &[f32]) {
            let bytes = write_f32_bytes(data);
            self.tensors.push(PackedTensorEntry {
                name: name.to_string(),
                pack_offset: self.offset,
                byte_len: bytes.len() as u64,
                source_file_offset: 0,
                dimensions: dims,
                ggml_type: quants::GGML_TYPE_F32,
            });
            self.pack_data.extend_from_slice(&bytes);
            self.offset += bytes.len() as u64;
        }

        fn write(
            self,
            dir: &std::path::Path,
            stage_name: &str,
            role: &str,
            stage_index: u32,
        ) -> std::path::PathBuf {
            let index_path = dir.join(format!("{}-required.index.json", stage_name));
            let pack_path = dir.join(format!("{}-required.pack", stage_name));
            fs::write(&pack_path, &self.pack_data).unwrap();
            fs::write(
                &index_path,
                serde_json::to_vec_pretty(&PackedStageIndex {
                    model_name: "test-gemma".into(),
                    architecture: "gemma4".into(),
                    stage_index,
                    role: role.into(),
                    total_bytes: self.offset,
                    tensor_count: self.tensors.len(),
                    tensors: self.tensors,
                })
                .unwrap(),
            )
            .unwrap();
            index_path
        }
    }

    fn add_layer_tensors(
        b: &mut TestStageBuilder,
        layer: usize,
        hidden_dim: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        ffn_dim: usize,
    ) {
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv_heads * head_dim;
        let prefix = format!("blk.{}", layer);

        let norm_w = vec![1.0f32; hidden_dim];
        b.add_f32(
            &format!("{prefix}.attn_norm.weight"),
            vec![hidden_dim as u64],
            &norm_w,
        );

        let identity: Vec<f32> = (0..hidden_dim * q_dim)
            .map(|i| if i / q_dim == i % q_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(
            &format!("{prefix}.attn_q.weight"),
            vec![hidden_dim as u64, q_dim as u64],
            &identity[..hidden_dim * q_dim],
        );

        let k_w: Vec<f32> = (0..hidden_dim * k_dim)
            .map(|i| if i / k_dim == i % k_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(
            &format!("{prefix}.attn_k.weight"),
            vec![hidden_dim as u64, k_dim as u64],
            &k_w,
        );
        b.add_f32(
            &format!("{prefix}.attn_v.weight"),
            vec![hidden_dim as u64, k_dim as u64],
            &k_w,
        );

        let q_norm = vec![1.0f32; head_dim];
        b.add_f32(
            &format!("{prefix}.attn_q_norm.weight"),
            vec![head_dim as u64],
            &q_norm,
        );
        b.add_f32(
            &format!("{prefix}.attn_k_norm.weight"),
            vec![head_dim as u64],
            &q_norm,
        );

        let out_w: Vec<f32> = (0..q_dim * hidden_dim)
            .map(|i| {
                if i / hidden_dim == i % hidden_dim {
                    0.1
                } else {
                    0.0
                }
            })
            .collect();
        b.add_f32(
            &format!("{prefix}.attn_output.weight"),
            vec![q_dim as u64, hidden_dim as u64],
            &out_w,
        );

        b.add_f32(
            &format!("{prefix}.post_attention_norm.weight"),
            vec![hidden_dim as u64],
            &norm_w,
        );
        b.add_f32(
            &format!("{prefix}.ffn_norm.weight"),
            vec![hidden_dim as u64],
            &norm_w,
        );

        let ffn_w: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| if i / ffn_dim == i % ffn_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(
            &format!("{prefix}.ffn_gate.weight"),
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_w,
        );
        b.add_f32(
            &format!("{prefix}.ffn_up.weight"),
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_w,
        );

        let ffn_down: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| {
                if i / hidden_dim == i % hidden_dim {
                    0.1
                } else {
                    0.0
                }
            })
            .collect();
        b.add_f32(
            &format!("{prefix}.ffn_down.weight"),
            vec![ffn_dim as u64, hidden_dim as u64],
            &ffn_down,
        );

        b.add_f32(
            &format!("{prefix}.post_ffw_norm.weight"),
            vec![hidden_dim as u64],
            &norm_w,
        );
        b.add_f32(
            &format!("{prefix}.layer_output_scale.weight"),
            vec![1],
            &[1.0],
        );
    }

    #[test]
    fn real_two_stage_roundtrip_produces_finite_output() {
        let temp = tempdir().unwrap();
        let hidden_dim = 8;
        let n_heads = 2;
        let head_dim = 4;
        let n_kv_heads = 1;
        let ffn_dim = 16;
        let vocab_size = 32;

        let mut head_builder = TestStageBuilder::new();
        let embd: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();
        head_builder.add_f32(
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
        );
        let rope = vec![1.0f32; head_dim / 2];
        head_builder.add_f32("rope_freqs.weight", vec![rope.len() as u64], &rope);
        add_layer_tensors(
            &mut head_builder,
            0,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut head_builder,
            1,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let head_path = head_builder.write(temp.path(), "stage-0", "head", 0);

        let mut tail_builder = TestStageBuilder::new();
        let tail_rope = vec![1.0f32; head_dim / 2];
        tail_builder.add_f32(
            "rope_freqs.weight",
            vec![tail_rope.len() as u64],
            &tail_rope,
        );
        add_layer_tensors(
            &mut tail_builder,
            2,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut tail_builder,
            3,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let norm_w = vec![1.0f32; hidden_dim];
        tail_builder.add_f32("output_norm.weight", vec![hidden_dim as u64], &norm_w);
        tail_builder.add_f32(
            "output.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
        );
        let tail_path = tail_builder.write(temp.path(), "stage-1", "tail", 1);

        let mut head = RealGemmaBackend::new(&head_path);
        head.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 1,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = RealGemmaBackend::new(&tail_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 2,
            end_layer: 3,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let head_output = head
            .begin_prompt("req-1", "hello world", Some(1), 0)
            .unwrap();
        assert_eq!(head_output.kind, PayloadKind::HiddenState);
        assert_eq!(head_output.hidden_dim, hidden_dim);
        assert!(head_output.bytes.len() > hidden_dim * 4);

        let head_state =
            RealGemmaBackend::decode_hidden_state(&head_output.bytes, hidden_dim).unwrap();
        assert!(
            head_state.iter().all(|v| v.is_finite()),
            "Head output has non-finite values: {:?}",
            head_state
        );

        let tail_input = tail.continue_forward(head_output).unwrap();
        assert_eq!(tail_input.kind, PayloadKind::HiddenState);
        assert_eq!(tail_input.hidden_dim, hidden_dim);
        assert_eq!(tail_input.stage_trace, vec!["stage-0", "stage-1"]);

        let tail_state =
            RealGemmaBackend::decode_hidden_state(&tail_input.bytes, hidden_dim).unwrap();
        assert!(
            tail_state.iter().all(|v| v.is_finite()),
            "Tail output has non-finite values: {:?}",
            tail_state
        );
        let sample = tail.sample_tail(tail_input).unwrap();
        assert_eq!(sample.token_ids.len() as u32, sample.completion_tokens);
        assert!(!sample.token_ids.is_empty());

        assert_ne!(
            head_state, tail_state,
            "Tail stage should transform the hidden state"
        );
    }

    #[test]
    fn cached_decode_step_matches_fresh_full_rerun() {
        let temp = tempdir().unwrap();
        let hidden_dim = 8;
        let ple_dim = 2;
        let n_heads = 2;
        let head_dim = 4;
        let n_kv_heads = 1;
        let ffn_dim = 16;
        let vocab_size = 32;

        let mut head_builder = TestStageBuilder::new();
        let embd: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();
        head_builder.add_f32(
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
        );
        head_builder.add_f32(
            "per_layer_token_embd.weight",
            vec![(ple_dim * 2) as u64, vocab_size as u64],
            &vec![0.5; ple_dim * 2 * vocab_size],
        );
        head_builder.add_f32(
            "per_layer_model_proj.weight",
            vec![hidden_dim as u64, (ple_dim * 2) as u64],
            &vec![0.0; hidden_dim * ple_dim * 2],
        );
        head_builder.add_f32(
            "per_layer_proj_norm.weight",
            vec![ple_dim as u64],
            &vec![1.0; ple_dim],
        );
        let rope = vec![1.0f32; head_dim / 2];
        head_builder.add_f32("rope_freqs.weight", vec![rope.len() as u64], &rope);
        add_layer_tensors(
            &mut head_builder,
            0,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut head_builder,
            1,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let head_path = head_builder.write(temp.path(), "stage-0", "head", 0);

        let mut tail_builder = TestStageBuilder::new();
        let tail_rope = vec![1.0f32; head_dim / 2];
        tail_builder.add_f32(
            "rope_freqs.weight",
            vec![tail_rope.len() as u64],
            &tail_rope,
        );
        add_layer_tensors(
            &mut tail_builder,
            2,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut tail_builder,
            3,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let norm_w = vec![1.0f32; hidden_dim];
        tail_builder.add_f32("output_norm.weight", vec![hidden_dim as u64], &norm_w);
        tail_builder.add_f32(
            "output.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
        );
        let tail_path = tail_builder.write(temp.path(), "stage-1", "tail", 1);

        let mut head = RealGemmaBackend::new(&head_path);
        head.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 1,
            is_head: true,
            is_tail: false,
        })
        .unwrap();
        let mut tail = RealGemmaBackend::new(&tail_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 2,
            end_layer: 3,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let prompt = "hello world";
        let head_output = head.begin_prompt("req-cache", prompt, Some(1), 0).unwrap();
        let tail_output = tail.continue_forward(head_output).unwrap();
        let first_sample = tail.sample_tail(tail_output).unwrap();
        let next_token = first_sample.token_ids[0];

        let cached_head_output = head
            .begin_token_ids("req-cache", &[next_token], Some(1), 0)
            .unwrap();
        let cached_tail_output = tail.continue_forward(cached_head_output).unwrap();
        let cached_sample = tail.sample_tail(cached_tail_output.clone()).unwrap();

        let mut fresh_head = RealGemmaBackend::new(&head_path);
        fresh_head
            .load_layout(StageLayout {
                model_id: "test-gemma".into(),
                stage_id: "stage-0".into(),
                start_layer: 0,
                end_layer: 1,
                is_head: true,
                is_tail: false,
            })
            .unwrap();
        let mut fresh_tail = RealGemmaBackend::new(&tail_path);
        fresh_tail
            .load_layout(StageLayout {
                model_id: "test-gemma".into(),
                stage_id: "stage-1".into(),
                start_layer: 2,
                end_layer: 3,
                is_head: false,
                is_tail: true,
            })
            .unwrap();

        let mut full_tokens = fresh_head.tokenize_text(prompt);
        full_tokens.push(next_token);
        let fresh_head_output = fresh_head
            .begin_token_ids("req-fresh", &full_tokens, Some(1), 0)
            .unwrap();
        let fresh_tail_output = fresh_tail.continue_forward(fresh_head_output).unwrap();
        let fresh_sample = fresh_tail.sample_tail(fresh_tail_output.clone()).unwrap();

        let cached_state =
            RealGemmaBackend::decode_hidden_state(&cached_tail_output.bytes, hidden_dim).unwrap();
        let fresh_state =
            RealGemmaBackend::decode_hidden_state(&fresh_tail_output.bytes, hidden_dim).unwrap();
        assert_eq!(cached_state, fresh_state);
        assert_eq!(cached_sample.token_ids, fresh_sample.token_ids);
        assert_eq!(cached_sample.text, fresh_sample.text);
    }

    #[test]
    fn two_stage_prompt_transfer_preserves_sequence_context_for_tail_attention() {
        let temp = tempdir().unwrap();
        let hidden_dim = 2;
        let vocab_size = 128;

        let mut head_builder = TestStageBuilder::new();
        let mut embd = vec![0.0f32; hidden_dim * vocab_size];
        embd['a' as usize * hidden_dim] = 2.0;
        embd['b' as usize * hidden_dim + 1] = 1.0;
        head_builder.add_f32(
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd,
        );
        head_builder.add_f32(
            "blk.0.attn_norm.weight",
            vec![hidden_dim as u64],
            &[1.0, 1.0],
        );
        head_builder.add_f32(
            "blk.0.attn_q.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 4],
        );
        head_builder.add_f32(
            "blk.0.attn_k.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 4],
        );
        head_builder.add_f32(
            "blk.0.attn_v.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0; 4],
        );
        let head_path = head_builder.write(temp.path(), "stage-0", "head", 0);

        let mut tail_builder = TestStageBuilder::new();
        tail_builder.add_f32(
            "blk.1.attn_norm.weight",
            vec![hidden_dim as u64],
            &[1.0, 1.0],
        );
        tail_builder.add_f32(
            "blk.1.attn_q.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0, 4.0, 4.0, 0.0],
        );
        tail_builder.add_f32(
            "blk.1.attn_k.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[4.0, 0.0, 0.0, 4.0],
        );
        tail_builder.add_f32(
            "blk.1.attn_v.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[8.0, 0.0, 0.0, 8.0],
        );
        tail_builder.add_f32(
            "blk.1.attn_output.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[4.0, 0.0, 0.0, 4.0],
        );
        tail_builder.add_f32("output_norm.weight", vec![hidden_dim as u64], &[1.0, 1.0]);
        let mut output = vec![0.0f32; hidden_dim * vocab_size];
        output['A' as usize * hidden_dim] = 1.0;
        output['B' as usize * hidden_dim + 1] = 1.0;
        tail_builder.add_f32(
            "output.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &output,
        );
        let tail_path = tail_builder.write(temp.path(), "stage-1", "tail", 1);

        let mut head = RealGemmaBackend::new(&head_path);
        head.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = RealGemmaBackend::new(&tail_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 1,
            end_layer: 1,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let head_output = head.begin_prompt("req-seq", "ab", Some(1), 0).unwrap();
        assert_eq!(head_output.hidden_state_len(), 2 * hidden_dim * 4);
        let staged_output = tail.continue_forward(head_output.clone()).unwrap();
        assert_eq!(staged_output.hidden_state_len(), 2 * hidden_dim * 4);
        let staged_sample = tail.sample_tail(staged_output).unwrap();
        assert_eq!(staged_sample.token_ids, vec!['A' as u32]);

        let last_state_only =
            RealGemmaBackend::decode_hidden_state(&head_output.bytes, hidden_dim).unwrap();
        let collapsed_output = tail
            .continue_forward(StageTensor {
                request_id: head_output.request_id.clone(),
                kind: head_output.kind,
                stage_trace: head_output.stage_trace.clone(),
                hidden_dim,
                bytes: RealGemmaBackend::encode_hidden_state(&last_state_only),
                prompt_text: head_output.prompt_text.clone(),
                max_tokens: head_output.max_tokens,
                continuation: None,
                transient: None,
                carry: None,
            })
            .unwrap();
        let collapsed_sample = tail.sample_tail(collapsed_output).unwrap();
        assert_eq!(collapsed_sample.token_ids, vec!['B' as u32]);
    }

    #[test]
    fn hidden_state_payload_frames_prompt_aux_without_forwarding_prompt_text() {
        let temp = tempdir().unwrap();
        let hidden_dim = 2usize;
        let ple_dim = 2usize;
        let vocab_size = 256usize;

        let build_stage = |stage_name: &str, role: &str, stage_index: u32| {
            let mut builder = TestStageBuilder::new();
            let token_embd: Vec<f32> = (0..vocab_size)
                .flat_map(|token| [token as f32 * 0.01, 1.0])
                .collect();
            builder.add_f32(
                "token_embd.weight",
                vec![hidden_dim as u64, vocab_size as u64],
                &token_embd,
            );
            builder.add_f32(
                "per_layer_token_embd.weight",
                vec![ple_dim as u64, vocab_size as u64],
                &vec![0.5; ple_dim * vocab_size],
            );
            builder.add_f32(
                "per_layer_model_proj.weight",
                vec![hidden_dim as u64, ple_dim as u64],
                &[0.0, 0.0, 0.0, 0.0],
            );
            builder.add_f32(
                "per_layer_proj_norm.weight",
                vec![ple_dim as u64],
                &[1.0, 1.0],
            );
            builder.add_f32("rope_freqs.weight", vec![1], &[1.0]);
            builder.add_f32(
                "blk.0.attn_q.weight",
                vec![hidden_dim as u64, hidden_dim as u64],
                &[0.0, 0.0, 0.0, 0.0],
            );
            builder.add_f32(
                "blk.0.attn_k.weight",
                vec![hidden_dim as u64, hidden_dim as u64],
                &[0.0, 0.0, 0.0, 0.0],
            );
            builder.write(temp.path(), stage_name, role, stage_index)
        };

        let head_path = build_stage("stage-0", "head", 0);
        let tail_path = build_stage("stage-1", "tail", 1);

        let mut head = RealGemmaBackend::new(&head_path);
        head.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = RealGemmaBackend::new(&tail_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 1,
            end_layer: 1,
            is_head: false,
            is_tail: false,
        })
        .unwrap();

        let head_output = head.begin_prompt("req-private", "ab", Some(1), 0).unwrap();
        let head_sections = stage_tensor_byte_sections(&head_output.bytes).unwrap();
        assert!(head_sections.aux_bytes.is_some());
        assert_eq!(head_output.prompt_text, None);
        assert_eq!(head_output.hidden_state_len(), 2 * hidden_dim * 4);
        assert!(head_output.bytes.len() > head_output.hidden_state_len());
        assert_eq!(
            RealGemmaBackend::decode_hidden_states_payload(&head_output.bytes, hidden_dim)
                .unwrap()
                .len(),
            2
        );

        let tail_output = tail.continue_forward(head_output).unwrap();
        let tail_sections = stage_tensor_byte_sections(&tail_output.bytes).unwrap();
        assert!(tail_sections.aux_bytes.is_some());
        assert_eq!(tail_output.prompt_text, None);
        assert_eq!(tail_output.hidden_state_len(), 2 * hidden_dim * 4);
    }

    #[test]
    fn real_tail_sampling_prefers_output_weight_over_tied_embeddings() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let hidden_dim = 2;
        let mut builder = TestStageBuilder::new();
        builder.add_f32("output_norm.weight", vec![hidden_dim as u64], &[1.0, 1.0]);
        builder.add_f32(
            "token_embd.weight",
            vec![hidden_dim as u64, 3],
            &[9.0, 0.0, 0.0, 1.0, -1.0, -1.0],
        );
        builder.add_f32(
            "output.weight",
            vec![hidden_dim as u64, 3],
            &[0.0, 1.0, 9.0, 0.0, -1.0, -1.0],
        );
        fs::write(&pack_path, &builder.pack_data).unwrap();
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "test-gemma".into(),
                architecture: "gemma4".into(),
                stage_index: 1,
                role: "tail".into(),
                total_bytes: builder.offset,
                tensor_count: builder.tensors.len(),
                tensors: builder.tensors,
            })
            .unwrap(),
        )
        .unwrap();

        let mut tail = RealGemmaBackend::new(&index_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: false,
            is_tail: true,
        })
        .unwrap();
        tail.set_debug_vocab_cap(Some(2));

        let input = StageTensor {
            request_id: "req-output-head".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["stage-1".into()],
            hidden_dim,
            bytes: RealGemmaBackend::encode_hidden_state(&[1.0, 0.0]),
            prompt_text: None,
            max_tokens: Some(1),
            continuation: None,
            transient: None,
            carry: None,
        };
        let trace = tail.trace_tail_logits(&input, 2).unwrap();
        assert_eq!(trace.projection_tensor, "output.weight");
        assert_eq!(trace.vocab_size, 2);
        assert_eq!(trace.selected_token_id, 1);
        assert_eq!(trace.top_logits.first().map(|(id, _)| *id), Some(1));

        let (combined_sample, combined_trace) =
            tail.sample_tail_with_trace(input.clone(), 2).unwrap();
        assert_eq!(combined_trace, trace);
        assert_eq!(combined_sample.token_ids, vec![1]);

        let sample = tail.sample_tail(input).unwrap();
        assert_eq!(sample.token_ids, vec![1]);
    }

    #[test]
    fn layer_config_preserves_model_level_rope_and_logit_settings() {
        let fallback = GemmaLayerConfig {
            hidden_dim: 2560,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            ffn_dim: 10240,
            eps: 1e-5,
            rope_base_theta: 1_000_000.0,
            logit_softcap: Some(30.0),
        };
        let layer = LayerOperatorView {
            attn_q: Some(PackedTensorEntry {
                name: "blk.0.attn_q.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 2048],
                ggml_type: 0,
            }),
            attn_k: Some(PackedTensorEntry {
                name: "blk.0.attn_k.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 512],
                ggml_type: 0,
            }),
            ffn_up: Some(PackedTensorEntry {
                name: "blk.0.ffn_up.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 10240],
                ggml_type: 0,
            }),
            ..LayerOperatorView::default()
        };

        let config = RealGemmaBackend::layer_config(&layer, &fallback);

        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_kv_heads, 2);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.eps, fallback.eps);
        assert_eq!(config.rope_base_theta, fallback.rope_base_theta);
        assert_eq!(config.logit_softcap, fallback.logit_softcap);
    }

    #[test]
    fn layer_config_uses_qk_norm_width_for_full_attention_layers() {
        let fallback = GemmaLayerConfig {
            hidden_dim: 2560,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            ffn_dim: 10240,
            eps: 1e-6,
            rope_base_theta: 1_000_000.0,
            logit_softcap: Some(30.0),
        };
        let layer = LayerOperatorView {
            attn_q: Some(PackedTensorEntry {
                name: "blk.5.attn_q.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 4096],
                ggml_type: 0,
            }),
            attn_k: Some(PackedTensorEntry {
                name: "blk.5.attn_k.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 1024],
                ggml_type: 0,
            }),
            attn_q_norm: Some(PackedTensorEntry {
                name: "blk.5.attn_q_norm.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![512],
                ggml_type: 0,
            }),
            attn_k_norm: Some(PackedTensorEntry {
                name: "blk.5.attn_k_norm.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![512],
                ggml_type: 0,
            }),
            ffn_up: Some(PackedTensorEntry {
                name: "blk.5.ffn_up.weight".into(),
                pack_offset: 0,
                byte_len: 0,
                source_file_offset: 0,
                dimensions: vec![2560, 10240],
                ggml_type: 0,
            }),
            ..LayerOperatorView::default()
        };

        let config = RealGemmaBackend::layer_config(&layer, &fallback);

        assert_eq!(config.head_dim, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_kv_heads, 2);
        assert_eq!(config.rope_base_theta, fallback.rope_base_theta);
    }

    #[test]
    fn layer_attention_config_matches_gemma4_e4b_hybrid_pattern() {
        let mut backend = RealGemmaBackend::new("ignored");
        backend.layout = Some(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        });
        let sliding_config = GemmaLayerConfig {
            hidden_dim: 2560,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            ffn_dim: 10240,
            eps: 1e-6,
            rope_base_theta: 1_000_000.0,
            logit_softcap: Some(30.0),
        };
        let full_config = GemmaLayerConfig {
            head_dim: 512,
            ..sliding_config.clone()
        };

        let full = backend.layer_attention_config(
            &LayerOperatorView {
                layer_index: 5,
                ..LayerOperatorView::default()
            },
            &full_config,
        );
        assert_eq!(full.rope_base_theta, 1_000_000.0);
        assert_eq!(full.rope_rotary_dim, 128);
        assert!(full.proportional_rope);
        assert_eq!(full.shared_kv_source_layer, None);
        assert_eq!(full.sliding_window, None);

        let sliding = backend.layer_attention_config(
            &LayerOperatorView {
                layer_index: 6,
                ..LayerOperatorView::default()
            },
            &sliding_config,
        );
        assert_eq!(sliding.rope_base_theta, 10_000.0);
        assert_eq!(sliding.rope_rotary_dim, 256);
        assert!(!sliding.proportional_rope);
        assert_eq!(sliding.shared_kv_source_layer, None);
        assert_eq!(sliding.sliding_window, Some(512));

        let shared_full = backend.layer_attention_config(
            &LayerOperatorView {
                layer_index: 29,
                ..LayerOperatorView::default()
            },
            &full_config,
        );
        assert_eq!(shared_full.shared_kv_source_layer, Some(23));

        let shared_sliding = backend.layer_attention_config(
            &LayerOperatorView {
                layer_index: 30,
                ..LayerOperatorView::default()
            },
            &sliding_config,
        );
        assert_eq!(shared_sliding.shared_kv_source_layer, Some(22));
    }

    #[test]
    fn load_layout_rejects_gemma4_e4b_stage_that_splits_shared_kv_sources() {
        let temp = tempdir().unwrap();
        let hidden_dim = 2560;
        let n_heads = 8;
        let head_dim = 256;
        let n_kv_heads = 2;
        let ffn_dim = 10240;

        let mut builder = TestStageBuilder::new();
        let rope = vec![1.0f32; head_dim / 2];
        builder.add_f32("rope_freqs.weight", vec![rope.len() as u64], &rope);
        add_layer_tensors(
            &mut builder,
            24,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let index_path = builder.write(temp.path(), "stage-2", "middle", 1);

        let mut backend = RealGemmaBackend::new(&index_path);
        let err = backend
            .load_layout(StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-24-24".into(),
                start_layer: 24,
                end_layer: 24,
                is_head: false,
                is_tail: false,
            })
            .expect_err("stage split should be rejected");

        let message = err.to_string();
        assert!(message.contains("shared KV"));
        assert!(message.contains("layer 24"));
        assert!(message.contains("layer 22"));
    }

    #[test]
    fn load_layout_accepts_gemma4_e4b_stage_that_keeps_shared_kv_sources_local() {
        let temp = tempdir().unwrap();
        let hidden_dim = 2560;
        let n_heads = 8;
        let head_dim = 256;
        let n_kv_heads = 2;
        let ffn_dim = 10240;

        let mut builder = TestStageBuilder::new();
        let rope = vec![1.0f32; head_dim / 2];
        builder.add_f32("rope_freqs.weight", vec![rope.len() as u64], &rope);
        add_layer_tensors(
            &mut builder,
            22,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut builder,
            23,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        add_layer_tensors(
            &mut builder,
            24,
            hidden_dim,
            n_heads,
            head_dim,
            n_kv_heads,
            ffn_dim,
        );
        let index_path = builder.write(temp.path(), "stage-1", "tail", 1);

        let mut backend = RealGemmaBackend::new(&index_path);
        backend
            .load_layout(StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-22-24".into(),
                start_layer: 22,
                end_layer: 24,
                is_head: false,
                is_tail: true,
            })
            .unwrap();
    }

    #[test]
    fn gemma4_e4b_ignores_packed_rope_freq_tensor() {
        let gemma4 = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let other = StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: true,
            is_tail: false,
        };

        assert!(!RealGemmaBackend::should_use_packed_rope_freqs(&gemma4));
        assert!(RealGemmaBackend::should_use_packed_rope_freqs(&other));
    }

    #[test]
    #[ignore = "local real-artifact regression; run explicitly"]
    fn local_real_e4b_two_stage_capped_output_is_deterministic_if_artifacts_present() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../compute-backend/out/gemma-e4b-2stage");
        let stage1_path = base.join("packed-stage-1/stage-1-required.index.json");
        let stage2_path = base.join("packed-stage-2/stage-2-required.index.json");
        let vocab_path = base.join("vocab.json");
        let scores_path = base.join("vocab_scores.json");
        if !stage1_path.exists() || !stage2_path.exists() || !vocab_path.exists() {
            eprintln!("skipping local real-artifact regression; packed Gemma artifacts not found");
            return;
        }

        let run_once = |disable_ple: bool| -> StageSample {
            let mut head = RealGemmaBackend::new(&stage1_path);
            head.set_debug_layer_cap(Some(6));
            head.set_debug_vocab_cap(Some(8192));
            head.set_debug_disable_ple(disable_ple);
            let sp = if scores_path.exists() {
                Some(scores_path.as_path())
            } else {
                None
            };
            head.load_tokenizer(&vocab_path, sp).unwrap();
            head.load_layout(StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-1".into(),
                start_layer: 0,
                end_layer: 20,
                is_head: true,
                is_tail: false,
            })
            .unwrap();

            let mut tail = RealGemmaBackend::new(&stage2_path);
            tail.set_debug_layer_cap(Some(6));
            tail.set_debug_vocab_cap(Some(8192));
            tail.set_debug_disable_ple(disable_ple);
            tail.load_tokenizer(&vocab_path, sp).unwrap();
            tail.load_layout(StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-2".into(),
                start_layer: 21,
                end_layer: 41,
                is_head: false,
                is_tail: true,
            })
            .unwrap();

            let head_output = head
                .begin_prompt(
                    "req-local-determinism",
                    "The capital of France is",
                    Some(1),
                    0,
                )
                .unwrap();
            let tail_output = tail.continue_forward(head_output).unwrap();
            tail.sample_tail(tail_output).unwrap()
        };

        let with_ple_a = run_once(false);
        let with_ple_b = run_once(false);
        assert_eq!(with_ple_a.token_ids, with_ple_b.token_ids);
        assert_eq!(with_ple_a.text, with_ple_b.text);
        assert!(!with_ple_a.token_ids.is_empty());
        assert!(!with_ple_a.text.is_empty());

        let without_ple_a = run_once(true);
        let without_ple_b = run_once(true);
        assert_eq!(without_ple_a.token_ids, without_ple_b.token_ids);
        assert_eq!(without_ple_a.text, without_ple_b.text);
        assert!(!without_ple_a.token_ids.is_empty());
        assert!(!without_ple_a.text.is_empty());
    }

    #[test]
    #[ignore = "local real-artifact correctness regression; run explicitly"]
    fn local_real_e4b_two_stage_output_matches_paris_if_artifacts_present() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../compute-backend/out/gemma-e4b-2stage");
        let stage1_path = base.join("packed-stage-1/stage-1-required.index.json");
        let stage2_path = base.join("packed-stage-2/stage-2-required.index.json");
        let vocab_path = base.join("vocab.json");
        let scores_path = base.join("vocab_scores.json");
        if !stage1_path.exists() || !stage2_path.exists() || !vocab_path.exists() {
            eprintln!(
                "skipping local real-artifact correctness regression; packed Gemma artifacts not found"
            );
            return;
        }

        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };

        let mut head = RealGemmaBackend::new(&stage1_path);
        head.load_tokenizer(&vocab_path, sp).unwrap();
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = RealGemmaBackend::new(&stage2_path);
        tail.load_tokenizer(&vocab_path, sp).unwrap();
        tail.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-2".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let head_output = head
            .begin_prompt(
                "req-local-correctness",
                "The capital of France is",
                Some(1),
                0,
            )
            .unwrap();
        let tail_output = tail.continue_forward(head_output).unwrap();
        let sample = tail.sample_tail(tail_output).unwrap();

        assert_eq!(sample.token_ids, vec![9079]);
        assert_eq!(sample.text, "Paris");
    }

    #[test]
    #[ignore = "local real-artifact prompt-set regression; run explicitly"]
    fn local_real_e4b_two_stage_outputs_match_small_prompt_set_if_artifacts_present() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../compute-backend/out/gemma-e4b-2stage");
        let stage1_path = base.join("packed-stage-1/stage-1-required.index.json");
        let stage2_path = base.join("packed-stage-2/stage-2-required.index.json");
        let vocab_path = base.join("vocab.json");
        let scores_path = base.join("vocab_scores.json");
        if !stage1_path.exists() || !stage2_path.exists() || !vocab_path.exists() {
            eprintln!(
                "skipping local real-artifact prompt-set regression; packed Gemma artifacts not found"
            );
            return;
        }

        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };

        let mut head = RealGemmaBackend::new(&stage1_path);
        head.load_tokenizer(&vocab_path, sp).unwrap();
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = RealGemmaBackend::new(&stage2_path);
        tail.load_tokenizer(&vocab_path, sp).unwrap();
        tail.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-2".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let cases = [
            ("The capital of Italy is", 13706, "Rome"),
            ("The capital of Japan is", 21038, "Tokyo"),
            ("The capital of Germany is", 15687, "Berlin"),
        ];

        for (idx, (prompt, expected_id, expected_text)) in cases.into_iter().enumerate() {
            let request_id = format!("req-local-prompt-set-{}", idx);
            let head_output = head.begin_prompt(&request_id, prompt, Some(1), 0).unwrap();
            let tail_output = tail.continue_forward(head_output).unwrap();
            let sample = tail.sample_tail(tail_output).unwrap();

            assert_eq!(sample.token_ids, vec![expected_id], "prompt={prompt:?}");
            assert_eq!(sample.text, expected_text, "prompt={prompt:?}");
        }
    }
}
