#![allow(
    clippy::collapsible_if,
    clippy::forget_non_drop,
    clippy::large_enum_variant,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::useless_conversion,
    unused_assignments,
    unused_mut,
    unused_variables
)]

use anyhow::{Context, Result, bail};
use libloading::Library;
use serde::{Deserialize, Serialize};
use stage_forward_lab::{PayloadKind, StageForwardBackend, StageLayout, StageSample, StageTensor};

pub use stage_forward_lab::{PayloadKind as StagePayloadKind, StageTensor as StageTensorPayload};
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::env::consts::EXE_SUFFIX;
use std::ffi::{CString, c_char, c_void};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::slice;
use std::thread;
use std::time::{Duration, Instant};

#[allow(non_camel_case_types, dead_code)]
mod ffi {
    use super::{c_char, c_void};

    pub enum LlamaModel {}
    pub enum LlamaContext {}
    pub enum LlamaVocab {}
    pub enum LlamaMemory {}

    pub type GgmlBackendDev = *mut c_void;
    pub type GgmlBackendBufferType = *mut c_void;
    pub type GgmlBackendSchedEvalCallback = Option<unsafe extern "C" fn()>;
    pub type GgmlAbortCallback = Option<unsafe extern "C" fn(*mut c_void) -> bool>;
    pub type LlamaProgressCallback = Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_tensor_buft_override {
        pub pattern: *const c_char,
        pub buft: GgmlBackendBufferType,
    }

    #[repr(C)]
    pub struct llama_model_kv_override {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct llama_sampler_seq_config {
        _private: [u8; 0],
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_split_mode {
        None = 0,
        Layer = 1,
        Row = 2,
        Tensor = 3,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_rope_scaling_type {
        Unspecified = -1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_pooling_type {
        Unspecified = -1,
        None = 0,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_attention_type {
        Unspecified = -1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_flash_attn_type {
        Auto = -1,
        Disabled = 0,
        Enabled = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum ggml_type {
        F32 = 0,
        F16 = 1,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_params {
        pub devices: *mut GgmlBackendDev,
        pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
        pub n_gpu_layers: i32,
        pub split_mode: llama_split_mode,
        pub main_gpu: i32,
        pub tensor_split: *const f32,
        pub progress_callback: LlamaProgressCallback,
        pub progress_callback_user_data: *mut c_void,
        pub kv_overrides: *const llama_model_kv_override,
        pub vocab_only: bool,
        pub use_mmap: bool,
        pub use_direct_io: bool,
        pub use_mlock: bool,
        pub check_tensors: bool,
        pub use_extra_bufts: bool,
        pub no_host: bool,
        pub no_alloc: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_context_params {
        pub n_ctx: u32,
        pub n_batch: u32,
        pub n_ubatch: u32,
        pub n_seq_max: u32,
        pub n_threads: i32,
        pub n_threads_batch: i32,
        pub rope_scaling_type: llama_rope_scaling_type,
        pub pooling_type: llama_pooling_type,
        pub attention_type: llama_attention_type,
        pub flash_attn_type: llama_flash_attn_type,
        pub rope_freq_base: f32,
        pub rope_freq_scale: f32,
        pub yarn_ext_factor: f32,
        pub yarn_attn_factor: f32,
        pub yarn_beta_fast: f32,
        pub yarn_beta_slow: f32,
        pub yarn_orig_ctx: u32,
        pub defrag_thold: f32,
        pub cb_eval: GgmlBackendSchedEvalCallback,
        pub cb_eval_user_data: *mut c_void,
        pub type_k: ggml_type,
        pub type_v: ggml_type,
        pub abort_callback: GgmlAbortCallback,
        pub abort_callback_data: *mut c_void,
        pub embeddings: bool,
        pub offload_kqv: bool,
        pub no_perf: bool,
        pub op_offload: bool,
        pub swa_full: bool,
        pub kv_unified: bool,
        pub samplers: *mut llama_sampler_seq_config,
        pub n_samplers: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_batch {
        pub n_tokens: i32,
        pub token: *mut i32,
        pub embd: *mut f32,
        pub pos: *mut i32,
        pub n_seq_id: *mut i32,
        pub seq_id: *mut *mut i32,
        pub logits: *mut i8,
    }
}

type FnBackendInit = unsafe extern "C" fn();
type FnModelDefaultParams = unsafe extern "C" fn() -> ffi::llama_model_params;
type FnContextDefaultParams = unsafe extern "C" fn() -> ffi::llama_context_params;
type FnModelLoadFromFile =
    unsafe extern "C" fn(*const c_char, ffi::llama_model_params) -> *mut ffi::LlamaModel;
type FnModelFree = unsafe extern "C" fn(*mut ffi::LlamaModel);
type FnInitFromModel =
    unsafe extern "C" fn(*mut ffi::LlamaModel, ffi::llama_context_params) -> *mut ffi::LlamaContext;
type FnContextFree = unsafe extern "C" fn(*mut ffi::LlamaContext);
type FnGetMemory = unsafe extern "C" fn(*const ffi::LlamaContext) -> *mut ffi::LlamaMemory;
type FnMemoryClear = unsafe extern "C" fn(*mut ffi::LlamaMemory, bool);
type FnMemorySeqRm = unsafe extern "C" fn(*mut ffi::LlamaMemory, i32, i32, i32) -> bool;
type FnModelGetVocab = unsafe extern "C" fn(*const ffi::LlamaModel) -> *const ffi::LlamaVocab;
type FnModelNEmbdOut = unsafe extern "C" fn(*const ffi::LlamaModel) -> i32;
type FnVocabNTokens = unsafe extern "C" fn(*const ffi::LlamaVocab) -> i32;
type FnTokenize = unsafe extern "C" fn(
    *const ffi::LlamaVocab,
    *const c_char,
    i32,
    *mut i32,
    i32,
    bool,
    bool,
) -> i32;
type FnTokenToPiece =
    unsafe extern "C" fn(*const ffi::LlamaVocab, i32, *mut c_char, i32, i32, bool) -> i32;
type FnDecode = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch) -> i32;
type FnDecodeHead = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32) -> i32;
type FnDecodeMiddle =
    unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32, i32) -> i32;
type FnDecodeTail = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32) -> i32;
type FnGetEmbeddings = unsafe extern "C" fn(*mut ffi::LlamaContext) -> *mut f32;
type FnGetLogitsIth = unsafe extern "C" fn(*mut ffi::LlamaContext, i32) -> *mut f32;
type FnVocabIsEog = unsafe extern "C" fn(*const ffi::LlamaVocab, i32) -> bool;
type FnSynchronize = unsafe extern "C" fn(*mut ffi::LlamaContext);

pub(crate) struct LlamaApi {
    _deps: Vec<Library>,
    _llama: Library,
    backend_init: FnBackendInit,
    model_default_params: FnModelDefaultParams,
    context_default_params: FnContextDefaultParams,
    model_load_from_file: FnModelLoadFromFile,
    model_free: FnModelFree,
    init_from_model: FnInitFromModel,
    context_free: FnContextFree,
    get_memory: FnGetMemory,
    memory_clear: FnMemoryClear,
    memory_seq_rm: FnMemorySeqRm,
    model_get_vocab: FnModelGetVocab,
    model_n_embd_out: FnModelNEmbdOut,
    vocab_n_tokens: FnVocabNTokens,
    tokenize: FnTokenize,
    token_to_piece: FnTokenToPiece,
    decode: FnDecode,
    decode_head: FnDecodeHead,
    decode_middle: FnDecodeMiddle,
    decode_tail: FnDecodeTail,
    get_embeddings: FnGetEmbeddings,
    get_logits_ith: FnGetLogitsIth,
    vocab_is_eog: FnVocabIsEog,
    synchronize: FnSynchronize,
}

impl LlamaApi {
    pub(crate) fn load() -> Result<Self> {
        let lib_dir = resolve_vendor_lib_dir()?;
        let mut dylibs: Vec<PathBuf> = fs::read_dir(&lib_dir)
            .with_context(|| format!("reading {}", lib_dir.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| {
                        let ext = ext.to_ascii_lowercase();
                        ext == "dylib" || ext == "so" || ext == "dll"
                    })
                    .unwrap_or(false)
            })
            .collect();
        dylibs.sort();

        let llama_path = dylibs
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains("llama"))
                    .unwrap_or(false)
            })
            .cloned()
            .with_context(|| format!("no libllama found under {}", lib_dir.display()))?;

        let mut deps = Vec::new();
        for path in dylibs.iter().filter(|path| *path != &llama_path) {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            // Only pre-load ggml family deps. Any stray libllama* file (e.g.
            // left behind by a previous install with a different ABI version)
            // is skipped — we already resolved the single libllama we want.
            if !(name.contains("ggml")) {
                continue;
            }
            let lib = unsafe { Library::new(path) }
                .with_context(|| format!("loading dependency {}", path.display()))?;
            deps.push(lib);
        }

        let llama = unsafe { Library::new(&llama_path) }
            .with_context(|| format!("loading {}", llama_path.display()))?;

        unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
            Ok(*unsafe { lib.get::<T>(name)? })
        }

        let api = unsafe {
            Self {
                backend_init: load_symbol(&llama, b"llama_backend_init\0")?,
                model_default_params: load_symbol(&llama, b"llama_model_default_params\0")?,
                context_default_params: load_symbol(&llama, b"llama_context_default_params\0")?,
                model_load_from_file: load_symbol(&llama, b"llama_model_load_from_file\0")?,
                model_free: load_symbol(&llama, b"llama_model_free\0")?,
                init_from_model: load_symbol(&llama, b"llama_init_from_model\0")?,
                context_free: load_symbol(&llama, b"llama_free\0")?,
                get_memory: load_symbol(&llama, b"llama_get_memory\0")?,
                memory_clear: load_symbol(&llama, b"llama_memory_clear\0")?,
                memory_seq_rm: load_symbol(&llama, b"llama_memory_seq_rm\0")?,
                model_get_vocab: load_symbol(&llama, b"llama_model_get_vocab\0")?,
                model_n_embd_out: load_symbol(&llama, b"llama_model_n_embd_out\0")?,
                vocab_n_tokens: load_symbol(&llama, b"llama_vocab_n_tokens\0")?,
                tokenize: load_symbol(&llama, b"llama_tokenize\0")?,
                token_to_piece: load_symbol(&llama, b"llama_token_to_piece\0")?,
                decode: load_symbol(&llama, b"llama_decode\0")?,
                decode_head: load_symbol(&llama, b"llama_decode_head\0")?,
                decode_middle: load_symbol(&llama, b"llama_decode_middle\0")?,
                decode_tail: load_symbol(&llama, b"llama_decode_tail\0")?,
                get_embeddings: load_symbol(&llama, b"llama_get_embeddings\0")?,
                get_logits_ith: load_symbol(&llama, b"llama_get_logits_ith\0")?,
                vocab_is_eog: load_symbol(&llama, b"llama_vocab_is_eog\0")?,
                synchronize: load_symbol(&llama, b"llama_synchronize\0")?,
                _deps: deps,
                _llama: llama,
            }
        };

        unsafe { (api.backend_init)() };

        Ok(api)
    }
}

pub(crate) struct LlamaModelHandle {
    pub(crate) model: *mut ffi::LlamaModel,
}

impl LlamaModelHandle {
    pub(crate) fn new(api: &LlamaApi, model_path: &Path) -> Result<Self> {
        let path = CString::new(model_path.to_string_lossy().as_bytes())?;
        let force_cpu = std::env::var_os("LLAMA_STAGE_FORCE_CPU").is_some();

        let mut mparams = unsafe { (api.model_default_params)() };
        mparams.n_gpu_layers = if force_cpu { 0 } else { -1 };
        mparams.split_mode = ffi::llama_split_mode::None;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        let model = unsafe { (api.model_load_from_file)(path.as_ptr(), mparams) };
        if model.is_null() {
            bail!("failed to load model {}", model_path.display());
        }

        Ok(Self { model })
    }

    fn vocab(&self, api: &LlamaApi) -> *const ffi::LlamaVocab {
        unsafe { (api.model_get_vocab)(self.model) }
    }

    fn hidden_dim(&self, api: &LlamaApi) -> usize {
        unsafe { (api.model_n_embd_out)(self.model) as usize }
    }

    pub(crate) fn create_session(&self, api: &LlamaApi) -> Result<LlamaSession> {
        let n_ctx = env_u32("LLAMA_STAGE_N_CTX").unwrap_or(8192);
        let n_batch = env_u32("LLAMA_STAGE_N_BATCH").unwrap_or(2048);
        let n_ubatch = env_u32("LLAMA_STAGE_N_UBATCH").unwrap_or(n_batch);
        self.create_session_with_limits(api, n_ctx, n_batch, n_ubatch)
    }

    pub(crate) fn create_session_with_limits(
        &self,
        api: &LlamaApi,
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
    ) -> Result<LlamaSession> {
        let force_cpu = std::env::var_os("LLAMA_STAGE_FORCE_CPU").is_some();

        let mut cparams = unsafe { (api.context_default_params)() };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        cparams.n_ctx = n_ctx;
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_ubatch.min(n_batch);
        cparams.n_seq_max = 1;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
        cparams.pooling_type = ffi::llama_pooling_type::None;
        cparams.rope_scaling_type = ffi::llama_rope_scaling_type::Unspecified;
        cparams.attention_type = ffi::llama_attention_type::Unspecified;
        cparams.flash_attn_type = ffi::llama_flash_attn_type::Enabled;
        let force_f32_kv = std::env::var_os("LLAMA_STAGE_FORCE_F32_KV").is_some();
        let kv_type = if force_f32_kv {
            ffi::ggml_type::F32
        } else {
            ffi::ggml_type::F16
        };
        cparams.type_k = kv_type;
        cparams.type_v = kv_type;
        cparams.offload_kqv = !force_cpu;
        cparams.op_offload = !force_cpu;
        cparams.kv_unified = std::env::var_os("LLAMA_STAGE_KV_UNIFIED").is_some();
        cparams.embeddings = false;

        let ctx = unsafe { (api.init_from_model)(self.model, cparams) };
        if ctx.is_null() {
            bail!("failed to create context from loaded model");
        }

        Ok(LlamaSession { ctx })
    }
}

pub(crate) struct LlamaSession {
    pub(crate) ctx: *mut ffi::LlamaContext,
}

impl LlamaSession {
    fn clear_memory(&self, api: &LlamaApi) {
        let memory = unsafe { (api.get_memory)(self.ctx) };
        if !memory.is_null() {
            unsafe { (api.memory_clear)(memory, true) };
        }
    }

    fn destroy(self, api: &LlamaApi) {
        unsafe { (api.context_free)(self.ctx) };
    }
}

struct OwnedBatch {
    _tokens: Option<Vec<i32>>,
    _embd: Option<Vec<f32>>,
    _logits: Option<Vec<i8>>,
    raw: ffi::llama_batch,
}

impl OwnedBatch {
    fn token_only(tokens: Vec<i32>) -> Self {
        let n_tokens = tokens.len() as i32;
        let mut tokens = tokens;
        let raw = ffi::llama_batch {
            n_tokens,
            token: tokens.as_mut_ptr(),
            embd: std::ptr::null_mut(),
            pos: std::ptr::null_mut(),
            n_seq_id: std::ptr::null_mut(),
            seq_id: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
        };
        Self {
            _tokens: Some(tokens),
            _embd: None,
            _logits: None,
            raw,
        }
    }

    fn token_and_hidden(tokens: Option<Vec<i32>>, hidden: Vec<f32>, token_count: usize) -> Self {
        let mut token_buf = tokens;
        let mut embd = hidden;
        let n_tokens = token_count as i32;

        let raw = ffi::llama_batch {
            n_tokens,
            token: token_buf
                .as_mut()
                .map(|tokens| tokens.as_mut_ptr())
                .unwrap_or(std::ptr::null_mut()),
            embd: embd.as_mut_ptr(),
            pos: std::ptr::null_mut(),
            n_seq_id: std::ptr::null_mut(),
            seq_id: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
        };

        Self {
            _tokens: token_buf,
            _embd: Some(embd),
            _logits: None,
            raw,
        }
    }

    /// Builds a batch from a stack of pre-computed hidden states (k+1 wide for
    /// spec verification) and requests logits at every batch position. Used
    /// only by the tail's verify path — the per-position logits let us
    /// greedy-sample each candidate position to compare against the draft.
    /// Mirrors `token_and_hidden` (carries the token IDs as well) so the tail
    /// sees the same batch shape as in the production single-step path.
    fn hidden_with_per_pos_logits(tokens: Vec<i32>, hidden: Vec<f32>, token_count: usize) -> Self {
        let mut token_buf = tokens;
        let mut embd = hidden;
        let mut logits = vec![1i8; token_count];
        let n_tokens = token_count as i32;

        let raw = ffi::llama_batch {
            n_tokens,
            token: token_buf.as_mut_ptr(),
            embd: embd.as_mut_ptr(),
            pos: std::ptr::null_mut(),
            n_seq_id: std::ptr::null_mut(),
            seq_id: std::ptr::null_mut(),
            logits: logits.as_mut_ptr(),
        };

        Self {
            _tokens: Some(token_buf),
            _embd: Some(embd),
            _logits: Some(logits),
            raw,
        }
    }
}

/// Find the index of the maximum value in `logits`, returning (index, value).
///
/// Hot loop: this is called once per generated token (`greedy_sample`,
/// `greedy_step` for the draft) and `k+1` times per spec-decode round
/// (`verify_batch_at_tail`). For Gemma-4 (vocab=262144) the prior `max_by` +
/// `partial_cmp` closure pattern took ~8ms per scan; this tight `if v > best`
/// loop auto-vectorizes under `-O3` and drops it to <1ms, removing the
/// dominant `tail_verify_sample_us` bucket from the spec path.
///
/// Returns the lowest index of the max value when there are ties (matching
/// the prior iterator behavior). Returns `None` if the slice is empty.
#[inline]
fn argmax_f32(logits: &[f32]) -> Option<(usize, f32)> {
    if logits.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    // Skip i=0; the loop body is branch-predictor-friendly because `if v > best`
    // is rarely taken once we're past the first few iterations.
    for i in 1..logits.len() {
        let v = unsafe { *logits.get_unchecked(i) };
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some((best_idx, best_val))
}

#[derive(Clone)]
struct CachedSample {
    sample: StageSample,
    token_id: i32,
    is_eog: bool,
}

struct SessionState {
    session: LlamaSession,
    cached_sample: Option<CachedSample>,
    /// Next free KV-cache position for this sequence. Maintained at every
    /// successful decode site so speculative-decode verification can compute
    /// the rollback target (`keep_count = n_pos_before + accepted + 1`) after
    /// partial draft acceptance.
    n_pos: i32,
    last_profile: StageNodeProfile,
}

struct BackendState {
    layout: Option<StageLayout>,
    model: Option<LlamaModelHandle>,
    sessions: HashMap<String, SessionState>,
    token_piece_cache: HashMap<i32, String>,
}

pub struct LlamaStageBackend {
    api: LlamaApi,
    model_path: PathBuf,
    state: RefCell<BackendState>,
}

// SAFETY:
// Access to a backend instance is serialized by the caller. The sidecar binaries
// handle one request at a time, and the daemon stage runtime wraps the backend
// behind a mutex before sharing it across async tasks. The underlying llama
// handles are not thread-safe for concurrent mutation; callers must not use the
// same backend instance concurrently without external synchronization.
unsafe impl Send for LlamaStageBackend {}
unsafe impl Sync for LlamaStageBackend {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyTokenSample {
    pub token_id: i32,
    pub piece: String,
    pub is_eog: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyCompletion {
    pub text: String,
    pub completion_tokens: u32,
    pub token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StageNodeRequest {
    Info,
    Tokenize {
        text: String,
    },
    BeginPrompt {
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
    },
    ContinueHeadTokens {
        request_id: String,
        token_ids: Vec<i32>,
        max_tokens: Option<u32>,
    },
    ContinueForward {
        tensor: StageTensor,
    },
    ContinueForwardTokens {
        tensor: StageTensor,
        token_ids: Vec<i32>,
        clear_memory: bool,
    },
    SampleTail {
        tensor: StageTensor,
    },
    SampleTailToken {
        tensor: StageTensor,
    },
    ClearDecodeSession {
        request_id: String,
    },
    // Speculative decoding (spec_decode_v1). Head produces `draft_k` candidate
    // tokens via a small draft model, then runs the head stage of the target
    // model on those tokens in a single batched forward. Returns the stacked
    // hidden states + the candidate ids the tail will verify.
    ContinueHeadDraftK {
        request_id: String,
        last_token: i32,
        draft_k: u32,
        max_tokens: Option<u32>,
    },
    // Tail-side verification: runs the tail stage on `tensor` (k stacked hidden
    // states), greedy-samples each position, and accepts the longest prefix
    // that matches `draft_tokens`. KV writes for accepted positions are
    // committed; rejected suffix is rolled back before responding.
    ContinueForwardVerifyK {
        request_id: String,
        tensor: StageTensor,
        last_token: i32,
        draft_tokens: Vec<i32>,
        clear_memory: bool,
    },
    // KV-cache rollback after partial draft acceptance. `keep_count` is the
    // number of token positions to retain from the start of the sequence.
    RollbackKv {
        request_id: String,
        keep_count: u32,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageNodeProfile {
    #[serde(default)]
    pub raw_request_bytes: usize,
    #[serde(default)]
    pub tail_decode_kernel_us: u64,
    #[serde(default)]
    pub tail_sample_us: u64,
    #[serde(default)]
    pub tail_verify_sample_us: u64,
    #[serde(default)]
    pub tail_verify_detok_us: u64,
    #[serde(default)]
    pub tail_verify_rollback_us: u64,
    /// Time spent in `llama_synchronize` after `decode_tail` to wait for
    /// pending Metal GPU work. Surfaces what `tail_decode_kernel_us` was
    /// previously hiding (decode_tail submits async on Metal and returns
    /// before completion). Tail-only; zero for head/middle stages.
    #[serde(default)]
    pub tail_sync_us: u64,
    #[serde(default)]
    pub raw_response_bytes: usize,
    #[serde(default)]
    pub request_json_encode_ms: u64,
    #[serde(default)]
    pub request_json_encode_us: u64,
    #[serde(default)]
    pub response_json_decode_ms: u64,
    #[serde(default)]
    pub response_json_decode_us: u64,
    #[serde(default)]
    pub request_write_ms: u64,
    #[serde(default)]
    pub request_write_us: u64,
    #[serde(default)]
    pub response_read_ms: u64,
    #[serde(default)]
    pub response_read_us: u64,
    #[serde(default)]
    pub server_request_json_decode_ms: u64,
    #[serde(default)]
    pub server_request_json_decode_us: u64,
    #[serde(default)]
    pub server_handle_ms: u64,
    #[serde(default)]
    pub server_handle_us: u64,
    #[serde(default)]
    pub server_response_json_encode_ms: u64,
    #[serde(default)]
    pub server_response_json_encode_us: u64,
    #[serde(default)]
    pub server_response_write_ms: u64,
    #[serde(default)]
    pub server_response_write_us: u64,
    #[serde(default)]
    pub tensor_pack_ms: u64,
    #[serde(default)]
    pub tensor_pack_us: u64,
    #[serde(default)]
    pub tensor_unpack_ms: u64,
    #[serde(default)]
    pub tensor_unpack_us: u64,
    #[serde(default)]
    pub hidden_bytes: usize,
    #[serde(default)]
    pub token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageNodeResponse {
    Info {
        info: StageNodeInfo,
    },
    TokenIds {
        token_ids: Vec<i32>,
    },
    // `tail_sample` is populated when the tail stage produces a token as a
    // side-effect of the decode (i.e. on ContinueForwardTokens / BeginPrompt
    // that lands on the tail). It lets the gateway skip a separate
    // SampleTailToken round-trip per token. Optional so mixed-version pairs
    // still interoperate: if absent, the gateway falls back to the old RTT.
    Tensor {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tensor: Option<StageTensor>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tail_sample: Option<GreedyTokenSample>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<StageNodeProfile>,
    },
    Sample {
        sample: StageSample,
    },
    TokenSample {
        sample: GreedyTokenSample,
    },
    // Result of ContinueForwardVerifyK. `accepted_count` is the number of
    // draft tokens that matched the target's prediction (0..=draft_k).
    // `accepted_token_ids` has length accepted_count + 1: the accepted draft
    // prefix plus one bonus token (the target's own prediction at the
    // divergence point, or after the full match). `is_eog` is true if the
    // bonus token is an end-of-generation marker.
    VerifiedBatch {
        accepted_count: u32,
        accepted_token_ids: Vec<i32>,
        // Detokenized piece for each id in `accepted_token_ids`, in matching
        // order. Populated so the gateway can stream text without an extra
        // round-trip; same `token_to_piece` path as `cached_tail_sample`.
        #[serde(default)]
        accepted_pieces: Vec<String>,
        is_eog: bool,
        // Tail-side profile for the verify call: decode kernel, sample loop,
        // detokenize, rollback. Optional so older peers without these fields
        // still interoperate (gateway treats absent as zero).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<StageNodeProfile>,
    },
    Ack,
    Error {
        message: String,
    },
}

#[derive(Debug, Clone)]
pub struct StageNodeConfig {
    pub model_path: PathBuf,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
}

pub const LLAMA_STAGE_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageNodeInfo {
    pub protocol_version: u32,
    pub model_id: String,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
    // Capability flag for spec_decode_v1 (ContinueHeadDraftK /
    // ContinueForwardVerifyK / RollbackKv). Defaulted false for back-compat
    // with v≤0.2.34 peers that omit the field.
    #[serde(default)]
    pub spec_decode_v1: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageTimings {
    pub head_prefill_ms: u64,
    pub head_decode_ms: u64,
    pub tail_decode_ms: u64,
    pub sample_ms: u64,
    pub transfer_bytes: usize,
    pub ttft_ms: u64,
    pub total_ms: u64,
    #[serde(default)]
    pub head_prefill_us: u64,
    #[serde(default)]
    pub head_decode_us: u64,
    #[serde(default)]
    pub tail_decode_us: u64,
    #[serde(default)]
    pub sample_us: u64,
    #[serde(default)]
    pub ttft_us: u64,
    #[serde(default)]
    pub total_us: u64,
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub decode_steps: u32,
    #[serde(default)]
    pub total_transfer_bytes: usize,
    #[serde(default)]
    pub head_hidden_bytes_prefill: usize,
    #[serde(default)]
    pub head_hidden_bytes_decode: usize,
    #[serde(default)]
    pub head_pack_ms: u64,
    #[serde(default)]
    pub head_pack_us: u64,
    #[serde(default)]
    pub tail_unpack_ms: u64,
    #[serde(default)]
    pub tail_unpack_us: u64,
    #[serde(default)]
    pub stage_request_json_encode_ms: u64,
    #[serde(default)]
    pub stage_request_json_encode_us: u64,
    #[serde(default)]
    pub stage_response_json_decode_ms: u64,
    #[serde(default)]
    pub stage_response_json_decode_us: u64,
    #[serde(default)]
    pub stage_request_write_ms: u64,
    #[serde(default)]
    pub stage_request_write_us: u64,
    #[serde(default)]
    pub stage_response_read_ms: u64,
    #[serde(default)]
    pub stage_response_read_us: u64,
    #[serde(default)]
    pub stage_server_request_json_decode_ms: u64,
    #[serde(default)]
    pub stage_server_request_json_decode_us: u64,
    #[serde(default)]
    pub stage_server_handle_ms: u64,
    #[serde(default)]
    pub stage_server_handle_us: u64,
    #[serde(default)]
    pub stage_server_response_json_encode_ms: u64,
    #[serde(default)]
    pub stage_server_response_json_encode_us: u64,
    #[serde(default)]
    pub stage_server_response_write_ms: u64,
    #[serde(default)]
    pub stage_server_response_write_us: u64,
    #[serde(default)]
    pub inline_sample_hits: u64,
    #[serde(default)]
    pub sample_rpc_fallbacks: u64,
    // True iff the gateway took the speculative-decode path for this run
    // (draft engine loaded AND both peers advertise spec_decode_v1 AND spec
    // config enabled). Surfaces into gateway_timings JSON so callers can
    // self-diagnose without log access.
    #[serde(default)]
    pub spec_active: bool,
    // Number of spec rounds executed (one head batch + one tail verify each).
    #[serde(default)]
    pub spec_rounds: u64,
    // Total draft tokens proposed across all rounds (sum of K).
    #[serde(default)]
    pub spec_drafts_proposed: u64,
    // Total drafts accepted by tail verify (bonus token excluded — bonus is
    // always +1 per round). spec_drafts_accepted / spec_drafts_proposed gives
    // the per-token acceptance rate.
    #[serde(default)]
    pub spec_drafts_accepted: u64,
    #[serde(default)]
    pub spec_draft_ms: u64,
    #[serde(default)]
    pub spec_draft_us: u64,
    #[serde(default)]
    pub spec_verify_ms: u64,
    #[serde(default)]
    pub spec_verify_us: u64,
    #[serde(default)]
    pub spec_rollback_ms: u64,
    #[serde(default)]
    pub spec_rollback_us: u64,
    #[serde(default)]
    pub tail_decode_kernel_us: u64,
    #[serde(default)]
    pub tail_sample_us: u64,
    #[serde(default)]
    pub tail_verify_sample_us: u64,
    #[serde(default)]
    pub tail_verify_detok_us: u64,
    #[serde(default)]
    pub tail_verify_rollback_us: u64,
    #[serde(default)]
    pub tail_sync_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageCompletion {
    pub text: String,
    pub completion_tokens: u32,
    pub token_ids: Vec<i32>,
    pub timings: RemoteStageTimings,
}

pub struct TcpStageClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RpcProfile {
    pub raw_request_bytes: usize,
    pub raw_response_bytes: usize,
    pub request_json_encode_ms: u64,
    pub request_json_encode_us: u64,
    pub response_json_decode_ms: u64,
    pub response_json_decode_us: u64,
    pub request_write_ms: u64,
    pub request_write_us: u64,
    pub response_read_ms: u64,
    pub response_read_us: u64,
    pub used_binary_framing: bool,
}

pub struct TcpGatewayClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayServiceInfo {
    pub protocol_version: u32,
    pub head_info: StageNodeInfo,
    pub tail_info: StageNodeInfo,
    pub reconnect_after_prompt: bool,
}

pub struct GatewayServiceClient {
    client: TcpGatewayClient,
    info: GatewayServiceInfo,
}

pub struct RemoteStageNodeClient {
    addr: String,
    client: Option<TcpStageClient>,
}

pub struct RemoteStagePair {
    head: RemoteStageNodeClient,
    tail: RemoteStageNodeClient,
    pub head_info: StageNodeInfo,
    pub tail_info: StageNodeInfo,
}

struct GatewaySessionState {
    max_tokens: u32,
    head_tensor: StageTensor,
    tail_tensor: Option<StageTensor>,
    text: String,
    token_ids: Vec<i32>,
    context_token_ids: Vec<i32>,
    timings: RemoteStageTimings,
    // Sample produced by the tail as a side-effect of the most recent
    // ContinueForwardTokens call, ready for the next step_completion to
    // consume without a SampleTailToken round-trip. None on the very first
    // step only if the tail is an old build that doesn't ship inline samples.
    pending_tail_sample: Option<GreedyTokenSample>,
    // Tokens that the most recent spec round committed but haven't been
    // surfaced through step_completion yet. step_completion drains this
    // before scheduling the next forward pass, which is what lets us amortize
    // a single (head_batch + tail_verify) round-trip over multiple emissions.
    pending_committed: std::collections::VecDeque<GreedyTokenSample>,
    // Tracked head-side KV position. Required for partial-accept rollback
    // (`head.rollback_kv(head_n_pos_before_batch + accepted + 1)`); the head
    // is remote so we can't query it cheaply per round.
    head_n_pos: i32,
    // Per-session adaptive spec-decode state. `current_k` shrinks on
    // consecutive low-acceptance rounds and recovers on full accepts.
    current_k: u32,
    consec_low_accept: u32,
    spec_suspended: bool,
    pending_draft_commit: Vec<i32>,
}

/// Tunables for the adaptive speculative-decoding loop. A request can opt out
/// entirely by passing `enabled=false`; otherwise `start_k` is the initial
/// draft window and the loop will shrink it on consecutive low-acceptance
/// rounds. Defaults are tuned for the current Gemma-4 E4B head/tail shards
/// with the Gemma-3 270M draft: use k=1 continuously. Even zero-accept k=1
/// rounds still return a verified target bonus token, which remains faster
/// than falling back to the legacy split single-step path.
#[derive(Debug, Clone, Copy)]
pub struct SpecDecodeConfig {
    pub enabled: bool,
    pub start_k: u32,
    pub min_k: u32,
    pub max_k: u32,
    /// After this many consecutive 0-accept rounds at the smallest k, suspend
    /// speculation for the remainder of the session.
    pub disable_after_consec_zero: u32,
    /// Opportunistic prompt-lookup drafting. When non-zero, the gateway first
    /// tries exact suffix matches in the already verified token stream before
    /// paying for the model draft. Tail verification still decides acceptance.
    pub lookup_min_ngram: u32,
    pub lookup_max_ngram: u32,
    pub lookup_max_tokens: u32,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        // Real Gemma-4 head/tail shards have low draft agreement with the
        // Gemma-3 270M draft. Keep the lookup window conservative by default:
        // 2-token repeats are common in lists/translations but were too weak
        // to safely carry multi-token guesses in varied prompt tests.
        Self {
            enabled: true,
            start_k: 1,
            min_k: 1,
            max_k: 1,
            disable_after_consec_zero: 0,
            lookup_min_ngram: 3,
            lookup_max_ngram: 16,
            lookup_max_tokens: 2,
        }
    }
}

impl SpecDecodeConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env();
        config
    }

    fn apply_env(&mut self) {
        if let Ok(raw) = env::var("LLAMA_STAGE_SPEC_ENABLED") {
            self.enabled = !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            );
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_START_K") {
            self.start_k = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_MIN_K") {
            self.min_k = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_MAX_K") {
            self.max_k = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_DISABLE_AFTER_ZERO") {
            self.disable_after_consec_zero = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_LOOKUP_MIN_NGRAM") {
            self.lookup_min_ngram = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_LOOKUP_MAX_NGRAM") {
            self.lookup_max_ngram = value;
        }
        if let Some(value) = env_u32("LLAMA_STAGE_SPEC_LOOKUP_MAX_TOKENS") {
            self.lookup_max_tokens = value;
        }

        self.min_k = self.min_k.max(1);
        self.max_k = self.max_k.max(self.min_k);
        self.start_k = self.start_k.max(self.min_k).min(self.max_k);
        if self.lookup_min_ngram == 0 || self.lookup_max_ngram == 0 {
            self.lookup_min_ngram = 0;
            self.lookup_max_ngram = 0;
        } else {
            self.lookup_min_ngram = self.lookup_min_ngram.max(1);
            self.lookup_max_ngram = self.lookup_max_ngram.max(self.lookup_min_ngram);
            self.lookup_max_tokens = self.lookup_max_tokens.max(1);
        }
    }
}

fn lookup_draft_tokens(context: &[i32], k: usize, min_ngram: usize, max_ngram: usize) -> Vec<i32> {
    if k == 0 || min_ngram == 0 || max_ngram == 0 || context.len() <= min_ngram {
        return Vec::new();
    }

    let max_n = max_ngram.min(context.len() - 1);
    if max_n < min_ngram {
        return Vec::new();
    }

    for n in (min_ngram..=max_n).rev() {
        let suffix_start = context.len() - n;
        let suffix = &context[suffix_start..];
        let last_start_with_next = context.len().saturating_sub(n + 1);
        for start in (0..=last_start_with_next).rev() {
            if &context[start..start + n] == suffix {
                let next_start = start + n;
                let next_end = (next_start + k.min(n)).min(context.len());
                if next_start < next_end {
                    return context[next_start..next_end].to_vec();
                }
            }
        }
    }

    Vec::new()
}

fn env_u32(name: &str) -> Option<u32> {
    env::var(name).ok().and_then(|raw| raw.parse::<u32>().ok())
}

pub struct RemoteStageGateway {
    pair: RemoteStagePair,
    reconnect_after_prompt: bool,
    sessions: HashMap<String, GatewaySessionState>,
    /// Optional draft engine for speculative decoding. When present AND both
    /// head/tail advertise spec_decode_v1, step_completion runs the batched
    /// spec path; otherwise it falls back to the per-token loop.
    draft: Option<DraftEngine>,
    spec_config: SpecDecodeConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GatewayStep {
    Token {
        request_id: String,
        sample: GreedyTokenSample,
        text: String,
        token_ids: Vec<i32>,
    },
    Complete {
        request_id: String,
        completion: RemoteStageCompletion,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StageGatewayRequest {
    Info,
    Tokenize {
        text: String,
    },
    Complete {
        request_id: String,
        prompt: String,
        max_tokens: u32,
    },
    BeginCompletion {
        request_id: String,
        prompt: String,
        max_tokens: u32,
    },
    StepCompletion {
        request_id: String,
    },
    ClearCompletion {
        request_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageGatewayResponse {
    Info {
        protocol_version: u32,
        head_info: StageNodeInfo,
        tail_info: StageNodeInfo,
        reconnect_after_prompt: bool,
    },
    TokenIds {
        token_ids: Vec<i32>,
    },
    Completion {
        completion: RemoteStageCompletion,
    },
    Started {
        request_id: String,
    },
    Step {
        step: GatewayStep,
    },
    Ack,
    Error {
        message: String,
    },
}

impl GatewayServiceClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let mut client = TcpGatewayClient::connect_with_timeout(addr, timeout)?;
        let info = match client.request(&StageGatewayRequest::Info)? {
            StageGatewayResponse::Info {
                protocol_version,
                head_info,
                tail_info,
                reconnect_after_prompt,
            } => GatewayServiceInfo {
                protocol_version,
                head_info,
                tail_info,
                reconnect_after_prompt,
            },
            other => bail!("expected info response, got {other:?}"),
        };
        if info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "gateway protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.protocol_version
            );
        }
        if info.head_info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "head protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.head_info.protocol_version
            );
        }
        if info.tail_info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "tail protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.tail_info.protocol_version
            );
        }
        Ok(Self { client, info })
    }

    pub fn info(&self) -> &GatewayServiceInfo {
        &self.info
    }

    pub fn complete(
        &mut self,
        request_id: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: u32,
    ) -> Result<RemoteStageCompletion> {
        match self.client.request(&StageGatewayRequest::Complete {
            request_id: request_id.into(),
            prompt: prompt.into(),
            max_tokens,
        })? {
            StageGatewayResponse::Completion { completion } => Ok(completion),
            other => bail!("expected completion response, got {other:?}"),
        }
    }

    pub fn begin_completion(
        &mut self,
        request_id: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: u32,
    ) -> Result<String> {
        let request_id = request_id.into();
        match self.client.request(&StageGatewayRequest::BeginCompletion {
            request_id: request_id.clone(),
            prompt: prompt.into(),
            max_tokens,
        })? {
            StageGatewayResponse::Started { request_id } => Ok(request_id),
            other => bail!("expected started response, got {other:?}"),
        }
    }

    pub fn step_completion(&mut self, request_id: impl Into<String>) -> Result<GatewayStep> {
        match self.client.request(&StageGatewayRequest::StepCompletion {
            request_id: request_id.into(),
        })? {
            StageGatewayResponse::Step { step } => Ok(step),
            other => bail!("expected step response, got {other:?}"),
        }
    }

    pub fn clear_completion(&mut self, request_id: impl Into<String>) -> Result<()> {
        match self.client.request(&StageGatewayRequest::ClearCompletion {
            request_id: request_id.into(),
        })? {
            StageGatewayResponse::Ack => Ok(()),
            other => bail!("expected ack response, got {other:?}"),
        }
    }

    pub fn tokenize(&mut self, text: impl Into<String>) -> Result<Vec<i32>> {
        match self
            .client
            .request(&StageGatewayRequest::Tokenize { text: text.into() })?
        {
            StageGatewayResponse::TokenIds { token_ids } => Ok(token_ids),
            other => bail!("expected token_ids response, got {other:?}"),
        }
    }

    pub fn request(&mut self, request: &StageGatewayRequest) -> Result<StageGatewayResponse> {
        self.client.request(request)
    }
}

pub fn handle_gateway_service_client_request(
    client: &mut GatewayServiceClient,
    request: StageGatewayRequest,
) -> StageGatewayResponse {
    let result: Result<StageGatewayResponse> = (|| match request {
        StageGatewayRequest::Info => Ok(StageGatewayResponse::Info {
            protocol_version: client.info.protocol_version,
            head_info: client.info.head_info.clone(),
            tail_info: client.info.tail_info.clone(),
            reconnect_after_prompt: client.info.reconnect_after_prompt,
        }),
        StageGatewayRequest::Tokenize { text } => Ok(StageGatewayResponse::TokenIds {
            token_ids: client.tokenize(text)?,
        }),
        StageGatewayRequest::Complete {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Completion {
            completion: client.complete(request_id, prompt, max_tokens)?,
        }),
        StageGatewayRequest::BeginCompletion {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Started {
            request_id: client.begin_completion(request_id, prompt, max_tokens)?,
        }),
        StageGatewayRequest::StepCompletion { request_id } => Ok(StageGatewayResponse::Step {
            step: client.step_completion(request_id)?,
        }),
        StageGatewayRequest::ClearCompletion { request_id } => {
            client.clear_completion(request_id)?;
            Ok(StageGatewayResponse::Ack)
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageGatewayResponse::Error {
            message: err.to_string(),
        },
    }
}

impl LlamaStageBackend {
    pub fn new(model_path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            api: LlamaApi::load()?,
            model_path: model_path.into(),
            state: RefCell::new(BackendState {
                layout: None,
                model: None,
                sessions: HashMap::new(),
                token_piece_cache: HashMap::new(),
            }),
        })
    }

    fn debug_flow_enabled() -> bool {
        std::env::var_os("LLAMA_STAGE_DEBUG_FLOW").is_some()
    }

    fn layout<'a>(&self, state: &'a BackendState) -> Result<&'a StageLayout> {
        state
            .layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no stage layout loaded"))
    }

    fn ensure_model<'a>(&'a self, state: &'a mut BackendState) -> Result<&'a LlamaModelHandle> {
        if state.model.is_none() {
            state.model = Some(LlamaModelHandle::new(&self.api, &self.model_path)?);
        }
        Ok(state.model.as_ref().expect("model initialized"))
    }

    fn tokenize_prompt(&self, model: &LlamaModelHandle, prompt: &str) -> Result<Vec<i32>> {
        let vocab = model.vocab(&self.api);
        if vocab.is_null() {
            bail!("model vocabulary is not available");
        }

        let prompt = CString::new(prompt)?;
        let mut tokens = vec![0i32; prompt.as_bytes().len() + 256];
        let mut n = unsafe {
            (self.api.tokenize)(
                vocab,
                prompt.as_ptr(),
                prompt.as_bytes().len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            )
        };

        if n < 0 {
            let need = (-n) as usize;
            tokens.resize(need, 0);
            n = unsafe {
                (self.api.tokenize)(
                    vocab,
                    prompt.as_ptr(),
                    prompt.as_bytes().len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    true,
                    true,
                )
            };
        }

        if n < 0 {
            bail!("tokenization failed for prompt");
        }

        tokens.truncate(n as usize);
        Ok(tokens)
    }

    fn tokenize_text(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut state = self.state.borrow_mut();
        let model = self.ensure_model(&mut state)?;
        self.tokenize_prompt(model, prompt)
    }

    fn ensure_session<'a>(
        &'a self,
        state: &'a mut BackendState,
        request_id: &str,
    ) -> Result<&'a mut SessionState> {
        if state.model.is_none() {
            state.model = Some(LlamaModelHandle::new(&self.api, &self.model_path)?);
        }
        if !state.sessions.contains_key(request_id) {
            let session = state
                .model
                .as_ref()
                .expect("model initialized")
                .create_session(&self.api)?;
            state.sessions.insert(
                request_id.to_string(),
                SessionState {
                    session,
                    cached_sample: None,
                    n_pos: 0,
                    last_profile: StageNodeProfile::default(),
                },
            );
        }
        Ok(state
            .sessions
            .get_mut(request_id)
            .expect("session initialized"))
    }

    fn clear_decode_session_inner(state: &mut BackendState, api: &LlamaApi, request_id: &str) {
        if let Some(session) = state.sessions.remove(request_id) {
            session.session.destroy(api);
        }
    }

    fn embeddings_to_tensor(
        &self,
        session: &LlamaSession,
        hidden_dim: usize,
        request_id: &str,
        prompt_text: Option<String>,
        stage_trace: Vec<String>,
        max_tokens: Option<u32>,
        token_count: usize,
    ) -> Result<(StageTensor, StageNodeProfile)> {
        let ptr = unsafe { (self.api.get_embeddings)(session.ctx) };
        if ptr.is_null() {
            bail!("llama_get_embeddings returned null");
        }

        let floats = unsafe { slice::from_raw_parts(ptr, token_count * hidden_dim) };
        if env::var_os("LLAMA_STAGE_DUMP_HIDDEN").is_some() {
            let n = 8.min(floats.len());
            tracing::debug!(
                "[llama-stage] HEAD_EMIT request={} token_count={} hidden_dim={} first{}={:?}",
                request_id,
                token_count,
                hidden_dim,
                n,
                &floats[..n]
            );
        }
        let pack_started = Instant::now();
        let byte_len = std::mem::size_of_val(floats);
        let bytes =
            unsafe { slice::from_raw_parts(floats.as_ptr() as *const u8, byte_len).to_vec() };
        let pack_elapsed = pack_started.elapsed();
        let tensor_pack_ms = pack_elapsed.as_millis() as u64;
        let tensor_pack_us = pack_elapsed.as_micros() as u64;
        let hidden_bytes = bytes.len();

        Ok((
            StageTensor {
                request_id: request_id.to_string(),
                kind: PayloadKind::HiddenState,
                stage_trace,
                hidden_dim,
                bytes,
                prompt_text,
                max_tokens,
                continuation: None,
                transient: None,
                carry: None,
            },
            StageNodeProfile {
                tensor_pack_ms,
                tensor_pack_us,
                hidden_bytes,
                token_count,
                ..StageNodeProfile::default()
            },
        ))
    }

    fn greedy_sample(
        &self,
        model: *mut ffi::LlamaModel,
        session: &LlamaSession,
        request_id: &str,
        model_id: &str,
    ) -> Result<CachedSample> {
        let vocab = unsafe { (self.api.model_get_vocab)(model) };
        let logits = unsafe { (self.api.get_logits_ith)(session.ctx, -1) };
        if logits.is_null() {
            bail!("no logits available for sampling");
        }

        let n_vocab = unsafe { (self.api.vocab_n_tokens)(vocab) as usize };
        let logits = unsafe { slice::from_raw_parts(logits, n_vocab) };
        let (token_id, _) = argmax_f32(logits).context("empty logits buffer")?;

        let token_id = token_id as i32;
        let text = self.token_to_piece(vocab, token_id)?;
        let is_eog = unsafe { (self.api.vocab_is_eog)(vocab, token_id) };
        Ok(CachedSample {
            sample: StageSample {
                request_id: request_id.to_string(),
                model_id: model_id.to_string(),
                text,
                token_ids: vec![token_id as u32],
                completion_tokens: 1,
            },
            token_id,
            is_eog,
        })
    }

    fn token_to_piece(&self, vocab: *const ffi::LlamaVocab, token: i32) -> Result<String> {
        if let Ok(state) = self.state.try_borrow() {
            if let Some(cached) = state.token_piece_cache.get(&token) {
                return Ok(cached.clone());
            }
        }

        let piece = Self::token_to_piece_uncached(&self.api, vocab, token)?;

        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.token_piece_cache.insert(token, piece.clone());
        }
        Ok(piece)
    }

    fn token_to_piece_uncached(
        api: &LlamaApi,
        vocab: *const ffi::LlamaVocab,
        token: i32,
    ) -> Result<String> {
        let mut buf = vec![0i8; 256];
        let mut n = unsafe {
            (api.token_to_piece)(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
        };

        if n < 0 {
            let need = (-n) as usize;
            buf.resize(need, 0);
            n = unsafe {
                (api.token_to_piece)(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
            };
        }

        if n < 0 {
            bail!("token_to_piece failed for token {}", token);
        }

        let bytes = buf[..n as usize]
            .iter()
            .map(|b| *b as u8)
            .collect::<Vec<u8>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn hidden_bytes_to_f32(input: &StageTensor) -> Result<(Vec<f32>, StageNodeProfile)> {
        let unpack_started = Instant::now();
        if input.hidden_dim == 0 {
            bail!("hidden_dim must be non-zero");
        }
        if input.bytes.len() % 4 != 0 {
            bail!("hidden-state bytes must be a multiple of 4");
        }
        let float_count = input.bytes.len() / 4;
        if float_count % input.hidden_dim != 0 {
            bail!(
                "hidden-state float count {} is not divisible by hidden_dim {}",
                float_count,
                input.hidden_dim
            );
        }
        let token_count = float_count / input.hidden_dim;
        let floats = unsafe {
            slice::from_raw_parts(input.bytes.as_ptr() as *const f32, float_count).to_vec()
        };
        Ok((floats, {
            let unpack_elapsed = unpack_started.elapsed();
            StageNodeProfile {
                tensor_unpack_ms: unpack_elapsed.as_millis() as u64,
                tensor_unpack_us: unpack_elapsed.as_micros() as u64,
                hidden_bytes: input.bytes.len(),
                token_count,
                ..StageNodeProfile::default()
            }
        }))
    }

    fn forward_head_tokens_impl(
        &self,
        request_id: &str,
        token_ids: Vec<i32>,
        prompt_text: Option<String>,
        max_tokens: Option<u32>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        if !layout.is_head {
            bail!(
                "head token ingress called on non-head stage {}",
                layout.stage_id
            );
        }

        let hidden_dim = self.ensure_model(&mut state)?.hidden_dim(&self.api);
        let session_state = self.ensure_session(&mut state, request_id)?;
        if clear_memory {
            session_state.cached_sample = None;
            session_state.session.clear_memory(&self.api);
            session_state.n_pos = 0;
        }

        let batch = OwnedBatch::token_only(token_ids.clone());
        let rc = unsafe {
            (self.api.decode_head)(
                session_state.session.ctx,
                batch.raw,
                layout.end_layer as i32,
            )
        };
        if rc != 0 && rc != 1 {
            bail!("llama_decode_head failed with {}", rc);
        }
        session_state.n_pos += token_ids.len() as i32;

        let (tensor, profile) = self.embeddings_to_tensor(
            &session_state.session,
            hidden_dim,
            request_id,
            prompt_text,
            vec![layout.stage_id],
            max_tokens,
            token_ids.len(),
        )?;
        session_state.last_profile = profile;
        Ok(tensor)
    }

    fn continue_forward_impl(
        &self,
        input: StageTensor,
        token_ids: Option<Vec<i32>>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        let hidden_dim = self.ensure_model(&mut state)?.hidden_dim(&self.api);
        let model_ptr = self.ensure_model(&mut state)?.model;

        let (hidden, unpack_profile) = Self::hidden_bytes_to_f32(&input)?;
        let token_count = hidden.len() / input.hidden_dim;
        if token_count == 0 {
            bail!("empty hidden-state payload");
        }

        if env::var_os("LLAMA_STAGE_DUMP_HIDDEN").is_some() {
            let n = 8.min(hidden.len());
            tracing::debug!(
                "[llama-stage] TAIL_RECV request={} stage={} token_count={} hidden_dim={} first{}={:?}",
                input.request_id,
                layout.stage_id,
                token_count,
                input.hidden_dim,
                n,
                &hidden[..n]
            );
        }

        let tokens = if let Some(token_ids) = token_ids {
            Some(token_ids)
        } else if let Some(prompt) = input.prompt_text.as_ref() {
            let model = self.ensure_model(&mut state)?;
            Some(self.tokenize_prompt(model, prompt)?)
        } else {
            None
        };

        if layout.start_layer > 0 && tokens.is_none() {
            bail!("downstream llama stage requires token ids or prompt_text");
        }

        if let Some(tokens) = tokens.as_ref() {
            if tokens.len() != token_count {
                bail!(
                    "token count {} does not match hidden token count {}",
                    tokens.len(),
                    token_count
                );
            }
        }

        let batch = OwnedBatch::token_and_hidden(tokens.clone(), hidden, token_count);
        let request_id = input.request_id.clone();
        let session_state = self.ensure_session(&mut state, &request_id)?;
        session_state.last_profile = unpack_profile;
        if clear_memory {
            session_state.cached_sample = None;
            session_state.session.clear_memory(&self.api);
            session_state.n_pos = 0;
        }

        if layout.is_tail {
            let decode_started = Instant::now();
            let rc = unsafe {
                (self.api.decode_tail)(
                    session_state.session.ctx,
                    batch.raw,
                    layout.start_layer as i32,
                )
            };
            let tail_decode_kernel_us = decode_started.elapsed().as_micros() as u64;
            if rc != 0 && rc != 1 {
                bail!("llama_decode_tail failed with {}", rc);
            }
            session_state.n_pos += token_count as i32;
            let sample_started = Instant::now();
            session_state.cached_sample = Some(self.greedy_sample(
                model_ptr,
                &session_state.session,
                &request_id,
                &layout.model_id,
            )?);
            let tail_sample_us = sample_started.elapsed().as_micros() as u64;

            let mut stage_trace = input.stage_trace.clone();
            stage_trace.push(layout.stage_id);
            session_state.last_profile.tail_decode_kernel_us = tail_decode_kernel_us;
            session_state.last_profile.tail_sample_us = tail_sample_us;
            return Ok(StageTensor {
                request_id,
                kind: PayloadKind::HiddenState,
                stage_trace,
                hidden_dim: input.hidden_dim,
                bytes: input.bytes,
                prompt_text: input.prompt_text,
                max_tokens: input.max_tokens,
                continuation: input.continuation,
                transient: input.transient,
                carry: input.carry,
            });
        }

        let rc = unsafe {
            (self.api.decode_middle)(
                session_state.session.ctx,
                batch.raw,
                layout.start_layer as i32,
                layout.end_layer as i32,
            )
        };
        if rc != 0 && rc != 1 {
            bail!("llama_decode_middle failed with {}", rc);
        }
        session_state.n_pos += token_count as i32;

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());
        let (tensor, mut profile) = self.embeddings_to_tensor(
            &session_state.session,
            hidden_dim,
            &request_id,
            input.prompt_text,
            stage_trace,
            input.max_tokens,
            token_count,
        )?;
        profile.tensor_unpack_ms = session_state.last_profile.tensor_unpack_ms;
        profile.hidden_bytes = session_state.last_profile.hidden_bytes;
        profile.token_count = session_state.last_profile.token_count;
        session_state.last_profile = profile;
        Ok(tensor)
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        self.tokenize_text(prompt)
    }

    pub fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let vocab = {
            let mut state = self.state.borrow_mut();
            let model = self.ensure_model(&mut state)?;
            unsafe { (self.api.model_get_vocab)(model.model) }
        };
        let mut text = String::new();
        for &token in tokens {
            let token = i32::try_from(token).context("token id exceeded i32 range")?;
            text.push_str(&self.token_to_piece(vocab, token)?);
        }
        Ok(text)
    }

    pub fn eos_token_id(&self) -> Result<Option<u32>> {
        let mut state = self.state.borrow_mut();
        let model = self.ensure_model(&mut state)?;
        let vocab = unsafe { (self.api.model_get_vocab)(model.model) };
        let n_vocab = unsafe { (self.api.vocab_n_tokens)(vocab) as usize };
        for token in 0..n_vocab {
            let token = token as i32;
            if unsafe { (self.api.vocab_is_eog)(vocab, token) } {
                return Ok(Some(token as u32));
            }
        }
        Ok(None)
    }

    pub fn clear_decode_session(&self, request_id: &str) -> Result<()> {
        let mut state = self.state.borrow_mut();
        Self::clear_decode_session_inner(&mut state, &self.api, request_id);
        Ok(())
    }

    pub fn node_info(&self) -> Result<StageNodeInfo> {
        let state = self.state.borrow();
        let layout = self.layout(&state)?;
        Ok(StageNodeInfo {
            protocol_version: LLAMA_STAGE_PROTOCOL_VERSION,
            model_id: layout.model_id.clone(),
            stage_id: layout.stage_id.clone(),
            start_layer: layout.start_layer,
            end_layer: layout.end_layer,
            is_head: layout.is_head,
            is_tail: layout.is_tail,
            // Phase 3 lands tail-side verify_batch_at_tail + rollback_kv, and
            // the head already supports batched continue_head_tokens. Both
            // sides can serve the spec round; gateway opts in via its own
            // config flag once both peers advertise the capability.
            spec_decode_v1: true,
        })
    }

    pub fn begin_prompt_session(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.clear_decode_session(request_id)?;
        let token_ids = self.tokenize(prompt)?;
        self.forward_head_tokens_impl(
            request_id,
            token_ids,
            Some(prompt.to_string()),
            max_tokens,
            true,
        )
    }

    pub fn continue_head_tokens(
        &self,
        request_id: &str,
        token_ids: Vec<i32>,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.forward_head_tokens_impl(request_id, token_ids, None, max_tokens, false)
    }

    pub fn continue_forward_with_tokens(
        &self,
        input: StageTensor,
        token_ids: Vec<i32>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        self.continue_forward_impl(input, Some(token_ids), clear_memory)
    }

    pub fn sample_tail_token(&self, input: StageTensor) -> Result<GreedyTokenSample> {
        let sample = self.sample_tail(input)?;
        let state = self.state.borrow();
        let cached = state
            .sessions
            .get(&sample.request_id)
            .and_then(|session| session.cached_sample.as_ref())
            .context("no cached tail token; call continue_forward first")?;
        Ok(GreedyTokenSample {
            token_id: cached.token_id,
            piece: sample.text,
            is_eog: cached.is_eog,
        })
    }

    /// Returns the cached greedy sample for the tail stage, if one is
    /// available. Used to piggyback the sample onto ContinueForwardTokens /
    /// BeginPrompt responses so the gateway can skip a separate
    /// SampleTailToken round-trip. Returns None when the backend is not a
    /// tail, the session does not exist, or no sample has been cached yet.
    pub fn cached_tail_sample(&self, request_id: &str) -> Option<GreedyTokenSample> {
        let state = self.state.borrow();
        let layout = state.layout.as_ref()?;
        if !layout.is_tail {
            return None;
        }
        let session = state.sessions.get(request_id)?;
        let cached = session.cached_sample.as_ref()?;
        let piece = cached.sample.text.clone();
        Some(GreedyTokenSample {
            token_id: cached.token_id,
            piece,
            is_eog: cached.is_eog,
        })
    }

    pub fn last_profile(&self, request_id: &str) -> Option<StageNodeProfile> {
        let state = self.state.borrow();
        state
            .sessions
            .get(request_id)
            .map(|session| session.last_profile.clone())
    }

    /// Returns the next free KV-cache position for `request_id`, or 0 if no
    /// session exists yet. Used by the gateway to compute rollback targets
    /// (`keep_count = pre_decode_n_pos + accepted_count + 1`) without having
    /// to mirror the n_pos accounting itself.
    pub fn session_n_pos(&self, request_id: &str) -> i32 {
        let state = self.state.borrow();
        state.sessions.get(request_id).map(|s| s.n_pos).unwrap_or(0)
    }

    /// Tail-side speculative-decode verification.
    ///
    /// Input layout: `tensor` carries `k+1` stacked hidden states for
    /// `[last_token, D_1, ..., D_k]`, computed by the head from the same
    /// k+1 inputs. `draft_tokens` is `[D_1, ..., D_k]` — the candidates
    /// produced by the draft model.
    ///
    /// Decodes the full k+1 batch through the tail in one call (amortizing
    /// the per-call Metal setup cost), then greedy-samples each position to
    /// find the longest prefix of drafts that matches the target's choice.
    /// Always returns one bonus token: either the target's mismatch sample
    /// at the divergence point (partial accept) or a fresh prediction past
    /// the last accepted draft (full accept).
    ///
    /// KV-cache rollback: writes for accepted positions stay; rejected suffix
    /// is removed via `memory_seq_rm`. The bonus token itself is NOT in the
    /// KV — it'll be the `last_token` for the next round and gets committed
    /// when its hidden state is decoded.
    pub fn verify_batch_at_tail(
        &self,
        request_id: &str,
        input: StageTensor,
        last_token: i32,
        draft_tokens: Vec<i32>,
        clear_memory: bool,
    ) -> Result<VerifiedOutcome> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        if !layout.is_tail {
            bail!(
                "verify_batch_at_tail requires tail stage, got {}",
                layout.stage_id
            );
        }

        let model_handle = self.ensure_model(&mut state)?;
        let model_ptr = model_handle.model;
        let hidden_dim = model_handle.hidden_dim(&self.api);

        let k = draft_tokens.len();
        if k == 0 {
            bail!("verify_batch_at_tail requires at least one draft token");
        }
        let expected_tokens = k + 1;

        if input.hidden_dim != hidden_dim {
            bail!(
                "hidden_dim mismatch: backend={hidden_dim}, tensor={}",
                input.hidden_dim
            );
        }

        let (hidden, unpack_profile) = Self::hidden_bytes_to_f32(&input)?;
        if hidden.len() != expected_tokens * hidden_dim {
            bail!(
                "verify expected {expected_tokens} tokens × hidden_dim {hidden_dim} = {} f32 entries, got {}",
                expected_tokens * hidden_dim,
                hidden.len()
            );
        }

        let session_state = self.ensure_session(&mut state, request_id)?;
        session_state.last_profile = unpack_profile;
        if clear_memory {
            session_state.cached_sample = None;
            session_state.session.clear_memory(&self.api);
            session_state.n_pos = 0;
        }

        let n_pos_before = session_state.n_pos;

        let batch_tokens: Vec<i32> = std::iter::once(last_token)
            .chain(draft_tokens.iter().copied())
            .collect();
        let batch = OwnedBatch::hidden_with_per_pos_logits(batch_tokens, hidden, expected_tokens);
        let decode_started = Instant::now();
        let rc = unsafe {
            (self.api.decode_tail)(
                session_state.session.ctx,
                batch.raw,
                layout.start_layer as i32,
            )
        };
        let tail_decode_kernel_us = decode_started.elapsed().as_micros() as u64;
        if rc != 0 && rc != 1 {
            bail!("llama_decode_tail (verify) failed with {rc}");
        }
        // Explicit GPU barrier: decode_tail submits the kernel async on Metal
        // and returns before completion, so the cost previously leaked into
        // the first get_logits_ith call. Calling synchronize here attributes
        // it honestly to its own bucket.
        let sync_started = Instant::now();
        unsafe { (self.api.synchronize)(session_state.session.ctx) };
        let tail_sync_us = sync_started.elapsed().as_micros() as u64;
        session_state.n_pos += expected_tokens as i32;

        let vocab = unsafe { (self.api.model_get_vocab)(model_ptr) };
        if vocab.is_null() {
            bail!("model vocab unavailable");
        }
        let n_vocab = unsafe { (self.api.vocab_n_tokens)(vocab) as usize };

        // Fetch all per-position logit pointers up front. On Metal,
        // `llama_get_logits_ith` may force a GPU→CPU sync the first time it's
        // called after `llama_decode_tail` (which submits the kernel async and
        // returns before completion). Pulling all `expected_tokens` pointers
        // here, then priming the buffer with a single deref, forces the GPU
        // sync to happen once before the argmax loop instead of mid-loop —
        // measured at ~50ms saved per round on Gemma-4 (vocab=262K).
        let verify_sample_started = Instant::now();
        let mut logit_ptrs: Vec<*const f32> = Vec::with_capacity(expected_tokens);
        for pos in 0..expected_tokens {
            let logits_ptr =
                unsafe { (self.api.get_logits_ith)(session_state.session.ctx, pos as i32) };
            if logits_ptr.is_null() {
                bail!("tail logits null at batch position {pos}");
            }
            logit_ptrs.push(logits_ptr);
        }
        if !logit_ptrs.is_empty() {
            let prime = unsafe { *logit_ptrs[0] };
            std::hint::black_box(prime);
        }

        let sample_at = |pos: usize| -> Result<i32> {
            let logits = unsafe { slice::from_raw_parts(logit_ptrs[pos], n_vocab) };
            let (token_id, _) = argmax_f32(logits).context("empty logits at verify position")?;
            Ok(token_id as i32)
        };

        let mut accepted_count: u32 = 0;
        let mut accepted_token_ids: Vec<i32> = Vec::with_capacity(expected_tokens);
        let mut bonus_token: i32 = 0;
        let mut all_matched = true;

        for i in 0..k {
            let sampled = sample_at(i)?;
            if sampled == draft_tokens[i] {
                accepted_count += 1;
                accepted_token_ids.push(sampled);
            } else {
                bonus_token = sampled;
                accepted_token_ids.push(sampled);
                all_matched = false;
                break;
            }
        }

        if all_matched {
            bonus_token = sample_at(k)?;
            accepted_token_ids.push(bonus_token);
        }
        let tail_verify_sample_us = verify_sample_started.elapsed().as_micros() as u64;
        let is_eog = unsafe { (self.api.vocab_is_eog)(vocab, bonus_token) };

        // KV rollback. We retain n_pos_before + accepted + 1 positions: the
        // pre-existing committed state, the last_token (always the first batch
        // input), and `accepted` matched drafts. The bonus token is NOT
        // committed to KV — it's only an output sample.
        let rollback_started = Instant::now();
        let keep_count = n_pos_before + accepted_count as i32 + 1;
        let n_pos_after_decode = n_pos_before + expected_tokens as i32;
        if keep_count < n_pos_after_decode {
            let memory = unsafe { (self.api.get_memory)(session_state.session.ctx) };
            if memory.is_null() {
                bail!("tail session memory unavailable for rollback");
            }
            let ok = unsafe { (self.api.memory_seq_rm)(memory, 0, keep_count, -1) };
            if !ok {
                bail!("tail memory_seq_rm failed (keep_count={keep_count})");
            }
        }
        session_state.n_pos = keep_count;
        let tail_verify_rollback_us = rollback_started.elapsed().as_micros() as u64;

        // Detokenize each accepted id (and the bonus) so the gateway can
        // stream text without an extra round-trip back to the head's vocab.
        // Drop the BackendState borrow before calling token_to_piece so the
        // cache lookup/insert can grab its own borrow without panicking.
        drop(state);
        let detok_started = Instant::now();
        let mut accepted_pieces = Vec::with_capacity(accepted_token_ids.len());
        for tid in &accepted_token_ids {
            accepted_pieces.push(self.token_to_piece(vocab, *tid)?);
        }
        let tail_verify_detok_us = detok_started.elapsed().as_micros() as u64;

        // Cache the bonus token so legacy `cached_tail_sample` callers still
        // see a valid "last sampled" entry after a verify round. Re-borrow now
        // that the cache calls are done.
        let bonus_piece = accepted_pieces.last().cloned().unwrap_or_default();
        {
            let mut state = self.state.borrow_mut();
            let session_state = self.ensure_session(&mut state, request_id)?;
            session_state.cached_sample = Some(CachedSample {
                sample: StageSample {
                    request_id: request_id.to_string(),
                    model_id: layout.model_id.clone(),
                    text: bonus_piece,
                    token_ids: vec![bonus_token as u32],
                    completion_tokens: 1,
                },
                token_id: bonus_token,
                is_eog,
            });
            session_state.last_profile = StageNodeProfile {
                tail_decode_kernel_us,
                tail_verify_sample_us,
                tail_verify_detok_us,
                tail_verify_rollback_us,
                tail_sync_us,
                ..StageNodeProfile::default()
            };
        }

        Ok(VerifiedOutcome {
            accepted_count,
            accepted_token_ids,
            accepted_pieces,
            is_eog,
            tail_decode_kernel_us,
            tail_verify_sample_us,
            tail_verify_detok_us,
            tail_verify_rollback_us,
            tail_sync_us,
        })
    }

    /// Roll the KV cache for `request_id` back to `keep_count` committed
    /// positions. No-op when `keep_count == n_pos`. Errors if `keep_count`
    /// exceeds the current n_pos. Used by the gateway to align the head's
    /// KV with the tail's after a partial spec-decode acceptance.
    pub fn rollback_kv(&self, request_id: &str, keep_count: u32) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let session_state = self.ensure_session(&mut state, request_id)?;
        let keep = keep_count as i32;
        if keep > session_state.n_pos {
            bail!(
                "rollback keep_count={keep} exceeds current n_pos={}",
                session_state.n_pos
            );
        }
        if keep == session_state.n_pos {
            return Ok(());
        }
        let memory = unsafe { (self.api.get_memory)(session_state.session.ctx) };
        if memory.is_null() {
            bail!("session memory unavailable for rollback");
        }
        let ok = unsafe { (self.api.memory_seq_rm)(memory, 0, keep, -1) };
        if !ok {
            bail!("memory_seq_rm failed (keep_count={keep})");
        }
        session_state.n_pos = keep;
        // Cached sample's position no longer aligns; invalidate.
        session_state.cached_sample = None;
        Ok(())
    }
}

/// Result of a tail-side speculative-decode verification round.
///
/// `accepted_count` ranges over `0..=draft_k`. `accepted_token_ids` always has
/// length `accepted_count + 1`: the matched draft prefix followed by one bonus
/// token (the target's own prediction at the divergence point on partial
/// accept, or a fresh prediction past the last draft on full accept).
#[derive(Debug, Clone)]
pub struct VerifiedOutcome {
    pub accepted_count: u32,
    pub accepted_token_ids: Vec<i32>,
    pub accepted_pieces: Vec<String>,
    pub is_eog: bool,
    pub tail_decode_kernel_us: u64,
    pub tail_verify_sample_us: u64,
    pub tail_verify_detok_us: u64,
    pub tail_verify_rollback_us: u64,
    pub tail_sync_us: u64,
}

pub fn build_stage_backend(config: &StageNodeConfig) -> Result<LlamaStageBackend> {
    let mut backend = LlamaStageBackend::new(&config.model_path)?;
    backend.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: config.stage_id.clone(),
        start_layer: config.start_layer,
        end_layer: config.end_layer,
        is_head: config.is_head,
        is_tail: config.is_tail,
    })?;
    Ok(backend)
}

impl TcpStageClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let stream = connect_tcp_stream(addr, timeout)?;
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    pub fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        self.request_profiled(request).map(|(response, _)| response)
    }

    pub fn request_profiled(
        &mut self,
        request: &StageNodeRequest,
    ) -> Result<(StageNodeResponse, RpcProfile)> {
        let encode_started = Instant::now();
        let request_bytes = rmp_serde::to_vec_named(request)?;
        let encode_elapsed = encode_started.elapsed();
        let request_json_encode_ms = encode_elapsed.as_millis() as u64;
        let request_json_encode_us = encode_elapsed.as_micros() as u64;

        let write_started = Instant::now();
        let len = u32::try_from(request_bytes.len()).context("stage request too large")?;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&request_bytes)?;
        self.stream.flush()?;
        let write_elapsed = write_started.elapsed();
        let request_write_ms = write_elapsed.as_millis() as u64;
        let request_write_us = write_elapsed.as_micros() as u64;

        let mut len_buf = [0u8; 4];
        let read_started = Instant::now();
        self.reader.read_exact(&mut len_buf)?;
        let response_len = u32::from_le_bytes(len_buf) as usize;
        let mut response_bytes = vec![0u8; response_len];
        self.reader.read_exact(&mut response_bytes)?;
        let read_elapsed = read_started.elapsed();
        let response_read_ms = read_elapsed.as_millis() as u64;
        let response_read_us = read_elapsed.as_micros() as u64;
        if response_bytes.is_empty() {
            bail!("tcp stage returned empty response");
        }

        let raw_response_bytes = response_bytes.len() + len_buf.len();
        let decode_started = Instant::now();
        let response: StageNodeResponse = rmp_serde::from_slice(&response_bytes)?;
        let decode_elapsed = decode_started.elapsed();
        let response_json_decode_ms = decode_elapsed.as_millis() as u64;
        let response_json_decode_us = decode_elapsed.as_micros() as u64;
        if let StageNodeResponse::Error { message } = &response {
            bail!("tcp stage error: {message}");
        }
        Ok((
            response,
            RpcProfile {
                raw_request_bytes: request_bytes.len() + std::mem::size_of::<u32>(),
                raw_response_bytes,
                request_json_encode_ms,
                request_json_encode_us,
                response_json_decode_ms,
                response_json_decode_us,
                request_write_ms,
                request_write_us,
                response_read_ms,
                response_read_us,
                used_binary_framing: true,
            },
        ))
    }
}

impl TcpGatewayClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let stream = connect_tcp_stream(addr, timeout)?;
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    pub fn request(&mut self, request: &StageGatewayRequest) -> Result<StageGatewayResponse> {
        let encode_started = Instant::now();
        let request_bytes = rmp_serde::to_vec_named(request)?;
        let _request_json_encode_ms = encode_started.elapsed().as_millis() as u64;

        let _write_started = Instant::now();
        let len = u32::try_from(request_bytes.len()).context("gateway request too large")?;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&request_bytes)?;
        self.stream.flush()?;
        let _request_write_ms = _write_started.elapsed().as_millis() as u64;

        let mut len_buf = [0u8; 4];
        let _read_started = Instant::now();
        self.reader.read_exact(&mut len_buf)?;
        let response_len = u32::from_le_bytes(len_buf) as usize;
        let mut response_bytes = vec![0u8; response_len];
        self.reader.read_exact(&mut response_bytes)?;
        let _response_read_ms = _read_started.elapsed().as_millis() as u64;
        if response_bytes.is_empty() {
            bail!("tcp gateway returned empty response");
        }

        let decode_started = Instant::now();
        let response: StageGatewayResponse = rmp_serde::from_slice(&response_bytes)?;
        let _response_json_decode_ms = decode_started.elapsed().as_millis() as u64;
        if let StageGatewayResponse::Error { message } = &response {
            bail!("tcp gateway error: {message}");
        }
        Ok(response)
    }
}

fn connect_tcp_stream(addr: &str, timeout: Option<Duration>) -> Result<TcpStream> {
    match timeout {
        Some(timeout) => {
            let socket_addr = addr
                .to_socket_addrs()
                .with_context(|| format!("resolving {addr}"))?
                .next()
                .with_context(|| format!("no socket addresses resolved for {addr}"))?;
            TcpStream::connect_timeout(&socket_addr, timeout).with_context(|| {
                format!(
                    "connecting to {addr} with timeout {}ms",
                    timeout.as_millis()
                )
            })
        }
        None => TcpStream::connect(addr).with_context(|| format!("connecting to {addr}")),
    }
}

impl RemoteStageNodeClient {
    pub fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let client = TcpStageClient::connect(&addr)?;
        Ok(Self {
            addr,
            client: Some(client),
        })
    }

    fn ensure_client(&mut self) -> Result<&mut TcpStageClient> {
        if self.client.is_none() {
            self.client = Some(TcpStageClient::connect(&self.addr)?);
        }
        Ok(self.client.as_mut().expect("client initialized"))
    }

    pub fn reconnect(&mut self) -> Result<()> {
        self.client = Some(TcpStageClient::connect(&self.addr)?);
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.client = None;
    }

    pub fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        self.request_profiled(request).map(|(response, _)| response)
    }

    pub fn request_profiled(
        &mut self,
        request: &StageNodeRequest,
    ) -> Result<(StageNodeResponse, RpcProfile)> {
        let first_try = self.ensure_client()?.request_profiled(request);
        match first_try {
            Ok(response) => Ok(response),
            Err(_) => {
                self.reconnect()?;
                self.ensure_client()?.request_profiled(request)
            }
        }
    }

    pub fn info(&mut self) -> Result<StageNodeInfo> {
        match self.request(&StageNodeRequest::Info)? {
            StageNodeResponse::Info { info } => {
                if info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
                    bail!(
                        "stage protocol mismatch at {}: expected {}, got {}",
                        self.addr,
                        LLAMA_STAGE_PROTOCOL_VERSION,
                        info.protocol_version
                    );
                }
                Ok(info)
            }
            other => bail!("expected info response, got {other:?}"),
        }
    }
}

fn connect_stage_node_with_retry(
    addr: &str,
    total_timeout: Duration,
) -> Result<RemoteStageNodeClient> {
    // The orchestrator pushes the head and tail assignments in parallel, so
    // when the gateway spins up on the head machine it can race the tail
    // sidecar's listener bind — and a bare TcpStream::connect just errors
    // with ECONNREFUSED. Retry until the peer accepts, or bail after the
    // total deadline elapses.
    let deadline = Instant::now() + total_timeout;
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match RemoteStageNodeClient::connect(addr) {
            Ok(client) => {
                if attempt > 1 {
                    tracing::info!("[gateway] connected to {addr} after {attempt} attempts");
                }
                return Ok(client);
            }
            Err(err) => {
                let now = Instant::now();
                let is_refused = err
                    .downcast_ref::<std::io::Error>()
                    .map(|io| io.kind() == std::io::ErrorKind::ConnectionRefused)
                    .unwrap_or(false)
                    || err.to_string().contains("Connection refused");
                if !is_refused || now >= deadline {
                    return Err(err.context(format!("connecting to {addr} (attempt {attempt})")));
                }
                std::thread::sleep(Duration::from_millis(500));
                if attempt % 10 == 0 {
                    tracing::info!(
                        "[gateway] waiting for {addr} (attempt {attempt}, {:?} remaining)",
                        deadline.saturating_duration_since(now)
                    );
                }
            }
        }
    }
}

impl RemoteStagePair {
    pub fn connect(head_addr: impl Into<String>, tail_addr: impl Into<String>) -> Result<Self> {
        let head_addr: String = head_addr.into();
        let tail_addr: String = tail_addr.into();
        let retry_window = Duration::from_secs(60);
        let mut head = connect_stage_node_with_retry(&head_addr, retry_window)?;
        let mut tail = connect_stage_node_with_retry(&tail_addr, retry_window)?;
        let head_info = head.info()?;
        let tail_info = tail.info()?;

        if !head_info.is_head {
            bail!("head endpoint {} is not marked as a head stage", head.addr);
        }
        if !tail_info.is_tail {
            bail!("tail endpoint {} is not marked as a tail stage", tail.addr);
        }
        if head_info.model_id != tail_info.model_id {
            bail!(
                "head model {} does not match tail model {}",
                head_info.model_id,
                tail_info.model_id
            );
        }
        if head_info.end_layer + 1 != tail_info.start_layer {
            // With per-stage shards each side renumbers its layers locally
            // (e.g., both report 0..N-1). The decode functions only need each
            // shard's own local indices, so a non-contiguous report is fine
            // as long as the model_ids and hidden dims line up.
            tracing::info!(
                "[gateway] note: head/tail layer ranges not contiguous in advertised indices: {}-{} then {}-{} (assuming per-stage shards)",
                head_info.start_layer,
                head_info.end_layer,
                tail_info.start_layer,
                tail_info.end_layer
            );
        }

        Ok(Self {
            head,
            tail,
            head_info,
            tail_info,
        })
    }

    pub fn reconnect(&mut self) -> Result<()> {
        self.head.reconnect()?;
        self.tail.reconnect()?;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.head.disconnect();
        self.tail.disconnect();
    }

    pub fn clear_decode_session(&mut self, request_id: &str) -> Result<()> {
        self.head.request(&StageNodeRequest::ClearDecodeSession {
            request_id: request_id.to_string(),
        })?;
        self.tail.request(&StageNodeRequest::ClearDecodeSession {
            request_id: request_id.to_string(),
        })?;
        Ok(())
    }

    pub fn run_greedy_completion(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        reconnect_after_prompt: bool,
    ) -> Result<RemoteStageCompletion> {
        let request_id = format!(
            "remote-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        self.clear_decode_session(&request_id)?;

        let prompt_tokens = match self.head.request(&StageNodeRequest::Tokenize {
            text: prompt.to_string(),
        })? {
            StageNodeResponse::TokenIds { token_ids } => token_ids,
            other => bail!("expected token_ids response, got {other:?}"),
        };

        let prompt_tokens_count = prompt_tokens.len() as u32;
        let t_head = std::time::Instant::now();
        let ((mut head_tensor, head_profile), head_rpc) =
            match self.head.request_profiled(&StageNodeRequest::BeginPrompt {
                request_id: request_id.clone(),
                prompt: prompt.to_string(),
                max_tokens: Some(max_tokens),
            })? {
                (
                    StageNodeResponse::Tensor {
                        tensor, profile, ..
                    },
                    rpc,
                ) => (
                    (
                        tensor.context("direct head begin missing tensor")?,
                        profile.unwrap_or_default(),
                    ),
                    rpc,
                ),
                (other, _) => bail!("expected tensor response, got {other:?}"),
            };
        let head_prefill_elapsed = t_head.elapsed();
        let head_prefill_ms = head_prefill_elapsed.as_millis() as u64;
        let head_prefill_us = head_prefill_elapsed.as_micros() as u64;
        let transfer_bytes = head_tensor.bytes.len();
        let mut total_transfer_bytes =
            transfer_bytes + head_rpc.raw_request_bytes + head_rpc.raw_response_bytes;
        let head_hidden_bytes_prefill = head_profile.hidden_bytes;
        let mut head_hidden_bytes_decode = 0usize;
        let mut head_pack_ms_total = head_profile.tensor_pack_ms;
        let mut head_pack_us_total = head_profile.tensor_pack_us;
        let mut tail_unpack_ms_total = 0u64;
        let mut tail_unpack_us_total = 0u64;
        let mut stage_request_json_encode_ms = head_rpc.request_json_encode_ms;
        let mut stage_request_json_encode_us = head_rpc.request_json_encode_us;
        let mut stage_response_json_decode_ms = head_rpc.response_json_decode_ms;
        let mut stage_response_json_decode_us = head_rpc.response_json_decode_us;
        let mut stage_request_write_ms = head_rpc.request_write_ms;
        let mut stage_request_write_us = head_rpc.request_write_us;
        let mut stage_response_read_ms = head_rpc.response_read_ms;
        let mut stage_response_read_us = head_rpc.response_read_us;
        let mut stage_server_request_json_decode_ms = head_profile.server_request_json_decode_ms;
        let mut stage_server_request_json_decode_us = head_profile.server_request_json_decode_us;
        let mut stage_server_handle_ms = head_profile.server_handle_ms;
        let mut stage_server_handle_us = head_profile.server_handle_us;
        let mut stage_server_response_json_encode_ms = head_profile.server_response_json_encode_ms;
        let mut stage_server_response_json_encode_us = head_profile.server_response_json_encode_us;
        let mut stage_server_response_write_ms = head_profile.server_response_write_ms;
        let mut stage_server_response_write_us = head_profile.server_response_write_us;
        let mut tail_decode_kernel_us_total = 0u64;
        let mut tail_sample_us_total = 0u64;
        let mut tail_verify_sample_us_total = 0u64;
        let mut tail_verify_detok_us_total = 0u64;
        let mut tail_verify_rollback_us_total = 0u64;

        let t_tail = std::time::Instant::now();
        let head_tensor_for_tail = head_tensor.clone();
        let ((mut tail_tensor, mut pending_sample, tail_profile), tail_rpc) = match self
            .tail
            .request_profiled(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor_for_tail,
                token_ids: prompt_tokens,
                clear_memory: true,
            })? {
            (
                StageNodeResponse::Tensor {
                    tensor,
                    tail_sample,
                    profile,
                },
                rpc,
            ) => ((tensor, tail_sample, profile.unwrap_or_default()), rpc),
            (other, _) => bail!("expected tensor response, got {other:?}"),
        };
        let tail_prefill_elapsed = t_tail.elapsed();
        let mut tail_decode_ms_total = tail_prefill_elapsed.as_millis() as u64;
        let mut tail_decode_us_total = tail_prefill_elapsed.as_micros() as u64;
        total_transfer_bytes +=
            head_tensor.bytes.len() + tail_rpc.raw_request_bytes + tail_rpc.raw_response_bytes;
        tail_unpack_ms_total += tail_profile.tensor_unpack_ms;
        tail_unpack_us_total += tail_profile.tensor_unpack_us;
        stage_request_json_encode_ms += tail_rpc.request_json_encode_ms;
        stage_request_json_encode_us += tail_rpc.request_json_encode_us;
        stage_response_json_decode_ms += tail_rpc.response_json_decode_ms;
        stage_response_json_decode_us += tail_rpc.response_json_decode_us;
        stage_request_write_ms += tail_rpc.request_write_ms;
        stage_request_write_us += tail_rpc.request_write_us;
        stage_response_read_ms += tail_rpc.response_read_ms;
        stage_response_read_us += tail_rpc.response_read_us;
        stage_server_request_json_decode_ms += tail_profile.server_request_json_decode_ms;
        stage_server_request_json_decode_us += tail_profile.server_request_json_decode_us;
        stage_server_handle_ms += tail_profile.server_handle_ms;
        stage_server_handle_us += tail_profile.server_handle_us;
        stage_server_response_json_encode_ms += tail_profile.server_response_json_encode_ms;
        stage_server_response_json_encode_us += tail_profile.server_response_json_encode_us;
        stage_server_response_write_ms += tail_profile.server_response_write_ms;
        stage_server_response_write_us += tail_profile.server_response_write_us;
        tail_decode_kernel_us_total += tail_profile.tail_decode_kernel_us;
        tail_sample_us_total += tail_profile.tail_sample_us;

        if reconnect_after_prompt {
            self.disconnect();
            self.reconnect()?;
        }

        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut head_decode_ms_total = 0u64;
        let mut head_decode_us_total = 0u64;
        let mut sample_ms_total = 0u64;
        let mut sample_us_total = 0u64;
        let mut ttft_ms = 0u64;
        let mut ttft_us = 0u64;
        let mut inline_sample_hits = if pending_sample.is_some() { 1u64 } else { 0u64 };
        let mut sample_rpc_fallbacks = 0u64;

        for step in 0..max_tokens {
            let t_sample = std::time::Instant::now();
            let sampled = if let Some(sampled) = pending_sample.take() {
                inline_sample_hits += 1;
                sampled
            } else {
                sample_rpc_fallbacks += 1;
                let tail_tensor_ref = tail_tensor
                    .clone()
                    .context("session tail tensor unavailable for sample fallback")?;
                let (response, rpc) =
                    self.tail
                        .request_profiled(&StageNodeRequest::SampleTailToken {
                            tensor: tail_tensor_ref.clone(),
                        })?;
                total_transfer_bytes +=
                    tail_tensor_ref.bytes.len() + rpc.raw_request_bytes + rpc.raw_response_bytes;
                stage_request_json_encode_ms += rpc.request_json_encode_ms;
                stage_response_json_decode_ms += rpc.response_json_decode_ms;
                stage_request_write_ms += rpc.request_write_ms;
                stage_response_read_ms += rpc.response_read_ms;
                match response {
                    StageNodeResponse::TokenSample { sample } => sample,
                    other => bail!("expected token_sample response, got {other:?}"),
                }
            };
            let sample_elapsed = t_sample.elapsed();
            sample_ms_total += sample_elapsed.as_millis() as u64;
            sample_us_total += sample_elapsed.as_micros() as u64;

            if ttft_ms == 0 {
                ttft_ms = head_prefill_ms + tail_decode_ms_total + sample_ms_total;
                ttft_us = head_prefill_us + tail_decode_us_total + sample_us_total;
            }

            text.push_str(&sampled.piece);
            token_ids.push(sampled.token_id);

            if sampled.is_eog || step + 1 >= max_tokens {
                break;
            }

            let t_head_step = std::time::Instant::now();
            let ((new_head_tensor, head_profile), head_rpc) =
                match self
                    .head
                    .request_profiled(&StageNodeRequest::ContinueHeadTokens {
                        request_id: request_id.clone(),
                        token_ids: vec![sampled.token_id],
                        max_tokens: Some(max_tokens),
                    })? {
                    (
                        StageNodeResponse::Tensor {
                            tensor, profile, ..
                        },
                        rpc,
                    ) => (
                        (
                            tensor.context("head continue missing tensor")?,
                            profile.unwrap_or_default(),
                        ),
                        rpc,
                    ),
                    (other, _) => bail!("expected tensor response, got {other:?}"),
                };
            head_tensor = new_head_tensor;
            let head_step_elapsed = t_head_step.elapsed();
            head_decode_ms_total += head_step_elapsed.as_millis() as u64;
            head_decode_us_total += head_step_elapsed.as_micros() as u64;
            total_transfer_bytes +=
                head_tensor.bytes.len() + head_rpc.raw_request_bytes + head_rpc.raw_response_bytes;
            head_hidden_bytes_decode += head_profile.hidden_bytes;
            head_pack_ms_total += head_profile.tensor_pack_ms;
            head_pack_us_total += head_profile.tensor_pack_us;
            stage_request_json_encode_ms += head_rpc.request_json_encode_ms;
            stage_request_json_encode_us += head_rpc.request_json_encode_us;
            stage_response_json_decode_ms += head_rpc.response_json_decode_ms;
            stage_response_json_decode_us += head_rpc.response_json_decode_us;
            stage_request_write_ms += head_rpc.request_write_ms;
            stage_request_write_us += head_rpc.request_write_us;
            stage_response_read_ms += head_rpc.response_read_ms;
            stage_response_read_us += head_rpc.response_read_us;
            stage_server_request_json_decode_ms += head_profile.server_request_json_decode_ms;
            stage_server_request_json_decode_us += head_profile.server_request_json_decode_us;
            stage_server_handle_ms += head_profile.server_handle_ms;
            stage_server_handle_us += head_profile.server_handle_us;
            stage_server_response_json_encode_ms += head_profile.server_response_json_encode_ms;
            stage_server_response_json_encode_us += head_profile.server_response_json_encode_us;
            stage_server_response_write_ms += head_profile.server_response_write_ms;
            stage_server_response_write_us += head_profile.server_response_write_us;

            let t_tail_step = std::time::Instant::now();
            let head_tensor_for_tail = head_tensor.clone();
            let ((new_tail_tensor, new_pending, tail_profile), tail_rpc) = match self
                .tail
                .request_profiled(&StageNodeRequest::ContinueForwardTokens {
                    tensor: head_tensor_for_tail.clone(),
                    token_ids: vec![sampled.token_id],
                    clear_memory: false,
                })? {
                (
                    StageNodeResponse::Tensor {
                        tensor,
                        tail_sample,
                        profile,
                    },
                    rpc,
                ) => ((tensor, tail_sample, profile.unwrap_or_default()), rpc),
                (other, _) => bail!("expected tensor response, got {other:?}"),
            };
            tail_tensor = new_tail_tensor;
            pending_sample = new_pending;
            let tail_step_elapsed = t_tail_step.elapsed();
            tail_decode_ms_total += tail_step_elapsed.as_millis() as u64;
            tail_decode_us_total += tail_step_elapsed.as_micros() as u64;
            total_transfer_bytes +=
                head_tensor.bytes.len() + tail_rpc.raw_request_bytes + tail_rpc.raw_response_bytes;
            tail_unpack_ms_total += tail_profile.tensor_unpack_ms;
            tail_unpack_us_total += tail_profile.tensor_unpack_us;
            stage_request_json_encode_ms += tail_rpc.request_json_encode_ms;
            stage_request_json_encode_us += tail_rpc.request_json_encode_us;
            stage_response_json_decode_ms += tail_rpc.response_json_decode_ms;
            stage_response_json_decode_us += tail_rpc.response_json_decode_us;
            stage_request_write_ms += tail_rpc.request_write_ms;
            stage_request_write_us += tail_rpc.request_write_us;
            stage_response_read_ms += tail_rpc.response_read_ms;
            stage_response_read_us += tail_rpc.response_read_us;
            stage_server_request_json_decode_ms += tail_profile.server_request_json_decode_ms;
            stage_server_request_json_decode_us += tail_profile.server_request_json_decode_us;
            stage_server_handle_ms += tail_profile.server_handle_ms;
            stage_server_handle_us += tail_profile.server_handle_us;
            stage_server_response_json_encode_ms += tail_profile.server_response_json_encode_ms;
            stage_server_response_json_encode_us += tail_profile.server_response_json_encode_us;
            stage_server_response_write_ms += tail_profile.server_response_write_ms;
            stage_server_response_write_us += tail_profile.server_response_write_us;
            tail_decode_kernel_us_total += tail_profile.tail_decode_kernel_us;
            tail_sample_us_total += tail_profile.tail_sample_us;
        }

        let total_ms =
            head_prefill_ms + head_decode_ms_total + tail_decode_ms_total + sample_ms_total;
        let total_us =
            head_prefill_us + head_decode_us_total + tail_decode_us_total + sample_us_total;

        let completion_tokens = token_ids.len() as u32;
        let result = RemoteStageCompletion {
            text,
            completion_tokens,
            token_ids,
            timings: RemoteStageTimings {
                head_prefill_ms,
                head_decode_ms: head_decode_ms_total,
                tail_decode_ms: tail_decode_ms_total,
                sample_ms: sample_ms_total,
                transfer_bytes,
                ttft_ms,
                total_ms,
                head_prefill_us: head_prefill_ms * 1000,
                head_decode_us: head_decode_ms_total * 1000,
                tail_decode_us: tail_decode_ms_total * 1000,
                sample_us: sample_ms_total * 1000,
                ttft_us: ttft_ms * 1000,
                total_us: total_ms * 1000,
                prompt_tokens: prompt_tokens_count,
                decode_steps: completion_tokens,
                total_transfer_bytes,
                head_hidden_bytes_prefill,
                head_hidden_bytes_decode,
                head_pack_ms: head_pack_ms_total,
                head_pack_us: head_pack_ms_total * 1000,
                tail_unpack_ms: tail_unpack_ms_total,
                tail_unpack_us: tail_unpack_ms_total * 1000,
                stage_request_json_encode_ms,
                stage_request_json_encode_us: stage_request_json_encode_ms * 1000,
                stage_response_json_decode_ms,
                stage_response_json_decode_us: stage_response_json_decode_ms * 1000,
                stage_request_write_ms,
                stage_request_write_us: stage_request_write_ms * 1000,
                stage_response_read_ms,
                stage_response_read_us: stage_response_read_ms * 1000,
                stage_server_request_json_decode_ms,
                stage_server_request_json_decode_us: stage_server_request_json_decode_ms * 1000,
                stage_server_handle_ms,
                stage_server_handle_us: stage_server_handle_ms * 1000,
                stage_server_response_json_encode_ms,
                stage_server_response_json_encode_us: stage_server_response_json_encode_ms * 1000,
                stage_server_response_write_ms,
                stage_server_response_write_us: stage_server_response_write_ms * 1000,
                inline_sample_hits,
                sample_rpc_fallbacks,
                spec_active: false,
                spec_rounds: 0,
                spec_drafts_proposed: 0,
                spec_drafts_accepted: 0,
                spec_draft_ms: 0,
                spec_draft_us: 0,
                spec_verify_ms: 0,
                spec_verify_us: 0,
                spec_rollback_ms: 0,
                spec_rollback_us: 0,
                tail_decode_kernel_us: tail_decode_kernel_us_total,
                tail_sample_us: tail_sample_us_total,
                tail_verify_sample_us: tail_verify_sample_us_total,
                tail_verify_detok_us: tail_verify_detok_us_total,
                tail_verify_rollback_us: tail_verify_rollback_us_total,
                tail_sync_us: 0,
            },
        };

        self.clear_decode_session(&request_id)?;
        Ok(result)
    }
}

impl RemoteStageGateway {
    pub fn connect(
        head_addr: impl Into<String>,
        tail_addr: impl Into<String>,
        reconnect_after_prompt: bool,
    ) -> Result<Self> {
        Ok(Self {
            pair: RemoteStagePair::connect(head_addr, tail_addr)?,
            reconnect_after_prompt,
            sessions: HashMap::new(),
            draft: None,
            spec_config: SpecDecodeConfig {
                enabled: false,
                ..SpecDecodeConfig::default()
            },
        })
    }

    /// Construct a gateway with a draft engine for speculative decoding.
    /// `draft_path` must point to a tokenizer-compatible draft GGUF (the canary
    /// check at startup catches catastrophic mismatches; per-vocab id parity is
    /// the caller's responsibility — see the draft_model_compat memory).
    pub fn connect_with_draft(
        head_addr: impl Into<String>,
        tail_addr: impl Into<String>,
        reconnect_after_prompt: bool,
        draft_path: impl AsRef<Path>,
        spec_config: SpecDecodeConfig,
    ) -> Result<Self> {
        let pair = RemoteStagePair::connect(head_addr, tail_addr)?;
        let draft = DraftEngine::load(draft_path)?;
        let mut spec_config = spec_config;
        if !pair.head_info.spec_decode_v1 || !pair.tail_info.spec_decode_v1 {
            // One of the peers is too old to verify batched drafts. Keep the
            // draft engine around (callers may re-negotiate after upgrade)
            // but don't take the spec path.
            spec_config.enabled = false;
        }
        Ok(Self {
            pair,
            reconnect_after_prompt,
            sessions: HashMap::new(),
            draft: Some(draft),
            spec_config,
        })
    }

    pub fn spec_config(&self) -> &SpecDecodeConfig {
        &self.spec_config
    }

    pub fn spec_active(&self) -> bool {
        self.spec_config.enabled
            && self.draft.is_some()
            && self.pair.head_info.spec_decode_v1
            && self.pair.tail_info.spec_decode_v1
    }

    pub fn head_info(&self) -> &StageNodeInfo {
        &self.pair.head_info
    }

    pub fn tail_info(&self) -> &StageNodeInfo {
        &self.pair.tail_info
    }

    pub fn reconnect_after_prompt(&self) -> bool {
        self.reconnect_after_prompt
    }

    pub fn clear_completion(&mut self, request_id: &str) -> Result<()> {
        self.sessions.remove(request_id);
        self.pair.clear_decode_session(request_id)
    }

    pub fn begin_completion(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<()> {
        if self.sessions.contains_key(request_id) {
            bail!("completion session already exists for request_id {request_id}");
        }

        self.pair.clear_decode_session(request_id)?;

        let (tokenize_response, tokenize_rpc) =
            self.pair
                .head
                .request_profiled(&StageNodeRequest::Tokenize {
                    text: prompt.to_string(),
                })?;
        let prompt_tokens = match tokenize_response {
            StageNodeResponse::TokenIds { token_ids } => token_ids,
            other => bail!("expected token_ids response, got {other:?}"),
        };

        let t_head = std::time::Instant::now();
        let ((head_tensor, head_profile), head_rpc) =
            match self
                .pair
                .head
                .request_profiled(&StageNodeRequest::BeginPrompt {
                    request_id: request_id.to_string(),
                    prompt: prompt.to_string(),
                    max_tokens: Some(max_tokens),
                })? {
                (
                    StageNodeResponse::Tensor {
                        tensor, profile, ..
                    },
                    rpc,
                ) => (
                    (
                        tensor.context("gateway begin head tensor missing")?,
                        profile.unwrap_or_default(),
                    ),
                    rpc,
                ),
                (other, _) => bail!("expected tensor response, got {other:?}"),
            };
        let head_prefill_elapsed = t_head.elapsed();
        let head_prefill_ms = head_prefill_elapsed.as_millis() as u64;
        let head_prefill_us = head_prefill_elapsed.as_micros() as u64;
        let transfer_bytes = head_tensor.bytes.len();
        let mut total_transfer_bytes = transfer_bytes
            + tokenize_rpc.raw_request_bytes
            + tokenize_rpc.raw_response_bytes
            + head_rpc.raw_request_bytes
            + head_rpc.raw_response_bytes;
        let head_hidden_bytes_prefill = head_profile.hidden_bytes;
        let head_pack_ms = head_profile.tensor_pack_ms;
        let head_pack_us = head_profile.tensor_pack_us;
        let mut tail_unpack_ms = 0u64;
        let mut tail_unpack_us = 0u64;
        let mut stage_request_json_encode_ms =
            tokenize_rpc.request_json_encode_ms + head_rpc.request_json_encode_ms;
        let mut stage_request_json_encode_us =
            tokenize_rpc.request_json_encode_us + head_rpc.request_json_encode_us;
        let mut stage_response_json_decode_ms =
            tokenize_rpc.response_json_decode_ms + head_rpc.response_json_decode_ms;
        let mut stage_response_json_decode_us =
            tokenize_rpc.response_json_decode_us + head_rpc.response_json_decode_us;
        let mut stage_request_write_ms = tokenize_rpc.request_write_ms + head_rpc.request_write_ms;
        let mut stage_request_write_us = tokenize_rpc.request_write_us + head_rpc.request_write_us;
        let mut stage_response_read_ms = tokenize_rpc.response_read_ms + head_rpc.response_read_ms;
        let mut stage_response_read_us = tokenize_rpc.response_read_us + head_rpc.response_read_us;
        let mut stage_server_request_json_decode_ms = head_profile.server_request_json_decode_ms;
        let mut stage_server_request_json_decode_us = head_profile.server_request_json_decode_us;
        let mut stage_server_handle_ms = head_profile.server_handle_ms;
        let mut stage_server_handle_us = head_profile.server_handle_us;
        let mut stage_server_response_json_encode_ms = head_profile.server_response_json_encode_ms;
        let mut stage_server_response_json_encode_us = head_profile.server_response_json_encode_us;
        let mut stage_server_response_write_ms = head_profile.server_response_write_ms;
        let mut stage_server_response_write_us = head_profile.server_response_write_us;
        let mut tail_decode_kernel_us_total = 0u64;
        let mut tail_sample_us_total = 0u64;
        let mut tail_verify_sample_us_total = 0u64;
        let mut tail_verify_detok_us_total = 0u64;
        let mut tail_verify_rollback_us_total = 0u64;

        let t_tail = std::time::Instant::now();
        let head_tensor_for_tail = head_tensor.clone();
        let ((tail_tensor, pending_tail_sample, tail_profile), tail_rpc) = match self
            .pair
            .tail
            .request_profiled(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor_for_tail,
                token_ids: prompt_tokens.clone(),
                clear_memory: true,
            })? {
            (
                StageNodeResponse::Tensor {
                    tensor,
                    tail_sample,
                    profile,
                },
                rpc,
            ) => ((tensor, tail_sample, profile.unwrap_or_default()), rpc),
            (other, _) => bail!("expected tensor response, got {other:?}"),
        };
        let tail_prefill_elapsed = t_tail.elapsed();
        let tail_decode_ms = tail_prefill_elapsed.as_millis() as u64;
        let tail_decode_us = tail_prefill_elapsed.as_micros() as u64;
        total_transfer_bytes +=
            head_tensor.bytes.len() + tail_rpc.raw_request_bytes + tail_rpc.raw_response_bytes;
        tail_unpack_ms += tail_profile.tensor_unpack_ms;
        tail_unpack_us += tail_profile.tensor_unpack_us;
        stage_request_json_encode_ms += tail_rpc.request_json_encode_ms;
        stage_request_json_encode_us += tail_rpc.request_json_encode_us;
        stage_response_json_decode_ms += tail_rpc.response_json_decode_ms;
        stage_response_json_decode_us += tail_rpc.response_json_decode_us;
        stage_request_write_ms += tail_rpc.request_write_ms;
        stage_request_write_us += tail_rpc.request_write_us;
        stage_response_read_ms += tail_rpc.response_read_ms;
        stage_response_read_us += tail_rpc.response_read_us;
        stage_server_request_json_decode_ms += tail_profile.server_request_json_decode_ms;
        stage_server_request_json_decode_us += tail_profile.server_request_json_decode_us;
        stage_server_handle_ms += tail_profile.server_handle_ms;
        stage_server_handle_us += tail_profile.server_handle_us;
        stage_server_response_json_encode_ms += tail_profile.server_response_json_encode_ms;
        stage_server_response_json_encode_us += tail_profile.server_response_json_encode_us;
        stage_server_response_write_ms += tail_profile.server_response_write_ms;
        stage_server_response_write_us += tail_profile.server_response_write_us;
        tail_decode_kernel_us_total += tail_profile.tail_decode_kernel_us;
        tail_sample_us_total += tail_profile.tail_sample_us;

        if self.reconnect_after_prompt {
            self.pair.disconnect();
            self.pair.reconnect()?;
        }

        let n_pos_after_prefill = prompt_tokens.len() as i32;
        let spec_active_now = self.spec_active();
        if spec_active_now {
            // Prime the draft engine with prompt[..n-1]; greedy_step_k() seeds
            // with the last prompt token on the first round so its logit is
            // computed against the full prompt context.
            let draft = self
                .draft
                .as_mut()
                .expect("spec_active() implies draft is Some");
            draft.reset();
            if prompt_tokens.len() > 1 {
                draft.prefill(&prompt_tokens[..prompt_tokens.len() - 1])?;
            }
        }

        self.sessions.insert(
            request_id.to_string(),
            GatewaySessionState {
                max_tokens,
                head_tensor,
                tail_tensor,
                text: String::new(),
                token_ids: Vec::new(),
                context_token_ids: prompt_tokens.clone(),
                timings: RemoteStageTimings {
                    head_prefill_ms,
                    head_decode_ms: 0,
                    tail_decode_ms,
                    sample_ms: 0,
                    transfer_bytes,
                    ttft_ms: 0,
                    total_ms: 0,
                    head_prefill_us,
                    head_decode_us: 0,
                    tail_decode_us,
                    sample_us: 0,
                    ttft_us: 0,
                    total_us: 0,
                    prompt_tokens: prompt_tokens.len() as u32,
                    decode_steps: 0,
                    total_transfer_bytes,
                    head_hidden_bytes_prefill,
                    head_hidden_bytes_decode: 0,
                    head_pack_ms,
                    head_pack_us,
                    tail_unpack_ms,
                    tail_unpack_us,
                    stage_request_json_encode_ms,
                    stage_request_json_encode_us,
                    stage_response_json_decode_ms,
                    stage_response_json_decode_us,
                    stage_request_write_ms,
                    stage_request_write_us,
                    stage_response_read_ms,
                    stage_response_read_us,
                    stage_server_request_json_decode_ms,
                    stage_server_request_json_decode_us,
                    stage_server_handle_ms,
                    stage_server_handle_us,
                    stage_server_response_json_encode_ms,
                    stage_server_response_json_encode_us,
                    stage_server_response_write_ms,
                    stage_server_response_write_us,
                    inline_sample_hits: if pending_tail_sample.is_some() { 1 } else { 0 },
                    sample_rpc_fallbacks: 0,
                    spec_active: spec_active_now,
                    spec_rounds: 0,
                    spec_drafts_proposed: 0,
                    spec_drafts_accepted: 0,
                    spec_draft_ms: 0,
                    spec_draft_us: 0,
                    spec_verify_ms: 0,
                    spec_verify_us: 0,
                    spec_rollback_ms: 0,
                    spec_rollback_us: 0,
                    tail_decode_kernel_us: tail_profile.tail_decode_kernel_us,
                    tail_sample_us: tail_profile.tail_sample_us,
                    tail_verify_sample_us: 0,
                    tail_verify_detok_us: 0,
                    tail_verify_rollback_us: 0,
                    tail_sync_us: 0,
                },
                pending_tail_sample,
                pending_committed: std::collections::VecDeque::new(),
                head_n_pos: n_pos_after_prefill,
                current_k: self.spec_config.start_k.max(self.spec_config.min_k),
                consec_low_accept: 0,
                spec_suspended: false,
                pending_draft_commit: Vec::new(),
            },
        );

        Ok(())
    }

    pub fn step_completion(&mut self, request_id: &str) -> Result<GatewayStep> {
        let mut session = self
            .sessions
            .remove(request_id)
            .with_context(|| format!("no completion session for request_id {request_id}"))?;

        let t_sample = std::time::Instant::now();
        let sampled = if let Some(sampled) = session.pending_tail_sample.take() {
            // Tail shipped the sample inline with the previous forward pass —
            // zero-cost consume, no round-trip.
            session.timings.inline_sample_hits += 1;
            sampled
        } else {
            // Fallback for older tail builds that don't populate tail_sample.
            session.timings.sample_rpc_fallbacks += 1;
            let tail_tensor = session
                .tail_tensor
                .clone()
                .context("tail tensor unavailable for sample fallback")?;
            let (response, rpc) =
                self.pair
                    .tail
                    .request_profiled(&StageNodeRequest::SampleTailToken {
                        tensor: tail_tensor.clone(),
                    })?;
            session.timings.total_transfer_bytes +=
                tail_tensor.bytes.len() + rpc.raw_request_bytes + rpc.raw_response_bytes;
            session.timings.stage_request_json_encode_ms += rpc.request_json_encode_ms;
            session.timings.stage_response_json_decode_ms += rpc.response_json_decode_ms;
            session.timings.stage_request_write_ms += rpc.request_write_ms;
            session.timings.stage_response_read_ms += rpc.response_read_ms;
            match response {
                StageNodeResponse::TokenSample { sample } => sample,
                other => bail!("expected token_sample response, got {other:?}"),
            }
        };
        let sample_elapsed = t_sample.elapsed();
        session.timings.sample_ms += sample_elapsed.as_millis() as u64;
        session.timings.sample_us += sample_elapsed.as_micros() as u64;

        // Record TTFT on the first token only. Later tokens accumulate into
        // sample_ms / tail_decode_ms, so overwriting ttft at `done` would
        // report cumulative (not first-token) latency.
        if session.timings.ttft_ms == 0 {
            session.timings.ttft_ms = session.timings.head_prefill_ms
                + session.timings.tail_decode_ms
                + session.timings.sample_ms;
            session.timings.ttft_us = session.timings.head_prefill_us
                + session.timings.tail_decode_us
                + session.timings.sample_us;
        }

        session.text.push_str(&sampled.piece);
        session.token_ids.push(sampled.token_id);
        session.context_token_ids.push(sampled.token_id);

        let done = sampled.is_eog || session.token_ids.len() as u32 >= session.max_tokens;
        if done {
            session.timings.total_ms = session.timings.head_prefill_ms
                + session.timings.head_decode_ms
                + session.timings.tail_decode_ms
                + session.timings.sample_ms;
            session.timings.total_us = session.timings.head_prefill_us
                + session.timings.head_decode_us
                + session.timings.tail_decode_us
                + session.timings.sample_us;
            let completion = RemoteStageCompletion {
                text: session.text,
                completion_tokens: session.token_ids.len() as u32,
                token_ids: session.token_ids,
                timings: session.timings,
            };
            self.pair.clear_decode_session(request_id)?;
            return Ok(GatewayStep::Complete {
                request_id: request_id.to_string(),
                completion,
            });
        }

        // Refill pending_tail_sample for the next call. Three paths:
        //   1. Drain pending_committed (zero new work — these were committed
        //      by an earlier spec round).
        //   2. Spec round (batched draft + verify) when spec is active.
        //   3. Single-token decode (legacy path) otherwise.
        if let Some(next) = session.pending_committed.pop_front() {
            session.pending_tail_sample = Some(next);
        } else if self.spec_active() && !session.spec_suspended {
            self.run_spec_round(request_id, &mut session, sampled.token_id)?;
        } else {
            self.run_single_step(request_id, &mut session, sampled.token_id)?;
        }

        let text = session.text.clone();
        let token_ids = session.token_ids.clone();
        self.sessions.insert(request_id.to_string(), session);

        Ok(GatewayStep::Token {
            request_id: request_id.to_string(),
            sample: sampled,
            text,
            token_ids,
        })
    }

    fn run_single_step(
        &mut self,
        request_id: &str,
        session: &mut GatewaySessionState,
        last_token: i32,
    ) -> Result<()> {
        let t_head = std::time::Instant::now();
        let ((new_head_tensor, head_profile), head_rpc) =
            match self
                .pair
                .head
                .request_profiled(&StageNodeRequest::ContinueHeadTokens {
                    request_id: request_id.to_string(),
                    token_ids: vec![last_token],
                    max_tokens: Some(session.max_tokens),
                })? {
                (
                    StageNodeResponse::Tensor {
                        tensor, profile, ..
                    },
                    rpc,
                ) => (
                    (
                        tensor.context("single-step head tensor missing")?,
                        profile.unwrap_or_default(),
                    ),
                    rpc,
                ),
                (other, _) => bail!("expected tensor response, got {other:?}"),
            };
        session.head_n_pos += 1;
        let head_step_elapsed = t_head.elapsed();
        session.timings.head_decode_ms += head_step_elapsed.as_millis() as u64;
        session.timings.head_decode_us += head_step_elapsed.as_micros() as u64;
        session.timings.total_transfer_bytes +=
            new_head_tensor.bytes.len() + head_rpc.raw_request_bytes + head_rpc.raw_response_bytes;
        session.timings.head_hidden_bytes_decode += head_profile.hidden_bytes;
        session.timings.head_pack_ms += head_profile.tensor_pack_ms;
        session.timings.head_pack_us += head_profile.tensor_pack_us;
        session.timings.stage_request_json_encode_ms += head_rpc.request_json_encode_ms;
        session.timings.stage_request_json_encode_us += head_rpc.request_json_encode_us;
        session.timings.stage_response_json_decode_ms += head_rpc.response_json_decode_ms;
        session.timings.stage_response_json_decode_us += head_rpc.response_json_decode_us;
        session.timings.stage_request_write_ms += head_rpc.request_write_ms;
        session.timings.stage_request_write_us += head_rpc.request_write_us;
        session.timings.stage_response_read_ms += head_rpc.response_read_ms;
        session.timings.stage_response_read_us += head_rpc.response_read_us;
        session.timings.stage_server_request_json_decode_ms +=
            head_profile.server_request_json_decode_ms;
        session.timings.stage_server_request_json_decode_us +=
            head_profile.server_request_json_decode_us;
        session.timings.stage_server_handle_ms += head_profile.server_handle_ms;
        session.timings.stage_server_handle_us += head_profile.server_handle_us;
        session.timings.stage_server_response_json_encode_ms +=
            head_profile.server_response_json_encode_ms;
        session.timings.stage_server_response_json_encode_us +=
            head_profile.server_response_json_encode_us;
        session.timings.stage_server_response_write_ms += head_profile.server_response_write_ms;
        session.timings.stage_server_response_write_us += head_profile.server_response_write_us;

        let t_tail = std::time::Instant::now();
        let head_tensor_for_tail = new_head_tensor.clone();
        let ((new_tail_tensor, new_pending, tail_profile), tail_rpc) = match self
            .pair
            .tail
            .request_profiled(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor_for_tail,
                token_ids: vec![last_token],
                clear_memory: false,
            })? {
            (
                StageNodeResponse::Tensor {
                    tensor,
                    tail_sample,
                    profile,
                },
                rpc,
            ) => ((tensor, tail_sample, profile.unwrap_or_default()), rpc),
            (other, _) => bail!("expected tensor response, got {other:?}"),
        };
        session.head_tensor = new_head_tensor;
        session.pending_tail_sample = new_pending;
        session.tail_tensor = new_tail_tensor;
        let tail_step_elapsed = t_tail.elapsed();
        session.timings.tail_decode_ms += tail_step_elapsed.as_millis() as u64;
        session.timings.tail_decode_us += tail_step_elapsed.as_micros() as u64;
        let tail_tensor_bytes = session
            .tail_tensor
            .as_ref()
            .map(|tensor| tensor.bytes.len())
            .unwrap_or(0);
        session.timings.total_transfer_bytes +=
            tail_tensor_bytes + tail_rpc.raw_request_bytes + tail_rpc.raw_response_bytes;
        session.timings.tail_unpack_ms += tail_profile.tensor_unpack_ms;
        session.timings.tail_unpack_us += tail_profile.tensor_unpack_us;
        session.timings.stage_request_json_encode_ms += tail_rpc.request_json_encode_ms;
        session.timings.stage_request_json_encode_us += tail_rpc.request_json_encode_us;
        session.timings.stage_response_json_decode_ms += tail_rpc.response_json_decode_ms;
        session.timings.stage_response_json_decode_us += tail_rpc.response_json_decode_us;
        session.timings.stage_request_write_ms += tail_rpc.request_write_ms;
        session.timings.stage_request_write_us += tail_rpc.request_write_us;
        session.timings.stage_response_read_ms += tail_rpc.response_read_ms;
        session.timings.stage_response_read_us += tail_rpc.response_read_us;
        session.timings.stage_server_request_json_decode_ms +=
            tail_profile.server_request_json_decode_ms;
        session.timings.stage_server_request_json_decode_us +=
            tail_profile.server_request_json_decode_us;
        session.timings.stage_server_handle_ms += tail_profile.server_handle_ms;
        session.timings.stage_server_handle_us += tail_profile.server_handle_us;
        session.timings.stage_server_response_json_encode_ms +=
            tail_profile.server_response_json_encode_ms;
        session.timings.stage_server_response_json_encode_us +=
            tail_profile.server_response_json_encode_us;
        session.timings.stage_server_response_write_ms += tail_profile.server_response_write_ms;
        session.timings.stage_server_response_write_us += tail_profile.server_response_write_us;
        Ok(())
    }

    fn run_spec_round(
        &mut self,
        request_id: &str,
        session: &mut GatewaySessionState,
        last_token: i32,
    ) -> Result<()> {
        let draft_started = Instant::now();
        let requested_k = session
            .current_k
            .max(self.spec_config.min_k)
            .min(self.spec_config.max_k);
        let lookup_k = if self.spec_config.lookup_max_tokens > 0 {
            self.spec_config.lookup_max_tokens
        } else {
            requested_k
        };
        let mut used_lookup_draft = false;
        let mut pending_draft_commit = Vec::new();
        let mut draft_pos_before = 0;
        let lookup_drafts = lookup_draft_tokens(
            &session.context_token_ids,
            lookup_k as usize,
            self.spec_config.lookup_min_ngram as usize,
            self.spec_config.lookup_max_ngram as usize,
        );
        let drafts = if !lookup_drafts.is_empty() {
            used_lookup_draft = true;
            lookup_drafts
        } else {
            let draft = self
                .draft
                .as_mut()
                .expect("run_spec_round invoked while spec_active() is true");
            pending_draft_commit = std::mem::take(&mut session.pending_draft_commit);
            draft_pos_before = draft.current_pos();
            draft.greedy_step_k_with_prefix(&pending_draft_commit, last_token, requested_k)?
        };
        let effective_k = drafts.len() as u32;
        let draft_elapsed = draft_started.elapsed();
        session.timings.spec_draft_ms += draft_elapsed.as_millis() as u64;
        session.timings.spec_draft_us += draft_elapsed.as_micros() as u64;

        let mut batch_tokens = Vec::with_capacity(drafts.len() + 1);
        batch_tokens.push(last_token);
        batch_tokens.extend(drafts.iter().copied());

        let head_n_pos_before = session.head_n_pos;

        let t_head = std::time::Instant::now();
        let ((head_tensor, head_profile), head_rpc) =
            match self
                .pair
                .head
                .request_profiled(&StageNodeRequest::ContinueHeadTokens {
                    request_id: request_id.to_string(),
                    token_ids: batch_tokens.clone(),
                    max_tokens: Some(session.max_tokens),
                })? {
                (
                    StageNodeResponse::Tensor {
                        tensor, profile, ..
                    },
                    rpc,
                ) => (
                    (
                        tensor.context("spec head tensor missing")?,
                        profile.unwrap_or_default(),
                    ),
                    rpc,
                ),
                (other, _) => bail!("expected tensor response, got {other:?}"),
            };
        session.head_n_pos += batch_tokens.len() as i32;
        let head_batch_elapsed = t_head.elapsed();
        session.timings.head_decode_ms += head_batch_elapsed.as_millis() as u64;
        session.timings.head_decode_us += head_batch_elapsed.as_micros() as u64;
        session.timings.total_transfer_bytes +=
            head_tensor.bytes.len() + head_rpc.raw_request_bytes + head_rpc.raw_response_bytes;
        session.timings.head_hidden_bytes_decode += head_profile.hidden_bytes;
        session.timings.head_pack_ms += head_profile.tensor_pack_ms;
        session.timings.head_pack_us += head_profile.tensor_pack_us;
        session.timings.stage_request_json_encode_ms += head_rpc.request_json_encode_ms;
        session.timings.stage_request_json_encode_us += head_rpc.request_json_encode_us;
        session.timings.stage_response_json_decode_ms += head_rpc.response_json_decode_ms;
        session.timings.stage_response_json_decode_us += head_rpc.response_json_decode_us;
        session.timings.stage_request_write_ms += head_rpc.request_write_ms;
        session.timings.stage_request_write_us += head_rpc.request_write_us;
        session.timings.stage_response_read_ms += head_rpc.response_read_ms;
        session.timings.stage_response_read_us += head_rpc.response_read_us;
        session.timings.stage_server_request_json_decode_ms +=
            head_profile.server_request_json_decode_ms;
        session.timings.stage_server_request_json_decode_us +=
            head_profile.server_request_json_decode_us;
        session.timings.stage_server_handle_ms += head_profile.server_handle_ms;
        session.timings.stage_server_handle_us += head_profile.server_handle_us;
        session.timings.stage_server_response_json_encode_ms +=
            head_profile.server_response_json_encode_ms;
        session.timings.stage_server_response_json_encode_us +=
            head_profile.server_response_json_encode_us;
        session.timings.stage_server_response_write_ms += head_profile.server_response_write_ms;
        session.timings.stage_server_response_write_us += head_profile.server_response_write_us;

        let t_tail = std::time::Instant::now();
        let head_tensor_for_tail = head_tensor.clone();
        let head_tensor_bytes = head_tensor_for_tail.bytes.len();
        let ((accepted_count, accepted_token_ids, accepted_pieces, is_eog, tail_profile), tail_rpc) =
            match self
                .pair
                .tail
                .request_profiled(&StageNodeRequest::ContinueForwardVerifyK {
                    request_id: request_id.to_string(),
                    tensor: head_tensor_for_tail,
                    last_token,
                    draft_tokens: drafts.clone(),
                    clear_memory: false,
                })? {
                (
                    StageNodeResponse::VerifiedBatch {
                        accepted_count,
                        accepted_token_ids,
                        accepted_pieces,
                        is_eog,
                        profile,
                    },
                    rpc,
                ) => (
                    (
                        accepted_count,
                        accepted_token_ids,
                        accepted_pieces,
                        is_eog,
                        profile.unwrap_or_default(),
                    ),
                    rpc,
                ),
                (other, _) => bail!("expected verified_batch response, got {other:?}"),
            };
        let tail_verify_elapsed = t_tail.elapsed();
        session.timings.tail_decode_ms += tail_verify_elapsed.as_millis() as u64;
        session.timings.tail_decode_us += tail_verify_elapsed.as_micros() as u64;
        session.timings.spec_verify_ms += tail_verify_elapsed.as_millis() as u64;
        session.timings.spec_verify_us += tail_verify_elapsed.as_micros() as u64;
        session.head_tensor = head_tensor;
        session.timings.spec_rounds += 1;
        session.timings.spec_drafts_proposed += drafts.len() as u64;
        session.timings.spec_drafts_accepted += accepted_count as u64;

        session.timings.total_transfer_bytes +=
            head_tensor_bytes + tail_rpc.raw_request_bytes + tail_rpc.raw_response_bytes;
        session.timings.stage_request_json_encode_ms += tail_rpc.request_json_encode_ms;
        session.timings.stage_request_json_encode_us += tail_rpc.request_json_encode_us;
        session.timings.stage_response_json_decode_ms += tail_rpc.response_json_decode_ms;
        session.timings.stage_response_json_decode_us += tail_rpc.response_json_decode_us;
        session.timings.stage_request_write_ms += tail_rpc.request_write_ms;
        session.timings.stage_request_write_us += tail_rpc.request_write_us;
        session.timings.stage_response_read_ms += tail_rpc.response_read_ms;
        session.timings.stage_response_read_us += tail_rpc.response_read_us;
        session.timings.stage_server_request_json_decode_ms +=
            tail_profile.server_request_json_decode_ms;
        session.timings.stage_server_request_json_decode_us +=
            tail_profile.server_request_json_decode_us;
        session.timings.stage_server_handle_ms += tail_profile.server_handle_ms;
        session.timings.stage_server_handle_us += tail_profile.server_handle_us;
        session.timings.stage_server_response_json_encode_ms +=
            tail_profile.server_response_json_encode_ms;
        session.timings.stage_server_response_json_encode_us +=
            tail_profile.server_response_json_encode_us;
        session.timings.stage_server_response_write_ms += tail_profile.server_response_write_ms;
        session.timings.stage_server_response_write_us += tail_profile.server_response_write_us;
        session.timings.tail_decode_kernel_us += tail_profile.tail_decode_kernel_us;
        session.timings.tail_verify_sample_us += tail_profile.tail_verify_sample_us;
        session.timings.tail_verify_detok_us += tail_profile.tail_verify_detok_us;
        session.timings.tail_verify_rollback_us += tail_profile.tail_verify_rollback_us;
        session.timings.tail_sync_us += tail_profile.tail_sync_us;

        if accepted_pieces.len() != accepted_token_ids.len() {
            bail!(
                "verify response shape mismatch: ids={} pieces={}",
                accepted_token_ids.len(),
                accepted_pieces.len()
            );
        }

        // Roll head's KV back to match what tail kept (T + accepted drafts).
        // Bonus is sampled-only, never written to either KV.
        let head_keep = head_n_pos_before + accepted_count as i32 + 1;
        if head_keep < session.head_n_pos {
            let rollback_started = Instant::now();
            self.pair.head.request(&StageNodeRequest::RollbackKv {
                request_id: request_id.to_string(),
                keep_count: head_keep as u32,
            })?;
            let rollback_elapsed = rollback_started.elapsed();
            session.timings.spec_rollback_ms += rollback_elapsed.as_millis() as u64;
            session.timings.spec_rollback_us += rollback_elapsed.as_micros() as u64;
            session.head_n_pos = head_keep;
        }

        if used_lookup_draft {
            // The model draft did no work this round. Keep it catch-up aligned
            // by deferring the verified target tokens that precede the next
            // round's `last_token`; a later model-draft fallback will prefill
            // these in one chunk before sampling.
            session.pending_draft_commit.push(last_token);
            session
                .pending_draft_commit
                .extend(drafts.iter().copied().take(accepted_count as usize));
        } else {
            let draft = self
                .draft
                .as_mut()
                .expect("run_spec_round invoked while spec_active() is true");
            // Align draft KV to (state-before-greedy) + pending + last_token + accepted.
            // greedy_step writes its INPUT to KV then samples the next token, so
            // after greedy_step_k(T, k) the chain wrote T, D_1, ..., D_{k-1} but
            // NOT D_k (D_k was sampled past the written window). On full accept,
            // defer D_k and batch it into the next draft decode with the bonus
            // token instead of paying an extra one-token draft call here.
            let target_pos =
                draft_pos_before + pending_draft_commit.len() as i32 + 1 + accepted_count as i32;
            if target_pos > draft.current_pos() {
                if let Some(&last_draft) = drafts.last() {
                    session.pending_draft_commit.push(last_draft);
                }
            } else {
                draft.rollback_to(target_pos)?;
            }
        }

        // Adaptive k: shrink on consecutive low accepts, recover on full
        // accepts. Bound by [min_k, max_k] from spec_config.
        if accepted_count == 0 {
            session.consec_low_accept = session.consec_low_accept.saturating_add(1);
            if effective_k <= self.spec_config.min_k
                && self.spec_config.disable_after_consec_zero > 0
                && session.consec_low_accept >= self.spec_config.disable_after_consec_zero
            {
                session.spec_suspended = true;
            } else if session.consec_low_accept >= 2 {
                session.current_k = (session.current_k / 2).max(self.spec_config.min_k);
            }
        } else if accepted_count == effective_k {
            session.consec_low_accept = 0;
            session.current_k = (session.current_k.saturating_mul(2))
                .min(self.spec_config.max_k)
                .max(self.spec_config.min_k);
        } else {
            session.consec_low_accept = 0;
        }

        // Emit plan: pending_tail_sample = first id (next call's emit), and
        // queue the rest in pending_committed so subsequent step_completion
        // calls can drain them without another forward pass. The very last id
        // is the bonus and carries `is_eog` for the round.
        let total = accepted_token_ids.len();
        for (idx, (tid, piece)) in accepted_token_ids
            .into_iter()
            .zip(accepted_pieces.into_iter())
            .enumerate()
        {
            let is_last = idx == total - 1;
            let sample = GreedyTokenSample {
                token_id: tid,
                piece,
                is_eog: is_last && is_eog,
            };
            if idx == 0 {
                session.pending_tail_sample = Some(sample);
            } else {
                session.pending_committed.push_back(sample);
            }
        }

        Ok(())
    }

    pub fn complete(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<RemoteStageCompletion> {
        self.begin_completion(request_id, prompt, max_tokens)?;
        loop {
            match self.step_completion(request_id)? {
                GatewayStep::Token { .. } => continue,
                GatewayStep::Complete { completion, .. } => return Ok(completion),
            }
        }
    }
}

pub fn handle_stage_gateway_request(
    gateway: &mut RemoteStageGateway,
    request: StageGatewayRequest,
) -> StageGatewayResponse {
    let result: Result<StageGatewayResponse> = (|| match request {
        StageGatewayRequest::Info => Ok(StageGatewayResponse::Info {
            protocol_version: LLAMA_STAGE_PROTOCOL_VERSION,
            head_info: gateway.head_info().clone(),
            tail_info: gateway.tail_info().clone(),
            reconnect_after_prompt: gateway.reconnect_after_prompt(),
        }),
        StageGatewayRequest::Tokenize { text } => Ok(StageGatewayResponse::TokenIds {
            token_ids: gateway
                .pair
                .head
                .request(&StageNodeRequest::Tokenize { text })
                .and_then(|response| match response {
                    StageNodeResponse::TokenIds { token_ids } => Ok(token_ids),
                    other => bail!("expected token_ids response, got {other:?}"),
                })?,
        }),
        StageGatewayRequest::Complete {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Completion {
            completion: gateway.complete(&request_id, &prompt, max_tokens)?,
        }),
        StageGatewayRequest::BeginCompletion {
            request_id,
            prompt,
            max_tokens,
        } => {
            gateway.begin_completion(&request_id, &prompt, max_tokens)?;
            Ok(StageGatewayResponse::Started { request_id })
        }
        StageGatewayRequest::StepCompletion { request_id } => Ok(StageGatewayResponse::Step {
            step: gateway.step_completion(&request_id)?,
        }),
        StageGatewayRequest::ClearCompletion { request_id } => {
            gateway.clear_completion(&request_id)?;
            Ok(StageGatewayResponse::Ack)
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageGatewayResponse::Error {
            message: err.to_string(),
        },
    }
}

pub fn handle_stage_node_request(
    backend: &LlamaStageBackend,
    request: StageNodeRequest,
) -> StageNodeResponse {
    let result: Result<StageNodeResponse> = (|| match request {
        StageNodeRequest::Info => Ok(StageNodeResponse::Info {
            info: backend.node_info()?,
        }),
        StageNodeRequest::Tokenize { text } => Ok(StageNodeResponse::TokenIds {
            token_ids: backend.tokenize(&text)?,
        }),
        StageNodeRequest::BeginPrompt {
            request_id,
            prompt,
            max_tokens,
        } => {
            let tensor = backend.begin_prompt_session(&request_id, &prompt, max_tokens)?;
            let tail_sample = backend.cached_tail_sample(&request_id);
            let profile = backend.last_profile(&request_id);
            let state = backend.state.borrow();
            let is_tail = state
                .layout
                .as_ref()
                .map(|layout| layout.is_tail)
                .unwrap_or(false);
            Ok(StageNodeResponse::Tensor {
                tensor: if is_tail { None } else { Some(tensor) },
                tail_sample,
                profile,
            })
        }
        StageNodeRequest::ContinueHeadTokens {
            request_id,
            token_ids,
            max_tokens,
        } => {
            let tensor = backend.continue_head_tokens(&request_id, token_ids, max_tokens)?;
            let profile = backend.last_profile(&request_id);
            // Head never caches a tail sample; cached_tail_sample returns None.
            Ok(StageNodeResponse::Tensor {
                tensor: Some(tensor),
                tail_sample: None,
                profile,
            })
        }
        StageNodeRequest::ContinueForward { tensor } => {
            let request_id = tensor.request_id.clone();
            let tensor = backend.continue_forward(tensor)?;
            let tail_sample = backend.cached_tail_sample(&request_id);
            let profile = backend.last_profile(&request_id);
            let state = backend.state.borrow();
            let is_tail = state
                .layout
                .as_ref()
                .map(|layout| layout.is_tail)
                .unwrap_or(false);
            Ok(StageNodeResponse::Tensor {
                tensor: if is_tail { None } else { Some(tensor) },
                tail_sample,
                profile,
            })
        }
        StageNodeRequest::ContinueForwardTokens {
            tensor,
            token_ids,
            clear_memory,
        } => {
            let request_id = tensor.request_id.clone();
            let tensor = backend.continue_forward_with_tokens(tensor, token_ids, clear_memory)?;
            let tail_sample = backend.cached_tail_sample(&request_id);
            let profile = backend.last_profile(&request_id);
            let state = backend.state.borrow();
            let is_tail = state
                .layout
                .as_ref()
                .map(|layout| layout.is_tail)
                .unwrap_or(false);
            Ok(StageNodeResponse::Tensor {
                tensor: if is_tail { None } else { Some(tensor) },
                tail_sample,
                profile,
            })
        }
        StageNodeRequest::SampleTail { tensor } => Ok(StageNodeResponse::Sample {
            sample: backend.sample_tail(tensor)?,
        }),
        StageNodeRequest::SampleTailToken { tensor } => Ok(StageNodeResponse::TokenSample {
            sample: backend.sample_tail_token(tensor)?,
        }),
        StageNodeRequest::ClearDecodeSession { request_id } => {
            backend.clear_decode_session(&request_id)?;
            Ok(StageNodeResponse::Ack)
        }
        StageNodeRequest::ContinueForwardVerifyK {
            request_id,
            tensor,
            last_token,
            draft_tokens,
            clear_memory,
        } => {
            let outcome = backend.verify_batch_at_tail(
                &request_id,
                tensor,
                last_token,
                draft_tokens,
                clear_memory,
            )?;
            let profile = StageNodeProfile {
                tail_decode_kernel_us: outcome.tail_decode_kernel_us,
                tail_verify_sample_us: outcome.tail_verify_sample_us,
                tail_verify_detok_us: outcome.tail_verify_detok_us,
                tail_verify_rollback_us: outcome.tail_verify_rollback_us,
                tail_sync_us: outcome.tail_sync_us,
                ..StageNodeProfile::default()
            };
            Ok(StageNodeResponse::VerifiedBatch {
                accepted_count: outcome.accepted_count,
                accepted_token_ids: outcome.accepted_token_ids,
                accepted_pieces: outcome.accepted_pieces,
                is_eog: outcome.is_eog,
                profile: Some(profile),
            })
        }
        StageNodeRequest::RollbackKv {
            request_id,
            keep_count,
        } => {
            backend.rollback_kv(&request_id, keep_count)?;
            Ok(StageNodeResponse::Ack)
        }
        // ContinueHeadDraftK is reserved for a future variant where the head
        // node owns the draft engine. v0.3.0 keeps drafting on the gateway
        // (which calls ContinueHeadTokens with [last_token, D_1..D_k]
        // directly), so this remains unimplemented here.
        StageNodeRequest::ContinueHeadDraftK { .. } => {
            bail!("ContinueHeadDraftK is not implemented on this node (gateway-side drafting)")
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageNodeResponse::Error {
            message: err.to_string(),
        },
    }
}

pub fn default_gemma_model_path() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        let cached = PathBuf::from(home).join(".compute/models/gemma-4-E4B-it-Q4_K_M.gguf");
        if cached.exists() {
            return cached;
        }
    }

    PathBuf::from("models/gemma-4-E4B-it-Q4_K_M.gguf")
}

pub fn default_compute_bin_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".compute/bin"))
}

fn dir_has_libllama(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if !lower.contains("llama") {
            continue;
        }
        let is_dylib = lower.ends_with(".dylib")
            || lower.ends_with(".so")
            || lower.ends_with(".dll")
            || lower.contains(".so.");
        if is_dylib {
            return true;
        }
    }
    false
}

fn resolve_vendor_lib_dir() -> Result<PathBuf> {
    if let Some(override_dir) = env::var_os("LLAMA_STAGE_VENDOR_LIB_DIR") {
        let path = PathBuf::from(override_dir);
        if dir_has_libllama(&path) {
            return Ok(path);
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            for candidate in [
                parent.to_path_buf(),
                parent.join("lib"),
                parent.join("../lib"),
            ] {
                if dir_has_libllama(&candidate) {
                    return Ok(candidate);
                }
            }
        }
    }

    let baked = PathBuf::from(env!("LLAMA_STAGE_VENDOR_LIB_DIR"));
    if dir_has_libllama(&baked) {
        return Ok(baked);
    }

    if let Some(bin_dir) = default_compute_bin_dir() {
        for candidate in [bin_dir.clone(), bin_dir.join("lib")] {
            if dir_has_libllama(&candidate) {
                return Ok(candidate);
            }
        }
    }

    bail!(
        "could not locate libllama dylib; checked LLAMA_STAGE_VENDOR_LIB_DIR, executable dir, ~/.compute/bin, and the build-time path {}",
        baked.display()
    );
}

pub fn resolve_model_arg(args: &[String]) -> (PathBuf, usize) {
    if let Some(candidate) = args.get(1) {
        let path = PathBuf::from(candidate);
        let looks_like_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if looks_like_gguf {
            return (path, 2);
        }
    }

    (default_gemma_model_path(), 1)
}

fn compute_workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .context("failed to resolve compute-app workspace root")
}

fn resolve_managed_binary_path(explicit: Option<&Path>, name: &str) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        bail!("configured binary path does not exist: {}", path.display());
    }

    let file_name = format!("{name}{EXE_SUFFIX}");
    if let Some(bin_dir) = default_compute_bin_dir() {
        let installed = bin_dir.join(&file_name);
        if installed.exists() {
            return Ok(installed);
        }
    }

    let root = compute_workspace_root()?;
    for profile in ["debug", "release"] {
        let candidate = root.join("target").join(profile).join(&file_name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!(
        "could not find {name} binary in ~/.compute/bin or under {}/target/{{debug,release}}; install sidecars into ~/.compute/bin, build compute-app with `cargo build -p llama-stage-backend --bins`, or configure an explicit binary path",
        root.display()
    )
}

fn read_listening_addr(stderr: ChildStderr) -> Result<String> {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    let mut captured: Vec<String> = Vec::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            let tail = captured
                .iter()
                .rev()
                .take(20)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            if tail.is_empty() {
                bail!("child exited before announcing listening address (stderr was empty)");
            }
            bail!("child exited before announcing listening address; last stderr lines: {tail}");
        }
        let trimmed = line.trim();
        if let Some(addr) = trimmed.strip_prefix("listening=") {
            let addr = addr.to_string();
            // Forward only our own banner lines (draft_model=, spec_active=,
            // etc.) plus anything that looks like a real error. Skip the
            // llama.cpp startup firehose (`llama_kv_cache: layer 7: ...`,
            // `sched_reserve: ...`, `ggml_metal_: ...`, backend enumeration,
            // etc.) — on a 42-layer model that's 100+ lines per spawn and
            // it floods the TUI log panel.
            for prior in &captured {
                if child_log_is_interesting(prior) {
                    forward_child_log_line(&addr, prior);
                }
            }
            let forward = env::var_os("LLAMA_STAGE_FORWARD_STDERR").is_some();
            let label = addr.clone();
            thread::spawn(move || {
                let mut reader = reader;
                let mut buf = String::new();
                loop {
                    buf.clear();
                    match reader.read_line(&mut buf) {
                        Ok(0) | Err(_) => return,
                        Ok(_) => {
                            // Unconditional path: we always surface errors
                            // so a crash loop is visible without needing
                            // LLAMA_STAGE_FORWARD_STDERR=1 set.
                            let trimmed = buf.trim_end();
                            if forward || child_log_is_error(trimmed) {
                                forward_child_log_line(&label, trimmed);
                            }
                        }
                    }
                }
            });
            return Ok(addr);
        }
        if !trimmed.is_empty() {
            captured.push(trimmed.to_string());
        }
    }
}

/// Is a child stderr line worth forwarding to the parent's log by default?
/// Matches our own banner prefixes (`draft_model=`, `spec_active=`, etc.)
/// and obvious error/panic lines; rejects the llama.cpp / ggml verbose
/// startup chatter.
fn child_log_is_interesting(line: &str) -> bool {
    if child_log_is_banner(line) || child_log_is_error(line) {
        return true;
    }
    false
}

/// Matches our own sidecar-emitted banner lines. Add new prefixes here as
/// the sidecars gain more pre-listening status output.
fn child_log_is_banner(line: &str) -> bool {
    const BANNER_PREFIXES: &[&str] = &[
        "draft_model=",
        "spec_active=",
        "spec_config=",
        "stage_id=",
        "gateway_addr=",
        "node_info=",
    ];
    BANNER_PREFIXES.iter().any(|p| line.starts_with(p))
}

/// Heuristic: does this line look like an actionable error we want the user
/// to see even when forwarding is off?
fn child_log_is_error(line: &str) -> bool {
    // Avoid false positives on "error_count=0" etc by matching common
    // error-log phrasings, not the bare word "error".
    const ERROR_NEEDLES: &[&str] = &[
        "panic",
        "thread 'main' panicked",
        "FATAL",
        "Fatal",
        "fatal:",
        "error:",
        "Error:",
        " failed:",
        " failed (",
        "Traceback",
        "assertion",
        "segmentation fault",
        "Segmentation fault",
    ];
    ERROR_NEEDLES.iter().any(|needle| line.contains(needle))
}

fn forward_child_log_line(label: &str, line: &str) {
    if child_log_is_error(line) {
        tracing::warn!("[child:{}] {}", label, line);
    } else {
        tracing::info!("[child:{}] {}", label, line);
    }
}

struct ManagedServiceChild {
    child: Child,
    addr: String,
}

impl ManagedServiceChild {
    fn spawn_stage_from_bin(
        bin_path: &Path,
        model_path: &Path,
        bind_addr: &str,
        stage_id: &str,
        start_layer: u32,
        end_layer: u32,
        is_head: bool,
        is_tail: bool,
    ) -> Result<Self> {
        let mut command = Command::new(bin_path);
        command
            .arg("--model")
            .arg(model_path)
            .arg("--bind")
            .arg(bind_addr)
            .arg("--stage-id")
            .arg(stage_id)
            .arg("--start-layer")
            .arg(start_layer.to_string())
            .arg("--end-layer")
            .arg(end_layer.to_string());
        if is_head {
            command.arg("--head");
        }
        if is_tail {
            command.arg("--tail");
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning tcp stage node from {}", bin_path.display()))?;
        let stderr = child.stderr.take().context("missing child stderr")?;
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }

    fn spawn_gateway_from_bin(
        bin_path: &Path,
        bind_addr: &str,
        head_addr: &str,
        tail_addr: &str,
        reconnect_after_prompt: bool,
        draft_model: Option<&Path>,
    ) -> Result<Self> {
        let mut command = Command::new(bin_path);
        command
            .arg("--bind")
            .arg(bind_addr)
            .arg("--head")
            .arg(head_addr)
            .arg("--tail")
            .arg(tail_addr);
        if reconnect_after_prompt {
            command.arg("--reconnect-after-prompt");
        }
        if let Some(path) = draft_model {
            command.arg("--draft-model").arg(path);
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning tcp gateway node from {}", bin_path.display()))?;
        let stderr = child.stderr.take().context("missing gateway stderr")?;
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }
}

impl Drop for ManagedServiceChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ManagedGatewayStack {
    _head: ManagedServiceChild,
    _tail: ManagedServiceChild,
    gateway: ManagedServiceChild,
}

#[derive(Debug, Clone, Default)]
pub struct ManagedGatewayLaunchSpec {
    pub stage_node_bin: Option<PathBuf>,
    pub gateway_bin: Option<PathBuf>,
    pub head_bind: Option<String>,
    pub tail_bind: Option<String>,
    pub gateway_bind: Option<String>,
    /// Optional draft model GGUF for speculative decoding. When set, the
    /// gateway child opens it via `RemoteStageGateway::connect_with_draft`
    /// and runs the spec path as long as both peers advertise spec_decode_v1.
    pub draft_model: Option<PathBuf>,
}

impl ManagedGatewayStack {
    pub fn spawn_local(
        model_path: impl Into<PathBuf>,
        reconnect_after_prompt: bool,
    ) -> Result<Self> {
        Self::spawn_local_with_spec(
            model_path,
            reconnect_after_prompt,
            &ManagedGatewayLaunchSpec::default(),
        )
    }

    pub fn spawn_local_with_spec(
        model_path: impl Into<PathBuf>,
        reconnect_after_prompt: bool,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let gateway_bin = resolve_managed_binary_path(
            launch_spec.gateway_bin.as_deref(),
            "llama_stage_gateway_tcp_node",
        )?;
        let head_bind = launch_spec.head_bind.as_deref().unwrap_or("127.0.0.1:0");
        let tail_bind = launch_spec.tail_bind.as_deref().unwrap_or("127.0.0.1:0");
        let gateway_bind = launch_spec.gateway_bind.as_deref().unwrap_or("127.0.0.1:0");

        let head = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            head_bind,
            "stage-1",
            0,
            20,
            true,
            false,
        )?;
        let tail = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            tail_bind,
            "stage-2",
            21,
            41,
            false,
            true,
        )?;
        let gateway = ManagedServiceChild::spawn_gateway_from_bin(
            &gateway_bin,
            gateway_bind,
            &head.addr,
            &tail.addr,
            reconnect_after_prompt,
            launch_spec.draft_model.as_deref(),
        )?;

        let deadline = Instant::now() + Duration::from_secs(30);
        let mut last_err = None;
        while Instant::now() < deadline {
            match GatewayServiceClient::connect(&gateway.addr) {
                Ok(_) => {
                    return Ok(Self {
                        _head: head,
                        _tail: tail,
                        gateway,
                    });
                }
                Err(err) => {
                    last_err = Some(err);
                    thread::sleep(Duration::from_millis(250));
                }
            }
        }

        let err = last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "gateway did not become ready".to_string());
        bail!("timed out waiting for gateway readiness: {err}")
    }

    pub fn gateway_addr(&self) -> &str {
        &self.gateway.addr
    }
}

/// A standalone tail-only stage worker (the rear half of a 2-machine split).
/// Spawns just `llama_stage_tcp_node --tail` so a remote head can connect.
pub struct ManagedTailNode {
    _child: ManagedServiceChild,
    addr: String,
}

impl ManagedTailNode {
    pub fn spawn(
        model_path: impl Into<PathBuf>,
        bind_addr: impl Into<String>,
        start_layer: u32,
        end_layer: u32,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let bind_addr = bind_addr.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let child = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            &bind_addr,
            "stage-tail",
            start_layer,
            end_layer,
            false,
            true,
        )?;
        let addr = child.addr.clone();
        Ok(Self {
            _child: child,
            addr,
        })
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }
}

/// A standalone head-only stage worker. Mirrors `ManagedTailNode` so probes
/// and integration tests can stand up a head + tail pair without the bundled
/// gateway binary (e.g., when the gateway lives in-process — spec decode).
pub struct ManagedHeadNode {
    _child: ManagedServiceChild,
    addr: String,
}

impl ManagedHeadNode {
    pub fn spawn(
        model_path: impl Into<PathBuf>,
        bind_addr: impl Into<String>,
        start_layer: u32,
        end_layer: u32,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let bind_addr = bind_addr.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let child = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            &bind_addr,
            "stage-head",
            start_layer,
            end_layer,
            true,
            false,
        )?;
        let addr = child.addr.clone();
        Ok(Self {
            _child: child,
            addr,
        })
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }
}

/// A head + gateway stack that talks to a remote tail worker over TCP.
/// Spawns the local head stage node and a gateway pointing at `tail_remote_addr`.
pub struct ManagedHeadGatewayStack {
    _head: ManagedServiceChild,
    gateway: ManagedServiceChild,
}

impl ManagedHeadGatewayStack {
    pub fn spawn_with_remote_tail(
        model_path: impl Into<PathBuf>,
        head_start_layer: u32,
        head_end_layer: u32,
        tail_remote_addr: impl Into<String>,
        reconnect_after_prompt: bool,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let tail_remote_addr = tail_remote_addr.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let gateway_bin = resolve_managed_binary_path(
            launch_spec.gateway_bin.as_deref(),
            "llama_stage_gateway_tcp_node",
        )?;
        let head_bind = launch_spec.head_bind.as_deref().unwrap_or("127.0.0.1:0");
        let gateway_bind = launch_spec.gateway_bind.as_deref().unwrap_or("127.0.0.1:0");

        let head = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            head_bind,
            "stage-head",
            head_start_layer,
            head_end_layer,
            true,
            false,
        )?;
        let gateway = ManagedServiceChild::spawn_gateway_from_bin(
            &gateway_bin,
            gateway_bind,
            &head.addr,
            &tail_remote_addr,
            reconnect_after_prompt,
            launch_spec.draft_model.as_deref(),
        )?;

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut last_err = None;
        while Instant::now() < deadline {
            match GatewayServiceClient::connect(&gateway.addr) {
                Ok(_) => {
                    return Ok(Self {
                        _head: head,
                        gateway,
                    });
                }
                Err(err) => {
                    last_err = Some(err);
                    thread::sleep(Duration::from_millis(250));
                }
            }
        }

        let err = last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "gateway did not become ready".to_string());
        bail!("timed out waiting for head-only gateway readiness: {err}")
    }

    pub fn gateway_addr(&self) -> &str {
        &self.gateway.addr
    }
}

impl Drop for LlamaStageBackend {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        for (_, session) in state.sessions.drain() {
            session.session.destroy(&self.api);
        }
        if let Some(model) = state.model.take() {
            unsafe {
                (self.api.model_free)(model.model);
            }
        }
    }
}

impl StageForwardBackend for LlamaStageBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let mut state = self.state.borrow_mut();
        for (_, session) in state.sessions.drain() {
            session.session.destroy(&self.api);
        }
        state.layout = Some(layout);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        if !layout.is_head {
            bail!("begin_prompt called on non-head stage {}", layout.stage_id);
        }

        let model = self.ensure_model(&mut state)?;
        let tokens = self.tokenize_prompt(model, prompt)?;
        drop(state);
        if Self::debug_flow_enabled() {
            tracing::debug!(
                "[llama-stage] begin_prompt stage={} tokens={}",
                layout.stage_id,
                tokens.len()
            );
        }
        self.forward_head_tokens_impl(
            request_id,
            tokens,
            Some(prompt.to_string()),
            max_tokens,
            true,
        )
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        self.continue_forward_impl(input, None, true)
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let state = self.state.borrow();
        let layout = self.layout(&state)?.clone();
        if !layout.is_tail {
            bail!("sample_tail called on non-tail stage {}", layout.stage_id);
        }

        let cached = state
            .sessions
            .get(&input.request_id)
            .and_then(|session| session.cached_sample.clone())
            .context("no cached tail sample; call continue_forward first")?;

        if Self::debug_flow_enabled() {
            tracing::debug!(
                "[llama-stage] sample_tail returning stage={} text={:?}",
                layout.stage_id,
                cached.sample.text
            );
        }

        Ok(cached.sample)
    }
}

pub fn greedy_single_node_baseline(
    model_path: impl Into<PathBuf>,
    prompt: &str,
) -> Result<StageSample> {
    let api = LlamaApi::load()?;
    let model_path = model_path.into();
    let model = LlamaModelHandle::new(&api, &model_path)?;
    let session = model.create_session(&api)?;
    session.clear_memory(&api);

    let backend = LlamaStageBackend {
        api,
        model_path,
        state: RefCell::new(BackendState {
            layout: None,
            model: None,
            sessions: HashMap::new(),
            token_piece_cache: HashMap::new(),
        }),
    };

    let tokens = backend.tokenize_prompt(&model, prompt)?;
    let batch = OwnedBatch::token_only(tokens);
    let rc = unsafe { (backend.api.decode)(session.ctx, batch.raw) };
    if rc != 0 && rc != 1 {
        bail!("llama_decode baseline failed with {}", rc);
    }

    let sample = backend.greedy_sample(model.model, &session, "baseline", "baseline")?;
    session.destroy(&backend.api);
    unsafe { (backend.api.model_free)(model.model) };
    Ok(sample.sample)
}

pub fn greedy_single_node_completion(
    model_path: impl Into<PathBuf>,
    prompt: &str,
    max_tokens: u32,
) -> Result<GreedyCompletion> {
    let api = LlamaApi::load()?;
    let model_path = model_path.into();
    let model = LlamaModelHandle::new(&api, &model_path)?;
    let session = model.create_session(&api)?;
    session.clear_memory(&api);

    let backend = LlamaStageBackend {
        api,
        model_path,
        state: RefCell::new(BackendState {
            layout: None,
            model: None,
            sessions: HashMap::new(),
            token_piece_cache: HashMap::new(),
        }),
    };

    let prompt_tokens = backend.tokenize_prompt(&model, prompt)?;
    let prompt_batch = OwnedBatch::token_only(prompt_tokens);
    let rc = unsafe { (backend.api.decode)(session.ctx, prompt_batch.raw) };
    if rc != 0 && rc != 1 {
        bail!("llama_decode baseline prompt failed with {}", rc);
    }

    let mut text = String::new();
    let mut token_ids = Vec::new();
    for _ in 0..max_tokens.max(1) {
        let token = backend.greedy_sample(model.model, &session, "baseline", "baseline")?;
        text.push_str(&token.sample.text);
        token_ids.push(token.token_id);
        if token.is_eog {
            break;
        }

        let step_batch = OwnedBatch::token_only(vec![token.token_id]);
        let rc = unsafe { (backend.api.decode)(session.ctx, step_batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("llama_decode baseline continuation failed with {}", rc);
        }
    }

    session.destroy(&backend.api);
    unsafe { (backend.api.model_free)(model.model) };

    Ok(GreedyCompletion {
        completion_tokens: token_ids.len() as u32,
        text,
        token_ids,
    })
}

// ---------------------------------------------------------------------------
// Phase 0 spike helpers — temporary scaffolding for batched_decode_bench.
//
// These exist purely to time llama_decode at varying batch sizes from a
// minimal probe binary. Not used by the production gateway code; safe to
// delete once the spike concludes.
// ---------------------------------------------------------------------------

pub struct BenchHandles {
    pub model_ptr: *mut ffi::LlamaModel,
    pub session_ctx: *mut ffi::LlamaContext,
}

impl LlamaStageBackend {
    pub fn single_node_for_bench(model_path: PathBuf) -> Result<Self> {
        let api = LlamaApi::load()?;
        Ok(Self {
            api,
            model_path,
            state: RefCell::new(BackendState {
                layout: None,
                model: None,
                sessions: HashMap::new(),
                token_piece_cache: HashMap::new(),
            }),
        })
    }

    pub fn bench_prefill_and_seed(
        &self,
        prompt: &str,
    ) -> Result<(*mut ffi::LlamaModel, *mut ffi::LlamaContext, i32)> {
        let model = LlamaModelHandle::new(&self.api, &self.model_path)?;
        let session = model.create_session(&self.api)?;
        session.clear_memory(&self.api);

        let prompt_tokens = self.tokenize_prompt(&model, prompt)?;
        let prefill = OwnedBatch::token_only(prompt_tokens.clone());
        let rc = unsafe { (self.api.decode)(session.ctx, prefill.raw) };
        if rc != 0 && rc != 1 {
            bail!("bench prefill failed with {}", rc);
        }
        let seed = self.greedy_sample(model.model, &session, "bench", "bench")?;

        let session_ctx = session.ctx;
        let model_ptr = model.model;
        // Intentionally leak — bench_cleanup releases.
        std::mem::forget(session);
        std::mem::forget(model);
        Ok((model_ptr, session_ctx, seed.token_id))
    }

    pub fn bench_decode_batch(
        &self,
        session_ctx: *mut ffi::LlamaContext,
        token_id: i32,
        batch_size: usize,
    ) -> Result<()> {
        let tokens = vec![token_id; batch_size];
        let batch = OwnedBatch::token_only(tokens);
        let rc = unsafe { (self.api.decode)(session_ctx, batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("bench decode failed with {}", rc);
        }
        Ok(())
    }

    pub fn bench_cleanup(
        &self,
        model_ptr: *mut ffi::LlamaModel,
        session_ctx: *mut ffi::LlamaContext,
    ) {
        unsafe { (self.api.context_free)(session_ctx) };
        unsafe { (self.api.model_free)(model_ptr) };
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use stage_forward_lab::PayloadKind;

    fn sample_tensor() -> StageTensor {
        StageTensor {
            request_id: "req-1".to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["head".to_string()],
            hidden_dim: 8,
            bytes: vec![1, 2, 3, 4, 5, 6, 7, 8],
            prompt_text: None,
            max_tokens: Some(64),
            continuation: None,
            transient: None,
            carry: None,
        }
    }

    fn round_trip_request(req: &StageNodeRequest) -> StageNodeRequest {
        let json = serde_json::to_string(req).expect("serialize request");
        serde_json::from_str(&json).expect("deserialize request")
    }

    fn round_trip_response(resp: &StageNodeResponse) -> StageNodeResponse {
        let json = serde_json::to_string(resp).expect("serialize response");
        serde_json::from_str(&json).expect("deserialize response")
    }

    #[test]
    fn continue_head_draft_k_round_trips() {
        let req = StageNodeRequest::ContinueHeadDraftK {
            request_id: "req-1".to_string(),
            last_token: 42,
            draft_k: 4,
            max_tokens: Some(128),
        };
        match round_trip_request(&req) {
            StageNodeRequest::ContinueHeadDraftK {
                request_id,
                last_token,
                draft_k,
                max_tokens,
            } => {
                assert_eq!(request_id, "req-1");
                assert_eq!(last_token, 42);
                assert_eq!(draft_k, 4);
                assert_eq!(max_tokens, Some(128));
            }
            other => panic!("unexpected variant after round-trip: {other:?}"),
        }
    }

    #[test]
    fn continue_forward_verify_k_round_trips() {
        let req = StageNodeRequest::ContinueForwardVerifyK {
            request_id: "req-1".to_string(),
            tensor: sample_tensor(),
            last_token: 7,
            draft_tokens: vec![10, 20, 30, 40],
            clear_memory: false,
        };
        match round_trip_request(&req) {
            StageNodeRequest::ContinueForwardVerifyK {
                request_id,
                tensor,
                last_token,
                draft_tokens,
                clear_memory,
            } => {
                assert_eq!(request_id, "req-1");
                assert_eq!(tensor.hidden_dim, 8);
                assert_eq!(last_token, 7);
                assert_eq!(draft_tokens, vec![10, 20, 30, 40]);
                assert!(!clear_memory);
            }
            other => panic!("unexpected variant after round-trip: {other:?}"),
        }
    }

    #[test]
    fn rollback_kv_round_trips() {
        let req = StageNodeRequest::RollbackKv {
            request_id: "req-1".to_string(),
            keep_count: 5,
        };
        match round_trip_request(&req) {
            StageNodeRequest::RollbackKv {
                request_id,
                keep_count,
            } => {
                assert_eq!(request_id, "req-1");
                assert_eq!(keep_count, 5);
            }
            other => panic!("unexpected variant after round-trip: {other:?}"),
        }
    }

    #[test]
    fn verified_batch_round_trips() {
        let resp = StageNodeResponse::VerifiedBatch {
            accepted_count: 3,
            accepted_token_ids: vec![10, 20, 30, 99],
            accepted_pieces: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            is_eog: false,
            profile: None,
        };
        match round_trip_response(&resp) {
            StageNodeResponse::VerifiedBatch {
                accepted_count,
                accepted_token_ids,
                accepted_pieces,
                is_eog,
                profile,
            } => {
                assert_eq!(accepted_count, 3);
                assert_eq!(accepted_token_ids, vec![10, 20, 30, 99]);
                assert_eq!(accepted_pieces, vec!["a", "b", "c", "d"]);
                assert!(!is_eog);
                assert!(profile.is_none());
            }
            other => panic!("unexpected variant after round-trip: {other:?}"),
        }
    }

    #[test]
    fn stage_node_info_back_compat_default_spec_decode_flag() {
        // Old peers (v≤0.2.34) ship StageNodeInfo without the spec_decode_v1
        // field. Make sure the new code accepts those payloads and defaults to
        // false rather than failing deserialization.
        let legacy_json = r#"{
            "protocol_version": 1,
            "model_id": "gemma-4-e4b",
            "stage_id": "head",
            "start_layer": 0,
            "end_layer": 17,
            "is_head": true,
            "is_tail": false
        }"#;
        let info: StageNodeInfo =
            serde_json::from_str(legacy_json).expect("legacy info should deserialize");
        assert_eq!(info.protocol_version, 1);
        assert!(!info.spec_decode_v1);
    }

    #[test]
    fn tensor_response_back_compat_omits_tail_sample() {
        // Skip-serializing-if-none keeps the wire size identical for peers that
        // don't ship a tail sample or profile. Confirm the optional fields are
        // absent in the JSON when None.
        let resp = StageNodeResponse::Tensor {
            tensor: Some(sample_tensor()),
            tail_sample: None,
            profile: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(
            !json.contains("tail_sample"),
            "tail_sample should be skipped when None: {json}"
        );
        assert!(
            !json.contains("profile"),
            "profile should be skipped when None: {json}"
        );
    }

    #[test]
    fn lookup_draft_uses_most_recent_exact_suffix_match() {
        let context = [1, 2, 3, 4, 9, 2, 3, 4, 10, 2, 3, 4];
        assert_eq!(lookup_draft_tokens(&context, 2, 3, 8), vec![10, 2]);
    }

    #[test]
    fn lookup_draft_respects_min_ngram() {
        let context = [1, 7, 2, 7];
        assert!(lookup_draft_tokens(&context, 1, 2, 8).is_empty());
        assert_eq!(lookup_draft_tokens(&context, 1, 1, 8), vec![2]);
    }

    #[test]
    fn lookup_draft_caps_guess_to_suffix_len() {
        let context = [1, 2, 10, 11, 12, 1, 2];
        assert_eq!(lookup_draft_tokens(&context, 8, 2, 8), vec![10, 11]);
    }
}

// ---------------------------------------------------------------------------
// Speculative decoding: head-side draft engine.
//
// Wraps a small companion model (e.g. Gemma-3-270M when targeting Gemma-4-E4B)
// in its own llama context so it can predict k candidate tokens before the
// target model burns the per-call Metal command-buffer setup cost. Tail-side
// verification (Phase 3) decides which drafts to accept.
//
// Owns its FFI handles directly. Greedy-only sampling for now — temperature
// and top-k can land alongside production tuning.
// ---------------------------------------------------------------------------

pub struct DraftEngine {
    api: LlamaApi,
    model: LlamaModelHandle,
    session: LlamaSession,
    n_vocab: usize,
    max_batch_tokens: usize,
    next_pos: i32,
}

// SAFETY: Same constraints as LlamaStageBackend — caller must serialize
// access. The draft engine is owned per-process (head node) and accessed
// behind the same outer mutex that guards the stage backend.
unsafe impl Send for DraftEngine {}
unsafe impl Sync for DraftEngine {}

impl DraftEngine {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let api = LlamaApi::load()?;
        let model = LlamaModelHandle::new(&api, model_path.as_ref())?;
        let n_ctx = env_u32("LLAMA_STAGE_DRAFT_N_CTX").unwrap_or(8192);
        let n_batch = env_u32("LLAMA_STAGE_DRAFT_N_BATCH").unwrap_or(64).max(1);
        let n_ubatch = env_u32("LLAMA_STAGE_DRAFT_N_UBATCH")
            .unwrap_or(n_batch)
            .max(1);
        let session = model.create_session_with_limits(&api, n_ctx, n_batch, n_ubatch)?;
        session.clear_memory(&api);
        let vocab = unsafe { (api.model_get_vocab)(model.model) };
        if vocab.is_null() {
            bail!("draft model vocab unavailable");
        }
        let n_vocab = unsafe { (api.vocab_n_tokens)(vocab) as usize };
        let mut draft = Self {
            api,
            model,
            session,
            n_vocab,
            max_batch_tokens: n_batch as usize,
            next_pos: 0,
        };
        if std::env::var_os("LLAMA_STAGE_DRAFT_SKIP_WARMUP").is_none() {
            draft.warmup()?;
        }
        Ok(draft)
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn current_pos(&self) -> i32 {
        self.next_pos
    }

    /// Reset the KV cache and seek back to position 0. Use between independent
    /// requests (the gateway's session boundary).
    pub fn reset(&mut self) {
        self.session.clear_memory(&self.api);
        self.next_pos = 0;
    }

    fn warmup(&mut self) -> Result<()> {
        let warmup_token = 0;
        let batch = OwnedBatch::token_only(vec![warmup_token, warmup_token]);
        let rc = unsafe { (self.api.decode)(self.session.ctx, batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("draft warmup llama_decode returned {rc}");
        }
        let logits = unsafe { (self.api.get_logits_ith)(self.session.ctx, -1) };
        if logits.is_null() {
            bail!("draft warmup logits null after decode");
        }
        let first = unsafe { *logits };
        std::hint::black_box(first);
        self.reset();
        Ok(())
    }

    /// Tokenize text using the draft model's vocab. For canary-string startup
    /// checks against the target's tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let vocab = unsafe { (self.api.model_get_vocab)(self.model.model) };
        if vocab.is_null() {
            bail!("draft model vocab unavailable");
        }
        let prompt = CString::new(text)?;
        let max_tokens = (text.len() + 64) as i32;
        let mut buf = vec![0i32; max_tokens as usize];
        let n = unsafe {
            (self.api.tokenize)(
                vocab,
                prompt.as_ptr(),
                text.len() as i32,
                buf.as_mut_ptr(),
                max_tokens,
                true,
                true,
            )
        };
        if n < 0 {
            bail!("draft tokenize returned {n}");
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    /// Run a batch of `tokens` through the draft model. Advances next_pos by
    /// tokens.len(). Used for prompt prefill.
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        for chunk in tokens.chunks(self.max_batch_tokens) {
            let batch = OwnedBatch::token_only(chunk.to_vec());
            let rc = unsafe { (self.api.decode)(self.session.ctx, batch.raw) };
            if rc != 0 && rc != 1 {
                bail!("draft prefill llama_decode returned {rc}");
            }
            self.next_pos += chunk.len() as i32;
        }
        Ok(())
    }

    /// Decode one token and greedy-sample the next. Returns the sampled token
    /// id, advances next_pos by 1 (the position consumed by `last_token`).
    /// The sampled id is NOT yet in the KV cache — callers either feed it
    /// back via greedy_step (for chained drafting) or discard.
    pub fn greedy_step(&mut self, last_token: i32) -> Result<i32> {
        let batch = OwnedBatch::token_only(vec![last_token]);
        let rc = unsafe { (self.api.decode)(self.session.ctx, batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("draft greedy_step llama_decode returned {rc}");
        }
        self.next_pos += 1;
        let logits = unsafe { (self.api.get_logits_ith)(self.session.ctx, -1) };
        if logits.is_null() {
            bail!("draft logits null after decode");
        }
        let logits = unsafe { slice::from_raw_parts(logits, self.n_vocab) };
        let (token_id, _) = argmax_f32(logits).context("empty draft logits buffer")?;
        Ok(token_id as i32)
    }

    /// Chain k greedy steps starting from `last_token`. Returns the k draft
    /// tokens in order. Each iteration commits the previous draft to the KV
    /// cache so the next prediction is conditioned on it.
    pub fn greedy_step_k(&mut self, last_token: i32, k: u32) -> Result<Vec<i32>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(k as usize);
        let mut cur = last_token;
        for _ in 0..k {
            let next = self.greedy_step(cur)?;
            out.push(next);
            cur = next;
        }
        Ok(out)
    }

    /// Like `greedy_step_k`, but first commits target-accepted draft tokens
    /// that were deferred from a prior full-accept round. The prefix and the
    /// current `last_token` are decoded in one batch, avoiding an extra draft
    /// call while leaving the KV in the same state as separate greedy steps.
    pub fn greedy_step_k_with_prefix(
        &mut self,
        prefix_tokens: &[i32],
        last_token: i32,
        k: u32,
    ) -> Result<Vec<i32>> {
        if prefix_tokens.is_empty() {
            return self.greedy_step_k(last_token, k);
        }
        if k == 0 {
            self.prefill(prefix_tokens)?;
            return Ok(Vec::new());
        }
        if prefix_tokens.len() + 1 > self.max_batch_tokens {
            self.prefill(prefix_tokens)?;
            return self.greedy_step_k(last_token, k);
        }

        let mut tokens = Vec::with_capacity(prefix_tokens.len() + 1);
        tokens.extend_from_slice(prefix_tokens);
        tokens.push(last_token);
        let batch = OwnedBatch::token_only(tokens);
        let rc = unsafe { (self.api.decode)(self.session.ctx, batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("draft greedy_step_k_with_prefix llama_decode returned {rc}");
        }
        self.next_pos += prefix_tokens.len() as i32 + 1;

        let logits = unsafe { (self.api.get_logits_ith)(self.session.ctx, -1) };
        if logits.is_null() {
            bail!("draft logits null after prefix decode");
        }
        let logits = unsafe { slice::from_raw_parts(logits, self.n_vocab) };
        let (token_id, _) = argmax_f32(logits).context("empty draft logits buffer")?;

        let mut out = Vec::with_capacity(k as usize);
        let mut cur = token_id as i32;
        out.push(cur);
        for _ in 1..k {
            cur = self.greedy_step(cur)?;
            out.push(cur);
        }
        Ok(out)
    }

    /// Roll back the KV cache so only the first `keep_count` token positions
    /// remain. Tail-side verification calls this on the draft after rejecting
    /// a suffix so the next round starts from the accepted prefix.
    pub fn rollback_to(&mut self, keep_count: i32) -> Result<()> {
        if keep_count < 0 || keep_count > self.next_pos {
            bail!(
                "draft rollback keep_count={keep_count} out of range (current pos={})",
                self.next_pos
            );
        }
        if keep_count == self.next_pos {
            return Ok(());
        }
        let memory = unsafe { (self.api.get_memory)(self.session.ctx) };
        if memory.is_null() {
            bail!("draft session memory unavailable");
        }
        // seq_id 0 is the only sequence in single-sequence contexts. p0=keep_count
        // p1=-1 means "all positions >= keep_count". Returns true on success.
        let ok = unsafe { (self.api.memory_seq_rm)(memory, 0, keep_count, -1) };
        if !ok {
            bail!("memory_seq_rm failed for draft (keep_count={keep_count})");
        }
        self.next_pos = keep_count;
        Ok(())
    }
}

impl Drop for DraftEngine {
    fn drop(&mut self) {
        // Match the explicit destroy/free pattern used elsewhere — we own the
        // raw handles and they must be released before the LlamaApi (which
        // owns the dlopen) is dropped.
        unsafe { (self.api.context_free)(self.session.ctx) };
        unsafe { (self.api.model_free)(self.model.model) };
    }
}
