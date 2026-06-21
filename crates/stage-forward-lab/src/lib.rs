#![allow(
    clippy::clone_on_copy,
    clippy::collapsible_if,
    clippy::derivable_impls,
    clippy::identity_op,
    clippy::manual_clamp,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    clippy::map_flatten,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::ptr_arg,
    clippy::question_mark,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::useless_conversion
)]

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub mod gguf;
pub mod prompt_suite;
pub mod prompting;
pub mod quants;
pub mod real_forward;
pub mod real_math;
pub mod tokenizer;

const STAGE_TENSOR_BYTES_MAGIC: [u8; 4] = *b"stb1";
const STAGE_TENSOR_BYTES_HEADER_LEN: usize = 12;

/// Serde helper that encodes `Vec<u8>` as a base64 string (over JSON) instead
/// of serde_json's default array-of-numbers. For hot-path tensor payloads on
/// the stage wire this is ~4× smaller and dramatically cheaper to parse.
/// Applies only to JSON; binary formats (bincode/postcard) pass raw bytes
/// via `serialize_bytes`.
pub mod bytes_b64 {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use serde::de::{self, Visitor};
    use serde::{Deserializer, Serialize, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            STANDARD.encode(bytes).serialize(serializer)
        } else {
            serializer.serialize_bytes(bytes)
        }
    }

    struct BytesOrBase64Visitor;

    impl<'de> Visitor<'de> for BytesOrBase64Visitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a base64 string or raw bytes")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<Vec<u8>, E> {
            STANDARD.decode(value).map_err(E::custom)
        }

        fn visit_string<E: de::Error>(self, value: String) -> Result<Vec<u8>, E> {
            self.visit_str(&value)
        }

        fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Vec<u8>, E> {
            Ok(value.to_vec())
        }

        fn visit_byte_buf<E: de::Error>(self, value: Vec<u8>) -> Result<Vec<u8>, E> {
            Ok(value)
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
            let mut bytes = Vec::new();
            while let Some(byte) = seq.next_element::<u8>()? {
                bytes.push(byte);
            }
            Ok(bytes)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        deserializer.deserialize_any(BytesOrBase64Visitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageTensorByteSections<'a> {
    pub hidden_bytes: &'a [u8],
    pub aux_bytes: Option<&'a [u8]>,
}

pub fn stage_tensor_byte_sections(bytes: &[u8]) -> Option<StageTensorByteSections<'_>> {
    if bytes.len() < STAGE_TENSOR_BYTES_HEADER_LEN || bytes[..4] != STAGE_TENSOR_BYTES_MAGIC {
        return None;
    }
    let hidden_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let aux_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let total_len = STAGE_TENSOR_BYTES_HEADER_LEN + hidden_len + aux_len;
    if total_len != bytes.len() {
        return None;
    }
    let hidden_start = STAGE_TENSOR_BYTES_HEADER_LEN;
    let hidden_end = hidden_start + hidden_len;
    let aux_bytes = if aux_len == 0 {
        None
    } else {
        Some(&bytes[hidden_end..])
    };
    Some(StageTensorByteSections {
        hidden_bytes: &bytes[hidden_start..hidden_end],
        aux_bytes,
    })
}

pub fn encode_stage_tensor_bytes(hidden_bytes: &[u8], aux_bytes: Option<&[u8]>) -> Vec<u8> {
    let Some(aux_bytes) = aux_bytes.filter(|bytes| !bytes.is_empty()) else {
        return hidden_bytes.to_vec();
    };

    let hidden_len = u32::try_from(hidden_bytes.len()).unwrap_or(u32::MAX);
    let aux_len = u32::try_from(aux_bytes.len()).unwrap_or(u32::MAX);
    let mut framed =
        Vec::with_capacity(STAGE_TENSOR_BYTES_HEADER_LEN + hidden_bytes.len() + aux_bytes.len());
    framed.extend_from_slice(&STAGE_TENSOR_BYTES_MAGIC);
    framed.extend_from_slice(&hidden_len.to_le_bytes());
    framed.extend_from_slice(&aux_len.to_le_bytes());
    framed.extend_from_slice(hidden_bytes);
    framed.extend_from_slice(aux_bytes);
    framed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadKind {
    PromptIngress,
    HiddenState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTensor {
    pub request_id: String,
    pub kind: PayloadKind,
    pub stage_trace: Vec<String>,
    pub hidden_dim: usize,
    #[serde(with = "bytes_b64")]
    pub bytes: Vec<u8>,
    pub prompt_text: Option<String>,
    pub max_tokens: Option<u32>,
    pub continuation: Option<StageContinuation>,
    pub transient: Option<StageTransientState>,
    pub carry: Option<StageCarryState>,
}

impl StageTensor {
    pub fn hidden_state_len(&self) -> usize {
        stage_tensor_byte_sections(&self.bytes)
            .map(|sections| sections.hidden_bytes.len())
            .unwrap_or(self.bytes.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageContinuation {
    pub version: u32,
    pub stage_role: String,
    pub next_layer_index: Option<u32>,
    pub completed_layers: u32,
    pub operator_layers: usize,
    pub has_attention_path: bool,
    pub has_ffn_path: bool,
    pub has_projection_path: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTransientState {
    pub attention: Option<AttentionContinuation>,
    pub ffn: Option<FfnContinuation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionContinuation {
    pub width: usize,
    pub lane_indices: Vec<usize>,
    pub q_preview: Vec<f32>,
    pub k_preview: Vec<f32>,
    pub v_preview: Vec<f32>,
    pub score_preview: Vec<f32>,
    pub value_preview: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FfnContinuation {
    pub width: usize,
    pub lane_indices: Vec<usize>,
    pub gate_preview: Vec<f32>,
    pub up_preview: Vec<f32>,
    pub activation_preview: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageLayout {
    pub model_id: String,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSample {
    pub request_id: String,
    pub model_id: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub completion_tokens: u32,
}

impl StageSample {
    pub fn text_token_ids(text: &str) -> Vec<u32> {
        text.chars().map(|ch| ch as u32).collect()
    }
}

pub const STAGE_FORWARD_FRAME_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageStateEnvelope {
    pub continuation: Option<StageContinuation>,
    pub transient: Option<StageTransientState>,
    pub carry: Option<StageCarryState>,
}

impl StageStateEnvelope {
    pub fn is_empty(&self) -> bool {
        self.continuation.is_none() && self.transient.is_none() && self.carry.is_none()
    }

    pub fn from_tensor(tensor: &StageTensor) -> Self {
        Self {
            continuation: tensor.continuation.clone(),
            transient: tensor.transient.clone(),
            carry: tensor.carry.clone(),
        }
    }

    pub fn apply_to_tensor(&self, tensor: &mut StageTensor) {
        tensor.continuation = self.continuation.clone();
        tensor.transient = self.transient.clone();
        tensor.carry = self.carry.clone();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRoute {
    pub source_stage_id: String,
    pub source_stage_start: u32,
    pub source_stage_end: u32,
    pub source_stage_role: String,
    pub target_stage_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageForwardFrame {
    pub version: u32,
    pub model_id: String,
    pub route: StageRoute,
    pub payload: StageTensor,
    pub state: StageStateEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageForwardFrameSummary {
    pub request_id: String,
    pub model_id: String,
    pub source_stage_id: String,
    pub source_stage_role: String,
    pub target_stage_id: Option<String>,
    pub payload_kind: PayloadKind,
    pub trace_depth: usize,
    pub hidden_dim: usize,
    pub hidden_bytes: usize,
    pub completed_layers: Option<u32>,
    pub operator_layers: Option<usize>,
    pub has_transient: bool,
    pub has_attention_transient: bool,
    pub has_ffn_transient: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransientSignature {
    pub width: usize,
    pub preview_len: usize,
    pub mean_milli: i32,
    pub mean_abs_milli: u32,
    pub rms_milli: u32,
    pub checksum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageTransferTransientState {
    pub attention: Option<TransientSignature>,
    pub ffn: Option<TransientSignature>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionCheckpointProvenance {
    pub layer_index: u32,
    pub operator_kind: String,
    pub layer_distance_to_boundary: u32,
}

impl AttentionCheckpointProvenance {
    fn age_by_layers(&self, layers: u32) -> Self {
        Self {
            layer_index: self.layer_index,
            operator_kind: self.operator_kind.clone(),
            layer_distance_to_boundary: self.layer_distance_to_boundary.saturating_add(layers),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryableAttentionProjectionState {
    pub width: usize,
    pub q_provenance: Option<AttentionCheckpointProvenance>,
    pub k_provenance: Option<AttentionCheckpointProvenance>,
    pub v_provenance: Option<AttentionCheckpointProvenance>,
    pub q_lane_indices: Vec<usize>,
    pub k_lane_indices: Vec<usize>,
    pub v_lane_indices: Vec<usize>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryableAttentionMixState {
    pub width: usize,
    pub score_provenance: Option<AttentionCheckpointProvenance>,
    pub value_provenance: Option<AttentionCheckpointProvenance>,
    pub score_lane_indices: Vec<usize>,
    pub value_lane_indices: Vec<usize>,
    pub scores: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryableAttentionState {
    #[serde(default)]
    pub contract: StageResumeContractSummary,
    pub projection: Option<CarryableAttentionProjectionState>,
    pub mix: Option<CarryableAttentionMixState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryableFfnState {
    pub width: usize,
    pub lane_indices: Vec<usize>,
    pub gate_head: Vec<f32>,
    pub up_head: Vec<f32>,
    pub activation_head: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageCarryPolicy {
    pub carry_attention: bool,
    pub carry_ffn: bool,
}

impl StageCarryPolicy {
    pub fn attention_only() -> Self {
        Self {
            carry_attention: true,
            carry_ffn: false,
        }
    }

    pub fn full() -> Self {
        Self {
            carry_attention: true,
            carry_ffn: true,
        }
    }

    pub fn for_boundary(layout: &StageLayout, continuation: Option<&StageContinuation>) -> Self {
        let mut policy = if layout.is_head && !layout.is_tail {
            Self::attention_only()
        } else if !layout.is_head && !layout.is_tail {
            Self::full()
        } else {
            Self {
                carry_attention: false,
                carry_ffn: false,
            }
        };

        if let Some(continuation) = continuation {
            policy.carry_attention &= continuation.has_attention_path;
            policy.carry_ffn &= continuation.has_ffn_path;
        }

        policy
    }

    pub fn for_execution_boundary(
        layout: &StageLayout,
        continuation: Option<&StageContinuation>,
        execution_programs: &[LayerExecutionProgram],
    ) -> Self {
        let mut policy = Self::for_boundary(layout, continuation);
        let has_runnable_attention = execution_programs.iter().any(|program| {
            program.runnable_sketch
                && program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        ExecutionOpKind::AttentionQ
                            | ExecutionOpKind::AttentionK
                            | ExecutionOpKind::AttentionV
                            | ExecutionOpKind::AttentionOut
                    )
                })
        });
        let has_runnable_ffn = execution_programs.iter().any(|program| {
            program.runnable_sketch
                && program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        ExecutionOpKind::FfnGate
                            | ExecutionOpKind::FfnUp
                            | ExecutionOpKind::FfnDown
                    )
                })
        });

        policy.carry_attention &= has_runnable_attention;
        policy.carry_ffn &= has_runnable_ffn;
        policy
    }
}

impl Default for StageCarryPolicy {
    fn default() -> Self {
        Self::attention_only()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageCarryState {
    pub attention: Option<CarryableAttentionState>,
    pub ffn: Option<CarryableFfnState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTransferStateEnvelope {
    pub continuation: Option<StageContinuation>,
    pub transient: Option<StageTransferTransientState>,
    pub carry: Option<StageCarryState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTransferPayload {
    pub request_id: String,
    pub kind: PayloadKind,
    pub stage_trace: Vec<String>,
    pub hidden_dim: usize,
    #[serde(with = "bytes_b64")]
    pub bytes: Vec<u8>,
    pub prompt_text: Option<String>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTransferFrame {
    pub version: u32,
    pub model_id: String,
    pub route: StageRoute,
    pub payload: StageTransferPayload,
    pub state: StageTransferStateEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageBoundaryPlan {
    pub source_stage_id: String,
    pub source_stage_role: String,
    pub source_stage_start: u32,
    pub source_stage_end: u32,
    pub target_stage_id: Option<String>,
    pub next_layer_index: Option<u32>,
    pub expected_payload_kind: PayloadKind,
    pub expected_hidden_dim: usize,
    pub carry_policy: StageCarryPolicy,
    pub expects_attention_carry: bool,
    pub expects_ffn_carry: bool,
    pub expected_attention_width: Option<usize>,
    pub expected_attention_lanes: Option<usize>,
    pub expects_attention_projection_carry: bool,
    pub expects_attention_mix_carry: bool,
    pub expected_attention_q_lanes: Option<usize>,
    pub expected_attention_k_lanes: Option<usize>,
    pub expected_attention_v_lanes: Option<usize>,
    pub expected_attention_score_lanes: Option<usize>,
    pub expected_attention_value_lanes: Option<usize>,
    pub expected_attention_q_distance: Option<u32>,
    pub expected_attention_k_distance: Option<u32>,
    pub expected_attention_v_distance: Option<u32>,
    pub expected_attention_score_distance: Option<u32>,
    pub expected_attention_value_distance: Option<u32>,
    pub expected_attention_projection_lanes: Option<usize>,
    pub expected_attention_mix_lanes: Option<usize>,
    pub expected_ffn_width: Option<usize>,
    pub expected_ffn_lanes: Option<usize>,
    pub resumable_attention_path: bool,
    pub resumable_attention_q: bool,
    pub resumable_attention_k: bool,
    pub resumable_attention_v: bool,
    pub resumable_attention_q_lanes: Option<usize>,
    pub resumable_attention_k_lanes: Option<usize>,
    pub resumable_attention_v_lanes: Option<usize>,
    pub resumable_attention_q_max_distance: Option<u32>,
    pub resumable_attention_k_max_distance: Option<u32>,
    pub resumable_attention_v_max_distance: Option<u32>,
    pub resumable_attention_score_max_distance: Option<u32>,
    pub resumable_attention_value_max_distance: Option<u32>,
    pub resumable_attention_contract: StageResumeContractSummary,
    pub resumable_attention_projection: bool,
    pub resumable_attention_mix: bool,
    pub resumable_ffn_path: bool,
    pub resumable_projection_path: bool,
    pub operator_layers: usize,
    pub completed_layers: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageResumeRequest {
    pub version: u32,
    pub model_id: String,
    pub target_stage_id: Option<String>,
    pub boundary: StageBoundaryPlan,
    pub transfer: StageTransferFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageResumeDecision {
    pub accepted: bool,
    pub accepted_stage_id: Option<String>,
    pub accepted_payload_kind: PayloadKind,
    pub accepted_hidden_dim: usize,
    pub accepted_attention_carry: bool,
    pub accepted_ffn_carry: bool,
    pub accepted_attention_width: Option<usize>,
    pub accepted_attention_lanes: Option<usize>,
    pub accepted_attention_q_resume: bool,
    pub accepted_attention_k_resume: bool,
    pub accepted_attention_v_resume: bool,
    pub accepted_attention_q_lanes: Option<usize>,
    pub accepted_attention_k_lanes: Option<usize>,
    pub accepted_attention_v_lanes: Option<usize>,
    pub accepted_attention_q_max_distance: Option<u32>,
    pub accepted_attention_k_max_distance: Option<u32>,
    pub accepted_attention_v_max_distance: Option<u32>,
    pub accepted_attention_score_max_distance: Option<u32>,
    pub accepted_attention_value_max_distance: Option<u32>,
    pub accepted_attention_contract: StageResumeContractSummary,
    pub accepted_attention_score_lanes: Option<usize>,
    pub accepted_attention_value_lanes: Option<usize>,
    pub accepted_attention_q_distance: Option<u32>,
    pub accepted_attention_k_distance: Option<u32>,
    pub accepted_attention_v_distance: Option<u32>,
    pub accepted_attention_score_distance: Option<u32>,
    pub accepted_attention_value_distance: Option<u32>,
    pub accepted_attention_projection_carry: bool,
    pub accepted_attention_mix_carry: bool,
    pub accepted_attention_projection_lanes: Option<usize>,
    pub accepted_attention_mix_lanes: Option<usize>,
    pub accepted_ffn_width: Option<usize>,
    pub accepted_ffn_lanes: Option<usize>,
    pub accepted_next_layer_index: Option<u32>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageResumeReceipt {
    pub version: u32,
    pub accepted: bool,
    pub accepted_stage_id: Option<String>,
    pub accepted_payload_kind: PayloadKind,
    pub accepted_hidden_dim: usize,
    pub accepted_attention_carry: bool,
    pub accepted_ffn_carry: bool,
    pub accepted_attention_width: Option<usize>,
    pub accepted_attention_lanes: Option<usize>,
    pub accepted_attention_q_resume: bool,
    pub accepted_attention_k_resume: bool,
    pub accepted_attention_v_resume: bool,
    pub accepted_attention_q_lanes: Option<usize>,
    pub accepted_attention_k_lanes: Option<usize>,
    pub accepted_attention_v_lanes: Option<usize>,
    pub accepted_attention_q_max_distance: Option<u32>,
    pub accepted_attention_k_max_distance: Option<u32>,
    pub accepted_attention_v_max_distance: Option<u32>,
    pub accepted_attention_score_max_distance: Option<u32>,
    pub accepted_attention_value_max_distance: Option<u32>,
    pub accepted_attention_contract: StageResumeContractSummary,
    pub accepted_attention_score_lanes: Option<usize>,
    pub accepted_attention_value_lanes: Option<usize>,
    pub accepted_attention_q_distance: Option<u32>,
    pub accepted_attention_k_distance: Option<u32>,
    pub accepted_attention_v_distance: Option<u32>,
    pub accepted_attention_score_distance: Option<u32>,
    pub accepted_attention_value_distance: Option<u32>,
    pub accepted_attention_projection_carry: bool,
    pub accepted_attention_mix_carry: bool,
    pub accepted_attention_projection_lanes: Option<usize>,
    pub accepted_attention_mix_lanes: Option<usize>,
    pub accepted_ffn_width: Option<usize>,
    pub accepted_ffn_lanes: Option<usize>,
    pub accepted_next_layer_index: Option<u32>,
    pub reason: Option<String>,
}

impl StageForwardFrame {
    fn carry_with_policy(&self, policy: &StageCarryPolicy) -> Option<StageCarryState> {
        self.state
            .carry
            .as_ref()
            .map(|carry| carry.with_policy(policy))
            .or_else(|| {
                self.state
                    .transient
                    .as_ref()
                    .map(|transient| StageCarryState::from_transient(transient, policy))
            })
    }

    pub fn from_tensor(
        model_id: impl Into<String>,
        layout: &StageLayout,
        tensor: StageTensor,
        target_stage_id: Option<String>,
    ) -> Self {
        let state = StageStateEnvelope::from_tensor(&tensor);
        let source_stage_role = if layout.is_head {
            "head"
        } else if layout.is_tail {
            "tail"
        } else {
            "middle"
        };
        Self {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: model_id.into(),
            route: StageRoute {
                source_stage_id: layout.stage_id.clone(),
                source_stage_start: layout.start_layer,
                source_stage_end: layout.end_layer,
                source_stage_role: source_stage_role.to_string(),
                target_stage_id,
            },
            payload: tensor,
            state,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != STAGE_FORWARD_FRAME_VERSION {
            bail!("Unsupported stage forward frame version {}", self.version);
        }
        if self.route.source_stage_start > self.route.source_stage_end {
            bail!(
                "Invalid stage range {}..{}",
                self.route.source_stage_start,
                self.route.source_stage_end
            );
        }
        if self.payload.hidden_dim == 0 && self.payload.kind == PayloadKind::HiddenState {
            bail!("Hidden-state frame must declare hidden_dim");
        }
        if self.payload.kind == PayloadKind::HiddenState
            && self.payload.bytes.len() != self.payload.hidden_dim * 4
        {
            bail!(
                "Hidden-state frame bytes {} do not match hidden_dim {}",
                self.payload.bytes.len(),
                self.payload.hidden_dim
            );
        }
        if !self.payload.stage_trace.is_empty()
            && self.payload.stage_trace.last() != Some(&self.route.source_stage_id)
        {
            bail!(
                "Frame source stage {} does not match payload trace tail {:?}",
                self.route.source_stage_id,
                self.payload.stage_trace.last()
            );
        }
        if self.payload.continuation != self.state.continuation {
            bail!("Frame payload continuation and state envelope diverged");
        }
        if self.payload.transient != self.state.transient {
            bail!("Frame payload transient and state envelope diverged");
        }
        if self.payload.carry != self.state.carry {
            bail!("Frame payload carry and state envelope diverged");
        }
        Ok(())
    }

    pub fn into_tensor(mut self) -> StageTensor {
        self.state.apply_to_tensor(&mut self.payload);
        self.payload
    }

    pub fn summary(&self) -> StageForwardFrameSummary {
        StageForwardFrameSummary {
            request_id: self.payload.request_id.clone(),
            model_id: self.model_id.clone(),
            source_stage_id: self.route.source_stage_id.clone(),
            source_stage_role: self.route.source_stage_role.clone(),
            target_stage_id: self.route.target_stage_id.clone(),
            payload_kind: self.payload.kind,
            trace_depth: self.payload.stage_trace.len(),
            hidden_dim: self.payload.hidden_dim,
            hidden_bytes: self.payload.bytes.len(),
            completed_layers: self
                .state
                .continuation
                .as_ref()
                .map(|continuation| continuation.completed_layers),
            operator_layers: self
                .state
                .continuation
                .as_ref()
                .map(|continuation| continuation.operator_layers),
            has_transient: self.state.transient.is_some(),
            has_attention_transient: self
                .state
                .transient
                .as_ref()
                .and_then(|transient| transient.attention.as_ref())
                .is_some(),
            has_ffn_transient: self
                .state
                .transient
                .as_ref()
                .and_then(|transient| transient.ffn.as_ref())
                .is_some(),
        }
    }

    pub fn to_transfer_frame(&self) -> StageTransferFrame {
        self.to_transfer_frame_with_policy(&StageCarryPolicy::default())
    }

    pub fn to_transfer_frame_for_layout(&self, layout: &StageLayout) -> StageTransferFrame {
        let policy = StageCarryPolicy::for_boundary(layout, self.state.continuation.as_ref());
        self.to_transfer_frame_with_policy(&policy)
    }

    pub fn to_transfer_frame_for_execution_boundary(
        &self,
        layout: &StageLayout,
        execution_programs: &[LayerExecutionProgram],
    ) -> StageTransferFrame {
        let policy = StageCarryPolicy::for_execution_boundary(
            layout,
            self.state.continuation.as_ref(),
            execution_programs,
        );
        let capabilities = StageResumeCapabilities::from_execution_programs(
            execution_programs,
            self.state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
        );
        let budgets = StageResumeBudgets::from_execution_programs(
            execution_programs,
            self.state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
            &capabilities,
        );
        let carry = self
            .carry_with_policy(&policy)
            .and_then(|carry| carry.with_resume_contract(&capabilities, &budgets));
        StageTransferFrame {
            version: self.version,
            model_id: self.model_id.clone(),
            route: self.route.clone(),
            payload: StageTransferPayload {
                request_id: self.payload.request_id.clone(),
                kind: self.payload.kind,
                stage_trace: self.payload.stage_trace.clone(),
                hidden_dim: self.payload.hidden_dim,
                bytes: self.payload.bytes.clone(),
                prompt_text: self.payload.prompt_text.clone(),
                max_tokens: self.payload.max_tokens,
            },
            state: StageTransferStateEnvelope {
                continuation: self.state.continuation.clone(),
                transient: self
                    .state
                    .transient
                    .as_ref()
                    .map(StageTransferTransientState::from),
                carry,
            },
        }
    }

    pub fn to_transfer_frame_with_policy(&self, policy: &StageCarryPolicy) -> StageTransferFrame {
        StageTransferFrame {
            version: self.version,
            model_id: self.model_id.clone(),
            route: self.route.clone(),
            payload: StageTransferPayload {
                request_id: self.payload.request_id.clone(),
                kind: self.payload.kind,
                stage_trace: self.payload.stage_trace.clone(),
                hidden_dim: self.payload.hidden_dim,
                bytes: self.payload.bytes.clone(),
                prompt_text: self.payload.prompt_text.clone(),
                max_tokens: self.payload.max_tokens,
            },
            state: StageTransferStateEnvelope {
                continuation: self.state.continuation.clone(),
                transient: self
                    .state
                    .transient
                    .as_ref()
                    .map(StageTransferTransientState::from),
                carry: self.carry_with_policy(policy),
            },
        }
    }

    pub fn to_boundary_plan_for_execution_boundary(
        &self,
        layout: &StageLayout,
        execution_programs: &[LayerExecutionProgram],
    ) -> StageBoundaryPlan {
        let carry_policy = StageCarryPolicy::for_execution_boundary(
            layout,
            self.state.continuation.as_ref(),
            execution_programs,
        );
        StageBoundaryPlan::from_frame_with_execution(self, carry_policy, execution_programs)
    }
}

impl StageBoundaryPlan {
    fn attention_provenance_distance(
        provenance: &Option<AttentionCheckpointProvenance>,
    ) -> Option<u32> {
        provenance
            .as_ref()
            .map(|provenance| provenance.layer_distance_to_boundary)
    }

    pub fn from_frame_with_execution(
        frame: &StageForwardFrame,
        carry_policy: StageCarryPolicy,
        execution_programs: &[LayerExecutionProgram],
    ) -> Self {
        let capabilities = StageResumeCapabilities::from_execution_programs(
            execution_programs,
            frame
                .state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
        );
        let budgets = StageResumeBudgets::from_execution_programs(
            execution_programs,
            frame
                .state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
            &capabilities,
        );
        let freshness = StageResumeFreshnessPolicy::from_execution_programs(
            execution_programs,
            frame
                .state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
            &capabilities,
        );
        let contract = StageResumeContractSummary::from_execution_programs(
            execution_programs,
            frame
                .state
                .continuation
                .as_ref()
                .and_then(|continuation| continuation.next_layer_index),
            &capabilities,
        );
        let filtered_frame = StageForwardFrame {
            version: frame.version,
            model_id: frame.model_id.clone(),
            route: frame.route.clone(),
            payload: frame.payload.clone(),
            state: StageStateEnvelope {
                continuation: frame.state.continuation.clone(),
                transient: None,
                carry: frame
                    .carry_with_policy(&carry_policy)
                    .and_then(|carry| carry.with_resume_contract(&capabilities, &budgets)),
            },
        };
        Self::from_frame_with_capabilities(
            frame,
            &filtered_frame,
            carry_policy,
            capabilities.attention_path,
            capabilities.attention_q,
            capabilities.attention_k,
            capabilities.attention_v,
            budgets.attention_q_lanes,
            budgets.attention_k_lanes,
            budgets.attention_v_lanes,
            freshness.attention_q_max_distance,
            freshness.attention_k_max_distance,
            freshness.attention_v_max_distance,
            freshness.attention_score_max_distance,
            freshness.attention_value_max_distance,
            contract,
            capabilities.attention_projection,
            capabilities.attention_mix,
            capabilities.ffn_path,
            capabilities.projection_path,
        )
    }

    pub fn from_frame(frame: &StageForwardFrame, carry_policy: StageCarryPolicy) -> Self {
        let continuation = frame.state.continuation.as_ref();
        Self::from_frame_with_capabilities(
            frame,
            frame,
            carry_policy,
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            StageResumeContractSummary::default(),
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_attention_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_ffn_path)
                .unwrap_or(false),
            continuation
                .map(|continuation| continuation.has_projection_path)
                .unwrap_or(false),
        )
    }

    fn from_frame_with_capabilities(
        frame: &StageForwardFrame,
        capability_frame: &StageForwardFrame,
        carry_policy: StageCarryPolicy,
        resumable_attention_path: bool,
        resumable_attention_q: bool,
        resumable_attention_k: bool,
        resumable_attention_v: bool,
        resumable_attention_q_lanes: Option<usize>,
        resumable_attention_k_lanes: Option<usize>,
        resumable_attention_v_lanes: Option<usize>,
        resumable_attention_q_max_distance: Option<u32>,
        resumable_attention_k_max_distance: Option<u32>,
        resumable_attention_v_max_distance: Option<u32>,
        resumable_attention_score_max_distance: Option<u32>,
        resumable_attention_value_max_distance: Option<u32>,
        resumable_attention_contract: StageResumeContractSummary,
        resumable_attention_projection: bool,
        resumable_attention_mix: bool,
        resumable_ffn: bool,
        resumable_projection_path: bool,
    ) -> Self {
        let continuation = frame.state.continuation.as_ref();
        let attention_provenance =
            capability_frame
                .state
                .carry
                .as_ref()
                .and_then(|carry| carry.attention.as_ref())
                .map(|attention| {
                    (
                        attention.projection.as_ref().and_then(|projection| {
                            Self::attention_provenance_distance(&projection.q_provenance)
                        }),
                        attention.projection.as_ref().and_then(|projection| {
                            Self::attention_provenance_distance(&projection.k_provenance)
                        }),
                        attention.projection.as_ref().and_then(|projection| {
                            Self::attention_provenance_distance(&projection.v_provenance)
                        }),
                        attention.mix.as_ref().and_then(|mix| {
                            Self::attention_provenance_distance(&mix.score_provenance)
                        }),
                        attention.mix.as_ref().and_then(|mix| {
                            Self::attention_provenance_distance(&mix.value_provenance)
                        }),
                    )
                });
        let attention_substates = if carry_policy.carry_attention {
            capability_frame
                .state
                .carry
                .as_ref()
                .and_then(|carry| carry.attention.as_ref())
                .map(|attention| {
                    (
                        attention.width(),
                        attention.lane_count(),
                        attention
                            .projection
                            .as_ref()
                            .map(|projection| projection.q_lane_indices.len()),
                        attention
                            .projection
                            .as_ref()
                            .map(|projection| projection.k_lane_indices.len()),
                        attention
                            .projection
                            .as_ref()
                            .map(|projection| projection.v_lane_indices.len()),
                        attention.projection.as_ref().map(|projection| {
                            projection
                                .q_lane_indices
                                .len()
                                .max(projection.k_lane_indices.len())
                                .max(projection.v_lane_indices.len())
                        }),
                        attention
                            .mix
                            .as_ref()
                            .map(|mix| mix.score_lane_indices.len()),
                        attention
                            .mix
                            .as_ref()
                            .map(|mix| mix.value_lane_indices.len()),
                        attention.mix.as_ref().map(|mix| {
                            mix.score_lane_indices
                                .len()
                                .max(mix.value_lane_indices.len())
                        }),
                    )
                })
                .or_else(|| {
                    capability_frame
                        .state
                        .transient
                        .as_ref()
                        .and_then(|transient| {
                            transient.attention.as_ref().map(|attention| {
                                (
                                    attention.width,
                                    attention
                                        .q_preview
                                        .len()
                                        .max(attention.k_preview.len())
                                        .max(attention.v_preview.len())
                                        .max(attention.score_preview.len())
                                        .max(attention.value_preview.len()),
                                    Some(attention.q_preview.len()),
                                    Some(attention.k_preview.len()),
                                    Some(attention.v_preview.len()),
                                    Some(
                                        attention
                                            .q_preview
                                            .len()
                                            .max(attention.k_preview.len())
                                            .max(attention.v_preview.len()),
                                    ),
                                    Some(attention.score_preview.len()),
                                    Some(attention.value_preview.len()),
                                    Some(
                                        attention
                                            .score_preview
                                            .len()
                                            .max(attention.value_preview.len()),
                                    ),
                                )
                            })
                        })
                })
        } else {
            None
        };
        let attention_carry = if carry_policy.carry_attention {
            attention_substates.map(|(width, lanes, _, _, _, _, _, _, _)| (width, lanes))
        } else {
            None
        };
        let ffn_carry = if carry_policy.carry_ffn {
            capability_frame
                .state
                .carry
                .as_ref()
                .and_then(|carry| carry.ffn.as_ref().map(|ffn| (ffn.width, ffn.lane_count())))
                .or_else(|| {
                    capability_frame
                        .state
                        .transient
                        .as_ref()
                        .and_then(|transient| {
                            transient.ffn.as_ref().map(|ffn| {
                                (
                                    ffn.width,
                                    ffn.gate_preview
                                        .len()
                                        .max(ffn.up_preview.len())
                                        .max(ffn.activation_preview.len()),
                                )
                            })
                        })
                })
        } else {
            None
        };
        Self {
            source_stage_id: frame.route.source_stage_id.clone(),
            source_stage_role: frame.route.source_stage_role.clone(),
            source_stage_start: frame.route.source_stage_start,
            source_stage_end: frame.route.source_stage_end,
            target_stage_id: frame.route.target_stage_id.clone(),
            next_layer_index: continuation.and_then(|continuation| continuation.next_layer_index),
            expected_payload_kind: frame.payload.kind,
            expected_hidden_dim: frame.payload.hidden_dim,
            expects_attention_carry: attention_carry.is_some(),
            expects_ffn_carry: ffn_carry.is_some(),
            expected_attention_width: attention_carry.map(|(width, _)| width),
            expected_attention_lanes: attention_carry.map(|(_, lanes)| lanes),
            expects_attention_projection_carry: attention_substates
                .and_then(|(_, _, _, _, _, projection_lanes, _, _, _)| projection_lanes)
                .is_some(),
            expects_attention_mix_carry: attention_substates
                .and_then(|(_, _, _, _, _, _, _, _, mix_lanes)| mix_lanes)
                .is_some(),
            expected_attention_q_lanes: attention_substates
                .and_then(|(_, _, q_lanes, _, _, _, _, _, _)| q_lanes),
            expected_attention_k_lanes: attention_substates
                .and_then(|(_, _, _, k_lanes, _, _, _, _, _)| k_lanes),
            expected_attention_v_lanes: attention_substates
                .and_then(|(_, _, _, _, v_lanes, _, _, _, _)| v_lanes),
            expected_attention_score_lanes: attention_substates
                .and_then(|(_, _, _, _, _, _, score_lanes, _, _)| score_lanes),
            expected_attention_value_lanes: attention_substates
                .and_then(|(_, _, _, _, _, _, _, value_lanes, _)| value_lanes),
            expected_attention_q_distance: attention_provenance
                .and_then(|(q_distance, _, _, _, _)| q_distance),
            expected_attention_k_distance: attention_provenance
                .and_then(|(_, k_distance, _, _, _)| k_distance),
            expected_attention_v_distance: attention_provenance
                .and_then(|(_, _, v_distance, _, _)| v_distance),
            expected_attention_score_distance: attention_provenance
                .and_then(|(_, _, _, score_distance, _)| score_distance),
            expected_attention_value_distance: attention_provenance
                .and_then(|(_, _, _, _, value_distance)| value_distance),
            expected_attention_projection_lanes: attention_substates
                .and_then(|(_, _, _, _, _, projection_lanes, _, _, _)| projection_lanes),
            expected_attention_mix_lanes: attention_substates
                .and_then(|(_, _, _, _, _, _, _, _, mix_lanes)| mix_lanes),
            expected_ffn_width: ffn_carry.map(|(width, _)| width),
            expected_ffn_lanes: ffn_carry.map(|(_, lanes)| lanes),
            resumable_attention_path,
            resumable_attention_q,
            resumable_attention_k,
            resumable_attention_v,
            resumable_attention_q_lanes,
            resumable_attention_k_lanes,
            resumable_attention_v_lanes,
            resumable_attention_q_max_distance,
            resumable_attention_k_max_distance,
            resumable_attention_v_max_distance,
            resumable_attention_score_max_distance,
            resumable_attention_value_max_distance,
            resumable_attention_contract,
            resumable_attention_projection,
            resumable_attention_mix,
            resumable_ffn_path: resumable_ffn,
            resumable_projection_path,
            operator_layers: continuation
                .map(|continuation| continuation.operator_layers)
                .unwrap_or(0),
            completed_layers: continuation.map(|continuation| continuation.completed_layers),
            carry_policy,
        }
    }

    pub fn validate_against_transfer(&self, frame: &StageTransferFrame) -> Result<()> {
        self.resumable_attention_contract
            .validate_against_freshness(StageResumeFreshnessPolicy {
                attention_q_max_distance: self.resumable_attention_q_max_distance,
                attention_k_max_distance: self.resumable_attention_k_max_distance,
                attention_v_max_distance: self.resumable_attention_v_max_distance,
                attention_score_max_distance: self.resumable_attention_score_max_distance,
                attention_value_max_distance: self.resumable_attention_value_max_distance,
            })?;
        if self.source_stage_id != frame.route.source_stage_id {
            bail!(
                "Boundary plan source stage {} does not match transfer source stage {}",
                self.source_stage_id,
                frame.route.source_stage_id
            );
        }
        if self.source_stage_role != frame.route.source_stage_role {
            bail!(
                "Boundary plan source role {} does not match transfer role {}",
                self.source_stage_role,
                frame.route.source_stage_role
            );
        }
        if self.expected_payload_kind != frame.payload.kind {
            bail!(
                "Boundary plan payload kind {:?} does not match transfer kind {:?}",
                self.expected_payload_kind,
                frame.payload.kind
            );
        }
        if self.expected_hidden_dim != frame.payload.hidden_dim {
            bail!(
                "Boundary plan hidden_dim {} does not match transfer hidden_dim {}",
                self.expected_hidden_dim,
                frame.payload.hidden_dim
            );
        }
        if self.target_stage_id != frame.route.target_stage_id {
            bail!(
                "Boundary plan target {:?} does not match transfer target {:?}",
                self.target_stage_id,
                frame.route.target_stage_id
            );
        }
        let transfer_carry = frame.state.carry.as_ref();
        let has_attention = transfer_carry
            .and_then(|carry| carry.attention.as_ref())
            .is_some();
        let has_ffn = transfer_carry
            .and_then(|carry| carry.ffn.as_ref())
            .is_some();
        if self.expects_attention_carry != has_attention {
            bail!(
                "Boundary plan expects attention carry={} but transfer has {}",
                self.expects_attention_carry,
                has_attention
            );
        }
        if self.expects_ffn_carry != has_ffn {
            bail!(
                "Boundary plan expects ffn carry={} but transfer has {}",
                self.expects_ffn_carry,
                has_ffn
            );
        }
        if let (Some(attention), Some(expected_width)) = (
            transfer_carry.and_then(|carry| carry.attention.as_ref()),
            self.expected_attention_width,
        ) {
            if attention.width() != expected_width {
                bail!(
                    "Boundary plan expects attention width {} but transfer has {}",
                    expected_width,
                    attention.width()
                );
            }
        }
        if let (Some(attention), Some(expected_lanes)) = (
            transfer_carry.and_then(|carry| carry.attention.as_ref()),
            self.expected_attention_lanes,
        ) {
            if attention.lane_count() != expected_lanes {
                bail!(
                    "Boundary plan expects attention lanes {} but transfer has {}",
                    expected_lanes,
                    attention.lane_count()
                );
            }
        }
        let transfer_projection = transfer_carry
            .and_then(|carry| carry.attention.as_ref())
            .and_then(|attention| attention.projection.as_ref());
        let transfer_mix = transfer_carry
            .and_then(|carry| carry.attention.as_ref())
            .and_then(|attention| attention.mix.as_ref());
        if self.expects_attention_projection_carry != transfer_projection.is_some() {
            bail!(
                "Boundary plan expects attention projection carry={} but transfer has {}",
                self.expects_attention_projection_carry,
                transfer_projection.is_some()
            );
        }
        if self.expects_attention_mix_carry != transfer_mix.is_some() {
            bail!(
                "Boundary plan expects attention mix carry={} but transfer has {}",
                self.expects_attention_mix_carry,
                transfer_mix.is_some()
            );
        }
        if let (Some(projection), Some(expected_lanes)) = (
            transfer_projection,
            self.expected_attention_projection_lanes,
        ) {
            let actual_lanes = projection
                .q_lane_indices
                .len()
                .max(projection.k_lane_indices.len())
                .max(projection.v_lane_indices.len());
            if actual_lanes != expected_lanes {
                bail!(
                    "Boundary plan expects attention projection lanes {} but transfer has {}",
                    expected_lanes,
                    actual_lanes
                );
            }
        }
        if let (Some(projection), Some(expected_lanes)) =
            (transfer_projection, self.expected_attention_q_lanes)
        {
            if projection.q_lane_indices.len() != expected_lanes {
                bail!(
                    "Boundary plan expects attention q lanes {} but transfer has {}",
                    expected_lanes,
                    projection.q_lane_indices.len()
                );
            }
        }
        if let (Some(projection), Some(expected_distance)) =
            (transfer_projection, self.expected_attention_q_distance)
        {
            let actual_distance = projection
                .q_provenance
                .as_ref()
                .map(|provenance| provenance.layer_distance_to_boundary);
            if actual_distance != Some(expected_distance) {
                bail!(
                    "Boundary plan expects attention q distance {} but transfer has {:?}",
                    expected_distance,
                    actual_distance
                );
            }
        }
        if let (Some(projection), Some(expected_lanes)) =
            (transfer_projection, self.expected_attention_k_lanes)
        {
            if projection.k_lane_indices.len() != expected_lanes {
                bail!(
                    "Boundary plan expects attention k lanes {} but transfer has {}",
                    expected_lanes,
                    projection.k_lane_indices.len()
                );
            }
        }
        if let (Some(projection), Some(expected_distance)) =
            (transfer_projection, self.expected_attention_k_distance)
        {
            let actual_distance = projection
                .k_provenance
                .as_ref()
                .map(|provenance| provenance.layer_distance_to_boundary);
            if actual_distance != Some(expected_distance) {
                bail!(
                    "Boundary plan expects attention k distance {} but transfer has {:?}",
                    expected_distance,
                    actual_distance
                );
            }
        }
        if let (Some(projection), Some(expected_lanes)) =
            (transfer_projection, self.expected_attention_v_lanes)
        {
            if projection.v_lane_indices.len() != expected_lanes {
                bail!(
                    "Boundary plan expects attention v lanes {} but transfer has {}",
                    expected_lanes,
                    projection.v_lane_indices.len()
                );
            }
        }
        if let (Some(projection), Some(expected_distance)) =
            (transfer_projection, self.expected_attention_v_distance)
        {
            let actual_distance = projection
                .v_provenance
                .as_ref()
                .map(|provenance| provenance.layer_distance_to_boundary);
            if actual_distance != Some(expected_distance) {
                bail!(
                    "Boundary plan expects attention v distance {} but transfer has {:?}",
                    expected_distance,
                    actual_distance
                );
            }
        }
        if let (Some(mix), Some(expected_lanes)) = (transfer_mix, self.expected_attention_mix_lanes)
        {
            let actual_lanes = mix
                .score_lane_indices
                .len()
                .max(mix.value_lane_indices.len());
            if actual_lanes != expected_lanes {
                bail!(
                    "Boundary plan expects attention mix lanes {} but transfer has {}",
                    expected_lanes,
                    actual_lanes
                );
            }
        }
        if let (Some(mix), Some(expected_lanes)) =
            (transfer_mix, self.expected_attention_score_lanes)
        {
            if mix.score_lane_indices.len() != expected_lanes {
                bail!(
                    "Boundary plan expects attention score lanes {} but transfer has {}",
                    expected_lanes,
                    mix.score_lane_indices.len()
                );
            }
        }
        if let (Some(mix), Some(expected_distance)) =
            (transfer_mix, self.expected_attention_score_distance)
        {
            let actual_distance = mix
                .score_provenance
                .as_ref()
                .map(|provenance| provenance.layer_distance_to_boundary);
            if actual_distance != Some(expected_distance) {
                bail!(
                    "Boundary plan expects attention score distance {} but transfer has {:?}",
                    expected_distance,
                    actual_distance
                );
            }
        }
        if let (Some(mix), Some(expected_lanes)) =
            (transfer_mix, self.expected_attention_value_lanes)
        {
            if mix.value_lane_indices.len() != expected_lanes {
                bail!(
                    "Boundary plan expects attention value lanes {} but transfer has {}",
                    expected_lanes,
                    mix.value_lane_indices.len()
                );
            }
        }
        if let (Some(mix), Some(expected_distance)) =
            (transfer_mix, self.expected_attention_value_distance)
        {
            let actual_distance = mix
                .value_provenance
                .as_ref()
                .map(|provenance| provenance.layer_distance_to_boundary);
            if actual_distance != Some(expected_distance) {
                bail!(
                    "Boundary plan expects attention value distance {} but transfer has {:?}",
                    expected_distance,
                    actual_distance
                );
            }
        }
        if let (Some(ffn), Some(expected_width)) = (
            transfer_carry.and_then(|carry| carry.ffn.as_ref()),
            self.expected_ffn_width,
        ) {
            if ffn.width != expected_width {
                bail!(
                    "Boundary plan expects ffn width {} but transfer has {}",
                    expected_width,
                    ffn.width
                );
            }
        }
        if let (Some(ffn), Some(expected_lanes)) = (
            transfer_carry.and_then(|carry| carry.ffn.as_ref()),
            self.expected_ffn_lanes,
        ) {
            if ffn.lane_count() != expected_lanes {
                bail!(
                    "Boundary plan expects ffn lanes {} but transfer has {}",
                    expected_lanes,
                    ffn.lane_count()
                );
            }
        }
        Ok(())
    }

    pub fn to_resume_request(&self, transfer: StageTransferFrame) -> Result<StageResumeRequest> {
        self.validate_against_transfer(&transfer)?;
        Ok(StageResumeRequest {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: transfer.model_id.clone(),
            target_stage_id: self.target_stage_id.clone(),
            boundary: self.clone(),
            transfer,
        })
    }
}

impl StageResumeRequest {
    pub fn validate(&self) -> Result<()> {
        if self.version != STAGE_FORWARD_FRAME_VERSION {
            bail!("Unsupported stage resume request version {}", self.version);
        }
        self.transfer.validate()?;
        self.boundary.validate_against_transfer(&self.transfer)?;
        Ok(())
    }

    pub fn accept(&self, accepted_stage_id: Option<String>) -> StageResumeReceipt {
        StageResumeReceipt {
            version: self.version,
            accepted: true,
            accepted_stage_id,
            accepted_payload_kind: self.transfer.payload.kind,
            accepted_hidden_dim: self.transfer.payload.hidden_dim,
            accepted_attention_carry: self.boundary.expects_attention_carry,
            accepted_ffn_carry: self.boundary.expects_ffn_carry,
            accepted_attention_width: self.boundary.expected_attention_width,
            accepted_attention_lanes: self.boundary.expected_attention_lanes,
            accepted_attention_q_resume: self.boundary.resumable_attention_q,
            accepted_attention_k_resume: self.boundary.resumable_attention_k,
            accepted_attention_v_resume: self.boundary.resumable_attention_v,
            accepted_attention_q_lanes: self.boundary.resumable_attention_q_lanes,
            accepted_attention_k_lanes: self.boundary.resumable_attention_k_lanes,
            accepted_attention_v_lanes: self.boundary.resumable_attention_v_lanes,
            accepted_attention_q_max_distance: self.boundary.resumable_attention_q_max_distance,
            accepted_attention_k_max_distance: self.boundary.resumable_attention_k_max_distance,
            accepted_attention_v_max_distance: self.boundary.resumable_attention_v_max_distance,
            accepted_attention_score_max_distance: self
                .boundary
                .resumable_attention_score_max_distance,
            accepted_attention_value_max_distance: self
                .boundary
                .resumable_attention_value_max_distance,
            accepted_attention_contract: self.boundary.resumable_attention_contract.clone(),
            accepted_attention_score_lanes: self.boundary.expected_attention_score_lanes,
            accepted_attention_value_lanes: self.boundary.expected_attention_value_lanes,
            accepted_attention_q_distance: self.boundary.expected_attention_q_distance,
            accepted_attention_k_distance: self.boundary.expected_attention_k_distance,
            accepted_attention_v_distance: self.boundary.expected_attention_v_distance,
            accepted_attention_score_distance: self.boundary.expected_attention_score_distance,
            accepted_attention_value_distance: self.boundary.expected_attention_value_distance,
            accepted_attention_projection_carry: self.boundary.expects_attention_projection_carry,
            accepted_attention_mix_carry: self.boundary.expects_attention_mix_carry,
            accepted_attention_projection_lanes: self.boundary.expected_attention_projection_lanes,
            accepted_attention_mix_lanes: self.boundary.expected_attention_mix_lanes,
            accepted_ffn_width: self.boundary.expected_ffn_width,
            accepted_ffn_lanes: self.boundary.expected_ffn_lanes,
            accepted_next_layer_index: self.boundary.next_layer_index,
            reason: None,
        }
    }

    pub fn reject(&self, reason: impl Into<String>) -> StageResumeReceipt {
        StageResumeReceipt {
            version: self.version,
            accepted: false,
            accepted_stage_id: self.target_stage_id.clone(),
            accepted_payload_kind: self.transfer.payload.kind,
            accepted_hidden_dim: self.transfer.payload.hidden_dim,
            accepted_attention_carry: false,
            accepted_ffn_carry: false,
            accepted_attention_width: None,
            accepted_attention_lanes: None,
            accepted_attention_q_resume: false,
            accepted_attention_k_resume: false,
            accepted_attention_v_resume: false,
            accepted_attention_q_lanes: None,
            accepted_attention_k_lanes: None,
            accepted_attention_v_lanes: None,
            accepted_attention_q_max_distance: None,
            accepted_attention_k_max_distance: None,
            accepted_attention_v_max_distance: None,
            accepted_attention_score_max_distance: None,
            accepted_attention_value_max_distance: None,
            accepted_attention_contract: StageResumeContractSummary::default(),
            accepted_attention_score_lanes: None,
            accepted_attention_value_lanes: None,
            accepted_attention_q_distance: None,
            accepted_attention_k_distance: None,
            accepted_attention_v_distance: None,
            accepted_attention_score_distance: None,
            accepted_attention_value_distance: None,
            accepted_attention_projection_carry: false,
            accepted_attention_mix_carry: false,
            accepted_attention_projection_lanes: None,
            accepted_attention_mix_lanes: None,
            accepted_ffn_width: None,
            accepted_ffn_lanes: None,
            accepted_next_layer_index: self.boundary.next_layer_index,
            reason: Some(reason.into()),
        }
    }
}

impl StageResumeDecision {
    pub fn accept(
        accepted_stage_id: Option<String>,
        payload_kind: PayloadKind,
        hidden_dim: usize,
        attention_carry: bool,
        ffn_carry: bool,
        attention_width: Option<usize>,
        attention_lanes: Option<usize>,
        attention_q_resume: bool,
        attention_k_resume: bool,
        attention_v_resume: bool,
        attention_q_lanes: Option<usize>,
        attention_k_lanes: Option<usize>,
        attention_v_lanes: Option<usize>,
        attention_q_max_distance: Option<u32>,
        attention_k_max_distance: Option<u32>,
        attention_v_max_distance: Option<u32>,
        attention_score_max_distance: Option<u32>,
        attention_value_max_distance: Option<u32>,
        attention_contract: StageResumeContractSummary,
        attention_score_lanes: Option<usize>,
        attention_value_lanes: Option<usize>,
        attention_q_distance: Option<u32>,
        attention_k_distance: Option<u32>,
        attention_v_distance: Option<u32>,
        attention_score_distance: Option<u32>,
        attention_value_distance: Option<u32>,
        attention_projection_carry: bool,
        attention_mix_carry: bool,
        attention_projection_lanes: Option<usize>,
        attention_mix_lanes: Option<usize>,
        ffn_width: Option<usize>,
        ffn_lanes: Option<usize>,
        next_layer_index: Option<u32>,
    ) -> Self {
        Self {
            accepted: true,
            accepted_stage_id,
            accepted_payload_kind: payload_kind,
            accepted_hidden_dim: hidden_dim,
            accepted_attention_carry: attention_carry,
            accepted_ffn_carry: ffn_carry,
            accepted_attention_width: attention_width,
            accepted_attention_lanes: attention_lanes,
            accepted_attention_q_resume: attention_q_resume,
            accepted_attention_k_resume: attention_k_resume,
            accepted_attention_v_resume: attention_v_resume,
            accepted_attention_q_lanes: attention_q_lanes,
            accepted_attention_k_lanes: attention_k_lanes,
            accepted_attention_v_lanes: attention_v_lanes,
            accepted_attention_q_max_distance: attention_q_max_distance,
            accepted_attention_k_max_distance: attention_k_max_distance,
            accepted_attention_v_max_distance: attention_v_max_distance,
            accepted_attention_score_max_distance: attention_score_max_distance,
            accepted_attention_value_max_distance: attention_value_max_distance,
            accepted_attention_contract: attention_contract,
            accepted_attention_score_lanes: attention_score_lanes,
            accepted_attention_value_lanes: attention_value_lanes,
            accepted_attention_q_distance: attention_q_distance,
            accepted_attention_k_distance: attention_k_distance,
            accepted_attention_v_distance: attention_v_distance,
            accepted_attention_score_distance: attention_score_distance,
            accepted_attention_value_distance: attention_value_distance,
            accepted_attention_projection_carry: attention_projection_carry,
            accepted_attention_mix_carry: attention_mix_carry,
            accepted_attention_projection_lanes: attention_projection_lanes,
            accepted_attention_mix_lanes: attention_mix_lanes,
            accepted_ffn_width: ffn_width,
            accepted_ffn_lanes: ffn_lanes,
            accepted_next_layer_index: next_layer_index,
            reason: None,
        }
    }

    pub fn reject(
        accepted_stage_id: Option<String>,
        payload_kind: PayloadKind,
        hidden_dim: usize,
        next_layer_index: Option<u32>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            accepted: false,
            accepted_stage_id,
            accepted_payload_kind: payload_kind,
            accepted_hidden_dim: hidden_dim,
            accepted_attention_carry: false,
            accepted_ffn_carry: false,
            accepted_attention_width: None,
            accepted_attention_lanes: None,
            accepted_attention_q_resume: false,
            accepted_attention_k_resume: false,
            accepted_attention_v_resume: false,
            accepted_attention_q_lanes: None,
            accepted_attention_k_lanes: None,
            accepted_attention_v_lanes: None,
            accepted_attention_q_max_distance: None,
            accepted_attention_k_max_distance: None,
            accepted_attention_v_max_distance: None,
            accepted_attention_score_max_distance: None,
            accepted_attention_value_max_distance: None,
            accepted_attention_contract: StageResumeContractSummary::default(),
            accepted_attention_score_lanes: None,
            accepted_attention_value_lanes: None,
            accepted_attention_q_distance: None,
            accepted_attention_k_distance: None,
            accepted_attention_v_distance: None,
            accepted_attention_score_distance: None,
            accepted_attention_value_distance: None,
            accepted_attention_projection_carry: false,
            accepted_attention_mix_carry: false,
            accepted_attention_projection_lanes: None,
            accepted_attention_mix_lanes: None,
            accepted_ffn_width: None,
            accepted_ffn_lanes: None,
            accepted_next_layer_index: next_layer_index,
            reason: Some(reason.into()),
        }
    }

    pub fn into_receipt(self, version: u32) -> StageResumeReceipt {
        StageResumeReceipt {
            version,
            accepted: self.accepted,
            accepted_stage_id: self.accepted_stage_id,
            accepted_payload_kind: self.accepted_payload_kind,
            accepted_hidden_dim: self.accepted_hidden_dim,
            accepted_attention_carry: self.accepted_attention_carry,
            accepted_ffn_carry: self.accepted_ffn_carry,
            accepted_attention_width: self.accepted_attention_width,
            accepted_attention_lanes: self.accepted_attention_lanes,
            accepted_attention_q_resume: self.accepted_attention_q_resume,
            accepted_attention_k_resume: self.accepted_attention_k_resume,
            accepted_attention_v_resume: self.accepted_attention_v_resume,
            accepted_attention_q_lanes: self.accepted_attention_q_lanes,
            accepted_attention_k_lanes: self.accepted_attention_k_lanes,
            accepted_attention_v_lanes: self.accepted_attention_v_lanes,
            accepted_attention_q_max_distance: self.accepted_attention_q_max_distance,
            accepted_attention_k_max_distance: self.accepted_attention_k_max_distance,
            accepted_attention_v_max_distance: self.accepted_attention_v_max_distance,
            accepted_attention_score_max_distance: self.accepted_attention_score_max_distance,
            accepted_attention_value_max_distance: self.accepted_attention_value_max_distance,
            accepted_attention_contract: self.accepted_attention_contract,
            accepted_attention_score_lanes: self.accepted_attention_score_lanes,
            accepted_attention_value_lanes: self.accepted_attention_value_lanes,
            accepted_attention_q_distance: self.accepted_attention_q_distance,
            accepted_attention_k_distance: self.accepted_attention_k_distance,
            accepted_attention_v_distance: self.accepted_attention_v_distance,
            accepted_attention_score_distance: self.accepted_attention_score_distance,
            accepted_attention_value_distance: self.accepted_attention_value_distance,
            accepted_attention_projection_carry: self.accepted_attention_projection_carry,
            accepted_attention_mix_carry: self.accepted_attention_mix_carry,
            accepted_attention_projection_lanes: self.accepted_attention_projection_lanes,
            accepted_attention_mix_lanes: self.accepted_attention_mix_lanes,
            accepted_ffn_width: self.accepted_ffn_width,
            accepted_ffn_lanes: self.accepted_ffn_lanes,
            accepted_next_layer_index: self.accepted_next_layer_index,
            reason: self.reason,
        }
    }
}

impl StageResumeReceipt {
    pub fn validate_against_request(&self, request: &StageResumeRequest) -> Result<()> {
        if self.version != request.version {
            bail!(
                "Resume receipt version {} does not match request version {}",
                self.version,
                request.version
            );
        }
        if self.accepted_payload_kind != request.transfer.payload.kind {
            bail!(
                "Resume receipt payload kind {:?} does not match request kind {:?}",
                self.accepted_payload_kind,
                request.transfer.payload.kind
            );
        }
        if self.accepted_hidden_dim != request.transfer.payload.hidden_dim {
            bail!(
                "Resume receipt hidden_dim {} does not match request hidden_dim {}",
                self.accepted_hidden_dim,
                request.transfer.payload.hidden_dim
            );
        }
        if self.accepted {
            if self.accepted_attention_carry != request.boundary.expects_attention_carry {
                bail!("Resume receipt accepted_attention_carry diverged from boundary plan");
            }
            if self.accepted_ffn_carry != request.boundary.expects_ffn_carry {
                bail!("Resume receipt accepted_ffn_carry diverged from boundary plan");
            }
            if self.accepted_attention_width != request.boundary.expected_attention_width {
                bail!("Resume receipt accepted_attention_width diverged from boundary plan");
            }
            if self.accepted_attention_lanes != request.boundary.expected_attention_lanes {
                bail!("Resume receipt accepted_attention_lanes diverged from boundary plan");
            }
            if self.accepted_attention_q_resume != request.boundary.resumable_attention_q {
                bail!("Resume receipt accepted_attention_q_resume diverged from boundary plan");
            }
            if self.accepted_attention_k_resume != request.boundary.resumable_attention_k {
                bail!("Resume receipt accepted_attention_k_resume diverged from boundary plan");
            }
            if self.accepted_attention_v_resume != request.boundary.resumable_attention_v {
                bail!("Resume receipt accepted_attention_v_resume diverged from boundary plan");
            }
            if self.accepted_attention_q_lanes != request.boundary.resumable_attention_q_lanes {
                bail!("Resume receipt accepted_attention_q_lanes diverged from boundary plan");
            }
            if self.accepted_attention_k_lanes != request.boundary.resumable_attention_k_lanes {
                bail!("Resume receipt accepted_attention_k_lanes diverged from boundary plan");
            }
            if self.accepted_attention_v_lanes != request.boundary.resumable_attention_v_lanes {
                bail!("Resume receipt accepted_attention_v_lanes diverged from boundary plan");
            }
            if self.accepted_attention_q_max_distance
                != request.boundary.resumable_attention_q_max_distance
            {
                bail!(
                    "Resume receipt accepted_attention_q_max_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_k_max_distance
                != request.boundary.resumable_attention_k_max_distance
            {
                bail!(
                    "Resume receipt accepted_attention_k_max_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_v_max_distance
                != request.boundary.resumable_attention_v_max_distance
            {
                bail!(
                    "Resume receipt accepted_attention_v_max_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_score_max_distance
                != request.boundary.resumable_attention_score_max_distance
            {
                bail!(
                    "Resume receipt accepted_attention_score_max_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_value_max_distance
                != request.boundary.resumable_attention_value_max_distance
            {
                bail!(
                    "Resume receipt accepted_attention_value_max_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_contract != request.boundary.resumable_attention_contract {
                bail!("Resume receipt accepted_attention_contract diverged from boundary plan");
            }
            if self.accepted_attention_score_lanes
                != request.boundary.expected_attention_score_lanes
            {
                bail!("Resume receipt accepted_attention_score_lanes diverged from boundary plan");
            }
            if self.accepted_attention_value_lanes
                != request.boundary.expected_attention_value_lanes
            {
                bail!("Resume receipt accepted_attention_value_lanes diverged from boundary plan");
            }
            if self.accepted_attention_q_distance != request.boundary.expected_attention_q_distance
            {
                bail!("Resume receipt accepted_attention_q_distance diverged from boundary plan");
            }
            if self.accepted_attention_k_distance != request.boundary.expected_attention_k_distance
            {
                bail!("Resume receipt accepted_attention_k_distance diverged from boundary plan");
            }
            if self.accepted_attention_v_distance != request.boundary.expected_attention_v_distance
            {
                bail!("Resume receipt accepted_attention_v_distance diverged from boundary plan");
            }
            if self.accepted_attention_score_distance
                != request.boundary.expected_attention_score_distance
            {
                bail!(
                    "Resume receipt accepted_attention_score_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_value_distance
                != request.boundary.expected_attention_value_distance
            {
                bail!(
                    "Resume receipt accepted_attention_value_distance diverged from boundary plan"
                );
            }
            if self.accepted_attention_projection_carry
                != request.boundary.expects_attention_projection_carry
            {
                bail!(
                    "Resume receipt accepted_attention_projection_carry diverged from boundary plan"
                );
            }
            if self.accepted_attention_mix_carry != request.boundary.expects_attention_mix_carry {
                bail!("Resume receipt accepted_attention_mix_carry diverged from boundary plan");
            }
            if self.accepted_attention_projection_lanes
                != request.boundary.expected_attention_projection_lanes
            {
                bail!(
                    "Resume receipt accepted_attention_projection_lanes diverged from boundary plan"
                );
            }
            if self.accepted_attention_mix_lanes != request.boundary.expected_attention_mix_lanes {
                bail!("Resume receipt accepted_attention_mix_lanes diverged from boundary plan");
            }
            if self.accepted_ffn_width != request.boundary.expected_ffn_width {
                bail!("Resume receipt accepted_ffn_width diverged from boundary plan");
            }
            if self.accepted_ffn_lanes != request.boundary.expected_ffn_lanes {
                bail!("Resume receipt accepted_ffn_lanes diverged from boundary plan");
            }
            if self.accepted_next_layer_index != request.boundary.next_layer_index {
                bail!("Resume receipt next_layer_index diverged from boundary plan");
            }
        }
        Ok(())
    }
}

impl TransientSignature {
    fn from_preview(width: usize, preview: &[f32]) -> Self {
        let preview_len = preview.len();
        if preview_len == 0 {
            return Self {
                width,
                preview_len,
                mean_milli: 0,
                mean_abs_milli: 0,
                rms_milli: 0,
                checksum: 0,
            };
        }

        let mean = preview.iter().sum::<f32>() / preview_len as f32;
        let mean_abs = preview.iter().map(|value| value.abs()).sum::<f32>() / preview_len as f32;
        let rms =
            (preview.iter().map(|value| value * value).sum::<f32>() / preview_len as f32).sqrt();
        let checksum = preview.iter().fold(0u64, |acc, value| {
            acc.wrapping_mul(16777619)
                .wrapping_add(value.to_bits() as u64)
        });

        Self {
            width,
            preview_len,
            mean_milli: (mean * 1000.0).round() as i32,
            mean_abs_milli: (mean_abs * 1000.0).round() as u32,
            rms_milli: (rms * 1000.0).round() as u32,
            checksum,
        }
    }
}

impl From<&AttentionContinuation> for TransientSignature {
    fn from(value: &AttentionContinuation) -> Self {
        let mut preview = Vec::new();
        preview.extend(value.q_preview.iter().copied());
        preview.extend(value.k_preview.iter().copied());
        preview.extend(value.v_preview.iter().copied());
        preview.extend(value.score_preview.iter().copied());
        preview.extend(value.value_preview.iter().copied());
        Self::from_preview(value.width, &preview)
    }
}

impl From<&FfnContinuation> for TransientSignature {
    fn from(value: &FfnContinuation) -> Self {
        let mut preview = Vec::new();
        preview.extend(value.gate_preview.iter().copied());
        preview.extend(value.up_preview.iter().copied());
        preview.extend(value.activation_preview.iter().copied());
        Self::from_preview(value.width, &preview)
    }
}

impl From<&StageTransientState> for StageTransferTransientState {
    fn from(value: &StageTransientState) -> Self {
        Self {
            attention: value.attention.as_ref().map(TransientSignature::from),
            ffn: value.ffn.as_ref().map(TransientSignature::from),
        }
    }
}

impl CarryableAttentionState {
    fn contract_for_parts(
        contract: StageResumeContractSummary,
        projection: Option<&CarryableAttentionProjectionState>,
        mix: Option<&CarryableAttentionMixState>,
    ) -> StageResumeContractSummary {
        Self {
            contract,
            projection: projection.cloned(),
            mix: mix.cloned(),
        }
        .contract_for_present_substates()
    }

    fn contract_for_present_substates(&self) -> StageResumeContractSummary {
        StageResumeContractSummary {
            attention_q: self
                .projection
                .as_ref()
                .filter(|projection| !projection.q_lane_indices.is_empty())
                .and(self.contract.attention_q),
            attention_k: self
                .projection
                .as_ref()
                .filter(|projection| !projection.k_lane_indices.is_empty())
                .and(self.contract.attention_k),
            attention_v: self
                .projection
                .as_ref()
                .filter(|projection| !projection.v_lane_indices.is_empty())
                .and(self.contract.attention_v),
            attention_score: self
                .mix
                .as_ref()
                .filter(|mix| !mix.score_lane_indices.is_empty())
                .and(self.contract.attention_score),
            attention_value: self
                .mix
                .as_ref()
                .filter(|mix| !mix.value_lane_indices.is_empty())
                .and(self.contract.attention_value),
        }
    }

    fn validate_contract_entry(
        name: &str,
        entry: Option<StageAttentionResumeContractSummary>,
        expected_phase: StageAttentionResumePhase,
    ) -> Result<()> {
        let Some(entry) = entry else {
            return Ok(());
        };
        if entry.phase != expected_phase {
            bail!(
                "attention carry {name} contract phase {:?} does not match required {:?}",
                entry.phase,
                expected_phase
            );
        }
        match entry.blend {
            StageAttentionResumeBlend::Overwrite => {
                if entry.blend_weight_milli.is_some() {
                    bail!("attention carry {name} overwrite contract must not carry blend weight");
                }
            }
            StageAttentionResumeBlend::WeakBlend => {
                let Some(weight_milli) = entry.blend_weight_milli else {
                    bail!("attention carry {name} weak blend contract is missing blend weight");
                };
                if StageResumeFreshnessPolicy::blend_strength_for_weight_milli(weight_milli)
                    != AttentionBlendStrength::WeakBlend
                {
                    bail!(
                        "attention carry {name} weak blend contract has non-weak weight {}",
                        weight_milli
                    );
                }
            }
            StageAttentionResumeBlend::StrongBlend => {
                let Some(weight_milli) = entry.blend_weight_milli else {
                    bail!("attention carry {name} strong blend contract is missing blend weight");
                };
                if StageResumeFreshnessPolicy::blend_strength_for_weight_milli(weight_milli)
                    != AttentionBlendStrength::StrongBlend
                {
                    bail!(
                        "attention carry {name} strong blend contract has non-strong weight {}",
                        weight_milli
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_contract_for_present_substates(&self) -> Result<()> {
        if self.contract.is_empty() {
            return Ok(());
        }
        let present_contract = self.contract_for_present_substates();
        if present_contract != self.contract {
            bail!("attention carry contract does not match carried substates");
        }
        Self::validate_contract_entry(
            "q",
            self.contract.attention_q,
            StageAttentionResumePhase::Direct,
        )?;
        Self::validate_contract_entry(
            "k",
            self.contract.attention_k,
            StageAttentionResumePhase::Direct,
        )?;
        Self::validate_contract_entry(
            "v",
            self.contract.attention_v,
            StageAttentionResumePhase::Direct,
        )?;
        Self::validate_contract_entry(
            "score",
            self.contract.attention_score,
            StageAttentionResumePhase::AfterProjection,
        )?;
        Self::validate_contract_entry(
            "value",
            self.contract.attention_value,
            StageAttentionResumePhase::AfterProjection,
        )?;
        Ok(())
    }

    fn blend_weight_for_q(&self) -> Option<f32> {
        self.contract
            .attention_q
            .and_then(|entry| entry.blend_weight_milli)
            .map(|weight| weight as f32 / 1000.0)
    }

    fn blend_weight_for_k(&self) -> Option<f32> {
        self.contract
            .attention_k
            .and_then(|entry| entry.blend_weight_milli)
            .map(|weight| weight as f32 / 1000.0)
    }

    fn blend_weight_for_v(&self) -> Option<f32> {
        self.contract
            .attention_v
            .and_then(|entry| entry.blend_weight_milli)
            .map(|weight| weight as f32 / 1000.0)
    }

    fn blend_weight_for_score(&self) -> Option<f32> {
        self.contract
            .attention_score
            .and_then(|entry| entry.blend_weight_milli)
            .map(|weight| weight as f32 / 1000.0)
    }

    fn blend_weight_for_value(&self) -> Option<f32> {
        self.contract
            .attention_value
            .and_then(|entry| entry.blend_weight_milli)
            .map(|weight| weight as f32 / 1000.0)
    }

    fn truncate_values(values: &[f32], lane_count: usize) -> Vec<f32> {
        values.iter().take(lane_count).copied().collect()
    }

    pub fn with_lane_budgets(
        &self,
        q_budget: Option<usize>,
        k_budget: Option<usize>,
        v_budget: Option<usize>,
        mix_budget: Option<usize>,
    ) -> Option<Self> {
        let projection = self.projection.as_ref().and_then(|projection| {
            let q_budget = q_budget.unwrap_or(projection.q_lane_indices.len());
            let k_budget = k_budget.unwrap_or(projection.k_lane_indices.len());
            let v_budget = v_budget.unwrap_or(projection.v_lane_indices.len());
            let q_lane_count = projection.q_lane_indices.len().min(q_budget);
            let k_lane_count = projection.k_lane_indices.len().min(k_budget);
            let v_lane_count = projection.v_lane_indices.len().min(v_budget);
            if q_lane_count == 0 && k_lane_count == 0 && v_lane_count == 0 {
                None
            } else {
                Some(CarryableAttentionProjectionState {
                    width: projection.width,
                    q_provenance: projection.q_provenance.clone(),
                    k_provenance: projection.k_provenance.clone(),
                    v_provenance: projection.v_provenance.clone(),
                    q_lane_indices: projection
                        .q_lane_indices
                        .iter()
                        .take(q_lane_count)
                        .copied()
                        .collect(),
                    k_lane_indices: projection
                        .k_lane_indices
                        .iter()
                        .take(k_lane_count)
                        .copied()
                        .collect(),
                    v_lane_indices: projection
                        .v_lane_indices
                        .iter()
                        .take(v_lane_count)
                        .copied()
                        .collect(),
                    q: Self::truncate_values(&projection.q, q_lane_count),
                    k: Self::truncate_values(&projection.k, k_lane_count),
                    v: Self::truncate_values(&projection.v, v_lane_count),
                })
            }
        });
        let mix = self.mix.as_ref().and_then(|mix| {
            let score_budget = mix_budget.unwrap_or(mix.score_lane_indices.len());
            let value_budget = mix_budget.unwrap_or(mix.value_lane_indices.len());
            let score_lane_count = mix.score_lane_indices.len().min(score_budget);
            let value_lane_count = mix.value_lane_indices.len().min(value_budget);
            if score_lane_count == 0 && value_lane_count == 0 {
                None
            } else {
                Some(CarryableAttentionMixState {
                    width: mix.width,
                    score_provenance: mix.score_provenance.clone(),
                    value_provenance: mix.value_provenance.clone(),
                    score_lane_indices: mix
                        .score_lane_indices
                        .iter()
                        .take(score_lane_count)
                        .copied()
                        .collect(),
                    value_lane_indices: mix
                        .value_lane_indices
                        .iter()
                        .take(value_lane_count)
                        .copied()
                        .collect(),
                    scores: Self::truncate_values(&mix.scores, score_lane_count),
                    values: Self::truncate_values(&mix.values, value_lane_count),
                })
            }
        });
        if projection.is_none() && mix.is_none() {
            None
        } else {
            Some(Self {
                contract: Self::contract_for_parts(
                    self.contract,
                    projection.as_ref(),
                    mix.as_ref(),
                ),
                projection,
                mix,
            })
        }
    }

    fn clamp_lanes(lane_indices: &[usize], vectors: &[&[f32]], budget: usize) -> Vec<usize> {
        let available = vectors
            .iter()
            .map(|values| values.len())
            .max()
            .unwrap_or(0)
            .min(lane_indices.len());
        lane_indices
            .iter()
            .take(available.min(budget))
            .copied()
            .collect()
    }

    fn pick_lanes(values: &[f32], lane_count: usize) -> Vec<f32> {
        values.iter().take(lane_count).copied().collect()
    }

    fn merged_lane_indices(&self) -> Vec<usize> {
        let mut lanes = BTreeSet::new();
        if let Some(projection) = &self.projection {
            lanes.extend(projection.q_lane_indices.iter().copied());
            lanes.extend(projection.k_lane_indices.iter().copied());
            lanes.extend(projection.v_lane_indices.iter().copied());
        }
        if let Some(mix) = &self.mix {
            lanes.extend(mix.score_lane_indices.iter().copied());
            lanes.extend(mix.value_lane_indices.iter().copied());
        }
        lanes.into_iter().collect()
    }

    fn align_to_lanes(lanes: &[usize], substate_lanes: &[usize], values: &[f32]) -> Vec<f32> {
        if lanes.is_empty() {
            return Vec::new();
        }
        let lane_map = substate_lanes
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, lane)| (lane, idx))
            .collect::<BTreeMap<_, _>>();
        let mut out = vec![0.0f32; lanes.len()];
        for (idx, lane) in lanes.iter().copied().enumerate() {
            if let Some(value_idx) = lane_map.get(&lane).copied() {
                if value_idx < values.len() {
                    out[idx] = values[value_idx];
                }
            }
        }
        out
    }

    fn from_attention(value: &AttentionContinuation) -> Self {
        let lane_indices = if value.lane_indices.is_empty() {
            (0..value
                .q_preview
                .len()
                .max(value.k_preview.len())
                .max(value.v_preview.len())
                .max(value.score_preview.len())
                .max(value.value_preview.len()))
                .collect()
        } else {
            value.lane_indices.clone()
        };
        Self {
            contract: StageResumeContractSummary::default(),
            projection: {
                let q_lanes = Self::clamp_lanes(
                    &lane_indices,
                    &[&value.q_preview],
                    ATTENTION_PROJECTION_CARRY_BUDGET,
                );
                let k_lanes = Self::clamp_lanes(
                    &lane_indices,
                    &[&value.k_preview],
                    ATTENTION_PROJECTION_CARRY_BUDGET,
                );
                let v_lanes = Self::clamp_lanes(
                    &lane_indices,
                    &[&value.v_preview],
                    ATTENTION_PROJECTION_CARRY_BUDGET,
                );
                if q_lanes.is_empty() && k_lanes.is_empty() && v_lanes.is_empty() {
                    None
                } else {
                    Some(CarryableAttentionProjectionState {
                        width: value.width,
                        q_provenance: None,
                        k_provenance: None,
                        v_provenance: None,
                        q_lane_indices: q_lanes.clone(),
                        k_lane_indices: k_lanes.clone(),
                        v_lane_indices: v_lanes.clone(),
                        q: Self::pick_lanes(&value.q_preview, q_lanes.len()),
                        k: Self::pick_lanes(&value.k_preview, k_lanes.len()),
                        v: Self::pick_lanes(&value.v_preview, v_lanes.len()),
                    })
                }
            },
            mix: {
                let score_lanes = Self::clamp_lanes(
                    &lane_indices,
                    &[&value.score_preview],
                    ATTENTION_MIX_CARRY_BUDGET,
                );
                let value_lanes = Self::clamp_lanes(
                    &lane_indices,
                    &[&value.value_preview],
                    ATTENTION_MIX_CARRY_BUDGET,
                );
                if score_lanes.is_empty() && value_lanes.is_empty() {
                    None
                } else {
                    Some(CarryableAttentionMixState {
                        width: value.width,
                        score_provenance: None,
                        value_provenance: None,
                        score_lane_indices: score_lanes.clone(),
                        value_lane_indices: value_lanes.clone(),
                        scores: Self::pick_lanes(&value.score_preview, score_lanes.len()),
                        values: Self::pick_lanes(&value.value_preview, value_lanes.len()),
                    })
                }
            },
        }
    }

    pub fn width(&self) -> usize {
        self.projection
            .as_ref()
            .map(|projection| projection.width)
            .or_else(|| self.mix.as_ref().map(|mix| mix.width))
            .unwrap_or(0)
    }

    pub fn lane_count(&self) -> usize {
        self.projection_lane_count().max(self.mix_lane_count())
    }

    pub fn projection_lane_count(&self) -> usize {
        self.projection
            .as_ref()
            .map(|projection| {
                projection
                    .q_lane_indices
                    .len()
                    .max(projection.k_lane_indices.len())
                    .max(projection.v_lane_indices.len())
            })
            .unwrap_or(0)
    }

    pub fn mix_lane_count(&self) -> usize {
        self.mix
            .as_ref()
            .map(|mix| {
                mix.score_lane_indices
                    .len()
                    .max(mix.value_lane_indices.len())
            })
            .unwrap_or(0)
    }

    fn to_attention_continuation(&self) -> Option<AttentionContinuation> {
        let lane_indices = self.merged_lane_indices();
        if lane_indices.is_empty() {
            return None;
        }
        let projection = self.projection.as_ref();
        let mix = self.mix.as_ref();
        Some(AttentionContinuation {
            width: self.width(),
            lane_indices: lane_indices.clone(),
            q_preview: projection
                .map(|projection| {
                    Self::align_to_lanes(&lane_indices, &projection.q_lane_indices, &projection.q)
                })
                .unwrap_or_else(|| vec![0.0; lane_indices.len()]),
            k_preview: projection
                .map(|projection| {
                    Self::align_to_lanes(&lane_indices, &projection.k_lane_indices, &projection.k)
                })
                .unwrap_or_else(|| vec![0.0; lane_indices.len()]),
            v_preview: projection
                .map(|projection| {
                    Self::align_to_lanes(&lane_indices, &projection.v_lane_indices, &projection.v)
                })
                .unwrap_or_else(|| vec![0.0; lane_indices.len()]),
            score_preview: mix
                .map(|mix| {
                    Self::align_to_lanes(&lane_indices, &mix.score_lane_indices, &mix.scores)
                })
                .unwrap_or_else(|| vec![0.0; lane_indices.len()]),
            value_preview: mix
                .map(|mix| {
                    Self::align_to_lanes(&lane_indices, &mix.value_lane_indices, &mix.values)
                })
                .unwrap_or_else(|| vec![0.0; lane_indices.len()]),
        })
    }
}

impl CarryableFfnState {
    pub fn with_lane_budget(&self, lane_budget: Option<usize>) -> Option<Self> {
        let budget = lane_budget.unwrap_or(self.lane_indices.len());
        let lane_count = self.lane_indices.len().min(budget);
        if lane_count == 0 {
            None
        } else {
            Some(Self {
                width: self.width,
                lane_indices: self.lane_indices.iter().take(lane_count).copied().collect(),
                gate_head: self.gate_head.iter().take(lane_count).copied().collect(),
                up_head: self.up_head.iter().take(lane_count).copied().collect(),
                activation_head: self
                    .activation_head
                    .iter()
                    .take(lane_count)
                    .copied()
                    .collect(),
            })
        }
    }

    fn from_ffn(value: &FfnContinuation) -> Self {
        let lane_indices = if value.lane_indices.is_empty() {
            (0..value
                .gate_preview
                .len()
                .max(value.up_preview.len())
                .max(value.activation_preview.len()))
                .collect()
        } else {
            value.lane_indices.clone()
        };
        Self {
            width: value.width,
            lane_indices,
            gate_head: value.gate_preview.clone(),
            up_head: value.up_preview.clone(),
            activation_head: value.activation_preview.clone(),
        }
    }

    pub fn lane_count(&self) -> usize {
        if self.lane_indices.is_empty() {
            self.gate_head
                .len()
                .max(self.up_head.len())
                .max(self.activation_head.len())
        } else {
            self.lane_indices.len()
        }
    }
}

impl StageCarryState {
    fn age_attention_by_layers(&self, layers: u32) -> Self {
        let attention = self
            .attention
            .as_ref()
            .map(|attention| CarryableAttentionState {
                contract: attention.contract,
                projection: attention.projection.as_ref().map(|projection| {
                    let mut aged = projection.clone();
                    aged.q_provenance = aged.q_provenance.as_ref().map(|p| p.age_by_layers(layers));
                    aged.k_provenance = aged.k_provenance.as_ref().map(|p| p.age_by_layers(layers));
                    aged.v_provenance = aged.v_provenance.as_ref().map(|p| p.age_by_layers(layers));
                    aged
                }),
                mix: attention.mix.as_ref().map(|mix| {
                    let mut aged = mix.clone();
                    aged.score_provenance = aged
                        .score_provenance
                        .as_ref()
                        .map(|p| p.age_by_layers(layers));
                    aged.value_provenance = aged
                        .value_provenance
                        .as_ref()
                        .map(|p| p.age_by_layers(layers));
                    aged
                }),
            });
        Self {
            attention,
            ffn: self.ffn.clone(),
        }
    }

    pub fn with_resume_contract(
        &self,
        capabilities: &StageResumeCapabilities,
        budgets: &StageResumeBudgets,
    ) -> Option<Self> {
        let attention = self.attention.as_ref().and_then(|attention| {
            if capabilities.attention_projection || capabilities.attention_mix {
                let narrowed = attention.with_lane_budgets(
                    budgets.attention_q_lanes,
                    budgets.attention_k_lanes,
                    budgets.attention_v_lanes,
                    budgets.attention_mix_lanes,
                )?;
                let projection = narrowed.projection?;
                let mix = narrowed.mix.filter(|_| capabilities.attention_mix);
                Some(CarryableAttentionState {
                    contract: CarryableAttentionState::contract_for_parts(
                        narrowed.contract,
                        Some(&projection),
                        mix.as_ref(),
                    ),
                    projection: Some(projection),
                    mix,
                })
            } else {
                None
            }
        });
        let ffn = self.ffn.as_ref().and_then(|ffn| {
            if capabilities.ffn_path {
                ffn.with_lane_budget(budgets.ffn_lanes)
            } else {
                None
            }
        });
        if attention.is_none() && ffn.is_none() {
            None
        } else {
            Some(Self { attention, ffn })
        }
    }

    pub fn with_resume_capabilities(&self, capabilities: &StageResumeCapabilities) -> Option<Self> {
        let attention = self.attention.as_ref().and_then(|attention| {
            let projection = if capabilities.attention_projection {
                attention.projection.clone()
            } else {
                None
            };
            let mix = if capabilities.attention_mix {
                attention.mix.clone()
            } else {
                None
            };
            if projection.is_none() {
                return None;
            }
            Some(CarryableAttentionState {
                contract: CarryableAttentionState::contract_for_parts(
                    attention.contract,
                    projection.as_ref(),
                    mix.as_ref(),
                ),
                projection,
                mix,
            })
        });
        let ffn = if capabilities.ffn_path {
            self.ffn.clone()
        } else {
            None
        };
        if attention.is_none() && ffn.is_none() {
            None
        } else {
            Some(Self { attention, ffn })
        }
    }

    pub fn with_policy(&self, policy: &StageCarryPolicy) -> Self {
        Self {
            attention: if policy.carry_attention {
                self.attention.clone()
            } else {
                None
            },
            ffn: if policy.carry_ffn {
                self.ffn.clone()
            } else {
                None
            },
        }
    }

    pub fn from_transient(value: &StageTransientState, policy: &StageCarryPolicy) -> Self {
        Self {
            attention: if policy.carry_attention {
                value
                    .attention
                    .as_ref()
                    .map(CarryableAttentionState::from_attention)
            } else {
                None
            },
            ffn: if policy.carry_ffn {
                value.ffn.as_ref().map(CarryableFfnState::from_ffn)
            } else {
                None
            },
        }
    }

    pub fn to_transient_state(&self) -> Option<StageTransientState> {
        let attention = self
            .attention
            .as_ref()
            .and_then(CarryableAttentionState::to_attention_continuation);
        let ffn = self.ffn.as_ref().map(|ffn| FfnContinuation {
            width: ffn.width,
            lane_indices: ffn.lane_indices.clone(),
            gate_preview: ffn.gate_head.clone(),
            up_preview: ffn.up_head.clone(),
            activation_preview: ffn.activation_head.clone(),
        });
        if attention.is_none() && ffn.is_none() {
            None
        } else {
            Some(StageTransientState { attention, ffn })
        }
    }
}

impl StageTransferFrame {
    pub fn validate(&self) -> Result<()> {
        if self.version != STAGE_FORWARD_FRAME_VERSION {
            bail!("Unsupported stage transfer frame version {}", self.version);
        }
        if self.route.source_stage_start > self.route.source_stage_end {
            bail!(
                "Invalid transfer stage range {}..{}",
                self.route.source_stage_start,
                self.route.source_stage_end
            );
        }
        if self.payload.kind == PayloadKind::HiddenState
            && self.payload.bytes.len() != self.payload.hidden_dim * 4
        {
            bail!(
                "Hidden-state transfer frame bytes {} do not match hidden_dim {}",
                self.payload.bytes.len(),
                self.payload.hidden_dim
            );
        }
        if !self.payload.stage_trace.is_empty()
            && self.payload.stage_trace.last() != Some(&self.route.source_stage_id)
        {
            bail!(
                "Transfer frame source stage {} does not match payload trace tail {:?}",
                self.route.source_stage_id,
                self.payload.stage_trace.last()
            );
        }
        if let Some(attention) = self
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.attention.as_ref())
        {
            if attention.width() == 0 {
                bail!("Attention carry width must be non-zero");
            }
            if attention.lane_count() == 0 {
                bail!("Attention carry must include at least one lane");
            }
            if attention.lane_count() > attention.width() {
                bail!(
                    "Attention carry lane count {} exceeds width {}",
                    attention.lane_count(),
                    attention.width()
                );
            }
            if let Some(projection) = &attention.projection {
                if projection.width != attention.width() {
                    bail!("Attention projection carry width diverged from attention carry width");
                }
                for provenance in [
                    projection.q_provenance.as_ref(),
                    projection.k_provenance.as_ref(),
                    projection.v_provenance.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if provenance.layer_distance_to_boundary > provenance.layer_index {
                        bail!("Attention projection carry provenance has invalid layer distance");
                    }
                }
                if projection.q.len() != projection.q_lane_indices.len()
                    || projection.k.len() != projection.k_lane_indices.len()
                    || projection.v.len() != projection.v_lane_indices.len()
                {
                    bail!("Attention projection carry vectors must match projection lane count");
                }
                if projection
                    .q_lane_indices
                    .iter()
                    .any(|lane| *lane >= projection.width)
                {
                    bail!("Attention q carry lane index exceeds width");
                }
                if projection
                    .k_lane_indices
                    .iter()
                    .any(|lane| *lane >= projection.width)
                {
                    bail!("Attention k carry lane index exceeds width");
                }
                if projection
                    .v_lane_indices
                    .iter()
                    .any(|lane| *lane >= projection.width)
                {
                    bail!("Attention projection carry lane index exceeds width");
                }
            }
            if let Some(mix) = &attention.mix {
                if mix.width != attention.width() {
                    bail!("Attention mix carry width diverged from attention carry width");
                }
                for provenance in [mix.score_provenance.as_ref(), mix.value_provenance.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    if provenance.layer_distance_to_boundary > provenance.layer_index {
                        bail!("Attention mix carry provenance has invalid layer distance");
                    }
                }
                if mix.scores.len() != mix.score_lane_indices.len()
                    || mix.values.len() != mix.value_lane_indices.len()
                {
                    bail!("Attention mix carry vectors must match mix lane count");
                }
                if mix.score_lane_indices.iter().any(|lane| *lane >= mix.width) {
                    bail!("Attention score carry lane index exceeds width");
                }
                if mix.value_lane_indices.iter().any(|lane| *lane >= mix.width) {
                    bail!("Attention value carry lane index exceeds width");
                }
            }
        }
        if let Some(ffn) = self
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.ffn.as_ref())
        {
            if ffn.width == 0 {
                bail!("FFN carry width must be non-zero");
            }
            if ffn.lane_count() == 0 {
                bail!("FFN carry must include at least one lane");
            }
            if ffn.lane_count() > ffn.width {
                bail!(
                    "FFN carry lane count {} exceeds width {}",
                    ffn.lane_count(),
                    ffn.width
                );
            }
            if ffn.gate_head.len() != ffn.lane_count()
                || ffn.up_head.len() != ffn.lane_count()
                || ffn.activation_head.len() != ffn.lane_count()
            {
                bail!("FFN carry vectors must match lane count");
            }
            if ffn.lane_indices.iter().any(|lane| *lane >= ffn.width) {
                bail!("FFN carry lane index exceeds width");
            }
        }
        Ok(())
    }

    pub fn into_stage_tensor(self) -> StageTensor {
        let mut tensor = StageTensor {
            request_id: self.payload.request_id,
            kind: self.payload.kind,
            stage_trace: self.payload.stage_trace,
            hidden_dim: self.payload.hidden_dim,
            bytes: self.payload.bytes,
            prompt_text: self.payload.prompt_text,
            max_tokens: self.payload.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        };
        tensor.continuation = self.state.continuation;
        tensor.carry = self.state.carry;
        tensor.transient = tensor
            .carry
            .as_ref()
            .and_then(StageCarryState::to_transient_state);
        tensor
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedRuntimeStage {
    pub stage_index: u32,
    pub start_layer: u32,
    pub end_layer: u32,
    pub role: String,
    pub required: BTreeSet<String>,
    pub optional: BTreeSet<String>,
    pub required_bytes: u64,
    pub optional_bytes: u64,
    pub required_slices: Vec<gguf::TensorSlice>,
    pub optional_slices: Vec<gguf::TensorSlice>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedRuntimeBundle {
    pub model_name: String,
    pub architecture: String,
    pub stages: Vec<LoadedRuntimeStage>,
}

#[derive(Debug, Clone)]
pub struct StageResidencyAdapter {
    pub bundle: LoadedRuntimeBundle,
    pub gguf_path: PathBuf,
    pub gguf: gguf::GgufFile,
    pub stage: LoadedRuntimeStage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedTensorEntry {
    pub name: String,
    pub pack_offset: u64,
    pub byte_len: u64,
    pub source_file_offset: u64,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedStageIndex {
    pub model_name: String,
    pub architecture: String,
    pub stage_index: u32,
    pub role: String,
    pub total_bytes: u64,
    pub tensor_count: usize,
    pub tensors: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedStageArtifact {
    pub pack_path: PathBuf,
    pub index_path: PathBuf,
    pub index: PackedStageIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTensorStore {
    pub artifact: PackedStageArtifact,
    entries: BTreeMap<String, PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageLayerView {
    pub layer_index: u32,
    pub tensors: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LayerOperatorView {
    pub layer_index: u32,
    pub attn_q: Option<PackedTensorEntry>,
    pub attn_k: Option<PackedTensorEntry>,
    pub attn_v: Option<PackedTensorEntry>,
    pub attn_output: Option<PackedTensorEntry>,
    pub attn_norm: Option<PackedTensorEntry>,
    pub attn_q_norm: Option<PackedTensorEntry>,
    pub attn_k_norm: Option<PackedTensorEntry>,
    pub ffn_up: Option<PackedTensorEntry>,
    pub ffn_down: Option<PackedTensorEntry>,
    pub ffn_gate: Option<PackedTensorEntry>,
    pub ffn_norm: Option<PackedTensorEntry>,
    pub proj: Option<PackedTensorEntry>,
    pub inp_gate: Option<PackedTensorEntry>,
    pub post_attention_norm: Option<PackedTensorEntry>,
    pub post_ffw_norm: Option<PackedTensorEntry>,
    pub post_norm: Option<PackedTensorEntry>,
    pub layer_output_scale: Option<PackedTensorEntry>,
    pub unknown: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecutionSpec {
    pub layer_index: u32,
    pub hidden_dim: Option<u64>,
    pub q_out_dim: Option<u64>,
    pub k_out_dim: Option<u64>,
    pub v_out_dim: Option<u64>,
    pub ffn_inner_dim: Option<u64>,
    pub has_attention_core: bool,
    pub has_attention_norms: bool,
    pub has_ffn_core: bool,
    pub has_post_norms: bool,
    pub has_projection_path: bool,
    pub unknown_tensor_count: usize,
    pub runnable_sketch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOpKind {
    PromptIngress,
    Positional,
    SharedAuxiliary,
    AttentionNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOut,
    PostAttentionNorm,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    PostFfnNorm,
    InputGate,
    Projection,
    LayerOutputScale,
    TailOnly,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionBinding {
    F32Vector,
    F32Matrix,
    QuantizedMatrix,
    Mixed,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionOp {
    pub kind: ExecutionOpKind,
    pub tensor_names: Vec<String>,
    pub binding: ExecutionBinding,
    pub binding_reason: &'static str,
    resume_contract: ExecutionResumeContract,
}

impl ExecutionOp {
    fn blend_mode(weight: f32) -> ExecutionResumeBlendMode {
        ExecutionResumeBlendMode::Blend {
            weight_milli: (weight * 1000.0).round() as u16,
        }
    }

    fn resume_contract_for_kind(kind: ExecutionOpKind) -> ExecutionResumeContract {
        match kind {
            ExecutionOpKind::AttentionQ => ExecutionResumeContract {
                attention: vec![ExecutionAttentionResumeContract {
                    slice: AttentionResumeSlice::Q,
                    recompute_phase: AttentionSliceRecomputePhase::Direct,
                    blend_mode: Self::blend_mode(ATTENTION_PROJECTION_BLEND_WEIGHT),
                }],
            },
            ExecutionOpKind::AttentionK => ExecutionResumeContract {
                attention: vec![ExecutionAttentionResumeContract {
                    slice: AttentionResumeSlice::K,
                    recompute_phase: AttentionSliceRecomputePhase::Direct,
                    blend_mode: Self::blend_mode(ATTENTION_PROJECTION_BLEND_WEIGHT),
                }],
            },
            ExecutionOpKind::AttentionV => ExecutionResumeContract {
                attention: vec![ExecutionAttentionResumeContract {
                    slice: AttentionResumeSlice::V,
                    recompute_phase: AttentionSliceRecomputePhase::Direct,
                    blend_mode: Self::blend_mode(ATTENTION_PROJECTION_BLEND_WEIGHT),
                }],
            },
            ExecutionOpKind::AttentionOut => ExecutionResumeContract {
                attention: vec![
                    ExecutionAttentionResumeContract {
                        slice: AttentionResumeSlice::Score,
                        recompute_phase: AttentionSliceRecomputePhase::AfterProjection,
                        blend_mode: Self::blend_mode(ATTENTION_MIX_BLEND_WEIGHT),
                    },
                    ExecutionAttentionResumeContract {
                        slice: AttentionResumeSlice::Value,
                        recompute_phase: AttentionSliceRecomputePhase::AfterProjection,
                        blend_mode: Self::blend_mode(ATTENTION_MIX_BLEND_WEIGHT),
                    },
                ],
            },
            _ => ExecutionResumeContract::default(),
        }
    }

    fn descriptor_for_attention_contract(
        contract: ExecutionAttentionResumeContract,
    ) -> ExecutionResumeDescriptor {
        let path = AttentionSliceResumePath {
            recompute_phase: contract.recompute_phase,
            blend_strength: match contract.blend_mode {
                ExecutionResumeBlendMode::Overwrite => None,
                ExecutionResumeBlendMode::Blend { weight_milli } => Some(
                    StageResumeFreshnessPolicy::blend_strength_for_weight_milli(weight_milli),
                ),
            },
            blend_weight_milli: match contract.blend_mode {
                ExecutionResumeBlendMode::Overwrite => None,
                ExecutionResumeBlendMode::Blend { weight_milli } => Some(weight_milli),
            },
        };
        match contract.slice {
            AttentionResumeSlice::Q => ExecutionResumeDescriptor::AttentionQ(path),
            AttentionResumeSlice::K => ExecutionResumeDescriptor::AttentionK(path),
            AttentionResumeSlice::V => ExecutionResumeDescriptor::AttentionV(path),
            AttentionResumeSlice::Score => ExecutionResumeDescriptor::AttentionScore(path),
            AttentionResumeSlice::Value => ExecutionResumeDescriptor::AttentionValue(path),
        }
    }

    fn descriptors_for_contract(
        contract: &ExecutionResumeContract,
    ) -> Vec<ExecutionResumeDescriptor> {
        contract
            .attention
            .iter()
            .copied()
            .map(Self::descriptor_for_attention_contract)
            .collect()
    }

    #[cfg(test)]
    fn with_resume_contract(mut self, resume_contract: ExecutionResumeContract) -> Self {
        self.resume_contract = resume_contract;
        self
    }

    fn new(
        kind: ExecutionOpKind,
        tensor_names: Vec<String>,
        binding: ExecutionBinding,
        binding_reason: &'static str,
    ) -> Self {
        let resume_contract = Self::resume_contract_for_kind(kind.clone());
        Self {
            kind,
            tensor_names,
            binding,
            binding_reason,
            resume_contract,
        }
    }

    fn attention_resume_descriptors(&self) -> Vec<ExecutionResumeDescriptor> {
        Self::descriptors_for_contract(&self.resume_contract)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecutionProgram {
    pub layer_index: u32,
    pub hidden_dim: Option<u64>,
    pub q_out_dim: Option<u64>,
    pub k_out_dim: Option<u64>,
    pub v_out_dim: Option<u64>,
    pub ffn_inner_dim: Option<u64>,
    pub runnable_sketch: bool,
    pub ops: Vec<ExecutionOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageModelView {
    pub role: String,
    pub prompt_ingress: Vec<PackedTensorEntry>,
    pub positional: Vec<PackedTensorEntry>,
    pub shared_auxiliary: Vec<PackedTensorEntry>,
    pub layers: Vec<StageLayerView>,
    pub operator_layers: Vec<LayerOperatorView>,
    pub execution_layers: Vec<LayerExecutionSpec>,
    pub execution_programs: Vec<LayerExecutionProgram>,
    pub tail_only: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageResumeCapabilities {
    pub attention_path: bool,
    pub attention_q: bool,
    pub attention_k: bool,
    pub attention_v: bool,
    pub attention_projection: bool,
    pub attention_mix: bool,
    pub ffn_path: bool,
    pub projection_path: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageResumeBudgets {
    pub attention_q_lanes: Option<usize>,
    pub attention_k_lanes: Option<usize>,
    pub attention_v_lanes: Option<usize>,
    pub attention_projection_lanes: Option<usize>,
    pub attention_mix_lanes: Option<usize>,
    pub ffn_lanes: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageResumeFreshnessPolicy {
    pub attention_q_max_distance: Option<u32>,
    pub attention_k_max_distance: Option<u32>,
    pub attention_v_max_distance: Option<u32>,
    pub attention_score_max_distance: Option<u32>,
    pub attention_value_max_distance: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageAttentionResumePhase {
    Direct,
    AfterProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageAttentionResumeBlend {
    Overwrite,
    WeakBlend,
    StrongBlend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageAttentionResumeContractSummary {
    pub phase: StageAttentionResumePhase,
    pub blend: StageAttentionResumeBlend,
    pub blend_weight_milli: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StageResumeContractSummary {
    pub attention_q: Option<StageAttentionResumeContractSummary>,
    pub attention_k: Option<StageAttentionResumeContractSummary>,
    pub attention_v: Option<StageAttentionResumeContractSummary>,
    pub attention_score: Option<StageAttentionResumeContractSummary>,
    pub attention_value: Option<StageAttentionResumeContractSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionBlendStrength {
    WeakBlend,
    StrongBlend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionSliceRecomputePhase {
    Direct,
    AfterProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AttentionSliceResumePath {
    recompute_phase: AttentionSliceRecomputePhase,
    blend_strength: Option<AttentionBlendStrength>,
    blend_weight_milli: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionResumeSlice {
    Q,
    K,
    V,
    Score,
    Value,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionResumeBlendMode {
    Overwrite,
    Blend { weight_milli: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecutionAttentionResumeContract {
    slice: AttentionResumeSlice,
    recompute_phase: AttentionSliceRecomputePhase,
    blend_mode: ExecutionResumeBlendMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ExecutionResumeContract {
    attention: Vec<ExecutionAttentionResumeContract>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionResumeDescriptor {
    AttentionQ(AttentionSliceResumePath),
    AttentionK(AttentionSliceResumePath),
    AttentionV(AttentionSliceResumePath),
    AttentionScore(AttentionSliceResumePath),
    AttentionValue(AttentionSliceResumePath),
}

impl LoadedRuntimeBundle {
    pub fn load(root: &Path) -> Result<Self> {
        let manifest_path = root.join("model-shard-manifest.json");
        let manifest: gguf::ModelShardManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).map_err(|err| {
                anyhow::anyhow!("Failed to read {}: {err}", manifest_path.display())
            })?)?;

        let mut stages = Vec::with_capacity(manifest.runtime_plan.len());
        for runtime in &manifest.runtime_plan {
            let stage_manifest_path = root.join(format!("stage-{}.json", runtime.stage_index + 1));
            let runtime_path = root.join(format!("runtime-stage-{}.json", runtime.stage_index + 1));
            let stage_manifest: gguf::StageManifest = serde_json::from_str(
                &fs::read_to_string(&stage_manifest_path).map_err(|err| {
                    anyhow::anyhow!("Failed to read {}: {err}", stage_manifest_path.display())
                })?,
            )?;
            let file_runtime: gguf::StageRuntimePlan =
                serde_json::from_str(&fs::read_to_string(&runtime_path).map_err(|err| {
                    anyhow::anyhow!("Failed to read {}: {err}", runtime_path.display())
                })?)?;
            if stage_manifest.stage_index != runtime.stage_index {
                bail!(
                    "Stage manifest {} does not match runtime stage index {}",
                    stage_manifest_path.display(),
                    runtime.stage_index + 1
                );
            }
            if &file_runtime != runtime {
                bail!(
                    "Runtime plan file {} does not match top-level manifest for stage {}",
                    runtime_path.display(),
                    runtime.stage_index + 1
                );
            }

            let required_path = root.join(format!(
                "runtime-stage-{}-required.txt",
                runtime.stage_index + 1
            ));
            let optional_path = root.join(format!(
                "runtime-stage-{}-optional.txt",
                runtime.stage_index + 1
            ));
            let slices_path = root.join(format!(
                "runtime-stage-{}-slices.json",
                runtime.stage_index + 1
            ));

            let required = read_tensor_set(&required_path)?;
            let optional = read_tensor_set(&optional_path)?;
            let slices: gguf::StageRuntimeSliceManifest =
                serde_json::from_str(&fs::read_to_string(&slices_path).map_err(|err| {
                    anyhow::anyhow!("Failed to read {}: {err}", slices_path.display())
                })?)?;

            let expected_required: BTreeSet<_> =
                runtime.required.tensor_names.iter().cloned().collect();
            let expected_optional: BTreeSet<_> =
                runtime.optional.tensor_names.iter().cloned().collect();
            let slice_required: BTreeSet<_> = slices
                .required
                .iter()
                .map(|slice| slice.name.clone())
                .collect();
            let slice_optional: BTreeSet<_> = slices
                .optional
                .iter()
                .map(|slice| slice.name.clone())
                .collect();

            if required != expected_required {
                bail!(
                    "Required runtime tensor list {} does not match manifest for stage {}",
                    required_path.display(),
                    runtime.stage_index + 1
                );
            }
            if optional != expected_optional {
                bail!(
                    "Optional runtime tensor list {} does not match manifest for stage {}",
                    optional_path.display(),
                    runtime.stage_index + 1
                );
            }
            if slice_required != expected_required || slice_optional != expected_optional {
                bail!(
                    "Slice manifest {} does not match runtime tensor lists for stage {}",
                    slices_path.display(),
                    runtime.stage_index + 1
                );
            }

            stages.push(LoadedRuntimeStage {
                stage_index: runtime.stage_index,
                start_layer: stage_manifest.start_layer,
                end_layer: stage_manifest.end_layer,
                role: runtime.role.clone(),
                required,
                optional,
                required_bytes: runtime.required.total_bytes,
                optional_bytes: runtime.optional.total_bytes,
                required_slices: slices.required,
                optional_slices: slices.optional,
            });
        }

        Ok(Self {
            model_name: manifest.model_name,
            architecture: manifest.architecture,
            stages,
        })
    }

    pub fn validate_against_gguf(&self, file: &gguf::GgufFile) -> Result<()> {
        let known: BTreeSet<_> = file
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        for stage in &self.stages {
            for name in stage.required.iter().chain(stage.optional.iter()) {
                if !known.contains(name.as_str()) {
                    bail!(
                        "Stage {} references tensor {} which is not present in GGUF",
                        stage.stage_index + 1,
                        name
                    );
                }
            }
        }
        Ok(())
    }

    pub fn stage(&self, stage_index: u32) -> Option<&LoadedRuntimeStage> {
        self.stages
            .iter()
            .find(|stage| stage.stage_index == stage_index)
    }
}

impl StageResidencyAdapter {
    pub fn load(bundle_root: &Path, gguf_path: &Path, stage_index: u32) -> Result<Self> {
        let bundle = LoadedRuntimeBundle::load(bundle_root)?;
        let gguf = gguf::GgufFile::parse_file(gguf_path)?;
        Self::from_parts(bundle, gguf_path.to_path_buf(), gguf, stage_index)
    }

    pub fn from_parts(
        bundle: LoadedRuntimeBundle,
        gguf_path: PathBuf,
        gguf: gguf::GgufFile,
        stage_index: u32,
    ) -> Result<Self> {
        bundle.validate_against_gguf(&gguf)?;
        let stage = bundle.stage(stage_index).cloned().ok_or_else(|| {
            anyhow::anyhow!("Stage {} not found in runtime bundle", stage_index + 1)
        })?;
        Ok(Self {
            bundle,
            gguf_path,
            gguf,
            stage,
        })
    }

    pub fn stage_index(&self) -> u32 {
        self.stage.stage_index
    }

    pub fn start_layer(&self) -> u32 {
        self.stage.start_layer
    }

    pub fn end_layer(&self) -> u32 {
        self.stage.end_layer
    }

    pub fn stage_role(&self) -> &str {
        &self.stage.role
    }

    pub fn default_required_pack_dir_name(&self) -> String {
        format!("packed-stage-{}-{}", self.start_layer(), self.end_layer())
    }

    pub fn default_materialized_dir_name(&self) -> String {
        format!(
            "materialized-stage-{}-{}",
            self.start_layer(),
            self.end_layer()
        )
    }

    pub fn read_required_tensor(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let slice = self
            .stage
            .required_slices
            .iter()
            .find(|slice| slice.name == tensor_name)
            .ok_or_else(|| {
                anyhow::anyhow!("Tensor {} is not required for this stage", tensor_name)
            })?;
        self.read_slice(slice)
    }

    pub fn read_optional_tensor(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let slice = self
            .stage
            .optional_slices
            .iter()
            .find(|slice| slice.name == tensor_name)
            .ok_or_else(|| {
                anyhow::anyhow!("Tensor {} is not optional for this stage", tensor_name)
            })?;
        self.read_slice(slice)
    }

    pub fn materialize_required_tensors(&self, out_dir: &Path) -> Result<Vec<PathBuf>> {
        fs::create_dir_all(out_dir)?;
        let mut written = Vec::with_capacity(self.stage.required_slices.len());
        for slice in &self.stage.required_slices {
            let bytes = self.read_slice(slice)?;
            let path = out_dir.join(sanitize_tensor_filename(&slice.name));
            fs::write(&path, bytes)?;
            written.push(path);
        }
        Ok(written)
    }

    pub fn pack_required_tensors(&self, out_dir: &Path) -> Result<PackedStageArtifact> {
        fs::create_dir_all(out_dir)?;
        let pack_path = out_dir.join(format!(
            "stage-{}-required.pack",
            self.stage.stage_index + 1
        ));
        let index_path = out_dir.join(format!(
            "stage-{}-required.index.json",
            self.stage.stage_index + 1
        ));
        let mut pack_file = fs::File::create(&pack_path)
            .map_err(|err| anyhow::anyhow!("Failed to create {}: {err}", pack_path.display()))?;

        let mut pack_offset = 0u64;
        let mut tensors = Vec::with_capacity(self.stage.required_slices.len());
        for slice in &self.stage.required_slices {
            let bytes = self.read_slice(slice)?;
            pack_file.write_all(&bytes)?;
            tensors.push(PackedTensorEntry {
                name: slice.name.clone(),
                pack_offset,
                byte_len: slice.byte_len,
                source_file_offset: slice.file_offset,
                dimensions: slice.dimensions.clone(),
                ggml_type: slice.ggml_type,
            });
            pack_offset += slice.byte_len;
        }
        pack_file.flush()?;

        let index = PackedStageIndex {
            model_name: self.bundle.model_name.clone(),
            architecture: self.bundle.architecture.clone(),
            stage_index: self.stage.stage_index,
            role: self.stage.role.clone(),
            total_bytes: pack_offset,
            tensor_count: tensors.len(),
            tensors,
        };
        fs::write(&index_path, serde_json::to_vec_pretty(&index)?)?;

        Ok(PackedStageArtifact {
            pack_path,
            index_path,
            index,
        })
    }

    pub fn pack_all_tensors(&self, out_dir: &Path) -> Result<PackedStageArtifact> {
        fs::create_dir_all(out_dir)?;
        let pack_path = out_dir.join(format!(
            "stage-{}-required.pack",
            self.stage.stage_index + 1
        ));
        let index_path = out_dir.join(format!(
            "stage-{}-required.index.json",
            self.stage.stage_index + 1
        ));
        let mut pack_file = fs::File::create(&pack_path)
            .map_err(|err| anyhow::anyhow!("Failed to create {}: {err}", pack_path.display()))?;

        let all_slices: Vec<_> = self
            .stage
            .required_slices
            .iter()
            .chain(self.stage.optional_slices.iter())
            .collect();

        let mut pack_offset = 0u64;
        let mut tensors = Vec::with_capacity(all_slices.len());
        for slice in &all_slices {
            let bytes = self.read_slice(slice)?;
            pack_file.write_all(&bytes)?;
            tensors.push(PackedTensorEntry {
                name: slice.name.clone(),
                pack_offset,
                byte_len: slice.byte_len,
                source_file_offset: slice.file_offset,
                dimensions: slice.dimensions.clone(),
                ggml_type: slice.ggml_type,
            });
            pack_offset += slice.byte_len;
        }
        pack_file.flush()?;

        let index = PackedStageIndex {
            model_name: self.bundle.model_name.clone(),
            architecture: self.bundle.architecture.clone(),
            stage_index: self.stage.stage_index,
            role: self.stage.role.clone(),
            total_bytes: pack_offset,
            tensor_count: tensors.len(),
            tensors,
        };
        fs::write(&index_path, serde_json::to_vec_pretty(&index)?)?;

        Ok(PackedStageArtifact {
            pack_path,
            index_path,
            index,
        })
    }

    fn read_slice(&self, slice: &gguf::TensorSlice) -> Result<Vec<u8>> {
        let mut file = fs::File::open(&self.gguf_path)
            .map_err(|err| anyhow::anyhow!("Failed to open {}: {err}", self.gguf_path.display()))?;
        file.seek(SeekFrom::Start(slice.file_offset))?;
        let mut buf = vec![0u8; slice.byte_len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

fn read_tensor_set(path: &Path) -> Result<BTreeSet<String>> {
    Ok(fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("Failed to read {}: {err}", path.display()))?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn sanitize_tensor_filename(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

impl PackedStageArtifact {
    pub fn load(index_path: &Path) -> Result<Self> {
        let index: PackedStageIndex =
            serde_json::from_str(&fs::read_to_string(index_path).map_err(|err| {
                anyhow::anyhow!("Failed to read {}: {err}", index_path.display())
            })?)?;
        let file_name = index_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid packed stage index filename"))?;
        let pack_name = file_name
            .strip_suffix(".index.json")
            .map(|stem| format!("{stem}.pack"))
            .ok_or_else(|| anyhow::anyhow!("Packed stage index must end with .index.json"))?;
        let pack_path = index_path.with_file_name(pack_name);
        if !pack_path.exists() {
            bail!("Packed stage file {} does not exist", pack_path.display());
        }
        let size = fs::metadata(&pack_path)?.len();
        if size != index.total_bytes {
            bail!(
                "Packed stage size {} does not match index total_bytes {}",
                size,
                index.total_bytes
            );
        }
        Ok(Self {
            pack_path,
            index_path: index_path.to_path_buf(),
            index,
        })
    }

    pub fn read_tensor(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let entry = self
            .index
            .tensors
            .iter()
            .find(|entry| entry.name == tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in packed stage", tensor_name))?;
        let mut file = fs::File::open(&self.pack_path)?;
        file.seek(SeekFrom::Start(entry.pack_offset))?;
        let mut buf = vec![0u8; entry.byte_len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

impl StageTensorStore {
    pub fn load(index_path: &Path) -> Result<Self> {
        let artifact = PackedStageArtifact::load(index_path)?;
        let entries = artifact
            .index
            .tensors
            .iter()
            .cloned()
            .map(|entry| (entry.name.clone(), entry))
            .collect();
        Ok(Self { artifact, entries })
    }

    pub fn contains(&self, tensor_name: &str) -> bool {
        self.entries.contains_key(tensor_name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn entry(&self, tensor_name: &str) -> Option<&PackedTensorEntry> {
        self.entries.get(tensor_name)
    }

    pub fn read(&self, tensor_name: &str) -> Result<Vec<u8>> {
        self.artifact.read_tensor(tensor_name)
    }

    pub fn total_bytes(&self) -> u64 {
        self.artifact.index.total_bytes
    }

    pub fn tensor_count(&self) -> usize {
        self.artifact.index.tensor_count
    }

    pub fn validate_offsets(&self) -> Result<()> {
        let mut expected_offset = 0u64;
        for entry in &self.artifact.index.tensors {
            if entry.pack_offset != expected_offset {
                bail!(
                    "Packed tensor {} has pack_offset {} but expected {}",
                    entry.name,
                    entry.pack_offset,
                    expected_offset
                );
            }
            expected_offset = expected_offset.saturating_add(entry.byte_len);
        }
        if expected_offset != self.artifact.index.total_bytes {
            bail!(
                "Packed tensor coverage {} does not match total_bytes {}",
                expected_offset,
                self.artifact.index.total_bytes
            );
        }
        Ok(())
    }

    pub fn model_view(&self) -> StageModelView {
        let mut prompt_ingress = Vec::new();
        let mut positional = Vec::new();
        let mut shared_auxiliary = Vec::new();
        let mut tail_only = Vec::new();
        let mut layers: BTreeMap<u32, Vec<PackedTensorEntry>> = BTreeMap::new();

        for entry in self.artifact.index.tensors.iter().cloned() {
            match classify_packed_tensor(&entry.name) {
                PackedTensorClass::Layer(layer_index) => {
                    layers.entry(layer_index).or_default().push(entry);
                }
                PackedTensorClass::PromptIngress => prompt_ingress.push(entry),
                PackedTensorClass::Positional => positional.push(entry),
                PackedTensorClass::SharedAuxiliary => shared_auxiliary.push(entry),
                PackedTensorClass::TailOnly => tail_only.push(entry),
                PackedTensorClass::Unknown => shared_auxiliary.push(entry),
            }
        }

        let layers: Vec<StageLayerView> = layers
            .into_iter()
            .map(|(layer_index, tensors)| StageLayerView {
                layer_index,
                tensors,
            })
            .collect();
        let operator_layers = layers
            .iter()
            .map(|layer| build_operator_view(layer))
            .collect::<Vec<_>>();
        let execution_layers = operator_layers
            .iter()
            .map(build_execution_spec)
            .collect::<Vec<_>>();
        let execution_programs = operator_layers
            .iter()
            .zip(execution_layers.iter())
            .map(|(layer, spec)| build_execution_program(layer, spec))
            .collect();

        StageModelView {
            role: self.artifact.index.role.clone(),
            prompt_ingress,
            positional,
            shared_auxiliary,
            layers,
            operator_layers,
            execution_layers,
            execution_programs,
            tail_only,
        }
    }
}

impl StageResumeCapabilities {
    pub fn from_execution_programs(
        execution_programs: &[LayerExecutionProgram],
        resume_entry_layer: Option<u32>,
    ) -> Self {
        let entry_program = resume_entry_layer
            .and_then(|resume_entry_layer| {
                execution_programs.iter().find(|program| {
                    program.runnable_sketch && program.layer_index >= resume_entry_layer
                })
            })
            .or_else(|| {
                execution_programs
                    .iter()
                    .find(|program| program.runnable_sketch)
            });
        let Some(entry_program) = entry_program else {
            return Self::default();
        };

        let has_attention_q = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::AttentionQ));
        let has_attention_k = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::AttentionK));
        let has_attention_v = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::AttentionV));
        let has_attention_out = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::AttentionOut));
        let has_ffn_gate = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::FfnGate));
        let has_ffn_up = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::FfnUp));
        let has_ffn_down = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::FfnDown));
        let has_input_gate = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::InputGate));
        let has_projection = entry_program
            .ops
            .iter()
            .any(|op| matches!(op.kind, ExecutionOpKind::Projection));

        let attention_projection = has_attention_q && has_attention_k && has_attention_v;
        let attention_mix = attention_projection && has_attention_out;
        let attention_path =
            has_attention_q || has_attention_k || has_attention_v || has_attention_out;
        let ffn_path = has_ffn_gate && has_ffn_up && has_ffn_down;
        let projection_path = has_input_gate && has_projection;

        Self {
            attention_path,
            attention_q: has_attention_q,
            attention_k: has_attention_k,
            attention_v: has_attention_v,
            attention_projection,
            attention_mix,
            ffn_path,
            projection_path,
        }
    }
}

impl StageResumeBudgets {
    pub fn from_execution_programs(
        execution_programs: &[LayerExecutionProgram],
        resume_entry_layer: Option<u32>,
        capabilities: &StageResumeCapabilities,
    ) -> Self {
        let entry_program = resume_entry_layer
            .and_then(|resume_entry_layer| {
                execution_programs.iter().find(|program| {
                    program.runnable_sketch && program.layer_index >= resume_entry_layer
                })
            })
            .or_else(|| {
                execution_programs
                    .iter()
                    .find(|program| program.runnable_sketch)
            });
        let Some(entry_program) = entry_program else {
            return Self::default();
        };

        let q_width = entry_program.q_out_dim.map(|width| width as usize);
        let k_width = entry_program.k_out_dim.map(|width| width as usize);
        let v_width = entry_program.v_out_dim.map(|width| width as usize);
        let q_lanes = if capabilities.attention_q {
            q_width.map(|width| ATTENTION_PROJECTION_CARRY_BUDGET.min(width))
        } else {
            None
        };
        let k_lanes = if capabilities.attention_k {
            k_width.map(|width| ATTENTION_PROJECTION_CARRY_BUDGET.min(width))
        } else {
            None
        };
        let v_lanes = if capabilities.attention_v {
            v_width.map(|width| ATTENTION_PROJECTION_CARRY_BUDGET.min(width))
        } else {
            None
        };
        let projection_lanes = q_lanes
            .zip(k_lanes)
            .zip(v_lanes)
            .map(|((q, k), v)| q.min(k).min(v));
        let mix_width = entry_program
            .q_out_dim
            .or(entry_program.k_out_dim)
            .or(entry_program.v_out_dim)
            .or(entry_program.hidden_dim)
            .map(|width| width as usize);
        let ffn_width = entry_program
            .ffn_inner_dim
            .or(entry_program.hidden_dim)
            .map(|width| width as usize);

        Self {
            attention_q_lanes: q_lanes,
            attention_k_lanes: k_lanes,
            attention_v_lanes: v_lanes,
            attention_projection_lanes: if capabilities.attention_projection {
                projection_lanes
            } else {
                None
            },
            attention_mix_lanes: if capabilities.attention_mix {
                mix_width.map(|width| {
                    ATTENTION_MIX_CARRY_BUDGET
                        .min(width)
                        .min(projection_lanes.unwrap_or(0))
                })
            } else {
                None
            },
            ffn_lanes: if capabilities.ffn_path {
                ffn_width.map(|width| TRANSIENT_PREVIEW_LEN.min(width))
            } else {
                None
            },
        }
    }
}

impl StageResumeFreshnessPolicy {
    #[cfg(test)]
    fn blend_strength(weight: f32) -> AttentionBlendStrength {
        Self::blend_strength_for_weight_milli((weight * 1000.0).round() as u16)
    }

    fn blend_strength_for_weight_milli(weight_milli: u16) -> AttentionBlendStrength {
        if weight_milli < 300 {
            AttentionBlendStrength::WeakBlend
        } else {
            AttentionBlendStrength::StrongBlend
        }
    }

    fn max_distance_for_resume_path(path: Option<AttentionSliceResumePath>) -> Option<u32> {
        match path {
            Some(AttentionSliceResumePath {
                blend_strength: None,
                ..
            }) => Some(0),
            Some(AttentionSliceResumePath {
                blend_strength: Some(AttentionBlendStrength::WeakBlend),
                ..
            }) => Some(1),
            Some(AttentionSliceResumePath {
                blend_strength: Some(AttentionBlendStrength::StrongBlend),
                ..
            }) => Some(2),
            None => None,
        }
    }

    #[cfg(test)]
    fn direct_resume_path(
        supported: bool,
        blend_weight: Option<f32>,
    ) -> Option<AttentionSliceResumePath> {
        supported.then_some(AttentionSliceResumePath {
            recompute_phase: AttentionSliceRecomputePhase::Direct,
            blend_strength: blend_weight.map(Self::blend_strength),
            blend_weight_milli: blend_weight.map(|weight| (weight * 1000.0).round() as u16),
        })
    }

    #[cfg(test)]
    fn after_projection_resume_path(
        supported: bool,
        projection_ready: bool,
        blend_weight: Option<f32>,
    ) -> Option<AttentionSliceResumePath> {
        (supported && projection_ready).then_some(AttentionSliceResumePath {
            recompute_phase: AttentionSliceRecomputePhase::AfterProjection,
            blend_strength: blend_weight.map(Self::blend_strength),
            blend_weight_milli: blend_weight.map(|weight| (weight * 1000.0).round() as u16),
        })
    }

    fn attention_resume_paths(
        entry_program: Option<&LayerExecutionProgram>,
    ) -> (
        Option<AttentionSliceResumePath>,
        Option<AttentionSliceResumePath>,
        Option<AttentionSliceResumePath>,
        Option<AttentionSliceResumePath>,
        Option<AttentionSliceResumePath>,
    ) {
        let mut q_path = None;
        let mut k_path = None;
        let mut v_path = None;
        let mut score_path = None;
        let mut value_path = None;

        if let Some(program) = entry_program {
            for op in &program.ops {
                for descriptor in op.attention_resume_descriptors() {
                    match descriptor {
                        ExecutionResumeDescriptor::AttentionQ(path) => q_path = Some(path),
                        ExecutionResumeDescriptor::AttentionK(path) => k_path = Some(path),
                        ExecutionResumeDescriptor::AttentionV(path) => v_path = Some(path),
                        ExecutionResumeDescriptor::AttentionScore(path) => score_path = Some(path),
                        ExecutionResumeDescriptor::AttentionValue(path) => value_path = Some(path),
                    }
                }
            }
        }

        (q_path, k_path, v_path, score_path, value_path)
    }

    pub fn from_execution_programs(
        execution_programs: &[LayerExecutionProgram],
        next_layer_index: Option<u32>,
        capabilities: &StageResumeCapabilities,
    ) -> Self {
        let entry_program = next_layer_index.and_then(|next_layer_index| {
            execution_programs
                .iter()
                .find(|program| program.runnable_sketch && program.layer_index == next_layer_index)
        });
        let (mut q_path, mut k_path, mut v_path, mut score_path, mut value_path) =
            Self::attention_resume_paths(entry_program);
        if !capabilities.attention_q {
            q_path = None;
        }
        if !capabilities.attention_k {
            k_path = None;
        }
        if !capabilities.attention_v {
            v_path = None;
        }
        if !capabilities.attention_mix || !capabilities.attention_projection {
            score_path = None;
            value_path = None;
        }

        Self {
            attention_q_max_distance: Self::max_distance_for_resume_path(q_path),
            attention_k_max_distance: Self::max_distance_for_resume_path(k_path),
            attention_v_max_distance: Self::max_distance_for_resume_path(v_path),
            attention_score_max_distance: Self::max_distance_for_resume_path(score_path),
            attention_value_max_distance: Self::max_distance_for_resume_path(value_path),
        }
    }
}

impl StageResumeContractSummary {
    fn attention_projection_blend() -> StageAttentionResumeContractSummary {
        StageAttentionResumeContractSummary {
            phase: StageAttentionResumePhase::Direct,
            blend: StageAttentionResumeBlend::StrongBlend,
            blend_weight_milli: Some((ATTENTION_PROJECTION_BLEND_WEIGHT * 1000.0).round() as u16),
        }
    }

    fn attention_mix_blend() -> StageAttentionResumeContractSummary {
        StageAttentionResumeContractSummary {
            phase: StageAttentionResumePhase::AfterProjection,
            blend: StageAttentionResumeBlend::WeakBlend,
            blend_weight_milli: Some((ATTENTION_MIX_BLEND_WEIGHT * 1000.0).round() as u16),
        }
    }

    fn current_attention_defaults() -> Self {
        let projection = Some(Self::attention_projection_blend());
        let mix = Some(Self::attention_mix_blend());
        Self {
            attention_q: projection,
            attention_k: projection,
            attention_v: projection,
            attention_score: mix,
            attention_value: mix,
        }
    }

    fn is_empty(&self) -> bool {
        self.attention_q.is_none()
            && self.attention_k.is_none()
            && self.attention_v.is_none()
            && self.attention_score.is_none()
            && self.attention_value.is_none()
    }

    fn validate_entry(
        name: &str,
        entry: Option<StageAttentionResumeContractSummary>,
        max_distance: Option<u32>,
    ) -> Result<()> {
        let Some(entry) = entry else {
            return Ok(());
        };
        let expected_max_distance = match entry.blend {
            StageAttentionResumeBlend::Overwrite => {
                if entry.blend_weight_milli.is_some() {
                    bail!("attention {name} overwrite contract must not carry a blend weight");
                }
                0
            }
            StageAttentionResumeBlend::WeakBlend => {
                let Some(weight_milli) = entry.blend_weight_milli else {
                    bail!("attention {name} weak blend contract is missing blend weight");
                };
                if StageResumeFreshnessPolicy::blend_strength_for_weight_milli(weight_milli)
                    != AttentionBlendStrength::WeakBlend
                {
                    bail!(
                        "attention {name} weak blend contract has non-weak weight {}",
                        weight_milli
                    );
                }
                1
            }
            StageAttentionResumeBlend::StrongBlend => {
                let Some(weight_milli) = entry.blend_weight_milli else {
                    bail!("attention {name} strong blend contract is missing blend weight");
                };
                if StageResumeFreshnessPolicy::blend_strength_for_weight_milli(weight_milli)
                    != AttentionBlendStrength::StrongBlend
                {
                    bail!(
                        "attention {name} strong blend contract has non-strong weight {}",
                        weight_milli
                    );
                }
                2
            }
        };
        if max_distance != Some(expected_max_distance) {
            bail!(
                "attention {name} contract implies max distance {} but boundary advertises {:?}",
                expected_max_distance,
                max_distance
            );
        }
        Ok(())
    }

    fn validate_against_freshness(&self, freshness: StageResumeFreshnessPolicy) -> Result<()> {
        Self::validate_entry("q", self.attention_q, freshness.attention_q_max_distance)?;
        Self::validate_entry("k", self.attention_k, freshness.attention_k_max_distance)?;
        Self::validate_entry("v", self.attention_v, freshness.attention_v_max_distance)?;
        Self::validate_entry(
            "score",
            self.attention_score,
            freshness.attention_score_max_distance,
        )?;
        Self::validate_entry(
            "value",
            self.attention_value,
            freshness.attention_value_max_distance,
        )?;
        Ok(())
    }

    fn summarize_path(
        path: Option<AttentionSliceResumePath>,
    ) -> Option<StageAttentionResumeContractSummary> {
        let path = path?;
        let phase = match path.recompute_phase {
            AttentionSliceRecomputePhase::Direct => StageAttentionResumePhase::Direct,
            AttentionSliceRecomputePhase::AfterProjection => {
                StageAttentionResumePhase::AfterProjection
            }
        };
        let blend = match path.blend_strength {
            Some(AttentionBlendStrength::WeakBlend) => StageAttentionResumeBlend::WeakBlend,
            Some(AttentionBlendStrength::StrongBlend) => StageAttentionResumeBlend::StrongBlend,
            None => StageAttentionResumeBlend::Overwrite,
        };
        Some(StageAttentionResumeContractSummary {
            phase,
            blend,
            blend_weight_milli: path.blend_weight_milli,
        })
    }

    fn from_execution_programs(
        execution_programs: &[LayerExecutionProgram],
        next_layer_index: Option<u32>,
        capabilities: &StageResumeCapabilities,
    ) -> Self {
        let entry_program = next_layer_index.and_then(|next_layer_index| {
            execution_programs
                .iter()
                .find(|program| program.runnable_sketch && program.layer_index == next_layer_index)
        });
        let (mut q_path, mut k_path, mut v_path, mut score_path, mut value_path) =
            StageResumeFreshnessPolicy::attention_resume_paths(entry_program);
        if !capabilities.attention_q {
            q_path = None;
        }
        if !capabilities.attention_k {
            k_path = None;
        }
        if !capabilities.attention_v {
            v_path = None;
        }
        if !capabilities.attention_mix || !capabilities.attention_projection {
            score_path = None;
            value_path = None;
        }

        Self {
            attention_q: Self::summarize_path(q_path),
            attention_k: Self::summarize_path(k_path),
            attention_v: Self::summarize_path(v_path),
            attention_score: Self::summarize_path(score_path),
            attention_value: Self::summarize_path(value_path),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedTensorClass {
    Layer(u32),
    PromptIngress,
    Positional,
    SharedAuxiliary,
    TailOnly,
    Unknown,
}

fn classify_packed_tensor(name: &str) -> PackedTensorClass {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(idx) = rest
            .split('.')
            .next()
            .and_then(|part| part.parse::<u32>().ok())
        {
            return PackedTensorClass::Layer(idx);
        }
    }
    match name {
        "token_embd.weight" => PackedTensorClass::PromptIngress,
        "rope_freqs.weight" => PackedTensorClass::Positional,
        "per_layer_model_proj.weight"
        | "per_layer_proj_norm.weight"
        | "per_layer_token_embd.weight" => PackedTensorClass::SharedAuxiliary,
        name if name.starts_with("output") => PackedTensorClass::TailOnly,
        _ => PackedTensorClass::Unknown,
    }
}

fn build_operator_view(layer: &StageLayerView) -> LayerOperatorView {
    let mut view = LayerOperatorView {
        layer_index: layer.layer_index,
        ..LayerOperatorView::default()
    };

    for entry in &layer.tensors {
        let name = entry.name.as_str();
        if name.contains(".attn_q.weight") {
            view.attn_q = Some(entry.clone());
        } else if name.contains(".attn_k.weight") {
            view.attn_k = Some(entry.clone());
        } else if name.contains(".attn_v.weight") {
            view.attn_v = Some(entry.clone());
        } else if name.contains(".attn_output.weight") {
            view.attn_output = Some(entry.clone());
        } else if name.contains(".attn_norm.weight") {
            view.attn_norm = Some(entry.clone());
        } else if name.contains(".attn_q_norm.weight") {
            view.attn_q_norm = Some(entry.clone());
        } else if name.contains(".attn_k_norm.weight") {
            view.attn_k_norm = Some(entry.clone());
        } else if name.contains(".ffn_up.weight") {
            view.ffn_up = Some(entry.clone());
        } else if name.contains(".ffn_down.weight") {
            view.ffn_down = Some(entry.clone());
        } else if name.contains(".ffn_gate.weight") {
            view.ffn_gate = Some(entry.clone());
        } else if name.contains(".ffn_norm.weight") {
            view.ffn_norm = Some(entry.clone());
        } else if name.contains(".proj.weight") {
            view.proj = Some(entry.clone());
        } else if name.contains(".inp_gate.weight") {
            view.inp_gate = Some(entry.clone());
        } else if name.contains(".post_attention_norm.weight") {
            view.post_attention_norm = Some(entry.clone());
        } else if name.contains(".post_ffw_norm.weight") {
            view.post_ffw_norm = Some(entry.clone());
        } else if name.contains(".post_norm.weight") {
            view.post_norm = Some(entry.clone());
        } else if name.contains(".layer_output_scale.weight") {
            view.layer_output_scale = Some(entry.clone());
        } else {
            view.unknown.push(entry.clone());
        }
    }

    view
}

fn weight_out_dim(entry: &Option<PackedTensorEntry>) -> Option<u64> {
    entry
        .as_ref()
        .and_then(|entry| entry.dimensions.first().copied())
}

fn weight_in_dim(entry: &Option<PackedTensorEntry>) -> Option<u64> {
    entry
        .as_ref()
        .and_then(|entry| entry.dimensions.get(1).copied())
}

fn norm_dim(entry: &Option<PackedTensorEntry>) -> Option<u64> {
    entry
        .as_ref()
        .and_then(|entry| entry.dimensions.first().copied())
}

fn build_execution_spec(layer: &LayerOperatorView) -> LayerExecutionSpec {
    let has_attention_core = layer.attn_q.is_some()
        && layer.attn_k.is_some()
        && layer.attn_v.is_some()
        && layer.attn_output.is_some();
    let has_attention_norms =
        layer.attn_norm.is_some() && layer.attn_q_norm.is_some() && layer.attn_k_norm.is_some();
    let has_ffn_core =
        layer.ffn_up.is_some() && layer.ffn_down.is_some() && layer.ffn_gate.is_some();
    let has_post_norms = layer.post_attention_norm.is_some()
        && layer.post_ffw_norm.is_some()
        && layer.post_norm.is_some()
        && layer.ffn_norm.is_some();
    let has_projection_path = layer.proj.is_some() && layer.inp_gate.is_some();

    let hidden_dim = weight_in_dim(&layer.attn_q)
        .or_else(|| norm_dim(&layer.attn_norm))
        .or_else(|| weight_in_dim(&layer.ffn_up));
    let q_out_dim = weight_out_dim(&layer.attn_q);
    let k_out_dim = weight_out_dim(&layer.attn_k);
    let v_out_dim = weight_out_dim(&layer.attn_v);
    let ffn_inner_dim = weight_out_dim(&layer.ffn_up)
        .or_else(|| weight_out_dim(&layer.ffn_gate))
        .or_else(|| weight_in_dim(&layer.ffn_down));

    let runnable_sketch = has_attention_core
        && has_ffn_core
        && has_projection_path
        && hidden_dim.is_some()
        && q_out_dim.is_some()
        && k_out_dim.is_some()
        && v_out_dim.is_some()
        && ffn_inner_dim.is_some();

    LayerExecutionSpec {
        layer_index: layer.layer_index,
        hidden_dim,
        q_out_dim,
        k_out_dim,
        v_out_dim,
        ffn_inner_dim,
        has_attention_core,
        has_attention_norms,
        has_ffn_core,
        has_post_norms,
        has_projection_path,
        unknown_tensor_count: layer.unknown.len(),
        runnable_sketch,
    }
}

fn named_entries(entries: &[&PackedTensorEntry]) -> Vec<String> {
    entries.iter().map(|entry| entry.name.clone()).collect()
}

fn classify_binding(entries: &[&PackedTensorEntry]) -> (ExecutionBinding, &'static str) {
    if entries.is_empty() {
        return (ExecutionBinding::Unsupported, "no tensors");
    }
    let all_f32_vectors = entries.iter().all(|entry| {
        entry.ggml_type == 0 && entry.dimensions.len() == 1 && entry.byte_len % 4 == 0
    });
    if all_f32_vectors {
        return (ExecutionBinding::F32Vector, "plain f32 vector tensors");
    }
    let all_f32_matrices = entries.iter().all(|entry| {
        entry.ggml_type == 0 && entry.dimensions.len() == 2 && entry.byte_len % 4 == 0
    });
    if all_f32_matrices {
        return (ExecutionBinding::F32Matrix, "plain f32 matrix tensors");
    }

    let any_quantized = entries.iter().any(|entry| entry.ggml_type != 0);
    let any_matrix = entries.iter().any(|entry| entry.dimensions.len() >= 2);
    match (any_quantized, any_matrix) {
        (true, true) => (
            ExecutionBinding::QuantizedMatrix,
            "quantized matrix tensors",
        ),
        (true, false) => (ExecutionBinding::Mixed, "non-f32 non-matrix tensors"),
        (false, true) => (
            ExecutionBinding::Mixed,
            "matrix tensors without numeric binding",
        ),
        (false, false) => (
            ExecutionBinding::Unsupported,
            "unsupported tensor shape/type",
        ),
    }
}

fn build_execution_program(
    layer: &LayerOperatorView,
    spec: &LayerExecutionSpec,
) -> LayerExecutionProgram {
    let mut ops = Vec::new();

    if let Some(entry) = &layer.attn_norm {
        let mut names = vec![entry];
        if let Some(q_norm) = &layer.attn_q_norm {
            names.push(q_norm);
        }
        if let Some(k_norm) = &layer.attn_k_norm {
            names.push(k_norm);
        }
        let (binding, binding_reason) = classify_binding(&names);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::AttentionNorm,
            named_entries(&names),
            binding,
            binding_reason,
        ));
    }

    if spec.has_attention_core {
        if let Some(entry) = &layer.attn_q {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::AttentionQ,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.attn_k {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::AttentionK,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.attn_v {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::AttentionV,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.attn_output {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::AttentionOut,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
    }

    if let Some(entry) = &layer.post_attention_norm {
        let (binding, binding_reason) = classify_binding(&[entry]);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::PostAttentionNorm,
            vec![entry.name.clone()],
            binding,
            binding_reason,
        ));
    }

    if let Some(entry) = &layer.ffn_norm {
        let (binding, binding_reason) = classify_binding(&[entry]);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::FfnNorm,
            vec![entry.name.clone()],
            binding,
            binding_reason,
        ));
    }

    if spec.has_ffn_core {
        if let Some(entry) = &layer.ffn_gate {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::FfnGate,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.ffn_up {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::FfnUp,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.ffn_down {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::FfnDown,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
    }

    let mut post_ffn = Vec::new();
    if let Some(entry) = &layer.post_ffw_norm {
        post_ffn.push(entry);
    }
    if let Some(entry) = &layer.post_norm {
        post_ffn.push(entry);
    }
    if !post_ffn.is_empty() {
        let (binding, binding_reason) = classify_binding(&post_ffn);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::PostFfnNorm,
            named_entries(&post_ffn),
            binding,
            binding_reason,
        ));
    }

    if spec.has_projection_path {
        if let Some(entry) = &layer.inp_gate {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::InputGate,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
        if let Some(entry) = &layer.proj {
            let (binding, binding_reason) = classify_binding(&[entry]);
            ops.push(ExecutionOp::new(
                ExecutionOpKind::Projection,
                vec![entry.name.clone()],
                binding,
                binding_reason,
            ));
        }
    }

    if let Some(entry) = &layer.layer_output_scale {
        let (binding, binding_reason) = classify_binding(&[entry]);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::LayerOutputScale,
            vec![entry.name.clone()],
            binding,
            binding_reason,
        ));
    }

    if !layer.unknown.is_empty() {
        let refs = layer.unknown.iter().collect::<Vec<_>>();
        let (binding, binding_reason) = classify_binding(&refs);
        ops.push(ExecutionOp::new(
            ExecutionOpKind::Unknown,
            layer
                .unknown
                .iter()
                .map(|entry| entry.name.clone())
                .collect(),
            binding,
            binding_reason,
        ));
    }

    LayerExecutionProgram {
        layer_index: layer.layer_index,
        hidden_dim: spec.hidden_dim,
        q_out_dim: spec.q_out_dim,
        k_out_dim: spec.k_out_dim,
        v_out_dim: spec.v_out_dim,
        ffn_inner_dim: spec.ffn_inner_dim,
        runnable_sketch: spec.runnable_sketch,
        ops,
    }
}

const TOY_TOTAL_LAYERS: u32 = 4;
const TOY_HIDDEN_DIM: usize = 16;
const TOY_VOCAB: &[u8] = b" ABCDEFGHIJKLMNOPQRSTUVWXYZ:-";
const SKETCH_VOCAB: &[u8] = b" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyLayerArtifact {
    pub index: u32,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyProjectionArtifact {
    pub vocab: Vec<String>,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyShardArtifact {
    pub model_id: String,
    pub hidden_dim: usize,
    pub start_layer: u32,
    pub end_layer: u32,
    pub total_layers: u32,
    pub layers: Vec<ToyLayerArtifact>,
    pub projection: Option<ToyProjectionArtifact>,
}

pub trait StageForwardBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()>;
    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        hidden_dim_hint: usize,
    ) -> Result<StageTensor>;
    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor>;
    fn sample_tail(&self, input: StageTensor) -> Result<StageSample>;
}

#[derive(Debug, Default)]
pub struct DeterministicStubBackend {
    layout: Option<StageLayout>,
}

impl DeterministicStubBackend {
    fn layout(&self) -> Result<&StageLayout> {
        self.layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage layout loaded"))
    }

    fn synthesize_hidden_state(
        &self,
        prompt: &str,
        prior: &[u8],
        hidden_dim_hint: usize,
        stage_id: &str,
    ) -> Vec<u8> {
        let seed = if prior.is_empty() {
            let mut buf = prompt.as_bytes().to_vec();
            buf.extend_from_slice(stage_id.as_bytes());
            if buf.is_empty() {
                buf.extend_from_slice(b"compute-backend");
            }
            buf
        } else {
            prior.to_vec()
        };

        let width = hidden_dim_hint.max(64).min(4096);
        let mut out = Vec::with_capacity(width);
        for idx in 0..width {
            let base = seed[idx % seed.len()];
            let stage = stage_id.as_bytes()[idx % stage_id.len()];
            out.push(base.wrapping_add(stage).wrapping_add((idx % 251) as u8));
        }
        out
    }
}

impl StageForwardBackend for DeterministicStubBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        self.layout = Some(layout);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }

        let hidden = self.synthesize_hidden_state(prompt, &[], hidden_dim_hint, &layout.stage_id);
        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim: hidden_dim_hint.max(64).min(4096),
            bytes: hidden,
            prompt_text: Some(prompt.to_string()),
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let layout = self.layout()?;
        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }
        if input
            .stage_trace
            .iter()
            .any(|stage| stage == &layout.stage_id)
        {
            bail!("Stage {} already applied to payload", layout.stage_id);
        }

        let prompt = input.prompt_text.clone().unwrap_or_default();
        let bytes =
            self.synthesize_hidden_state(&prompt, &input.bytes, input.hidden_dim, &layout.stage_id);
        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());

        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: input.hidden_dim,
            bytes,
            prompt_text: input.prompt_text,
            max_tokens: input.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let layout = self.layout()?;
        if !layout.is_tail {
            bail!("Only the tail stage may sample output");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail sampling requires hidden-state payloads");
        }

        let prompt = input.prompt_text.unwrap_or_default();
        let trace = input.stage_trace.join(" -> ");
        let take = input.max_tokens.unwrap_or(48).min(96);
        let digest = input
            .bytes
            .iter()
            .take(16)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();

        let text = format!(
            "{} :: {} :: {} :: {}",
            prompt, trace, layout.model_id, digest
        )
        .chars()
        .take(take as usize)
        .collect::<String>();
        Ok(StageSample {
            request_id: input.request_id,
            model_id: layout.model_id.clone(),
            token_ids: StageSample::text_token_ids(&text),
            text,
            completion_tokens: take,
        })
    }
}

#[derive(Debug, Default)]
pub struct ToyLinearBackend {
    layout: Option<StageLayout>,
}

impl ToyLinearBackend {
    fn layout(&self) -> Result<&StageLayout> {
        self.layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage layout loaded"))
    }

    fn encode_hidden_state(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        if bytes.len() != hidden_dim * 4 {
            bail!(
                "Hidden-state byte length {} does not match hidden_dim {}",
                bytes.len(),
                hidden_dim
            );
        }
        let mut values = Vec::with_capacity(hidden_dim);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }

    fn prompt_embedding(prompt: &str) -> Vec<f32> {
        let mut state = vec![0.0f32; TOY_HIDDEN_DIM];
        if prompt.is_empty() {
            state[0] = 1.0;
            return state;
        }

        for (idx, byte) in prompt.bytes().enumerate() {
            let slot = idx % TOY_HIDDEN_DIM;
            let normalized = (byte as f32) / 255.0;
            state[slot] += normalized;
            state[(slot * 7 + 3) % TOY_HIDDEN_DIM] += normalized * 0.5;
        }

        let scale = 1.0 / prompt.len() as f32;
        for value in &mut state {
            *value *= scale;
        }
        state
    }

    fn apply_layer(mut state: Vec<f32>, layer_idx: u32) -> Vec<f32> {
        let len = state.len();
        let bias = 0.03125 * (layer_idx as f32 + 1.0);
        let scale = 1.05 + (layer_idx as f32 * 0.015);
        for idx in 0..len {
            let prev = state[(idx + len - 1) % len];
            let next = state[(idx + 1) % len];
            let mix = prev * 0.17 + next * 0.11;
            let pos = ((idx as f32 + 1.0) / len as f32) * 0.07;
            state[idx] = ((state[idx] * scale) + mix + bias + pos).tanh();
        }
        state
    }

    fn project_text(state: &[f32], max_tokens: Option<u32>) -> String {
        let take = max_tokens.unwrap_or(12).clamp(4, 16) as usize;
        let mut out = String::with_capacity(take);
        for token_idx in 0..take {
            let mut best_score = f32::MIN;
            let mut best_char = ' ';
            for (vocab_idx, byte) in TOY_VOCAB.iter().enumerate() {
                let mut score = 0.0f32;
                for (dim, value) in state.iter().enumerate() {
                    let weight = (((vocab_idx + 1) * (dim + 3) * (token_idx + 1)) % 17) as f32;
                    score += *value * (weight * 0.03125);
                }
                if score > best_score {
                    best_score = score;
                    best_char = *byte as char;
                }
            }
            out.push(best_char);
        }
        out.trim().to_string()
    }

    fn run_layer_range(mut state: Vec<f32>, start_layer: u32, end_layer: u32) -> Vec<f32> {
        for layer_idx in start_layer..=end_layer {
            state = Self::apply_layer(state, layer_idx);
        }
        state
    }
}

impl StageForwardBackend for ToyLinearBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        self.layout = Some(layout);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }

        let state = Self::run_layer_range(
            Self::prompt_embedding(prompt),
            layout.start_layer,
            layout.end_layer,
        );

        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim: TOY_HIDDEN_DIM,
            bytes: Self::encode_hidden_state(&state),
            prompt_text: Some(prompt.to_string()),
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let layout = self.layout()?;
        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }

        let state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;
        let next_state = Self::run_layer_range(state, layout.start_layer, layout.end_layer);
        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());

        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: input.hidden_dim,
            bytes: Self::encode_hidden_state(&next_state),
            prompt_text: input.prompt_text,
            max_tokens: input.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let layout = self.layout()?;
        if !layout.is_tail {
            bail!("Only the tail stage may sample output");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail sampling requires hidden-state payloads");
        }

        let state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;
        let text = Self::project_text(&state, input.max_tokens);

        Ok(StageSample {
            request_id: input.request_id,
            model_id: layout.model_id.clone(),
            completion_tokens: text.chars().count() as u32,
            token_ids: StageSample::text_token_ids(&text),
            text,
        })
    }
}

pub fn run_toy_single_node_reference(prompt: &str, max_tokens: Option<u32>) -> StageSample {
    let state = ToyLinearBackend::run_layer_range(
        ToyLinearBackend::prompt_embedding(prompt),
        0,
        TOY_TOTAL_LAYERS - 1,
    );
    let text = ToyLinearBackend::project_text(&state, max_tokens);
    StageSample {
        request_id: "single-node-reference".into(),
        model_id: "toy-linear-4l".into(),
        completion_tokens: text.chars().count() as u32,
        token_ids: StageSample::text_token_ids(&text),
        text,
    }
}

#[derive(Debug)]
pub struct ArtifactBackedToyBackend {
    artifact_path: PathBuf,
    layout: Option<StageLayout>,
    artifact: Option<ToyShardArtifact>,
}

#[derive(Debug)]
pub struct PackedResidencySketchBackend {
    index_path: PathBuf,
    layout: Option<StageLayout>,
    store: Option<StageTensorStore>,
    model_view: Option<StageModelView>,
    debug_layer_cap: Option<usize>,
}

#[derive(Debug, Default, Clone)]
struct LayerScratch {
    attention_contract: StageResumeContractSummary,
    attn_q: Option<Vec<f32>>,
    attn_k: Option<Vec<f32>>,
    attn_v: Option<Vec<f32>>,
    attn_score: Option<Vec<f32>>,
    attn_value: Option<Vec<f32>>,
    attn_q_blend_weight: Option<f32>,
    attn_k_blend_weight: Option<f32>,
    attn_v_blend_weight: Option<f32>,
    attn_score_blend_weight: Option<f32>,
    attn_value_blend_weight: Option<f32>,
    attn_q_lane_indices: Option<Vec<usize>>,
    attn_k_lane_indices: Option<Vec<usize>>,
    attn_v_lane_indices: Option<Vec<usize>>,
    attn_score_lane_indices: Option<Vec<usize>>,
    attn_value_lane_indices: Option<Vec<usize>>,
    ffn_gate: Option<Vec<f32>>,
    ffn_up: Option<Vec<f32>>,
    ffn_activation: Option<Vec<f32>>,
    ffn_lane_indices: Option<Vec<usize>>,
    input_gate: Option<Vec<f32>>,
}

impl LayerScratch {
    fn typed_or_legacy_attention_blend(
        &self,
        typed_weight: Option<f32>,
        legacy_weight: f32,
    ) -> Option<f32> {
        if self.attention_contract.is_empty() {
            Some(legacy_weight)
        } else {
            typed_weight
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionScratchState {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AttentionMixState {
    scores: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AttentionExecutionCheckpoint {
    projection: Option<AttentionScratchState>,
    mix: Option<AttentionMixState>,
    q_provenance: Option<AttentionCheckpointProvenance>,
    k_provenance: Option<AttentionCheckpointProvenance>,
    v_provenance: Option<AttentionCheckpointProvenance>,
    score_provenance: Option<AttentionCheckpointProvenance>,
    value_provenance: Option<AttentionCheckpointProvenance>,
}

#[derive(Debug, Clone)]
struct FfnScratchState {
    gate: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FfnMixState {
    activations: Vec<f32>,
}

const TRANSIENT_PREVIEW_LEN: usize = 8;
const ATTENTION_PROJECTION_CARRY_BUDGET: usize = 4;
const ATTENTION_MIX_CARRY_BUDGET: usize = TRANSIENT_PREVIEW_LEN;
const ATTENTION_PROJECTION_BLEND_WEIGHT: f32 = 0.35;
const ATTENTION_MIX_BLEND_WEIGHT: f32 = 0.25;

impl PackedResidencySketchBackend {
    fn merge_transient_checkpoint(
        previous: Option<StageTransientState>,
        current: Option<StageTransientState>,
    ) -> Option<StageTransientState> {
        match (previous, current) {
            (None, None) => None,
            (Some(previous), None) => Some(previous),
            (None, Some(current)) => Some(current),
            (Some(previous), Some(current)) => Some(StageTransientState {
                attention: current.attention.or(previous.attention),
                ffn: current.ffn.or(previous.ffn),
            }),
        }
    }

    fn merge_stage_carry(
        previous: Option<StageCarryState>,
        current: Option<StageCarryState>,
    ) -> Option<StageCarryState> {
        match (previous, current) {
            (None, None) => None,
            (Some(previous), None) => Some(previous.age_attention_by_layers(1)),
            (None, Some(current)) => Some(current),
            (Some(previous), Some(current)) => Some(StageCarryState {
                attention: current
                    .attention
                    .or_else(|| previous.age_attention_by_layers(1).attention),
                ffn: current.ffn.or(previous.ffn),
            }),
        }
    }

    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            index_path: path.into(),
            layout: None,
            store: None,
            model_view: None,
            debug_layer_cap: None,
        }
    }

    pub fn with_debug_layer_cap(mut self, layer_cap: usize) -> Self {
        self.debug_layer_cap = Some(layer_cap);
        self
    }

    pub fn set_debug_layer_cap(&mut self, layer_cap: Option<usize>) {
        self.debug_layer_cap = layer_cap;
    }

    fn layout(&self) -> Result<&StageLayout> {
        self.layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage layout loaded"))
    }

    fn store(&self) -> Result<&StageTensorStore> {
        self.store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage tensor store loaded"))
    }

    fn model_view(&self) -> Result<&StageModelView> {
        self.model_view
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage model view loaded"))
    }

    fn expected_attention_carry_width(view: &StageModelView) -> Option<usize> {
        view.execution_programs
            .iter()
            .find(|program| {
                program.runnable_sketch
                    && program.ops.iter().any(|op| {
                        matches!(
                            op.kind,
                            ExecutionOpKind::AttentionQ
                                | ExecutionOpKind::AttentionK
                                | ExecutionOpKind::AttentionV
                                | ExecutionOpKind::AttentionOut
                        )
                    })
            })
            .and_then(|program| {
                program
                    .q_out_dim
                    .or(program.k_out_dim)
                    .or(program.v_out_dim)
                    .or(program.hidden_dim)
            })
            .map(|width| width as usize)
    }

    fn expected_ffn_carry_width(view: &StageModelView) -> Option<usize> {
        view.execution_programs
            .iter()
            .find(|program| {
                program.runnable_sketch
                    && program.ops.iter().any(|op| {
                        matches!(
                            op.kind,
                            ExecutionOpKind::FfnGate
                                | ExecutionOpKind::FfnUp
                                | ExecutionOpKind::FfnDown
                        )
                    })
            })
            .and_then(|program| program.ffn_inner_dim.or(program.hidden_dim))
            .map(|width| width as usize)
    }

    pub fn admit_resume_request(&self, request: &StageResumeRequest) -> StageResumeDecision {
        let fallback_payload_kind = request.transfer.payload.kind;
        let fallback_hidden_dim = request.transfer.payload.hidden_dim;

        if let Err(err) = request.validate() {
            return StageResumeDecision::reject(
                None,
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                format!("invalid resume request: {err}"),
            );
        }

        let layout = match self.layout() {
            Ok(layout) => layout,
            Err(err) => {
                return StageResumeDecision::reject(
                    None,
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!("backend not ready: {err}"),
                );
            }
        };
        let view = match self.model_view() {
            Ok(view) => view,
            Err(err) => {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!("missing model view: {err}"),
                );
            }
        };

        if layout.is_head {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "head stage cannot admit downstream resume requests",
            );
        }
        if request.model_id != layout.model_id {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                format!(
                    "model mismatch: request {} vs layout {}",
                    request.model_id, layout.model_id
                ),
            );
        }
        if let Some(target_stage_id) = &request.target_stage_id {
            if target_stage_id != &layout.stage_id {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "target stage mismatch: request {} vs layout {}",
                        target_stage_id, layout.stage_id
                    ),
                );
            }
        }
        if request.transfer.payload.kind != PayloadKind::HiddenState {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "downstream resume requires hidden-state payload",
            );
        }
        if let Some(next_layer_index) = request.boundary.next_layer_index {
            if next_layer_index != layout.start_layer {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "next-layer mismatch: request {} vs layout start {}",
                        next_layer_index, layout.start_layer
                    ),
                );
            }
        }

        let capabilities = StageResumeCapabilities::from_execution_programs(
            &view.execution_programs,
            request.boundary.next_layer_index,
        );
        let budgets = StageResumeBudgets::from_execution_programs(
            &view.execution_programs,
            request.boundary.next_layer_index,
            &capabilities,
        );
        let freshness = StageResumeFreshnessPolicy::from_execution_programs(
            &view.execution_programs,
            request.boundary.next_layer_index,
            &capabilities,
        );
        let target_contract = StageResumeContractSummary::from_execution_programs(
            &view.execution_programs,
            request.boundary.next_layer_index,
            &capabilities,
        );
        if !request.boundary.resumable_attention_contract.is_empty()
            && request.boundary.resumable_attention_contract != target_contract
        {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "attention resume contract does not match target execution program",
            );
        }
        if request.boundary.expects_attention_carry && !capabilities.attention_path {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "attention carry requested but target stage has no runnable attention path",
            );
        }
        if request.boundary.expects_attention_projection_carry && !capabilities.attention_projection
        {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "attention projection carry requested but target stage cannot resume q/k/v projection substate",
            );
        }
        if request.boundary.expects_attention_mix_carry && !capabilities.attention_mix {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "attention mix carry requested but target stage cannot resume score/value mix substate",
            );
        }
        if request.boundary.expects_ffn_carry && !capabilities.ffn_path {
            return StageResumeDecision::reject(
                Some(layout.stage_id.clone()),
                fallback_payload_kind,
                fallback_hidden_dim,
                request.boundary.next_layer_index,
                "ffn carry requested but target stage has no runnable ffn path",
            );
        }
        if let (Some(expected_lanes), Some(max_lanes)) = (
            request.boundary.expected_attention_q_lanes,
            budgets.attention_q_lanes,
        ) {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention q carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let (Some(expected_distance), Some(max_distance)) = (
            request.boundary.expected_attention_q_distance,
            freshness.attention_q_max_distance,
        ) {
            if expected_distance > max_distance {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention q freshness distance {} exceeds target limit {}",
                        expected_distance, max_distance
                    ),
                );
            }
        }
        if let (Some(expected_lanes), Some(max_lanes)) = (
            request.boundary.expected_attention_k_lanes,
            budgets.attention_k_lanes,
        ) {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention k carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let (Some(expected_distance), Some(max_distance)) = (
            request.boundary.expected_attention_k_distance,
            freshness.attention_k_max_distance,
        ) {
            if expected_distance > max_distance {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention k freshness distance {} exceeds target limit {}",
                        expected_distance, max_distance
                    ),
                );
            }
        }
        if let (Some(expected_lanes), Some(max_lanes)) = (
            request.boundary.expected_attention_v_lanes,
            budgets.attention_v_lanes,
        ) {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention v carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let (Some(expected_distance), Some(max_distance)) = (
            request.boundary.expected_attention_v_distance,
            freshness.attention_v_max_distance,
        ) {
            if expected_distance > max_distance {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention v freshness distance {} exceeds target limit {}",
                        expected_distance, max_distance
                    ),
                );
            }
        }
        if let (Some(expected_lanes), Some(max_lanes)) = (
            request.boundary.expected_attention_projection_lanes,
            budgets.attention_projection_lanes,
        ) {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention projection carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let (Some(expected_lanes), Some(max_lanes)) = (
            request.boundary.expected_attention_mix_lanes,
            budgets.attention_mix_lanes,
        ) {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention mix carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let (Some(expected_distance), Some(max_distance)) = (
            request.boundary.expected_attention_score_distance,
            freshness.attention_score_max_distance,
        ) {
            if expected_distance > max_distance {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention score freshness distance {} exceeds target limit {}",
                        expected_distance, max_distance
                    ),
                );
            }
        }
        if let (Some(expected_distance), Some(max_distance)) = (
            request.boundary.expected_attention_value_distance,
            freshness.attention_value_max_distance,
        ) {
            if expected_distance > max_distance {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "attention value freshness distance {} exceeds target limit {}",
                        expected_distance, max_distance
                    ),
                );
            }
        }
        if let (Some(expected_lanes), Some(max_lanes)) =
            (request.boundary.expected_ffn_lanes, budgets.ffn_lanes)
        {
            if expected_lanes > max_lanes {
                return StageResumeDecision::reject(
                    Some(layout.stage_id.clone()),
                    fallback_payload_kind,
                    fallback_hidden_dim,
                    request.boundary.next_layer_index,
                    format!(
                        "ffn carry lanes {} exceed target budget {}",
                        expected_lanes, max_lanes
                    ),
                );
            }
        }
        if let Some(expected_width) = request.boundary.expected_attention_width {
            match Self::expected_attention_carry_width(view) {
                Some(actual_width) if actual_width == expected_width => {}
                Some(actual_width) => {
                    return StageResumeDecision::reject(
                        Some(layout.stage_id.clone()),
                        fallback_payload_kind,
                        fallback_hidden_dim,
                        request.boundary.next_layer_index,
                        format!(
                            "attention carry width mismatch: request {} vs target {}",
                            expected_width, actual_width
                        ),
                    );
                }
                None => {
                    return StageResumeDecision::reject(
                        Some(layout.stage_id.clone()),
                        fallback_payload_kind,
                        fallback_hidden_dim,
                        request.boundary.next_layer_index,
                        "attention carry width requested but target stage has no carry width",
                    );
                }
            }
        }
        if let Some(expected_width) = request.boundary.expected_ffn_width {
            match Self::expected_ffn_carry_width(view) {
                Some(actual_width) if actual_width == expected_width => {}
                Some(actual_width) => {
                    return StageResumeDecision::reject(
                        Some(layout.stage_id.clone()),
                        fallback_payload_kind,
                        fallback_hidden_dim,
                        request.boundary.next_layer_index,
                        format!(
                            "ffn carry width mismatch: request {} vs target {}",
                            expected_width, actual_width
                        ),
                    );
                }
                None => {
                    return StageResumeDecision::reject(
                        Some(layout.stage_id.clone()),
                        fallback_payload_kind,
                        fallback_hidden_dim,
                        request.boundary.next_layer_index,
                        "ffn carry width requested but target stage has no carry width",
                    );
                }
            }
        }

        StageResumeDecision::accept(
            Some(layout.stage_id.clone()),
            request.transfer.payload.kind,
            request.transfer.payload.hidden_dim,
            request.boundary.expects_attention_carry && capabilities.attention_path,
            request.boundary.expects_ffn_carry && capabilities.ffn_path,
            request.boundary.expected_attention_width,
            request.boundary.expected_attention_lanes,
            capabilities.attention_q,
            capabilities.attention_k,
            capabilities.attention_v,
            request.boundary.resumable_attention_q_lanes,
            request.boundary.resumable_attention_k_lanes,
            request.boundary.resumable_attention_v_lanes,
            request.boundary.resumable_attention_q_max_distance,
            request.boundary.resumable_attention_k_max_distance,
            request.boundary.resumable_attention_v_max_distance,
            request.boundary.resumable_attention_score_max_distance,
            request.boundary.resumable_attention_value_max_distance,
            request.boundary.resumable_attention_contract.clone(),
            request.boundary.expected_attention_score_lanes,
            request.boundary.expected_attention_value_lanes,
            request.boundary.expected_attention_q_distance,
            request.boundary.expected_attention_k_distance,
            request.boundary.expected_attention_v_distance,
            request.boundary.expected_attention_score_distance,
            request.boundary.expected_attention_value_distance,
            request.boundary.expects_attention_projection_carry
                && capabilities.attention_projection,
            request.boundary.expects_attention_mix_carry && capabilities.attention_mix,
            request.boundary.expected_attention_projection_lanes,
            request.boundary.expected_attention_mix_lanes,
            request.boundary.expected_ffn_width,
            request.boundary.expected_ffn_lanes,
            request.boundary.next_layer_index,
        )
    }

    pub fn resume_forward(
        &self,
        request: StageResumeRequest,
    ) -> Result<(StageTensor, StageResumeReceipt)> {
        let decision = self.admit_resume_request(&request);
        let receipt = decision.into_receipt(request.version);
        receipt.validate_against_request(&request)?;
        if !receipt.accepted {
            bail!(
                "{}",
                receipt
                    .reason
                    .clone()
                    .unwrap_or_else(|| "resume request rejected".to_string())
            );
        }

        let input = request.transfer.into_stage_tensor();
        let output = self.continue_forward(input)?;
        Ok((output, receipt))
    }

    fn hidden_dim(view: &StageModelView) -> usize {
        let base = (view
            .layers
            .len()
            .saturating_mul(16)
            .saturating_add(view.shared_auxiliary.len().saturating_mul(8))
            .max(64))
        .clamp(64, 512);
        base.next_multiple_of(16)
    }

    fn continuation_for_view(
        model_view: &StageModelView,
        completed_layers: u32,
        next_layer_index: Option<u32>,
    ) -> StageContinuation {
        StageContinuation {
            version: 1,
            stage_role: model_view.role.clone(),
            next_layer_index,
            completed_layers,
            operator_layers: model_view.operator_layers.len(),
            has_attention_path: model_view.execution_programs.iter().any(|program| {
                program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        ExecutionOpKind::AttentionQ
                            | ExecutionOpKind::AttentionK
                            | ExecutionOpKind::AttentionV
                            | ExecutionOpKind::AttentionOut
                    )
                })
            }),
            has_ffn_path: model_view.execution_programs.iter().any(|program| {
                program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        ExecutionOpKind::FfnGate
                            | ExecutionOpKind::FfnUp
                            | ExecutionOpKind::FfnDown
                    )
                })
            }),
            has_projection_path: model_view.execution_programs.iter().any(|program| {
                program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        ExecutionOpKind::InputGate | ExecutionOpKind::Projection
                    )
                })
            }),
        }
    }

    fn prompt_seed(prompt: &str, view: &StageModelView, stage_id: &str, width: usize) -> Vec<u8> {
        let mut seed = prompt.as_bytes().to_vec();
        seed.extend_from_slice(stage_id.as_bytes());
        seed.extend_from_slice(&(view.layers.len() as u64).to_le_bytes());
        seed.extend_from_slice(&(view.prompt_ingress.len() as u64).to_le_bytes());
        seed.extend_from_slice(&(view.shared_auxiliary.len() as u64).to_le_bytes());
        if seed.is_empty() {
            seed.extend_from_slice(b"packed-stage-backend");
        }

        let mut out = vec![0u8; width];
        for idx in 0..width {
            let a = seed[idx % seed.len()];
            let b = stage_id.as_bytes()[idx % stage_id.len()];
            out[idx] = a
                .wrapping_add(b)
                .wrapping_add((idx % 251) as u8)
                .rotate_left((idx % 7) as u32);
        }
        out
    }

    fn encode_hidden_state(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        if bytes.len() != hidden_dim * 4 {
            bail!(
                "Hidden-state byte length {} does not match hidden_dim {}",
                bytes.len(),
                hidden_dim
            );
        }
        let mut values = Vec::with_capacity(hidden_dim);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }

    fn prompt_state(prompt: &str, view: &StageModelView, stage_id: &str, width: usize) -> Vec<f32> {
        let seed = Self::prompt_seed(prompt, view, stage_id, width);
        seed.into_iter()
            .map(|byte| ((byte as f32) / 255.0) * 2.0 - 1.0)
            .collect()
    }

    fn continue_state(input: &[f32], stage_id: &str) -> Vec<f32> {
        input
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                let salt = stage_id.as_bytes()[idx % stage_id.len()] as f32 / 255.0;
                (value + salt + ((idx % 17) as f32 * 0.01)).tanh()
            })
            .collect()
    }

    fn entry_scalar(store: &StageTensorStore, entry: &PackedTensorEntry) -> Result<f32> {
        let bytes = store.read(&entry.name)?;
        let take = bytes.len().min(64);
        let acc = bytes
            .iter()
            .take(take)
            .enumerate()
            .fold(0u64, |acc, (idx, byte)| {
                acc + ((*byte as u64) * ((idx as u64 % 13) + 1))
            });
        let dim_mix = entry.dimensions.iter().copied().sum::<u64>() + entry.ggml_type as u64;
        let scalar = (((acc + dim_mix) % 4096) as f32 / 4096.0) * 2.0 - 1.0;
        Ok(scalar)
    }

    fn try_decode_f32_vector(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<Option<Vec<f32>>> {
        if entry.ggml_type != 0 || entry.dimensions.len() != 1 || entry.byte_len % 4 != 0 {
            return Ok(None);
        }
        let bytes = store.read(&entry.name)?;
        if bytes.len() != entry.byte_len as usize {
            bail!(
                "Packed tensor {} length {} did not match index length {}",
                entry.name,
                bytes.len(),
                entry.byte_len
            );
        }
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some(out))
    }

    fn try_decode_f32_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<Option<(usize, usize, Vec<f32>)>> {
        if entry.ggml_type != 0 || entry.dimensions.len() != 2 || entry.byte_len % 4 != 0 {
            return Ok(None);
        }
        let rows = entry.dimensions[0] as usize;
        let cols = entry.dimensions[1] as usize;
        let bytes = store.read(&entry.name)?;
        if bytes.len() != rows * cols * 4 {
            bail!(
                "Packed tensor {} byte size {} did not match rows*cols*4 {}",
                entry.name,
                bytes.len(),
                rows * cols * 4
            );
        }
        let mut out = Vec::with_capacity(rows * cols);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some((rows, cols, out)))
    }

    fn try_decode_numeric_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
    ) -> Result<Option<(usize, usize, Vec<f32>)>> {
        if entry.dimensions.len() != 2 {
            return Ok(None);
        }
        let rows = entry.dimensions[0] as usize;
        let cols = entry.dimensions[1] as usize;
        let bytes = store.read(&entry.name)?;
        let decoded = match entry.ggml_type {
            quants::GGML_TYPE_F32 | quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q6_K => {
                quants::dequantize_tensor(entry.ggml_type, &bytes)?
            }
            _ => return Ok(None),
        };
        if decoded.len() != rows * cols {
            bail!(
                "Decoded tensor {} length {} did not match rows*cols {}",
                entry.name,
                decoded.len(),
                rows * cols
            );
        }
        Ok(Some((rows, cols, decoded)))
    }

    fn apply_scalar_to_state(state: &mut [f32], scalar: f32, salt: &[u8]) {
        let snapshot = state.to_vec();
        let len = snapshot.len();
        for (idx, value) in state.iter_mut().enumerate() {
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            let neighbor = snapshot[(idx + len - 1) % len];
            let next = snapshot[(idx + 1) % len];
            let mix = neighbor * 0.11 + next * 0.07;
            *value =
                ((snapshot[idx] * (1.0 + scalar * 0.05)) + mix + salt_value * 0.03 + scalar * 0.1)
                    .tanh();
        }
    }

    fn apply_numeric_vector_to_state(
        state: &mut [f32],
        vector: &[f32],
        salt: &[u8],
        rms_style: bool,
    ) {
        let len = state.len().min(vector.len());
        if len == 0 {
            return;
        }
        let snapshot = state[..len].to_vec();
        let rms = if rms_style {
            let mean_sq = snapshot.iter().map(|v| v * v).sum::<f32>() / len as f32;
            (mean_sq + 1.0e-6).sqrt()
        } else {
            1.0
        };

        for idx in 0..len {
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            let normalized = snapshot[idx] / rms;
            let scale = vector[idx];
            state[idx] = (normalized * scale + salt_value * 0.01).tanh();
        }
        for idx in len..state.len() {
            state[idx] = (state[idx] + (salt[idx % salt.len()] as f32 / 255.0) * 0.005).tanh();
        }
    }

    fn apply_numeric_matrix_to_state(
        state: &mut [f32],
        rows: usize,
        cols: usize,
        matrix: &[f32],
        salt: &[u8],
    ) {
        if rows == 0 || cols == 0 || matrix.len() != rows * cols || state.is_empty() {
            return;
        }

        let input = (0..cols)
            .map(|idx| state[idx % state.len()])
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut acc = 0.0f32;
            let row_offset = row * cols;
            for col in 0..cols {
                acc += matrix[row_offset + col] * input[col];
            }
            output[row] = acc.tanh();
        }

        let snapshot = state.to_vec();
        for idx in 0..state.len() {
            let projected = output[idx % rows];
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            state[idx] = (snapshot[idx] * 0.7 + projected * 0.3 + salt_value * 0.01).tanh();
        }
    }

    fn apply_numeric_matrix_project(
        state: &[f32],
        rows: usize,
        cols: usize,
        matrix: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 || matrix.len() != rows * cols || state.is_empty() {
            return Vec::new();
        }
        let input = (0..cols)
            .map(|idx| state[idx % state.len()])
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut acc = 0.0f32;
            let row_offset = row * cols;
            for col in 0..cols {
                acc += matrix[row_offset + col] * input[col];
            }
            output[row] = acc;
        }
        output
    }

    pub fn try_project_decoded_attention_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        Ok(Some(Self::apply_numeric_matrix_project(
            state, rows, cols, &matrix,
        )))
    }

    fn try_project_decoded_attention_output_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
        scratch: &LayerScratch,
        salt: &[u8],
    ) -> Result<
        Option<(
            Option<AttentionScratchState>,
            Option<AttentionMixState>,
            Vec<f32>,
        )>,
    > {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        let attn_state = Self::build_attention_state(scratch);
        let (attn_state, mix, merged) = if let Some(attn_state) = attn_state {
            let mix = Self::build_resumed_attention_mix_state(&attn_state, rows, salt, scratch);
            let merged = Self::mix_attention_state(&mix, salt);
            (Some(attn_state), Some(mix), merged)
        } else {
            (None, None, Vec::new())
        };
        let merged = if merged.is_empty() {
            Self::apply_numeric_matrix_project(state, rows, cols, &matrix)
        } else {
            merged
        };
        let out_proj = Self::apply_numeric_matrix_project(&merged, rows, cols, &matrix);
        Ok(Some((attn_state, mix, out_proj)))
    }

    fn try_project_decoded_ffn_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        Ok(Some(Self::apply_numeric_matrix_project(
            state, rows, cols, &matrix,
        )))
    }

    fn try_project_decoded_ffn_down_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
        scratch: &LayerScratch,
        salt: &[u8],
    ) -> Result<Option<(Option<FfnScratchState>, Option<FfnMixState>, Vec<f32>)>> {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        let ffn_state = Self::build_ffn_state(scratch);
        let (ffn_state, mix, input) = if let Some(ffn_state) = ffn_state {
            let mix = Self::build_resumed_ffn_mix_state(&ffn_state, scratch);
            let input = Self::mix_ffn_state(&mix, salt);
            (Some(ffn_state), Some(mix), input)
        } else {
            (None, None, state.to_vec())
        };
        let down_proj = Self::apply_numeric_matrix_project(&input, rows, cols, &matrix);
        Ok(Some((ffn_state, mix, down_proj)))
    }

    fn try_project_decoded_input_gate_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        Ok(Some(Self::apply_numeric_matrix_project(
            state, rows, cols, &matrix,
        )))
    }

    fn try_project_decoded_projection_matrix(
        store: &StageTensorStore,
        entry: &PackedTensorEntry,
        state: &[f32],
        scratch: &LayerScratch,
    ) -> Result<Option<Vec<f32>>> {
        let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)? else {
            return Ok(None);
        };
        let mut proj = Self::apply_numeric_matrix_project(state, rows, cols, &matrix);
        if let Some(gate) = &scratch.input_gate {
            for idx in 0..proj.len() {
                proj[idx] *= gate[idx % gate.len()].tanh();
            }
        }
        Ok(Some(proj))
    }

    fn blend_projection_into_state(
        state: &mut [f32],
        projection: &[f32],
        salt: &[u8],
        weight: f32,
    ) {
        if state.is_empty() || projection.is_empty() {
            return;
        }
        let snapshot = state.to_vec();
        for idx in 0..state.len() {
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            let projected = projection[idx % projection.len()].tanh();
            state[idx] =
                (snapshot[idx] * (1.0 - weight) + projected * weight + salt_value * 0.01).tanh();
        }
    }

    fn build_attention_state(scratch: &LayerScratch) -> Option<AttentionScratchState> {
        Some(AttentionScratchState {
            q: scratch.attn_q.clone()?,
            k: scratch.attn_k.clone()?,
            v: scratch.attn_v.clone()?,
        })
    }

    fn build_attention_scores(
        attn: &AttentionScratchState,
        output_dim: usize,
        salt: &[u8],
    ) -> Vec<f32> {
        if output_dim == 0 || attn.q.is_empty() || attn.k.is_empty() {
            return Vec::new();
        }
        let q_len = attn.q.len();
        let k_len = attn.k.len();
        let scale = (q_len.max(1) as f32).sqrt();
        let mut scores = vec![0.0f32; output_dim];

        for idx in 0..output_dim {
            let q = attn.q[idx % q_len];
            let k = attn.k[idx % k_len];
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            scores[idx] = ((q * k) / scale + salt_value * 0.05).tanh();
        }

        scores
    }

    fn build_attention_mix_state(
        attn: &AttentionScratchState,
        output_dim: usize,
        salt: &[u8],
    ) -> AttentionMixState {
        let scores = Self::build_attention_scores(attn, output_dim, salt);
        let values = if output_dim == 0 || attn.v.is_empty() {
            Vec::new()
        } else {
            (0..output_dim)
                .map(|idx| attn.v[idx % attn.v.len()])
                .collect()
        };
        AttentionMixState { scores, values }
    }

    fn build_resumed_attention_mix_state(
        attn: &AttentionScratchState,
        output_dim: usize,
        salt: &[u8],
        scratch: &LayerScratch,
    ) -> AttentionMixState {
        let mut mix = Self::build_attention_mix_state(attn, output_dim, salt);
        if let (Some(carry), Some(weight)) = (
            scratch.attn_score.as_ref(),
            scratch.typed_or_legacy_attention_blend(
                scratch.attn_score_blend_weight,
                ATTENTION_MIX_BLEND_WEIGHT,
            ),
        ) {
            Self::blend_lane_state(
                &mut mix.scores,
                carry,
                scratch.attn_score_lane_indices.as_deref(),
                weight,
            );
        }
        if let (Some(carry), Some(weight)) = (
            scratch.attn_value.as_ref(),
            scratch.typed_or_legacy_attention_blend(
                scratch.attn_value_blend_weight,
                ATTENTION_MIX_BLEND_WEIGHT,
            ),
        ) {
            Self::blend_lane_state(
                &mut mix.values,
                carry,
                scratch.attn_value_lane_indices.as_deref(),
                weight,
            );
        }
        mix
    }

    fn mix_attention_state(mix: &AttentionMixState, salt: &[u8]) -> Vec<f32> {
        if mix.scores.is_empty() || mix.values.is_empty() {
            return Vec::new();
        }
        let len = mix.scores.len().min(mix.values.len());
        let mut mixed = vec![0.0f32; len];

        for idx in 0..len {
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            mixed[idx] = (mix.scores[idx] * mix.values[idx] + salt_value * 0.01).tanh();
        }

        mixed
    }

    fn build_ffn_state(scratch: &LayerScratch) -> Option<FfnScratchState> {
        Some(FfnScratchState {
            gate: scratch.ffn_gate.clone()?,
            up: scratch.ffn_up.clone()?,
        })
    }

    fn build_ffn_mix_state(ffn: &FfnScratchState) -> FfnMixState {
        let len = ffn.gate.len().max(ffn.up.len());
        if len == 0 {
            return FfnMixState {
                activations: Vec::new(),
            };
        }
        let mut activations = vec![0.0f32; len];
        for idx in 0..len {
            let gate = ffn.gate[idx % ffn.gate.len()].tanh();
            let up = ffn.up[idx % ffn.up.len()];
            activations[idx] = gate * up;
        }
        FfnMixState { activations }
    }

    fn mix_ffn_state(mix: &FfnMixState, salt: &[u8]) -> Vec<f32> {
        if mix.activations.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0f32; mix.activations.len()];
        for idx in 0..mix.activations.len() {
            let salt_value = salt[idx % salt.len()] as f32 / 255.0;
            out[idx] = (mix.activations[idx] + salt_value * 0.01).tanh();
        }
        out
    }

    fn build_resumed_ffn_mix_state(
        ffn_state: &FfnScratchState,
        scratch: &LayerScratch,
    ) -> FfnMixState {
        let mut mix = Self::build_ffn_mix_state(ffn_state);
        if let Some(carry) = scratch.ffn_activation.as_ref() {
            Self::blend_lane_state(
                &mut mix.activations,
                carry,
                scratch.ffn_lane_indices.as_deref(),
                0.25,
            );
        }
        mix
    }

    fn scattered_values(preview: &[f32], lane_indices: &[usize], width: usize) -> Vec<f32> {
        if preview.is_empty() || lane_indices.is_empty() || width == 0 {
            return Vec::new();
        }
        let mut out = vec![0.0f32; width];
        for (idx, lane) in lane_indices.iter().copied().enumerate() {
            if lane < width && idx < preview.len() {
                out[lane] = preview[idx];
            }
        }
        out
    }

    fn scratch_from_resume_carry(carry: &StageCarryState) -> Result<LayerScratch> {
        let mut scratch = LayerScratch::default();
        if let Some(attention) = &carry.attention {
            attention.validate_contract_for_present_substates()?;
            scratch.attention_contract = attention.contract;
            scratch.attn_q_blend_weight = attention.blend_weight_for_q();
            scratch.attn_k_blend_weight = attention.blend_weight_for_k();
            scratch.attn_v_blend_weight = attention.blend_weight_for_v();
            scratch.attn_score_blend_weight = attention.blend_weight_for_score();
            scratch.attn_value_blend_weight = attention.blend_weight_for_value();
            if let Some(projection) = &attention.projection {
                scratch.attn_q_lane_indices = Some(projection.q_lane_indices.clone());
                scratch.attn_k_lane_indices = Some(projection.k_lane_indices.clone());
                scratch.attn_v_lane_indices = Some(projection.v_lane_indices.clone());
                scratch.attn_q = Some(Self::scattered_values(
                    &projection.q,
                    &projection.q_lane_indices,
                    projection.width,
                ));
                scratch.attn_k = Some(Self::scattered_values(
                    &projection.k,
                    &projection.k_lane_indices,
                    projection.width,
                ));
                scratch.attn_v = Some(Self::scattered_values(
                    &projection.v,
                    &projection.v_lane_indices,
                    projection.width,
                ));
            }
            if let Some(mix) = &attention.mix {
                scratch.attn_score_lane_indices = Some(mix.score_lane_indices.clone());
                scratch.attn_value_lane_indices = Some(mix.value_lane_indices.clone());
                scratch.attn_score = Some(Self::scattered_values(
                    &mix.scores,
                    &mix.score_lane_indices,
                    mix.width,
                ));
                scratch.attn_value = Some(Self::scattered_values(
                    &mix.values,
                    &mix.value_lane_indices,
                    mix.width,
                ));
            }
        }
        if let Some(ffn) = &carry.ffn {
            let lane_indices = ffn.lane_indices.clone();
            scratch.ffn_lane_indices = Some(lane_indices.clone());
            scratch.ffn_gate = Some(Self::scattered_values(
                &ffn.gate_head,
                &lane_indices,
                ffn.width,
            ));
            scratch.ffn_up = Some(Self::scattered_values(
                &ffn.up_head,
                &lane_indices,
                ffn.width,
            ));
            scratch.ffn_activation = Some(Self::scattered_values(
                &ffn.activation_head,
                &lane_indices,
                ffn.width,
            ));
        }
        Ok(scratch)
    }

    fn blend_lane_state(
        current: &mut [f32],
        carry: &[f32],
        lane_indices: Option<&[usize]>,
        weight: f32,
    ) {
        if current.is_empty() || carry.is_empty() {
            return;
        }
        if let Some(indices) = lane_indices {
            for lane in indices {
                let idx = *lane;
                if idx < current.len() && idx < carry.len() {
                    let carried = carry[idx];
                    current[idx] = (current[idx] * (1.0 - weight) + carried * weight).tanh();
                }
            }
        } else {
            for idx in 0..current.len() {
                let carried = carry[idx % carry.len()];
                current[idx] = (current[idx] * (1.0 - weight) + carried * weight).tanh();
            }
        }
    }

    fn preview(values: &[f32]) -> Vec<f32> {
        values.iter().take(TRANSIENT_PREVIEW_LEN).copied().collect()
    }

    fn transient_for_states(
        attention: Option<(&AttentionScratchState, &AttentionMixState)>,
        ffn: Option<(&FfnScratchState, &FfnMixState)>,
    ) -> Option<StageTransientState> {
        let attention = attention.map(|(attn, mix)| AttentionContinuation {
            width: attn.q.len().max(attn.k.len()).max(attn.v.len()),
            lane_indices: (0..Self::preview(&attn.q).len()).collect(),
            q_preview: Self::preview(&attn.q),
            k_preview: Self::preview(&attn.k),
            v_preview: Self::preview(&attn.v),
            score_preview: Self::preview(&mix.scores),
            value_preview: Self::preview(&mix.values),
        });
        let ffn = ffn.map(|(ffn_state, mix)| FfnContinuation {
            width: ffn_state.gate.len().max(ffn_state.up.len()),
            lane_indices: (0..Self::preview(&ffn_state.gate).len()).collect(),
            gate_preview: Self::preview(&ffn_state.gate),
            up_preview: Self::preview(&ffn_state.up),
            activation_preview: Self::preview(&mix.activations),
        });
        if attention.is_none() && ffn.is_none() {
            None
        } else {
            Some(StageTransientState { attention, ffn })
        }
    }

    fn transient_for_attention_checkpoint(
        checkpoint: Option<&AttentionExecutionCheckpoint>,
    ) -> Option<AttentionContinuation> {
        let checkpoint = checkpoint?;
        let projection = checkpoint.projection.as_ref()?;
        let mix = checkpoint.mix.as_ref()?;
        Some(AttentionContinuation {
            width: projection
                .q
                .len()
                .max(projection.k.len())
                .max(projection.v.len()),
            lane_indices: (0..Self::preview(&projection.q).len()).collect(),
            q_preview: Self::preview(&projection.q),
            k_preview: Self::preview(&projection.k),
            v_preview: Self::preview(&projection.v),
            score_preview: Self::preview(&mix.scores),
            value_preview: Self::preview(&mix.values),
        })
    }

    fn carry_for_attention_checkpoint(
        checkpoint: Option<&AttentionExecutionCheckpoint>,
    ) -> Option<CarryableAttentionState> {
        let checkpoint = checkpoint?;
        let projection = checkpoint.projection.as_ref()?;
        let mix = checkpoint.mix.as_ref()?;
        let attention_width = projection
            .q
            .len()
            .max(projection.k.len())
            .max(projection.v.len());
        Some(CarryableAttentionState {
            contract: StageResumeContractSummary::current_attention_defaults(),
            projection: Some(CarryableAttentionProjectionState {
                width: attention_width,
                q_provenance: checkpoint.q_provenance.clone(),
                k_provenance: checkpoint.k_provenance.clone(),
                v_provenance: checkpoint.v_provenance.clone(),
                q_lane_indices: (0..projection.q.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .collect(),
                k_lane_indices: (0..projection.k.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .collect(),
                v_lane_indices: (0..projection.v.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .collect(),
                q: projection
                    .q
                    .iter()
                    .take(projection.q.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .copied()
                    .collect(),
                k: projection
                    .k
                    .iter()
                    .take(projection.k.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .copied()
                    .collect(),
                v: projection
                    .v
                    .iter()
                    .take(projection.v.len().min(ATTENTION_PROJECTION_CARRY_BUDGET))
                    .copied()
                    .collect(),
            }),
            mix: Some(CarryableAttentionMixState {
                width: attention_width,
                score_provenance: checkpoint.score_provenance.clone(),
                value_provenance: checkpoint.value_provenance.clone(),
                score_lane_indices: (0..mix.scores.len().min(ATTENTION_MIX_CARRY_BUDGET)).collect(),
                value_lane_indices: (0..mix.values.len().min(ATTENTION_MIX_CARRY_BUDGET)).collect(),
                scores: mix
                    .scores
                    .iter()
                    .take(mix.scores.len().min(ATTENTION_MIX_CARRY_BUDGET))
                    .copied()
                    .collect(),
                values: mix
                    .values
                    .iter()
                    .take(mix.values.len().min(ATTENTION_MIX_CARRY_BUDGET))
                    .copied()
                    .collect(),
            }),
        })
    }

    fn carry_for_states(
        attention: Option<(&AttentionScratchState, &AttentionMixState)>,
        ffn: Option<(&FfnScratchState, &FfnMixState)>,
        policy: &StageCarryPolicy,
    ) -> Option<StageCarryState> {
        let attention = if policy.carry_attention {
            attention.map(|(attn, mix)| {
                CarryableAttentionState::from_attention(&AttentionContinuation {
                    width: attn.q.len().max(attn.k.len()).max(attn.v.len()),
                    lane_indices: (0..attn
                        .q
                        .len()
                        .max(attn.k.len())
                        .max(attn.v.len())
                        .max(mix.scores.len())
                        .max(mix.values.len()))
                        .collect(),
                    q_preview: attn.q.clone(),
                    k_preview: attn.k.clone(),
                    v_preview: attn.v.clone(),
                    score_preview: mix.scores.clone(),
                    value_preview: mix.values.clone(),
                })
            })
        } else {
            None
        };
        let ffn = if policy.carry_ffn {
            ffn.map(|(ffn_state, mix)| {
                CarryableFfnState::from_ffn(&FfnContinuation {
                    width: ffn_state.gate.len().max(ffn_state.up.len()),
                    lane_indices: (0..ffn_state
                        .gate
                        .len()
                        .max(ffn_state.up.len())
                        .max(mix.activations.len()))
                        .collect(),
                    gate_preview: ffn_state.gate.clone(),
                    up_preview: ffn_state.up.clone(),
                    activation_preview: mix.activations.clone(),
                })
            })
        } else {
            None
        };
        if attention.is_none() && ffn.is_none() {
            None
        } else {
            Some(StageCarryState { attention, ffn })
        }
    }

    fn try_apply_numeric_op(
        store: &StageTensorStore,
        state: &mut [f32],
        op: &ExecutionOp,
        salt: &[u8],
    ) -> Result<bool> {
        match op.binding {
            ExecutionBinding::F32Vector => {
                let mut decoded = Vec::with_capacity(op.tensor_names.len());
                for tensor_name in &op.tensor_names {
                    let entry = store.entry(tensor_name).ok_or_else(|| {
                        anyhow::anyhow!("Tensor {} missing from stage store", tensor_name)
                    })?;
                    let Some(vector) = Self::try_decode_f32_vector(store, entry)? else {
                        return Ok(false);
                    };
                    decoded.push(vector);
                }

                let rms_style = matches!(
                    op.kind,
                    ExecutionOpKind::AttentionNorm
                        | ExecutionOpKind::PostAttentionNorm
                        | ExecutionOpKind::FfnNorm
                        | ExecutionOpKind::PostFfnNorm
                );
                for vector in &decoded {
                    Self::apply_numeric_vector_to_state(state, vector, salt, rms_style);
                }
                Ok(true)
            }
            ExecutionBinding::F32Matrix => {
                for tensor_name in &op.tensor_names {
                    let entry = store.entry(tensor_name).ok_or_else(|| {
                        anyhow::anyhow!("Tensor {} missing from stage store", tensor_name)
                    })?;
                    let Some((rows, cols, matrix)) = Self::try_decode_f32_matrix(store, entry)?
                    else {
                        return Ok(false);
                    };
                    Self::apply_numeric_matrix_to_state(state, rows, cols, &matrix, salt);
                }
                Ok(true)
            }
            ExecutionBinding::QuantizedMatrix => {
                for tensor_name in &op.tensor_names {
                    let entry = store.entry(tensor_name).ok_or_else(|| {
                        anyhow::anyhow!("Tensor {} missing from stage store", tensor_name)
                    })?;
                    let Some((rows, cols, matrix)) = Self::try_decode_numeric_matrix(store, entry)?
                    else {
                        return Ok(false);
                    };
                    Self::apply_numeric_matrix_to_state(state, rows, cols, &matrix, salt);
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn execute_layer_program(
        store: &StageTensorStore,
        state: &mut [f32],
        layer_program: &LayerExecutionProgram,
        layer: &LayerOperatorView,
        stage_id: &str,
        resume_carry: Option<&StageCarryState>,
        carry_policy: &StageCarryPolicy,
    ) -> Result<(Option<StageTransientState>, Option<StageCarryState>)> {
        let layer_salt = format!("{stage_id}:layer:{}", layer.layer_index);
        let base = layer_salt.as_bytes();
        let mut scratch = resume_carry
            .map(Self::scratch_from_resume_carry)
            .transpose()?
            .unwrap_or_default();
        let mut last_attention_state: Option<AttentionScratchState> = None;
        let mut last_attention_mix: Option<AttentionMixState> = None;
        let mut last_attention_q_provenance: Option<AttentionCheckpointProvenance> = None;
        let mut last_attention_k_provenance: Option<AttentionCheckpointProvenance> = None;
        let mut last_attention_v_provenance: Option<AttentionCheckpointProvenance> = None;
        let mut last_attention_score_provenance: Option<AttentionCheckpointProvenance> = None;
        let mut last_attention_value_provenance: Option<AttentionCheckpointProvenance> = None;
        let mut last_ffn_state: Option<FfnScratchState> = None;
        let mut last_ffn_mix: Option<FfnMixState> = None;

        for op in &layer_program.ops {
            let mut handled_semantically = false;
            if op.tensor_names.len() == 1 {
                let tensor_name = &op.tensor_names[0];
                let entry = store.entry(tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Tensor {} missing from stage store", tensor_name)
                })?;
                match op.kind {
                    ExecutionOpKind::AttentionQ => {
                        if let Some(mut projected) =
                            Self::try_project_decoded_attention_matrix(store, entry, state)?
                        {
                            if let (Some(carry), Some(weight)) = (
                                scratch.attn_q.as_ref(),
                                scratch.typed_or_legacy_attention_blend(
                                    scratch.attn_q_blend_weight,
                                    ATTENTION_PROJECTION_BLEND_WEIGHT,
                                ),
                            ) {
                                Self::blend_lane_state(
                                    &mut projected,
                                    carry,
                                    scratch.attn_q_lane_indices.as_deref(),
                                    weight,
                                );
                            }
                            scratch.attn_q = Some(projected);
                            last_attention_q_provenance = Some(AttentionCheckpointProvenance {
                                layer_index: layer.layer_index,
                                operator_kind: "AttentionQ".into(),
                                layer_distance_to_boundary: 0,
                            });
                            if let Some(attn_state) = Self::build_attention_state(&scratch) {
                                last_attention_state = Some(attn_state);
                            }
                            handled_semantically = true;
                        }
                    }
                    ExecutionOpKind::AttentionK => {
                        if let Some(mut projected) =
                            Self::try_project_decoded_attention_matrix(store, entry, state)?
                        {
                            if let (Some(carry), Some(weight)) = (
                                scratch.attn_k.as_ref(),
                                scratch.typed_or_legacy_attention_blend(
                                    scratch.attn_k_blend_weight,
                                    ATTENTION_PROJECTION_BLEND_WEIGHT,
                                ),
                            ) {
                                Self::blend_lane_state(
                                    &mut projected,
                                    carry,
                                    scratch.attn_k_lane_indices.as_deref(),
                                    weight,
                                );
                            }
                            scratch.attn_k = Some(projected);
                            last_attention_k_provenance = Some(AttentionCheckpointProvenance {
                                layer_index: layer.layer_index,
                                operator_kind: "AttentionK".into(),
                                layer_distance_to_boundary: 0,
                            });
                            if let Some(attn_state) = Self::build_attention_state(&scratch) {
                                last_attention_state = Some(attn_state);
                            }
                            handled_semantically = true;
                        }
                    }
                    ExecutionOpKind::AttentionV => {
                        if let Some(mut projected) =
                            Self::try_project_decoded_attention_matrix(store, entry, state)?
                        {
                            if let (Some(carry), Some(weight)) = (
                                scratch.attn_v.as_ref(),
                                scratch.typed_or_legacy_attention_blend(
                                    scratch.attn_v_blend_weight,
                                    ATTENTION_PROJECTION_BLEND_WEIGHT,
                                ),
                            ) {
                                Self::blend_lane_state(
                                    &mut projected,
                                    carry,
                                    scratch.attn_v_lane_indices.as_deref(),
                                    weight,
                                );
                            }
                            scratch.attn_v = Some(projected);
                            last_attention_v_provenance = Some(AttentionCheckpointProvenance {
                                layer_index: layer.layer_index,
                                operator_kind: "AttentionV".into(),
                                layer_distance_to_boundary: 0,
                            });
                            if let Some(attn_state) = Self::build_attention_state(&scratch) {
                                last_attention_state = Some(attn_state);
                            }
                            handled_semantically = true;
                        }
                    }
                    _ => {}
                }
                if !handled_semantically {
                    if Self::try_decode_numeric_matrix(store, entry)?.is_some() {
                        match op.kind {
                            ExecutionOpKind::AttentionOut => {
                                let (attn_state, mix, out_proj) =
                                    Self::try_project_decoded_attention_output_matrix(
                                        store, entry, state, &scratch, base,
                                    )?
                                    .expect("attention output matrix was decoded above");
                                if let (Some(attn_state), Some(mix)) = (attn_state, mix) {
                                    last_attention_state = Some(attn_state);
                                    last_attention_mix = Some(mix);
                                    last_attention_score_provenance =
                                        Some(AttentionCheckpointProvenance {
                                            layer_index: layer.layer_index,
                                            operator_kind: "AttentionOut".into(),
                                            layer_distance_to_boundary: 0,
                                        });
                                    last_attention_value_provenance =
                                        Some(AttentionCheckpointProvenance {
                                            layer_index: layer.layer_index,
                                            operator_kind: "AttentionOut".into(),
                                            layer_distance_to_boundary: 0,
                                        });
                                }
                                Self::blend_projection_into_state(
                                    state,
                                    &out_proj,
                                    base,
                                    ATTENTION_PROJECTION_BLEND_WEIGHT,
                                );
                                handled_semantically = true;
                            }
                            ExecutionOpKind::FfnGate => {
                                let mut projected =
                                    Self::try_project_decoded_ffn_matrix(store, entry, state)?
                                        .expect("ffn gate matrix was decoded above");
                                if let Some(carry) = scratch.ffn_gate.as_ref() {
                                    Self::blend_lane_state(
                                        &mut projected,
                                        carry,
                                        scratch.ffn_lane_indices.as_deref(),
                                        0.35,
                                    );
                                }
                                scratch.ffn_gate = Some(projected);
                                handled_semantically = true;
                            }
                            ExecutionOpKind::FfnUp => {
                                let mut projected =
                                    Self::try_project_decoded_ffn_matrix(store, entry, state)?
                                        .expect("ffn up matrix was decoded above");
                                if let Some(carry) = scratch.ffn_up.as_ref() {
                                    Self::blend_lane_state(
                                        &mut projected,
                                        carry,
                                        scratch.ffn_lane_indices.as_deref(),
                                        0.35,
                                    );
                                }
                                scratch.ffn_up = Some(projected);
                                handled_semantically = true;
                            }
                            ExecutionOpKind::FfnDown => {
                                let (ffn_state, mix, down_proj) =
                                    Self::try_project_decoded_ffn_down_matrix(
                                        store, entry, state, &scratch, base,
                                    )?
                                    .expect("ffn down matrix was decoded above");
                                if let (Some(ffn_state), Some(mix)) = (ffn_state, mix) {
                                    last_ffn_state = Some(ffn_state);
                                    last_ffn_mix = Some(mix);
                                }
                                Self::blend_projection_into_state(state, &down_proj, base, 0.4);
                                handled_semantically = true;
                            }
                            ExecutionOpKind::InputGate => {
                                scratch.input_gate = Self::try_project_decoded_input_gate_matrix(
                                    store, entry, state,
                                )?;
                                handled_semantically = true;
                            }
                            ExecutionOpKind::Projection => {
                                let proj = Self::try_project_decoded_projection_matrix(
                                    store, entry, state, &scratch,
                                )?
                                .expect("projection matrix was decoded above");
                                Self::blend_projection_into_state(state, &proj, base, 0.25);
                                handled_semantically = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
            if handled_semantically {
                continue;
            }
            if Self::try_apply_numeric_op(store, state, op, base)? {
                continue;
            }
            for tensor_name in &op.tensor_names {
                let entry = store.entry(tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Tensor {} missing from stage store", tensor_name)
                })?;
                Self::apply_scalar_to_state(state, Self::entry_scalar(store, entry)?, base);
            }
        }

        let shape_mix = [
            layer_program.hidden_dim.unwrap_or(0) as f32,
            layer_program.q_out_dim.unwrap_or(0) as f32,
            layer_program.k_out_dim.unwrap_or(0) as f32,
            layer_program.v_out_dim.unwrap_or(0) as f32,
            layer_program.ffn_inner_dim.unwrap_or(0) as f32,
        ];
        for (idx, value) in state.iter_mut().enumerate() {
            *value = (*value
                + ((shape_mix[idx % shape_mix.len()] % 4096.0) / 4096.0)
                + (layer.layer_index as f32 * 0.001))
                .tanh();
        }
        let attention_pair = match (last_attention_state.as_ref(), last_attention_mix.as_ref()) {
            (Some(state), Some(mix)) => Some((state, mix)),
            _ => None,
        };
        let ffn_pair = match (last_ffn_state.as_ref(), last_ffn_mix.as_ref()) {
            (Some(state), Some(mix)) => Some((state, mix)),
            _ => None,
        };
        let attention_checkpoint =
            attention_pair.map(|(state, mix)| AttentionExecutionCheckpoint {
                projection: Some(state.clone()),
                mix: Some(mix.clone()),
                q_provenance: last_attention_q_provenance,
                k_provenance: last_attention_k_provenance,
                v_provenance: last_attention_v_provenance,
                score_provenance: last_attention_score_provenance,
                value_provenance: last_attention_value_provenance,
            });
        let ffn_transient = Self::transient_for_states(None, ffn_pair).and_then(|state| state.ffn);
        let ffn_carry =
            Self::carry_for_states(None, ffn_pair, carry_policy).and_then(|state| state.ffn);
        Ok((
            Some(StageTransientState {
                attention: Self::transient_for_attention_checkpoint(attention_checkpoint.as_ref()),
                ffn: ffn_transient,
            })
            .filter(|state| state.attention.is_some() || state.ffn.is_some()),
            Some(StageCarryState {
                attention: if carry_policy.carry_attention {
                    Self::carry_for_attention_checkpoint(attention_checkpoint.as_ref())
                } else {
                    None
                },
                ffn: ffn_carry,
            })
            .filter(|state| state.attention.is_some() || state.ffn.is_some()),
        ))
    }

    fn sample_text(bytes: &[u8], max_tokens: Option<u32>) -> String {
        let take = max_tokens.unwrap_or(16).clamp(4, 32) as usize;
        let mut out = String::with_capacity(take);
        for idx in 0..take {
            let byte = bytes[idx % bytes.len()];
            let ch = SKETCH_VOCAB[(byte as usize) % SKETCH_VOCAB.len()] as char;
            out.push(ch);
        }
        out.trim().to_string()
    }
}

impl Default for PackedResidencySketchBackend {
    fn default() -> Self {
        Self::new("stage-required.index.json")
    }
}

impl ArtifactBackedToyBackend {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            artifact_path: path.into(),
            layout: None,
            artifact: None,
        }
    }

    fn layout(&self) -> Result<&StageLayout> {
        self.layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No stage layout loaded"))
    }

    fn artifact(&self) -> Result<&ToyShardArtifact> {
        self.artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No shard artifact loaded"))
    }

    fn prompt_embedding(prompt: &str, hidden_dim: usize) -> Vec<f32> {
        let mut state = vec![0.0f32; hidden_dim];
        if prompt.is_empty() {
            state[0] = 1.0;
            return state;
        }

        for (idx, byte) in prompt.bytes().enumerate() {
            let slot = idx % hidden_dim;
            let normalized = (byte as f32) / 255.0;
            state[slot] += normalized;
            state[(slot * 5 + 1) % hidden_dim] += normalized * 0.35;
        }

        let scale = 1.0 / prompt.len() as f32;
        for value in &mut state {
            *value *= scale;
        }
        state
    }

    fn apply_layer(state: &[f32], layer: &ToyLayerArtifact) -> Result<Vec<f32>> {
        let dim = state.len();
        if layer.bias.len() != dim || layer.weights.len() != dim {
            bail!("Layer {} shape mismatch", layer.index);
        }

        let mut out = vec![0.0f32; dim];
        for row in 0..dim {
            if layer.weights[row].len() != dim {
                bail!("Layer {} row {} shape mismatch", layer.index, row);
            }
            let mut acc = layer.bias[row];
            for (col, value) in state.iter().enumerate() {
                acc += layer.weights[row][col] * value;
            }
            out[row] = acc.tanh();
        }
        Ok(out)
    }

    fn encode_hidden_state(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        if bytes.len() != hidden_dim * 4 {
            bail!(
                "Hidden-state byte length {} does not match hidden_dim {}",
                bytes.len(),
                hidden_dim
            );
        }
        let mut values = Vec::with_capacity(hidden_dim);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }

    fn project_text(
        projection: &ToyProjectionArtifact,
        state: &[f32],
        max_tokens: Option<u32>,
    ) -> Result<String> {
        let dim = state.len();
        if projection.bias.len() != projection.vocab.len()
            || projection.weights.len() != projection.vocab.len()
        {
            bail!("Projection shape mismatch");
        }

        let take = max_tokens.unwrap_or(12).clamp(4, 16) as usize;
        let mut out = String::with_capacity(take);

        for token_idx in 0..take {
            let mut best_score = f32::MIN;
            let mut best_token = "";
            for row in 0..projection.vocab.len() {
                if projection.weights[row].len() != dim {
                    bail!("Projection row {} shape mismatch", row);
                }
                let mut score = projection.bias[row];
                for col in 0..dim {
                    let modifier = 1.0 + (token_idx as f32 * 0.015);
                    score += projection.weights[row][col] * state[col] * modifier;
                }
                if score > best_score {
                    best_score = score;
                    best_token = &projection.vocab[row];
                }
            }
            out.push_str(best_token);
        }

        Ok(out)
    }
}

impl Default for ArtifactBackedToyBackend {
    fn default() -> Self {
        Self::new("artifact.json")
    }
}

impl StageForwardBackend for ArtifactBackedToyBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let artifact: ToyShardArtifact =
            serde_json::from_str(&fs::read_to_string(&self.artifact_path).map_err(|err| {
                anyhow::anyhow!("Failed to read {}: {err}", self.artifact_path.display())
            })?)?;

        if artifact.model_id != layout.model_id {
            bail!(
                "Artifact model {} does not match layout model {}",
                artifact.model_id,
                layout.model_id
            );
        }
        if artifact.start_layer != layout.start_layer || artifact.end_layer != layout.end_layer {
            bail!(
                "Artifact layer range {}-{} does not match layout {}-{}",
                artifact.start_layer,
                artifact.end_layer,
                layout.start_layer,
                layout.end_layer
            );
        }
        if artifact.layers.len() as u32 != (layout.end_layer - layout.start_layer + 1) {
            bail!("Artifact layer count does not match layout range");
        }

        self.layout = Some(layout);
        self.artifact = Some(artifact);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        let artifact = self.artifact()?;
        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }

        let mut state = Self::prompt_embedding(prompt, artifact.hidden_dim);
        for layer in &artifact.layers {
            state = Self::apply_layer(&state, layer)?;
        }

        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim: artifact.hidden_dim,
            bytes: Self::encode_hidden_state(&state),
            prompt_text: Some(prompt.to_string()),
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let layout = self.layout()?;
        let artifact = self.artifact()?;
        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }

        let mut state = Self::decode_hidden_state(&input.bytes, artifact.hidden_dim)?;
        for layer in &artifact.layers {
            state = Self::apply_layer(&state, layer)?;
        }

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());

        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: artifact.hidden_dim,
            bytes: Self::encode_hidden_state(&state),
            prompt_text: input.prompt_text,
            max_tokens: input.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let layout = self.layout()?;
        let artifact = self.artifact()?;
        if !layout.is_tail {
            bail!("Only the tail stage may sample output");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail sampling requires hidden-state payloads");
        }

        let projection = artifact
            .projection
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tail artifact is missing a projection head"))?;
        let state = Self::decode_hidden_state(&input.bytes, artifact.hidden_dim)?;
        let text = Self::project_text(projection, &state, input.max_tokens)?;

        Ok(StageSample {
            request_id: input.request_id,
            model_id: layout.model_id.clone(),
            completion_tokens: text.chars().count() as u32,
            token_ids: StageSample::text_token_ids(&text),
            text,
        })
    }
}

impl StageForwardBackend for PackedResidencySketchBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let store = StageTensorStore::load(&self.index_path)?;
        store.validate_offsets()?;

        let role_ok = match store.artifact.index.role.as_str() {
            "head" => layout.is_head && !layout.is_tail,
            "tail" => layout.is_tail && !layout.is_head,
            "middle" => !layout.is_head && !layout.is_tail,
            _ => false,
        };
        if !role_ok {
            bail!(
                "Packed stage role {} does not match layout head={} tail={}",
                store.artifact.index.role,
                layout.is_head,
                layout.is_tail
            );
        }

        if store.artifact.index.model_name != layout.model_id
            && store.artifact.index.architecture != layout.model_id
        {
            // Keep the check permissive enough for the current lab layouts.
        }

        let model_view = store.model_view();
        self.layout = Some(layout);
        self.model_view = Some(model_view);
        self.store = Some(store);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        let model_view = self.model_view()?;
        let store = self.store()?;
        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }
        if model_view.prompt_ingress.is_empty() {
            bail!("Head packed stage is missing prompt ingress tensors");
        }

        let width = Self::hidden_dim(model_view);
        let mut state = Self::prompt_state(prompt, model_view, &layout.stage_id, width);
        let base_salt = layout.stage_id.as_bytes();
        for entry in &model_view.prompt_ingress {
            Self::apply_scalar_to_state(&mut state, Self::entry_scalar(store, entry)?, base_salt);
        }
        for entry in &model_view.positional {
            Self::apply_scalar_to_state(&mut state, Self::entry_scalar(store, entry)?, base_salt);
        }
        for entry in &model_view.shared_auxiliary {
            Self::apply_scalar_to_state(&mut state, Self::entry_scalar(store, entry)?, base_salt);
        }
        let mut last_transient = None;
        let mut last_carry = None;
        let execution_layer_count = self
            .debug_layer_cap
            .unwrap_or(model_view.execution_programs.len())
            .min(model_view.execution_programs.len());
        let carry_policy =
            StageCarryPolicy::for_execution_boundary(layout, None, &model_view.execution_programs);
        for (layer, program) in model_view
            .operator_layers
            .iter()
            .zip(model_view.execution_programs.iter())
            .take(execution_layer_count)
        {
            let (transient, carry) = Self::execute_layer_program(
                store,
                &mut state,
                program,
                layer,
                &layout.stage_id,
                None,
                &carry_policy,
            )?;
            last_transient = Self::merge_transient_checkpoint(last_transient, transient);
            last_carry = Self::merge_stage_carry(last_carry, carry);
        }
        let bytes = Self::encode_hidden_state(&state);
        let continuation = Some(Self::continuation_for_view(
            model_view,
            execution_layer_count as u32,
            if execution_layer_count < model_view.execution_programs.len() {
                Some(layout.start_layer + execution_layer_count as u32)
            } else {
                None
            },
        ));
        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim: width,
            bytes,
            prompt_text: Some(prompt.to_string()),
            max_tokens,
            continuation,
            transient: last_transient,
            carry: last_carry,
        })
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let layout = self.layout()?;
        let model_view = self.model_view()?;
        let store = self.store()?;
        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());
        let decoded =
            Self::decode_hidden_state(&input.bytes, input.hidden_dim).unwrap_or_else(|_| {
                Self::continue_state(
                    &Self::prompt_state(
                        input.prompt_text.as_deref().unwrap_or_default(),
                        model_view,
                        &layout.stage_id,
                        input.hidden_dim,
                    ),
                    &layout.stage_id,
                )
            });
        let mut state = Self::continue_state(&decoded, &layout.stage_id);
        let base_salt = layout.stage_id.as_bytes();
        for entry in &model_view.positional {
            Self::apply_scalar_to_state(&mut state, Self::entry_scalar(store, entry)?, base_salt);
        }
        for entry in &model_view.shared_auxiliary {
            Self::apply_scalar_to_state(&mut state, Self::entry_scalar(store, entry)?, base_salt);
        }
        let mut last_transient = None;
        let mut last_carry = None;
        let execution_layer_count = self
            .debug_layer_cap
            .unwrap_or(model_view.execution_programs.len())
            .min(model_view.execution_programs.len());
        let resume_carry = input.carry.clone().or_else(|| {
            input.transient.as_ref().map(|transient| {
                StageCarryState::from_transient(transient, &StageCarryPolicy::full())
            })
        });
        let continuation = Some(Self::continuation_for_view(
            model_view,
            execution_layer_count as u32,
            if execution_layer_count < model_view.execution_programs.len() {
                Some(layout.start_layer + execution_layer_count as u32)
            } else {
                None
            },
        ));
        let carry_policy = StageCarryPolicy::for_execution_boundary(
            layout,
            continuation.as_ref(),
            &model_view.execution_programs,
        );
        for (layer, program) in model_view
            .operator_layers
            .iter()
            .zip(model_view.execution_programs.iter())
            .take(execution_layer_count)
        {
            let resume_seed = if last_transient.is_none() {
                resume_carry.as_ref()
            } else {
                None
            };
            let (transient, carry) = Self::execute_layer_program(
                store,
                &mut state,
                program,
                layer,
                &layout.stage_id,
                resume_seed,
                &carry_policy,
            )?;
            last_transient = Self::merge_transient_checkpoint(last_transient, transient);
            last_carry = Self::merge_stage_carry(last_carry, carry);
        }
        let bytes = Self::encode_hidden_state(&state);

        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: input.hidden_dim,
            bytes,
            prompt_text: input.prompt_text,
            max_tokens: input.max_tokens,
            continuation,
            transient: last_transient,
            carry: last_carry,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let layout = self.layout()?;
        let store = self.store()?;
        let model_view = self.model_view()?;
        if !layout.is_tail {
            bail!("Only the tail stage may sample output");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail sampling requires hidden-state payloads");
        }
        if model_view.tail_only.is_empty() {
            bail!("Tail packed stage is missing tail-only tensors");
        }

        let mut state =
            Self::decode_hidden_state(&input.bytes, input.hidden_dim).unwrap_or_else(|_| {
                Self::continue_state(&vec![0.0; input.hidden_dim], &layout.stage_id)
            });
        let tail_salt = format!("{}:tail", layout.stage_id);
        for entry in &model_view.tail_only {
            Self::apply_scalar_to_state(
                &mut state,
                Self::entry_scalar(store, entry)?,
                tail_salt.as_bytes(),
            );
        }
        let bytes = Self::encode_hidden_state(&state);
        let text = Self::sample_text(&bytes, input.max_tokens);
        let completion_tokens = text.chars().count() as u32;
        Ok(StageSample {
            request_id: input.request_id,
            model_id: layout.model_id.clone(),
            token_ids: StageSample::text_token_ids(&text),
            text,
            completion_tokens,
        })
    }
}

pub fn write_sample_toy_artifacts(dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    fs::create_dir_all(dir)?;

    let head = ToyShardArtifact {
        model_id: "toy-linear-4l".into(),
        hidden_dim: 4,
        start_layer: 0,
        end_layer: 1,
        total_layers: 4,
        layers: vec![
            ToyLayerArtifact {
                index: 0,
                weights: vec![
                    vec![0.90, 0.10, 0.00, 0.00],
                    vec![0.05, 0.85, 0.05, 0.00],
                    vec![0.00, 0.10, 0.80, 0.10],
                    vec![0.00, 0.00, 0.15, 0.75],
                ],
                bias: vec![0.05, 0.02, -0.01, 0.03],
            },
            ToyLayerArtifact {
                index: 1,
                weights: vec![
                    vec![0.82, 0.12, 0.06, 0.00],
                    vec![0.04, 0.88, 0.06, 0.02],
                    vec![0.03, 0.09, 0.79, 0.09],
                    vec![0.01, 0.02, 0.12, 0.85],
                ],
                bias: vec![0.01, 0.03, 0.02, -0.02],
            },
        ],
        projection: None,
    };

    let tail = ToyShardArtifact {
        model_id: "toy-linear-4l".into(),
        hidden_dim: 4,
        start_layer: 2,
        end_layer: 3,
        total_layers: 4,
        layers: vec![
            ToyLayerArtifact {
                index: 2,
                weights: vec![
                    vec![0.84, 0.08, 0.05, 0.03],
                    vec![0.06, 0.86, 0.05, 0.03],
                    vec![0.02, 0.08, 0.83, 0.07],
                    vec![0.04, 0.02, 0.10, 0.84],
                ],
                bias: vec![0.00, 0.01, 0.04, 0.02],
            },
            ToyLayerArtifact {
                index: 3,
                weights: vec![
                    vec![0.88, 0.06, 0.03, 0.03],
                    vec![0.03, 0.90, 0.03, 0.04],
                    vec![0.02, 0.05, 0.89, 0.04],
                    vec![0.04, 0.03, 0.05, 0.88],
                ],
                bias: vec![0.03, 0.00, 0.02, 0.01],
            },
        ],
        projection: Some(ToyProjectionArtifact {
            vocab: vec![
                "A".into(),
                "B".into(),
                "C".into(),
                "D".into(),
                "E".into(),
                "F".into(),
            ],
            weights: vec![
                vec![0.9, 0.3, 0.2, 0.1],
                vec![0.2, 0.8, 0.1, 0.4],
                vec![0.1, 0.2, 0.9, 0.3],
                vec![0.3, 0.1, 0.3, 0.8],
                vec![0.6, 0.4, 0.2, 0.1],
                vec![0.1, 0.5, 0.4, 0.7],
            ],
            bias: vec![0.02, -0.01, 0.03, 0.01, 0.00, 0.02],
        }),
    };

    let full = ToyShardArtifact {
        model_id: "toy-linear-4l".into(),
        hidden_dim: 4,
        start_layer: 0,
        end_layer: 3,
        total_layers: 4,
        layers: head
            .layers
            .iter()
            .chain(tail.layers.iter())
            .cloned()
            .collect(),
        projection: tail.projection.clone(),
    };

    let head_path = dir.join("toy-head.json");
    let tail_path = dir.join("toy-tail.json");
    let full_path = dir.join("toy-full.json");
    fs::write(&head_path, serde_json::to_vec_pretty(&head)?)?;
    fs::write(&tail_path, serde_json::to_vec_pretty(&tail)?)?;
    fs::write(&full_path, serde_json::to_vec_pretty(&full)?)?;

    Ok((head_path, tail_path, full_path))
}

pub fn run_artifact_single_node_reference(
    artifact_path: &Path,
    prompt: &str,
    max_tokens: Option<u32>,
) -> Result<StageSample> {
    let artifact: ToyShardArtifact = serde_json::from_str(&fs::read_to_string(artifact_path)?)?;
    let mut state = ArtifactBackedToyBackend::prompt_embedding(prompt, artifact.hidden_dim);
    for layer in &artifact.layers {
        state = ArtifactBackedToyBackend::apply_layer(&state, layer)?;
    }
    let projection = artifact
        .projection
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Full artifact is missing a projection head"))?;
    let text = ArtifactBackedToyBackend::project_text(projection, &state, max_tokens)?;
    Ok(StageSample {
        request_id: "artifact-single-reference".into(),
        model_id: artifact.model_id,
        completion_tokens: text.chars().count() as u32,
        token_ids: StageSample::text_token_ids(&text),
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, MetadataValue, StageSplit, TensorInfo};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    fn head_layout() -> StageLayout {
        StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-13".into(),
            start_layer: 0,
            end_layer: 13,
            is_head: true,
            is_tail: false,
        }
    }

    fn tail_layout() -> StageLayout {
        StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "14-27".into(),
            start_layer: 14,
            end_layer: 27,
            is_head: false,
            is_tail: true,
        }
    }

    fn toy_head_layout() -> StageLayout {
        StageLayout {
            model_id: "toy-linear-4l".into(),
            stage_id: "0-1".into(),
            start_layer: 0,
            end_layer: 1,
            is_head: true,
            is_tail: false,
        }
    }

    fn toy_tail_layout() -> StageLayout {
        StageLayout {
            model_id: "toy-linear-4l".into(),
            stage_id: "2-3".into(),
            start_layer: 2,
            end_layer: 3,
            is_head: false,
            is_tail: true,
        }
    }

    fn packed_head_layout() -> StageLayout {
        StageLayout {
            model_id: "Toy".into(),
            stage_id: "packed-head".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: true,
            is_tail: false,
        }
    }

    fn packed_tail_layout() -> StageLayout {
        StageLayout {
            model_id: "Toy".into(),
            stage_id: "packed-tail".into(),
            start_layer: 1,
            end_layer: 1,
            is_head: false,
            is_tail: true,
        }
    }

    #[test]
    fn begin_prompt_emits_hidden_state_payload() {
        let mut backend = DeterministicStubBackend::default();
        backend.load_layout(head_layout()).unwrap();

        let tensor = backend
            .begin_prompt("req-1", "hello", Some(32), 2048)
            .unwrap();

        assert_eq!(tensor.kind, PayloadKind::HiddenState);
        assert_eq!(tensor.prompt_text.as_deref(), Some("hello"));
        assert_eq!(tensor.stage_trace, vec!["0-13"]);
        assert_eq!(tensor.hidden_state_len(), 2048);
    }

    #[test]
    fn continue_forward_requires_hidden_state() {
        let mut backend = DeterministicStubBackend::default();
        backend.load_layout(tail_layout()).unwrap();

        let err = backend
            .continue_forward(StageTensor {
                request_id: "req-1".into(),
                kind: PayloadKind::PromptIngress,
                stage_trace: vec![],
                hidden_dim: 2048,
                bytes: vec![],
                prompt_text: Some("hello".into()),
                max_tokens: Some(32),
                continuation: None,
                transient: None,
                carry: None,
            })
            .unwrap_err()
            .to_string();

        assert!(err.contains("hidden-state payloads"));
    }

    #[test]
    fn two_stage_roundtrip_samples_from_tail() {
        let mut head = DeterministicStubBackend::default();
        head.load_layout(head_layout()).unwrap();
        let mut tail = DeterministicStubBackend::default();
        tail.load_layout(tail_layout()).unwrap();

        let head_tensor = head
            .begin_prompt("req-1", "reply exactly STAGE LAB", Some(64), 2048)
            .unwrap();
        let tail_tensor = tail.continue_forward(head_tensor).unwrap();
        let sample = tail.sample_tail(tail_tensor).unwrap();

        assert!(sample.text.contains("STAGE LAB"));
        assert!(sample.text.contains("0-13 -> 14-27"));
        assert_eq!(sample.model_id, "gemma-4-e4b-q4");
    }

    #[test]
    fn toy_linear_two_stage_matches_single_node_reference() {
        let prompt = "reply exactly STAGE LAB";

        let mut head = ToyLinearBackend::default();
        head.load_layout(toy_head_layout()).unwrap();
        let mut tail = ToyLinearBackend::default();
        tail.load_layout(toy_tail_layout()).unwrap();

        let stage1 = head
            .begin_prompt("toy-req", prompt, Some(12), TOY_HIDDEN_DIM)
            .unwrap();
        let stage2 = tail.continue_forward(stage1).unwrap();
        let distributed = tail.sample_tail(stage2).unwrap();
        let single = run_toy_single_node_reference(prompt, Some(12));

        assert_eq!(distributed.text, single.text);
        assert_eq!(distributed.completion_tokens, single.completion_tokens);
    }

    #[test]
    fn artifact_backed_two_stage_matches_single_node_reference() {
        let temp = tempdir().unwrap();
        let (head_path, tail_path, full_path) = write_sample_toy_artifacts(temp.path()).unwrap();
        let prompt = "reply exactly STAGE LAB";

        let mut head = ArtifactBackedToyBackend::new(head_path);
        head.load_layout(StageLayout {
            model_id: "toy-linear-4l".into(),
            stage_id: "0-1".into(),
            start_layer: 0,
            end_layer: 1,
            is_head: true,
            is_tail: false,
        })
        .unwrap();

        let mut tail = ArtifactBackedToyBackend::new(tail_path);
        tail.load_layout(StageLayout {
            model_id: "toy-linear-4l".into(),
            stage_id: "2-3".into(),
            start_layer: 2,
            end_layer: 3,
            is_head: false,
            is_tail: true,
        })
        .unwrap();

        let stage1 = head
            .begin_prompt("artifact-req", prompt, Some(12), 4)
            .unwrap();
        let stage2 = tail.continue_forward(stage1).unwrap();
        let distributed = tail.sample_tail(stage2).unwrap();
        let single = run_artifact_single_node_reference(&full_path, prompt, Some(12)).unwrap();

        assert_eq!(distributed.text, single.text);
        assert_eq!(distributed.completion_tokens, single.completion_tokens);
    }

    #[test]
    fn runtime_bundle_loader_roundtrips_manifest_outputs() {
        let temp = tempdir().unwrap();
        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        assert_eq!(bundle.model_name, "Toy");
        assert_eq!(bundle.architecture, "toy");
        assert_eq!(bundle.stages.len(), 2);
        assert_eq!(bundle.stage(0).unwrap().role, "head");
        assert_eq!(bundle.stage(1).unwrap().role, "tail");
        assert!(
            bundle
                .stage(0)
                .unwrap()
                .required
                .contains("token_embd.weight")
        );
        assert!(
            bundle
                .stage(1)
                .unwrap()
                .required
                .contains("output_norm.weight")
        );
        assert!(
            bundle
                .stage(1)
                .unwrap()
                .optional
                .contains("token_embd.weight")
        );
        assert_eq!(bundle.stage(0).unwrap().required_slices.len(), 2);
        assert_eq!(bundle.stage(1).unwrap().required_slices.len(), 1);
        assert_eq!(bundle.stage(1).unwrap().optional_slices.len(), 1);
        bundle.validate_against_gguf(&file).unwrap();
    }

    #[test]
    fn stage_residency_adapter_reads_required_tensor_bytes() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(500).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 5,
            file_size_bytes: 500,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 0).unwrap();
        let token_embd = adapter.read_required_tensor("token_embd.weight").unwrap();
        let blk0 = adapter.read_required_tensor("blk.0.attn_q.weight").unwrap();

        assert_eq!(token_embd.len(), 100);
        assert_eq!(blk0.len(), 100);
        assert_eq!(token_embd[0], bytes[0]);
        assert_eq!(token_embd[99], bytes[99]);
        assert_eq!(blk0[0], bytes[100]);
        assert_eq!(blk0[99], bytes[199]);
    }

    #[test]
    fn packed_stage_artifact_roundtrips_required_tensors() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 0).unwrap();
        let packed = adapter.pack_required_tensors(temp.path()).unwrap();
        let reopened = PackedStageArtifact::load(&packed.index_path).unwrap();

        assert_eq!(reopened.index.tensor_count, 2);
        let token_embd = reopened.read_tensor("token_embd.weight").unwrap();
        let blk0 = reopened.read_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(token_embd.len(), 100);
        assert_eq!(blk0.len(), 100);
        assert_eq!(token_embd[0], bytes[0]);
        assert_eq!(blk0[0], bytes[100]);
    }

    #[test]
    fn stage_tensor_store_validates_and_reads_entries() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 0).unwrap();
        let packed = adapter.pack_required_tensors(temp.path()).unwrap();

        let store = StageTensorStore::load(&packed.index_path).unwrap();
        store.validate_offsets().unwrap();
        assert!(store.contains("token_embd.weight"));
        assert!(store.contains("blk.0.attn_q.weight"));
        assert!(!store.contains("output_norm.weight"));
        assert_eq!(store.tensor_count(), 2);
        let token_embd = store.read("token_embd.weight").unwrap();
        let blk0 = store.read("blk.0.attn_q.weight").unwrap();
        assert_eq!(token_embd[0], bytes[0]);
        assert_eq!(blk0[0], bytes[100]);
        let view = store.model_view();
        assert_eq!(view.prompt_ingress.len(), 1);
        assert_eq!(view.layers.len(), 1);
        assert_eq!(view.operator_layers.len(), 1);
        assert_eq!(view.execution_layers.len(), 1);
        assert_eq!(view.execution_programs.len(), 1);
        assert_eq!(view.layers[0].layer_index, 0);
        assert_eq!(view.layers[0].tensors.len(), 1);
        assert!(view.operator_layers[0].attn_q.is_some());
        assert!(view.execution_layers[0].q_out_dim.is_some());
        assert!(!view.execution_layers[0].runnable_sketch);
        assert!(view.execution_programs[0].ops.is_empty());
    }

    #[test]
    fn packed_residency_backend_runs_two_stage_roundtrip() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let head_adapter =
            StageResidencyAdapter::from_parts(bundle.clone(), gguf_path.clone(), file.clone(), 0)
                .unwrap();
        let tail_adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 1).unwrap();

        let head_pack_dir = temp.path().join("packed-head");
        let tail_pack_dir = temp.path().join("packed-tail");
        let head_pack = head_adapter.pack_required_tensors(&head_pack_dir).unwrap();
        let tail_pack = tail_adapter.pack_required_tensors(&tail_pack_dir).unwrap();

        let mut head = PackedResidencySketchBackend::new(head_pack.index_path);
        head.load_layout(packed_head_layout()).unwrap();
        let mut tail = PackedResidencySketchBackend::new(tail_pack.index_path);
        tail.load_layout(packed_tail_layout()).unwrap();

        let stage1 = head
            .begin_prompt("packed-req", "reply exactly STAGE PACK", Some(12), 0)
            .unwrap();
        let stage1_cont = stage1.continuation.clone().unwrap();
        assert_eq!(stage1_cont.version, 1);
        assert_eq!(stage1_cont.stage_role, "head");
        assert_eq!(stage1_cont.completed_layers, 1);
        assert_eq!(stage1_cont.operator_layers, 1);
        assert_eq!(stage1_cont.next_layer_index, None);
        assert!(stage1.transient.is_none());
        let stage2 = tail.continue_forward(stage1).unwrap();
        let stage2_cont = stage2.continuation.clone().unwrap();
        assert_eq!(stage2_cont.version, 1);
        assert_eq!(stage2_cont.stage_role, "tail");
        assert_eq!(
            stage2_cont.completed_layers,
            stage2_cont.operator_layers as u32
        );
        assert_eq!(stage2_cont.next_layer_index, None);
        assert!(stage2.transient.is_none());
        let sample = tail.sample_tail(stage2).unwrap();

        assert!(!sample.text.is_empty());
        assert!(sample.completion_tokens > 0);
    }

    #[test]
    fn packed_backend_debug_layer_cap_updates_continuation() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let head_adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 0).unwrap();
        let head_pack_dir = temp.path().join("packed-head");
        let head_pack = head_adapter.pack_required_tensors(&head_pack_dir).unwrap();

        let mut head =
            PackedResidencySketchBackend::new(head_pack.index_path).with_debug_layer_cap(0);
        head.load_layout(packed_head_layout()).unwrap();

        let stage1 = head
            .begin_prompt("packed-req", "reply exactly STAGE PACK", Some(12), 0)
            .unwrap();
        let cont = stage1.continuation.unwrap();
        assert_eq!(cont.completed_layers, 0);
        assert_eq!(cont.next_layer_index, Some(0));
        assert!(stage1.transient.is_none());
    }

    #[test]
    fn stage_state_envelope_roundtrips_tensor_state() {
        let tensor = StageTensor {
            request_id: "req-1".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["head".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(1),
                completed_layers: 1,
                operator_layers: 2,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 4,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, 0.2],
                    k_preview: vec![0.3, 0.4],
                    v_preview: vec![0.5, 0.6],
                    score_preview: vec![0.7, 0.8],
                    value_preview: vec![0.9, 1.0],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let envelope = StageStateEnvelope::from_tensor(&tensor);
        assert!(!envelope.is_empty());

        let mut stripped = tensor.clone();
        stripped.continuation = None;
        stripped.transient = None;
        stripped.carry = None;
        envelope.apply_to_tensor(&mut stripped);

        assert_eq!(stripped.continuation, tensor.continuation);
        assert_eq!(stripped.transient, tensor.transient);
        assert_eq!(stripped.carry, tensor.carry);
    }

    #[test]
    fn stage_forward_frame_roundtrips_tensor_and_validates() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-2".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: None,
                ffn: Some(FfnContinuation {
                    width: 4,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };

        let frame = StageForwardFrame::from_tensor(
            "gemma-4-e4b-q4",
            &layout,
            tensor.clone(),
            Some("21-41".into()),
        );
        frame.validate().unwrap();
        let summary = frame.summary();
        assert_eq!(summary.request_id, "req-2");
        assert_eq!(summary.source_stage_role, "head");
        assert_eq!(summary.target_stage_id.as_deref(), Some("21-41"));
        assert_eq!(summary.completed_layers, Some(21));
        assert!(summary.has_transient);
        assert!(!summary.has_attention_transient);
        assert!(summary.has_ffn_transient);

        let roundtrip = frame.into_tensor();
        assert_eq!(roundtrip, tensor);
    }

    #[test]
    fn stage_carry_policy_is_derived_from_boundary() {
        let head_layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let middle_layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let tail_layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "31-41".into(),
            start_layer: 31,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        };
        let continuation = StageContinuation {
            version: 1,
            stage_role: "head".into(),
            next_layer_index: Some(21),
            completed_layers: 21,
            operator_layers: 21,
            has_attention_path: true,
            has_ffn_path: true,
            has_projection_path: true,
        };

        let head = StageCarryPolicy::for_boundary(&head_layout, Some(&continuation));
        assert!(head.carry_attention);
        assert!(!head.carry_ffn);

        let middle = StageCarryPolicy::for_boundary(&middle_layout, Some(&continuation));
        assert!(middle.carry_attention);
        assert!(middle.carry_ffn);

        let tail = StageCarryPolicy::for_boundary(&tail_layout, Some(&continuation));
        assert!(!tail.carry_attention);
        assert!(!tail.carry_ffn);
    }

    #[test]
    fn transfer_frame_for_layout_uses_boundary_policy() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-4".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: Some(FfnContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };

        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let transfer = frame.to_transfer_frame_for_layout(&layout);
        transfer.validate().unwrap();

        let carry = transfer.state.carry.as_ref().unwrap();
        assert!(carry.attention.is_some());
        assert!(carry.ffn.is_none());
    }

    #[test]
    fn stage_carry_policy_is_constrained_by_execution_programs() {
        let middle_layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let continuation = StageContinuation {
            version: 1,
            stage_role: "middle".into(),
            next_layer_index: Some(31),
            completed_layers: 10,
            operator_layers: 10,
            has_attention_path: true,
            has_ffn_path: true,
            has_projection_path: true,
        };
        let ffn_only_program = LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(2048),
            q_out_dim: None,
            k_out_dim: None,
            v_out_dim: None,
            ffn_inner_dim: Some(2560),
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::FfnGate,
                    vec!["blk.21.ffn_gate.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::FfnUp,
                    vec!["blk.21.ffn_up.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::FfnDown,
                    vec!["blk.21.ffn_down.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        };

        let policy = StageCarryPolicy::for_execution_boundary(
            &middle_layout,
            Some(&continuation),
            &[ffn_only_program],
        );
        assert!(!policy.carry_attention);
        assert!(policy.carry_ffn);
    }

    #[test]
    fn transfer_frame_for_execution_boundary_uses_program_aware_policy() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-5".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["21-30".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "middle".into(),
                next_layer_index: Some(31),
                completed_layers: 10,
                operator_layers: 10,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: Some(FfnContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("31-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(2048),
            q_out_dim: None,
            k_out_dim: None,
            v_out_dim: None,
            ffn_inner_dim: Some(2560),
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::FfnGate,
                    vec!["blk.21.ffn_gate.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::FfnUp,
                    vec!["blk.21.ffn_up.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::FfnDown,
                    vec!["blk.21.ffn_down.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        transfer.validate().unwrap();
        let carry = transfer.state.carry.as_ref().unwrap();
        assert!(carry.attention.is_none());
        assert!(carry.ffn.is_some());
    }

    #[test]
    fn transfer_frame_rejects_attention_carry_with_more_lanes_than_width() {
        let transfer = StageTransferFrame {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: "gemma-4-e4b-q4".into(),
            route: StageRoute {
                source_stage_id: "0-20".into(),
                source_stage_start: 0,
                source_stage_end: 20,
                source_stage_role: "head".into(),
                target_stage_id: Some("21-41".into()),
            },
            payload: StageTransferPayload {
                request_id: "req-invalid-carry".into(),
                kind: PayloadKind::HiddenState,
                stage_trace: vec!["0-20".into()],
                hidden_dim: 4,
                bytes: vec![0; 16],
                prompt_text: None,
                max_tokens: Some(8),
            },
            state: StageTransferStateEnvelope {
                continuation: None,
                transient: None,
                carry: Some(StageCarryState {
                    attention: Some(CarryableAttentionState {
                        contract: StageResumeContractSummary::default(),
                        projection: Some(CarryableAttentionProjectionState {
                            width: 2,
                            q_provenance: None,
                            k_provenance: None,
                            v_provenance: None,
                            q_lane_indices: vec![0, 1, 2],
                            k_lane_indices: vec![0, 1, 2],
                            v_lane_indices: vec![0, 1, 2],
                            q: vec![0.1, 0.2, 0.3],
                            k: vec![0.0, 0.0, 0.0],
                            v: vec![0.0, 0.0, 0.0],
                        }),
                        mix: Some(CarryableAttentionMixState {
                            width: 2,
                            score_provenance: None,
                            value_provenance: None,
                            score_lane_indices: vec![0, 1, 2],
                            value_lane_indices: vec![0, 1, 2],
                            scores: vec![0.0, 0.0, 0.0],
                            values: vec![0.0, 0.0, 0.0],
                        }),
                    }),
                    ffn: None,
                }),
            },
        };

        let err = transfer.validate().unwrap_err().to_string();
        assert!(err.contains("Attention carry lane count"));
    }

    #[test]
    fn boundary_plan_tracks_execution_boundary_expectations() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-6".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: Some(FfnContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 0,
            hidden_dim: Some(2048),
            q_out_dim: Some(2560),
            k_out_dim: Some(2560),
            v_out_dim: Some(2560),
            ffn_inner_dim: Some(2560),
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.0.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.0.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        transfer.validate().unwrap();
        plan.validate_against_transfer(&transfer).unwrap();
        assert_eq!(plan.next_layer_index, Some(21));
        assert!(!plan.expects_attention_carry);
        assert!(!plan.expects_ffn_carry);
        assert_eq!(plan.expected_attention_width, None);
        assert_eq!(plan.expected_attention_lanes, None);
        assert!(!plan.expects_attention_projection_carry);
        assert!(!plan.expects_attention_mix_carry);
        assert_eq!(plan.expected_attention_projection_lanes, None);
        assert_eq!(plan.expected_attention_mix_lanes, None);
        assert_eq!(plan.expected_ffn_width, None);
        assert_eq!(plan.expected_ffn_lanes, None);
        assert!(plan.resumable_attention_path);
        assert!(!plan.resumable_attention_projection);
        assert!(!plan.resumable_attention_mix);
        assert!(!plan.resumable_ffn_path);
        assert!(!plan.resumable_projection_path);
    }

    #[test]
    fn resume_request_and_receipt_roundtrip_boundary_contract() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-7".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: Some(FfnContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 0,
            hidden_dim: Some(2048),
            q_out_dim: Some(2560),
            k_out_dim: Some(2560),
            v_out_dim: Some(2560),
            ffn_inner_dim: Some(2560),
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.0.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.0.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];
        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        let request = plan.to_resume_request(transfer).unwrap();
        request.validate().unwrap();

        let receipt = request.accept(Some("21-41".into()));
        receipt.validate_against_request(&request).unwrap();
        assert!(receipt.accepted);
        assert_eq!(receipt.accepted_stage_id.as_deref(), Some("21-41"));
        assert!(!receipt.accepted_attention_carry);
        assert!(!receipt.accepted_attention_projection_carry);
        assert!(!receipt.accepted_attention_mix_carry);
        assert_eq!(receipt.accepted_attention_projection_lanes, None);
        assert_eq!(receipt.accepted_attention_mix_lanes, None);
        assert!(!receipt.accepted_ffn_carry);

        let reject = request.reject("unsupported carry lane");
        assert!(!reject.accepted);
        assert_eq!(reject.reason.as_deref(), Some("unsupported carry lane"));
    }

    #[test]
    fn boundary_plan_derives_attention_substate_capabilities_from_execution_program() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-cap-shape".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("31-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(2048),
            q_out_dim: Some(2560),
            k_out_dim: Some(2560),
            v_out_dim: Some(2560),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![ExecutionOp::new(
                ExecutionOpKind::AttentionOut,
                vec!["blk.21.attn_output.weight".into()],
                ExecutionBinding::QuantizedMatrix,
                "quantized matrix tensors",
            )],
        }];

        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        assert!(plan.resumable_attention_path);
        assert!(!plan.resumable_attention_projection);
        assert!(!plan.resumable_attention_mix);
        assert!(!plan.resumable_ffn_path);
        assert!(!plan.resumable_projection_path);
    }

    #[test]
    fn boundary_plan_advertises_attention_projection_slices_independently() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-q-slice".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: false,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("31-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(2048),
            q_out_dim: Some(2560),
            k_out_dim: Some(2560),
            v_out_dim: Some(2560),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![ExecutionOp::new(
                ExecutionOpKind::AttentionQ,
                vec!["blk.21.attn_q.weight".into()],
                ExecutionBinding::QuantizedMatrix,
                "quantized matrix tensors",
            )],
        }];

        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        assert!(plan.resumable_attention_path);
        assert!(plan.resumable_attention_q);
        assert!(!plan.resumable_attention_k);
        assert!(!plan.resumable_attention_v);
        assert_eq!(
            plan.resumable_attention_q_lanes,
            Some(ATTENTION_PROJECTION_CARRY_BUDGET)
        );
        assert_eq!(plan.resumable_attention_k_lanes, None);
        assert_eq!(plan.resumable_attention_v_lanes, None);
        assert!(!plan.resumable_attention_projection);
        assert!(!plan.resumable_attention_mix);
    }

    #[test]
    fn boundary_plan_uses_resume_entry_layer_not_later_stage_layers_for_capabilities() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "21-30".into(),
            start_layer: 21,
            end_layer: 30,
            is_head: false,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-entry-cap".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("31-41".into()));
        let programs = vec![
            LayerExecutionProgram {
                layer_index: 21,
                hidden_dim: Some(2048),
                q_out_dim: Some(2560),
                k_out_dim: Some(2560),
                v_out_dim: Some(2560),
                ffn_inner_dim: Some(2560),
                runnable_sketch: true,
                ops: vec![ExecutionOp::new(
                    ExecutionOpKind::FfnGate,
                    vec!["blk.21.ffn_gate.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                )],
            },
            LayerExecutionProgram {
                layer_index: 22,
                hidden_dim: Some(2048),
                q_out_dim: Some(2560),
                k_out_dim: Some(2560),
                v_out_dim: Some(2560),
                ffn_inner_dim: Some(2560),
                runnable_sketch: true,
                ops: vec![
                    ExecutionOp::new(
                        ExecutionOpKind::AttentionQ,
                        vec!["blk.22.attn_q.weight".into()],
                        ExecutionBinding::QuantizedMatrix,
                        "quantized matrix tensors",
                    ),
                    ExecutionOp::new(
                        ExecutionOpKind::AttentionK,
                        vec!["blk.22.attn_k.weight".into()],
                        ExecutionBinding::QuantizedMatrix,
                        "quantized matrix tensors",
                    ),
                    ExecutionOp::new(
                        ExecutionOpKind::AttentionV,
                        vec!["blk.22.attn_v.weight".into()],
                        ExecutionBinding::QuantizedMatrix,
                        "quantized matrix tensors",
                    ),
                    ExecutionOp::new(
                        ExecutionOpKind::AttentionOut,
                        vec!["blk.22.attn_output.weight".into()],
                        ExecutionBinding::QuantizedMatrix,
                        "quantized matrix tensors",
                    ),
                ],
            },
        ];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        assert!(transfer.state.carry.is_none());
        assert!(!plan.expects_attention_carry);
        assert!(!plan.expects_attention_projection_carry);
        assert!(!plan.expects_attention_mix_carry);
        assert!(!plan.resumable_attention_path);
        assert!(!plan.resumable_attention_projection);
        assert!(!plan.resumable_attention_mix);
    }

    #[test]
    fn packed_backend_admits_matching_resume_request() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let tail_adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 1).unwrap();
        let tail_pack_dir = temp.path().join("packed-tail");
        let tail_pack = tail_adapter.pack_required_tensors(&tail_pack_dir).unwrap();

        let mut tail = PackedResidencySketchBackend::new(tail_pack.index_path);
        tail.load_layout(packed_tail_layout()).unwrap();

        let request = StageResumeRequest {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: "Toy".into(),
            target_stage_id: Some("packed-tail".into()),
            boundary: StageBoundaryPlan {
                source_stage_id: "packed-head".into(),
                source_stage_role: "head".into(),
                source_stage_start: 0,
                source_stage_end: 0,
                target_stage_id: Some("packed-tail".into()),
                next_layer_index: Some(1),
                expected_payload_kind: PayloadKind::HiddenState,
                expected_hidden_dim: 64,
                carry_policy: StageCarryPolicy {
                    carry_attention: false,
                    carry_ffn: false,
                },
                expects_attention_carry: false,
                expects_ffn_carry: false,
                expected_attention_width: None,
                expected_attention_lanes: None,
                expects_attention_projection_carry: false,
                expects_attention_mix_carry: false,
                expected_attention_q_lanes: None,
                expected_attention_k_lanes: None,
                expected_attention_v_lanes: None,
                expected_attention_score_lanes: None,
                expected_attention_value_lanes: None,
                expected_attention_q_distance: None,
                expected_attention_k_distance: None,
                expected_attention_v_distance: None,
                expected_attention_score_distance: None,
                expected_attention_value_distance: None,
                expected_attention_projection_lanes: None,
                expected_attention_mix_lanes: None,
                expected_ffn_width: None,
                expected_ffn_lanes: None,
                resumable_attention_path: false,
                resumable_attention_q: false,
                resumable_attention_k: false,
                resumable_attention_v: false,
                resumable_attention_q_lanes: None,
                resumable_attention_k_lanes: None,
                resumable_attention_v_lanes: None,
                resumable_attention_q_max_distance: None,
                resumable_attention_k_max_distance: None,
                resumable_attention_v_max_distance: None,
                resumable_attention_score_max_distance: None,
                resumable_attention_value_max_distance: None,
                resumable_attention_contract: StageResumeContractSummary::default(),
                resumable_attention_projection: false,
                resumable_attention_mix: false,
                resumable_ffn_path: false,
                resumable_projection_path: false,
                operator_layers: 0,
                completed_layers: Some(1),
            },
            transfer: StageTransferFrame {
                version: STAGE_FORWARD_FRAME_VERSION,
                model_id: "Toy".into(),
                route: StageRoute {
                    source_stage_id: "packed-head".into(),
                    source_stage_start: 0,
                    source_stage_end: 0,
                    source_stage_role: "head".into(),
                    target_stage_id: Some("packed-tail".into()),
                },
                payload: StageTransferPayload {
                    request_id: "req-8".into(),
                    kind: PayloadKind::HiddenState,
                    stage_trace: vec!["packed-head".into()],
                    hidden_dim: 64,
                    bytes: vec![0; 64 * 4],
                    prompt_text: Some("hello".into()),
                    max_tokens: Some(8),
                },
                state: StageTransferStateEnvelope {
                    continuation: Some(StageContinuation {
                        version: 1,
                        stage_role: "head".into(),
                        next_layer_index: Some(1),
                        completed_layers: 1,
                        operator_layers: 1,
                        has_attention_path: false,
                        has_ffn_path: false,
                        has_projection_path: false,
                    }),
                    transient: None,
                    carry: None,
                },
            },
        };

        let decision = tail.admit_resume_request(&request);
        assert!(decision.accepted);
        assert_eq!(decision.accepted_stage_id.as_deref(), Some("packed-tail"));
        assert_eq!(decision.accepted_next_layer_index, Some(1));
    }

    #[test]
    fn packed_backend_rejects_resume_request_with_wrong_next_layer() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let tail_adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 1).unwrap();
        let tail_pack_dir = temp.path().join("packed-tail");
        let tail_pack = tail_adapter.pack_required_tensors(&tail_pack_dir).unwrap();

        let mut tail = PackedResidencySketchBackend::new(tail_pack.index_path);
        tail.load_layout(packed_tail_layout()).unwrap();

        let request = StageResumeRequest {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: "Toy".into(),
            target_stage_id: Some("packed-tail".into()),
            boundary: StageBoundaryPlan {
                source_stage_id: "packed-head".into(),
                source_stage_role: "head".into(),
                source_stage_start: 0,
                source_stage_end: 0,
                target_stage_id: Some("packed-tail".into()),
                next_layer_index: Some(999),
                expected_payload_kind: PayloadKind::HiddenState,
                expected_hidden_dim: 64,
                carry_policy: StageCarryPolicy {
                    carry_attention: false,
                    carry_ffn: false,
                },
                expects_attention_carry: false,
                expects_ffn_carry: false,
                expected_attention_width: None,
                expected_attention_lanes: None,
                expects_attention_projection_carry: false,
                expects_attention_mix_carry: false,
                expected_attention_q_lanes: None,
                expected_attention_k_lanes: None,
                expected_attention_v_lanes: None,
                expected_attention_score_lanes: None,
                expected_attention_value_lanes: None,
                expected_attention_q_distance: None,
                expected_attention_k_distance: None,
                expected_attention_v_distance: None,
                expected_attention_score_distance: None,
                expected_attention_value_distance: None,
                expected_attention_projection_lanes: None,
                expected_attention_mix_lanes: None,
                expected_ffn_width: None,
                expected_ffn_lanes: None,
                resumable_attention_path: false,
                resumable_attention_q: false,
                resumable_attention_k: false,
                resumable_attention_v: false,
                resumable_attention_q_lanes: None,
                resumable_attention_k_lanes: None,
                resumable_attention_v_lanes: None,
                resumable_attention_q_max_distance: None,
                resumable_attention_k_max_distance: None,
                resumable_attention_v_max_distance: None,
                resumable_attention_score_max_distance: None,
                resumable_attention_value_max_distance: None,
                resumable_attention_contract: StageResumeContractSummary::default(),
                resumable_attention_projection: false,
                resumable_attention_mix: false,
                resumable_ffn_path: false,
                resumable_projection_path: false,
                operator_layers: 0,
                completed_layers: Some(1),
            },
            transfer: StageTransferFrame {
                version: STAGE_FORWARD_FRAME_VERSION,
                model_id: "Toy".into(),
                route: StageRoute {
                    source_stage_id: "packed-head".into(),
                    source_stage_start: 0,
                    source_stage_end: 0,
                    source_stage_role: "head".into(),
                    target_stage_id: Some("packed-tail".into()),
                },
                payload: StageTransferPayload {
                    request_id: "req-9".into(),
                    kind: PayloadKind::HiddenState,
                    stage_trace: vec!["packed-head".into()],
                    hidden_dim: 64,
                    bytes: vec![0; 64 * 4],
                    prompt_text: Some("hello".into()),
                    max_tokens: Some(8),
                },
                state: StageTransferStateEnvelope {
                    continuation: Some(StageContinuation {
                        version: 1,
                        stage_role: "head".into(),
                        next_layer_index: Some(999),
                        completed_layers: 1,
                        operator_layers: 1,
                        has_attention_path: false,
                        has_ffn_path: false,
                        has_projection_path: false,
                    }),
                    transient: None,
                    carry: None,
                },
            },
        };

        let decision = tail.admit_resume_request(&request);
        assert!(!decision.accepted);
        assert!(
            decision
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("next-layer mismatch")
        );
    }

    #[test]
    fn packed_backend_resume_forward_matches_direct_continue() {
        let temp = tempdir().unwrap();
        let gguf_path = temp.path().join("toy.gguf");
        let bytes: Vec<u8> = (0..=255).cycle().take(300).collect();
        fs::write(&gguf_path, &bytes).unwrap();

        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            file_size_bytes: 300,
            tensor_data_offset: 0,
            metadata: BTreeMap::from([
                ("general.name".into(), MetadataValue::String("Toy".into())),
                (
                    "general.architecture".into(),
                    MetadataValue::String("toy".into()),
                ),
                ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ]),
            tensors: vec![
                TensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 0,
                },
                TensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 100,
                },
                TensorInfo {
                    name: "output_norm.weight".into(),
                    dimensions: vec![1],
                    ggml_type: 0,
                    offset: 200,
                },
            ],
        };
        let plan = file.plan_for_splits(&[
            StageSplit {
                stage_index: 0,
                start_layer: 0,
                end_layer: 0,
            },
            StageSplit {
                stage_index: 1,
                start_layer: 1,
                end_layer: 1,
            },
        ]);
        plan.write_bundle(&file, temp.path()).unwrap();

        let bundle = LoadedRuntimeBundle::load(temp.path()).unwrap();
        let head_adapter =
            StageResidencyAdapter::from_parts(bundle.clone(), gguf_path.clone(), file.clone(), 0)
                .unwrap();
        let tail_adapter =
            StageResidencyAdapter::from_parts(bundle, gguf_path.clone(), file.clone(), 1).unwrap();
        let head_pack = head_adapter
            .pack_required_tensors(&temp.path().join("packed-head"))
            .unwrap();
        let tail_pack = tail_adapter
            .pack_required_tensors(&temp.path().join("packed-tail"))
            .unwrap();

        let mut head = PackedResidencySketchBackend::new(head_pack.index_path);
        head.load_layout(packed_head_layout()).unwrap();
        let mut tail = PackedResidencySketchBackend::new(tail_pack.index_path);
        tail.load_layout(packed_tail_layout()).unwrap();

        let stage1 = head
            .begin_prompt("packed-req", "reply exactly STAGE PACK", Some(12), 0)
            .unwrap();
        let direct = tail.continue_forward(stage1.clone()).unwrap();

        let frame = StageForwardFrame::from_tensor(
            "Toy",
            &packed_head_layout(),
            stage1,
            Some("packed-tail".into()),
        );
        let tail_store = StageTensorStore::load(&tail.index_path).unwrap();
        let tail_view = tail_store.model_view();
        let boundary = frame.to_boundary_plan_for_execution_boundary(
            &packed_tail_layout(),
            &tail_view.execution_programs,
        );
        let transfer = frame.to_transfer_frame_for_execution_boundary(
            &packed_tail_layout(),
            &tail_view.execution_programs,
        );
        let request = boundary.to_resume_request(transfer).unwrap();

        let (resumed, receipt) = tail.resume_forward(request).unwrap();
        assert!(receipt.accepted);
        assert_eq!(resumed.kind, direct.kind);
        assert_eq!(resumed.hidden_dim, direct.hidden_dim);
        assert_eq!(resumed.stage_trace, direct.stage_trace);
        assert_eq!(resumed.bytes, direct.bytes);
    }

    #[test]
    fn packed_backend_resume_carry_changes_downstream_execution() {
        let temp = tempdir().unwrap();
        let pack_path = temp.path().join("stage-2-required.pack");
        let index_path = temp.path().join("stage-2-required.index.json");
        let tensors = vec![
            (
                "blk.1.attn_norm.weight",
                vec![1.0f32, 0.9, 1.1, 1.05],
                vec![4u64],
                0u32,
            ),
            (
                "blk.1.attn_q.weight",
                vec![
                    1.0f32, 0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0,
                    1.0,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.attn_k.weight",
                vec![
                    0.9f32, 0.0, 0.0, 0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.1,
                    0.9,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.attn_v.weight",
                vec![
                    1.1f32, 0.0, 0.1, 0.0, 0.0, 1.1, 0.0, 0.1, 0.1, 0.0, 1.1, 0.0, 0.0, 0.1, 0.0,
                    1.1,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.attn_output.weight",
                vec![
                    0.8f32, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0,
                    0.8,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.ffn_norm.weight",
                vec![1.0f32, 1.0, 1.0, 1.0],
                vec![4u64],
                0u32,
            ),
            (
                "blk.1.ffn_gate.weight",
                vec![
                    1.2f32, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0,
                    1.2,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.ffn_up.weight",
                vec![
                    0.7f32, 0.1, 0.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.7, 0.1, 0.1, 0.0, 0.0,
                    0.7,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
            (
                "blk.1.ffn_down.weight",
                vec![
                    0.6f32, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0,
                    0.6,
                ],
                vec![4u64, 4u64],
                0u32,
            ),
        ];
        let mut pack_bytes = Vec::new();
        let mut entries = Vec::new();
        let mut pack_offset = 0u64;
        for (name, values, dimensions, ggml_type) in tensors {
            let mut bytes = Vec::new();
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let byte_len = bytes.len() as u64;
            pack_bytes.extend_from_slice(&bytes);
            entries.push(PackedTensorEntry {
                name: name.into(),
                pack_offset,
                byte_len,
                source_file_offset: pack_offset,
                dimensions,
                ggml_type,
            });
            pack_offset += byte_len;
        }
        fs::write(&pack_path, pack_bytes).unwrap();
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 1,
                role: "tail".into(),
                total_bytes: pack_offset,
                tensor_count: entries.len(),
                tensors: entries,
            })
            .unwrap(),
        )
        .unwrap();

        let mut tail = PackedResidencySketchBackend::new(index_path);
        tail.load_layout(packed_tail_layout()).unwrap();

        let with_carry = StageTensor {
            request_id: "packed-carry-req".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["packed-head".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("reply exactly STAGE PACK".into()),
            max_tokens: Some(12),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(1),
                completed_layers: 1,
                operator_layers: 1,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 4,
                    lane_indices: vec![0, 1, 2, 3],
                    q_preview: vec![0.8, -0.4, 0.2, 0.1],
                    k_preview: vec![0.2, 0.7, -0.3, 0.4],
                    v_preview: vec![0.6, 0.1, 0.5, -0.2],
                    score_preview: vec![0.3, 0.2, 0.1, 0.4],
                    value_preview: vec![0.5, -0.1, 0.4, 0.2],
                }),
                ffn: Some(FfnContinuation {
                    width: 4,
                    lane_indices: vec![0, 1, 2, 3],
                    gate_preview: vec![0.9, -0.7, 0.4, 0.2],
                    up_preview: vec![0.5, 0.3, -0.6, 0.7],
                    activation_preview: vec![0.4, -0.2, 0.3, 0.1],
                }),
            }),
            carry: None,
        };
        let mut without_carry = with_carry.clone();
        without_carry.transient = None;
        without_carry.carry = None;

        let carried = tail.continue_forward(with_carry).unwrap();
        let stripped = tail.continue_forward(without_carry).unwrap();

        assert_ne!(carried.bytes, stripped.bytes);
    }

    #[test]
    fn transfer_frame_rehydrates_transient_from_carry() {
        let transfer = StageTransferFrame {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: "gemma-4-e4b-q4".into(),
            route: StageRoute {
                source_stage_id: "0-20".into(),
                source_stage_start: 0,
                source_stage_end: 20,
                source_stage_role: "head".into(),
                target_stage_id: Some("21-41".into()),
            },
            payload: StageTransferPayload {
                request_id: "req-10".into(),
                kind: PayloadKind::HiddenState,
                stage_trace: vec!["0-20".into()],
                hidden_dim: 4,
                bytes: vec![0; 16],
                prompt_text: Some("hello".into()),
                max_tokens: Some(8),
            },
            state: StageTransferStateEnvelope {
                continuation: Some(StageContinuation {
                    version: 1,
                    stage_role: "head".into(),
                    next_layer_index: Some(21),
                    completed_layers: 21,
                    operator_layers: 21,
                    has_attention_path: true,
                    has_ffn_path: true,
                    has_projection_path: true,
                }),
                transient: None,
                carry: Some(StageCarryState {
                    attention: Some(CarryableAttentionState {
                        contract: StageResumeContractSummary::default(),
                        projection: Some(CarryableAttentionProjectionState {
                            width: 8,
                            q_provenance: None,
                            k_provenance: None,
                            v_provenance: None,
                            q_lane_indices: vec![0, 1],
                            k_lane_indices: vec![0, 1],
                            v_lane_indices: vec![0, 1],
                            q: vec![0.1, -0.2],
                            k: vec![0.3, -0.4],
                            v: vec![0.5, -0.6],
                        }),
                        mix: Some(CarryableAttentionMixState {
                            width: 8,
                            score_provenance: None,
                            value_provenance: None,
                            score_lane_indices: vec![0, 1],
                            value_lane_indices: vec![0, 1],
                            scores: vec![0.7, -0.8],
                            values: vec![0.9, -1.0],
                        }),
                    }),
                    ffn: Some(CarryableFfnState {
                        width: 8,
                        lane_indices: vec![0, 1],
                        gate_head: vec![0.1, 0.2],
                        up_head: vec![0.3, 0.4],
                        activation_head: vec![0.5, 0.6],
                    }),
                }),
            },
        };

        let tensor = transfer.into_stage_tensor();
        let transient = tensor.transient.unwrap();
        assert_eq!(
            transient.attention.as_ref().unwrap().q_preview,
            vec![0.1, -0.2]
        );
        assert_eq!(
            transient.ffn.as_ref().unwrap().activation_preview,
            vec![0.5, 0.6]
        );
    }

    #[test]
    fn stage_transfer_frame_compacts_transient_state() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-3".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: true,
                has_projection_path: true,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    q_preview: vec![0.1, -0.2],
                    k_preview: vec![0.3, -0.4],
                    v_preview: vec![0.5, -0.6],
                    score_preview: vec![0.7, -0.8],
                    value_preview: vec![0.9, -1.0],
                }),
                ffn: Some(FfnContinuation {
                    width: 8,
                    lane_indices: vec![0, 1],
                    gate_preview: vec![0.1, 0.2],
                    up_preview: vec![0.3, 0.4],
                    activation_preview: vec![0.5, 0.6],
                }),
            }),
            carry: None,
        };

        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let transfer = frame.to_transfer_frame();
        transfer.validate().unwrap();

        let attention = transfer
            .state
            .transient
            .as_ref()
            .and_then(|transient| transient.attention.as_ref())
            .unwrap();
        let ffn = transfer
            .state
            .transient
            .as_ref()
            .and_then(|transient| transient.ffn.as_ref())
            .unwrap();

        assert_eq!(attention.width, 8);
        assert_eq!(attention.preview_len, 10);
        assert!(attention.rms_milli > 0);
        assert_ne!(attention.checksum, 0);
        assert_eq!(ffn.width, 8);
        assert_eq!(ffn.preview_len, 6);
        assert!(ffn.rms_milli > 0);
        assert_ne!(ffn.checksum, 0);

        let carry_attention = transfer
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.attention.as_ref())
            .unwrap();
        assert_eq!(carry_attention.width(), 8);
        assert_eq!(carry_attention.lane_count(), 2);
        assert_eq!(carry_attention.projection_lane_count(), 2);
        assert_eq!(carry_attention.mix_lane_count(), 2);
        assert_eq!(
            carry_attention.projection.as_ref().unwrap().q,
            vec![0.1, -0.2]
        );
        assert_eq!(
            carry_attention.mix.as_ref().unwrap().scores,
            vec![0.7, -0.8]
        );
        assert!(
            transfer
                .state
                .carry
                .as_ref()
                .and_then(|carry| carry.ffn.as_ref())
                .is_none()
        );

        let full_transfer = frame.to_transfer_frame_with_policy(&StageCarryPolicy::full());
        full_transfer.validate().unwrap();
        let carry_ffn = full_transfer
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.ffn.as_ref())
            .unwrap();
        assert_eq!(carry_ffn.width, 8);
        assert_eq!(carry_ffn.lane_count(), 2);
        assert_eq!(carry_ffn.gate_head, vec![0.1, 0.2]);
        assert_eq!(carry_ffn.activation_head, vec![0.5, 0.6]);
    }

    #[test]
    fn attention_carry_uses_typed_budgeted_projection_and_mix_substates() {
        let attention = AttentionContinuation {
            width: 16,
            lane_indices: vec![0, 2, 4, 6, 8, 10, 12, 14],
            q_preview: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            k_preview: vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            v_preview: vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            score_preview: vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            value_preview: vec![3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
        };

        let carry = CarryableAttentionState::from_attention(&attention);
        assert_eq!(carry.width(), 16);
        assert_eq!(
            carry.projection_lane_count(),
            ATTENTION_PROJECTION_CARRY_BUDGET
        );
        assert_eq!(carry.mix_lane_count(), ATTENTION_MIX_CARRY_BUDGET);
        assert_eq!(
            carry.projection.as_ref().unwrap().q_lane_indices,
            vec![0, 2, 4, 6]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().k_lane_indices,
            vec![0, 2, 4, 6]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().v_lane_indices,
            vec![0, 2, 4, 6]
        );
        assert_eq!(
            carry.mix.as_ref().unwrap().score_lane_indices,
            vec![0, 2, 4, 6, 8, 10, 12, 14]
        );
        assert_eq!(
            carry.mix.as_ref().unwrap().value_lane_indices,
            vec![0, 2, 4, 6, 8, 10, 12, 14]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().q,
            vec![0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(
            carry.mix.as_ref().unwrap().scores,
            vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
        );

        let roundtrip = carry.to_attention_continuation().unwrap();
        assert_eq!(roundtrip.width, 16);
        assert_eq!(roundtrip.lane_indices, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(
            roundtrip.q_preview,
            vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            roundtrip.score_preview,
            vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
        );
    }

    #[test]
    fn execution_boundary_clamps_attention_carry_to_entry_budget() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-budget-clamp".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: false,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                    q_preview: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    k_preview: vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                    v_preview: vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
                    score_preview: vec![3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
                    value_preview: vec![4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(2),
            q_out_dim: Some(2),
            k_out_dim: Some(2),
            v_out_dim: Some(2),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.21.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionK,
                    vec!["blk.21.attn_k.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionV,
                    vec!["blk.21.attn_v.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.21.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        let carry = transfer
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.attention.as_ref())
            .unwrap();

        assert_eq!(carry.projection_lane_count(), 2);
        assert_eq!(carry.mix_lane_count(), 2);
        assert_eq!(
            carry.projection.as_ref().unwrap().q_lane_indices,
            vec![0, 1]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().k_lane_indices,
            vec![0, 1]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().v_lane_indices,
            vec![0, 1]
        );
        assert_eq!(carry.mix.as_ref().unwrap().score_lane_indices, vec![0, 1]);
        assert_eq!(carry.mix.as_ref().unwrap().value_lane_indices, vec![0, 1]);
        assert_eq!(plan.expected_attention_q_lanes, Some(2));
        assert_eq!(plan.expected_attention_k_lanes, Some(2));
        assert_eq!(plan.expected_attention_v_lanes, Some(2));
        assert_eq!(plan.expected_attention_score_lanes, Some(2));
        assert_eq!(plan.expected_attention_value_lanes, Some(2));
        assert_eq!(plan.resumable_attention_q_lanes, Some(2));
        assert_eq!(plan.resumable_attention_k_lanes, Some(2));
        assert_eq!(plan.resumable_attention_v_lanes, Some(2));
        assert_eq!(plan.expected_attention_projection_lanes, Some(2));
        assert_eq!(plan.expected_attention_mix_lanes, Some(2));
    }

    #[test]
    fn execution_boundary_caps_attention_mix_budget_by_projection_budget() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-mix-cap".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: false,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 16,
                    lane_indices: (0..8).collect(),
                    q_preview: vec![0.1; 8],
                    k_preview: vec![0.2; 8],
                    v_preview: vec![0.3; 8],
                    score_preview: vec![0.4; 8],
                    value_preview: vec![0.5; 8],
                }),
                ffn: None,
            }),
            carry: None,
        };
        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(16),
            q_out_dim: Some(16),
            k_out_dim: Some(16),
            v_out_dim: Some(16),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.21.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionK,
                    vec!["blk.21.attn_k.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionV,
                    vec!["blk.21.attn_v.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.21.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);
        let carry = transfer
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.attention.as_ref())
            .unwrap();

        assert_eq!(
            carry.projection_lane_count(),
            ATTENTION_PROJECTION_CARRY_BUDGET
        );
        assert_eq!(carry.mix_lane_count(), ATTENTION_PROJECTION_CARRY_BUDGET);
        assert_eq!(
            plan.expected_attention_mix_lanes,
            Some(ATTENTION_PROJECTION_CARRY_BUDGET)
        );
    }

    #[test]
    fn transfer_frame_prefers_execution_captured_carry_over_transient_derivation() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let tensor = StageTensor {
            request_id: "req-carry-pref".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["0-20".into()],
            hidden_dim: 4,
            bytes: vec![0; 16],
            prompt_text: Some("hello".into()),
            max_tokens: Some(8),
            continuation: Some(StageContinuation {
                version: 1,
                stage_role: "head".into(),
                next_layer_index: Some(21),
                completed_layers: 21,
                operator_layers: 21,
                has_attention_path: true,
                has_ffn_path: false,
                has_projection_path: false,
            }),
            transient: Some(StageTransientState {
                attention: Some(AttentionContinuation {
                    width: 8,
                    lane_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                    q_preview: vec![9.0; 8],
                    k_preview: vec![9.0; 8],
                    v_preview: vec![9.0; 8],
                    score_preview: vec![9.0; 8],
                    value_preview: vec![9.0; 8],
                }),
                ffn: None,
            }),
            carry: Some(StageCarryState {
                attention: Some(CarryableAttentionState {
                    contract: StageResumeContractSummary::default(),
                    projection: Some(CarryableAttentionProjectionState {
                        width: 8,
                        q_provenance: None,
                        k_provenance: None,
                        v_provenance: None,
                        q_lane_indices: vec![1, 3],
                        k_lane_indices: vec![1, 3],
                        v_lane_indices: vec![1, 3],
                        q: vec![0.1, 0.2],
                        k: vec![0.3, 0.4],
                        v: vec![0.5, 0.6],
                    }),
                    mix: Some(CarryableAttentionMixState {
                        width: 8,
                        score_provenance: None,
                        value_provenance: None,
                        score_lane_indices: vec![1, 3, 5],
                        value_lane_indices: vec![1, 3, 5],
                        scores: vec![0.7, 0.8, 0.9],
                        values: vec![1.0, 1.1, 1.2],
                    }),
                }),
                ffn: None,
            }),
        };

        let frame =
            StageForwardFrame::from_tensor("gemma-4-e4b-q4", &layout, tensor, Some("21-41".into()));
        let transfer = frame.to_transfer_frame();
        let carry = transfer
            .state
            .carry
            .as_ref()
            .and_then(|carry| carry.attention.as_ref())
            .unwrap();

        assert_eq!(
            carry.projection.as_ref().unwrap().q_lane_indices,
            vec![1, 3]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().k_lane_indices,
            vec![1, 3]
        );
        assert_eq!(
            carry.projection.as_ref().unwrap().v_lane_indices,
            vec![1, 3]
        );
        assert_eq!(
            carry.mix.as_ref().unwrap().score_lane_indices,
            vec![1, 3, 5]
        );
        assert_eq!(
            carry.mix.as_ref().unwrap().value_lane_indices,
            vec![1, 3, 5]
        );
        assert_eq!(carry.projection.as_ref().unwrap().q, vec![0.1, 0.2]);
        assert_eq!(carry.mix.as_ref().unwrap().scores, vec![0.7, 0.8, 0.9]);
    }

    #[test]
    fn resume_carry_rehydrates_attention_projection_and_mix_lanes_independently() {
        let scratch = PackedResidencySketchBackend::scratch_from_resume_carry(&StageCarryState {
            attention: Some(CarryableAttentionState {
                contract: StageResumeContractSummary::default(),
                projection: Some(CarryableAttentionProjectionState {
                    width: 8,
                    q_provenance: None,
                    k_provenance: None,
                    v_provenance: None,
                    q_lane_indices: vec![1, 6],
                    k_lane_indices: vec![1, 6],
                    v_lane_indices: vec![1, 6],
                    q: vec![0.1, 0.2],
                    k: vec![0.3, 0.4],
                    v: vec![0.5, 0.6],
                }),
                mix: Some(CarryableAttentionMixState {
                    width: 8,
                    score_provenance: None,
                    value_provenance: None,
                    score_lane_indices: vec![0, 3, 7],
                    value_lane_indices: vec![0, 3, 7],
                    scores: vec![1.0, 1.1, 1.2],
                    values: vec![2.0, 2.1, 2.2],
                }),
            }),
            ffn: None,
        })
        .unwrap();

        assert_eq!(scratch.attn_q_lane_indices, Some(vec![1, 6]));
        assert_eq!(scratch.attn_k_lane_indices, Some(vec![1, 6]));
        assert_eq!(scratch.attn_v_lane_indices, Some(vec![1, 6]));
        assert_eq!(scratch.attn_score_lane_indices, Some(vec![0, 3, 7]));
        assert_eq!(scratch.attn_value_lane_indices, Some(vec![0, 3, 7]));
        assert_eq!(
            scratch.attn_q,
            Some(vec![0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0])
        );
        assert_eq!(
            scratch.attn_score,
            Some(vec![1.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2])
        );
        assert_eq!(
            scratch.attn_value,
            Some(vec![2.0, 0.0, 0.0, 2.1, 0.0, 0.0, 0.0, 2.2])
        );
    }

    #[test]
    fn resume_carry_rejects_attention_contract_phase_mismatch() {
        let err = PackedResidencySketchBackend::scratch_from_resume_carry(&StageCarryState {
            attention: Some(CarryableAttentionState {
                contract: StageResumeContractSummary {
                    attention_q: Some(StageAttentionResumeContractSummary {
                        phase: StageAttentionResumePhase::AfterProjection,
                        blend: StageAttentionResumeBlend::StrongBlend,
                        blend_weight_milli: Some(350),
                    }),
                    ..StageResumeContractSummary::default()
                },
                projection: Some(CarryableAttentionProjectionState {
                    width: 4,
                    q_provenance: None,
                    k_provenance: None,
                    v_provenance: None,
                    q_lane_indices: vec![0],
                    k_lane_indices: Vec::new(),
                    v_lane_indices: Vec::new(),
                    q: vec![0.5],
                    k: Vec::new(),
                    v: Vec::new(),
                }),
                mix: None,
            }),
            ffn: None,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("attention carry q contract phase AfterProjection"));
    }

    #[test]
    fn typed_attention_overwrite_contract_disables_resume_blend() {
        let mut current = vec![1.0, 1.0, 1.0, 1.0];
        let scratch = LayerScratch {
            attention_contract: StageResumeContractSummary {
                attention_q: Some(StageAttentionResumeContractSummary {
                    phase: StageAttentionResumePhase::Direct,
                    blend: StageAttentionResumeBlend::Overwrite,
                    blend_weight_milli: None,
                }),
                ..StageResumeContractSummary::default()
            },
            attn_q_blend_weight: None,
            ..LayerScratch::default()
        };

        if let Some(weight) = scratch.typed_or_legacy_attention_blend(
            scratch.attn_q_blend_weight,
            ATTENTION_PROJECTION_BLEND_WEIGHT,
        ) {
            PackedResidencySketchBackend::blend_lane_state(
                &mut current,
                &[9.0, 9.0, 9.0, 9.0],
                Some(&[1, 2]),
                weight,
            );
        }

        assert_eq!(current, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn decoded_attention_projection_uses_real_f32_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [2.0f32, 0.0, 0.0, 3.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.attn_q.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let projected = PackedResidencySketchBackend::try_project_decoded_attention_matrix(
            &store,
            &entry,
            &[4.0, 5.0],
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected, vec![8.0, 15.0]);
    }

    #[test]
    fn decoded_attention_projection_uses_real_q4_k_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let bytes = vec![0u8; 144];
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.attn_q.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![256, 1],
            ggml_type: quants::GGML_TYPE_Q4_K,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let projected = PackedResidencySketchBackend::try_project_decoded_attention_matrix(
            &store,
            &entry,
            &[42.0],
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected.len(), quants::QK_K);
        assert!(projected.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn decoded_attention_projection_uses_real_q6_k_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let bytes = vec![0u8; 210];
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.attn_q.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![256, 1],
            ggml_type: quants::GGML_TYPE_Q6_K,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let projected = PackedResidencySketchBackend::try_project_decoded_attention_matrix(
            &store,
            &entry,
            &[42.0],
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected.len(), quants::QK_K);
        assert!(projected.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn decoded_attention_output_uses_real_f32_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [1.0f32, 0.0, 0.0, 1.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.attn_output.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let (attn, mix, projected) =
            PackedResidencySketchBackend::try_project_decoded_attention_output_matrix(
                &store,
                &entry,
                &[4.0, 5.0],
                &LayerScratch::default(),
                b"test",
            )
            .unwrap()
            .unwrap();

        assert!(attn.is_none());
        assert!(mix.is_none());
        assert_eq!(projected, vec![4.0, 5.0]);
    }

    #[test]
    fn decoded_ffn_gate_uses_real_f32_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [2.0f32, 0.0, 0.0, 3.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.ffn_gate.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let projected = PackedResidencySketchBackend::try_project_decoded_ffn_matrix(
            &store,
            &entry,
            &[4.0, 5.0],
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected, vec![8.0, 15.0]);
    }

    #[test]
    fn decoded_ffn_down_uses_real_f32_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [1.0f32, 0.0, 0.0, 1.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "blk.0.ffn_down.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let (ffn, mix, projected) =
            PackedResidencySketchBackend::try_project_decoded_ffn_down_matrix(
                &store,
                &entry,
                &[4.0, 5.0],
                &LayerScratch::default(),
                b"test",
            )
            .unwrap()
            .unwrap();

        assert!(ffn.is_none());
        assert!(mix.is_none());
        assert_eq!(projected, vec![4.0, 5.0]);
    }

    #[test]
    fn decoded_input_gate_uses_real_f32_matrix_from_pack() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [2.0f32, 0.0, 0.0, 3.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "input_gate.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();

        let projected = PackedResidencySketchBackend::try_project_decoded_input_gate_matrix(
            &store,
            &entry,
            &[4.0, 5.0],
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected, vec![8.0, 15.0]);
    }

    #[test]
    fn decoded_projection_applies_real_input_gate_from_scratch() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");
        let values = [1.0f32, 0.0, 0.0, 1.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&pack_path, &bytes).unwrap();
        let entry = PackedTensorEntry {
            name: "output.weight".into(),
            pack_offset: 0,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: vec![2, 2],
            ggml_type: quants::GGML_TYPE_F32,
        };
        fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "Toy".into(),
                architecture: "toy".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: bytes.len() as u64,
                tensor_count: 1,
                tensors: vec![entry.clone()],
            })
            .unwrap(),
        )
        .unwrap();
        let store = StageTensorStore::load(&index_path).unwrap();
        let scratch = LayerScratch {
            input_gate: Some(vec![0.0, 1.0]),
            ..LayerScratch::default()
        };

        let projected = PackedResidencySketchBackend::try_project_decoded_projection_matrix(
            &store,
            &entry,
            &[4.0, 5.0],
            &scratch,
        )
        .unwrap()
        .unwrap();

        assert_eq!(projected, vec![0.0, 5.0 * 1.0_f32.tanh()]);
    }

    #[test]
    fn resume_contract_drops_attention_mix_without_projection_resume() {
        let carry = StageCarryState {
            attention: Some(CarryableAttentionState {
                contract: StageResumeContractSummary::default(),
                projection: None,
                mix: Some(CarryableAttentionMixState {
                    width: 8,
                    score_provenance: None,
                    value_provenance: None,
                    score_lane_indices: vec![0, 2, 4],
                    value_lane_indices: vec![0, 2, 4],
                    scores: vec![0.1, 0.2, 0.3],
                    values: vec![0.4, 0.5, 0.6],
                }),
            }),
            ffn: None,
        };
        let capabilities = StageResumeCapabilities {
            attention_path: true,
            attention_q: false,
            attention_k: false,
            attention_v: false,
            attention_projection: false,
            attention_mix: true,
            ffn_path: false,
            projection_path: false,
        };
        let budgets = StageResumeBudgets {
            attention_q_lanes: None,
            attention_k_lanes: None,
            attention_v_lanes: None,
            attention_projection_lanes: None,
            attention_mix_lanes: Some(3),
            ffn_lanes: None,
        };

        assert!(
            carry
                .with_resume_contract(&capabilities, &budgets)
                .is_none()
        );
        assert!(carry.with_resume_capabilities(&capabilities).is_none());
    }

    #[test]
    fn stage_carry_merge_ages_attention_provenance_across_later_layers() {
        let previous = StageCarryState {
            attention: Some(CarryableAttentionState {
                contract: StageResumeContractSummary::default(),
                projection: Some(CarryableAttentionProjectionState {
                    width: 8,
                    q_provenance: Some(AttentionCheckpointProvenance {
                        layer_index: 3,
                        operator_kind: "AttentionQ".into(),
                        layer_distance_to_boundary: 0,
                    }),
                    k_provenance: Some(AttentionCheckpointProvenance {
                        layer_index: 3,
                        operator_kind: "AttentionK".into(),
                        layer_distance_to_boundary: 0,
                    }),
                    v_provenance: Some(AttentionCheckpointProvenance {
                        layer_index: 3,
                        operator_kind: "AttentionV".into(),
                        layer_distance_to_boundary: 0,
                    }),
                    q_lane_indices: vec![0, 1],
                    k_lane_indices: vec![0, 1],
                    v_lane_indices: vec![0, 1],
                    q: vec![0.1, 0.2],
                    k: vec![0.3, 0.4],
                    v: vec![0.5, 0.6],
                }),
                mix: Some(CarryableAttentionMixState {
                    width: 8,
                    score_provenance: Some(AttentionCheckpointProvenance {
                        layer_index: 3,
                        operator_kind: "AttentionOut".into(),
                        layer_distance_to_boundary: 0,
                    }),
                    value_provenance: Some(AttentionCheckpointProvenance {
                        layer_index: 3,
                        operator_kind: "AttentionOut".into(),
                        layer_distance_to_boundary: 0,
                    }),
                    score_lane_indices: vec![0, 1],
                    value_lane_indices: vec![0, 1],
                    scores: vec![0.7, 0.8],
                    values: vec![0.9, 1.0],
                }),
            }),
            ffn: None,
        };

        let merged = PackedResidencySketchBackend::merge_stage_carry(Some(previous), None).unwrap();
        let projection = merged
            .attention
            .as_ref()
            .unwrap()
            .projection
            .as_ref()
            .unwrap();
        let mix = merged.attention.as_ref().unwrap().mix.as_ref().unwrap();

        assert_eq!(
            projection
                .q_provenance
                .as_ref()
                .unwrap()
                .layer_distance_to_boundary,
            1
        );
        assert_eq!(
            projection
                .k_provenance
                .as_ref()
                .unwrap()
                .layer_distance_to_boundary,
            1
        );
        assert_eq!(
            projection
                .v_provenance
                .as_ref()
                .unwrap()
                .layer_distance_to_boundary,
            1
        );
        assert_eq!(
            mix.score_provenance
                .as_ref()
                .unwrap()
                .layer_distance_to_boundary,
            1
        );
        assert_eq!(
            mix.value_provenance
                .as_ref()
                .unwrap()
                .layer_distance_to_boundary,
            1
        );
    }

    #[test]
    fn execution_boundary_plan_surfaces_aged_attention_freshness() {
        let merged = PackedResidencySketchBackend::merge_stage_carry(
            Some(StageCarryState {
                attention: Some(CarryableAttentionState {
                    contract: StageResumeContractSummary::default(),
                    projection: Some(CarryableAttentionProjectionState {
                        width: 8,
                        q_provenance: Some(AttentionCheckpointProvenance {
                            layer_index: 3,
                            operator_kind: "AttentionQ".into(),
                            layer_distance_to_boundary: 0,
                        }),
                        k_provenance: Some(AttentionCheckpointProvenance {
                            layer_index: 3,
                            operator_kind: "AttentionK".into(),
                            layer_distance_to_boundary: 0,
                        }),
                        v_provenance: Some(AttentionCheckpointProvenance {
                            layer_index: 3,
                            operator_kind: "AttentionV".into(),
                            layer_distance_to_boundary: 0,
                        }),
                        q_lane_indices: vec![0, 1],
                        k_lane_indices: vec![0, 1],
                        v_lane_indices: vec![0, 1],
                        q: vec![0.1, 0.2],
                        k: vec![0.3, 0.4],
                        v: vec![0.5, 0.6],
                    }),
                    mix: Some(CarryableAttentionMixState {
                        width: 8,
                        score_provenance: Some(AttentionCheckpointProvenance {
                            layer_index: 3,
                            operator_kind: "AttentionOut".into(),
                            layer_distance_to_boundary: 0,
                        }),
                        value_provenance: Some(AttentionCheckpointProvenance {
                            layer_index: 3,
                            operator_kind: "AttentionOut".into(),
                            layer_distance_to_boundary: 0,
                        }),
                        score_lane_indices: vec![0, 1],
                        value_lane_indices: vec![0, 1],
                        scores: vec![0.7, 0.8],
                        values: vec![0.9, 1.0],
                    }),
                }),
                ffn: None,
            }),
            None,
        )
        .unwrap();
        let layout = StageLayout {
            model_id: "Toy".into(),
            stage_id: "packed-head".into(),
            start_layer: 0,
            end_layer: 4,
            is_head: true,
            is_tail: false,
        };
        let frame = StageForwardFrame::from_tensor(
            "Toy",
            &layout,
            StageTensor {
                request_id: "req-aged-boundary".into(),
                kind: PayloadKind::HiddenState,
                stage_trace: vec!["packed-head".into()],
                hidden_dim: 8,
                bytes: vec![0; 32],
                prompt_text: None,
                max_tokens: None,
                continuation: Some(StageContinuation {
                    version: 1,
                    stage_role: "head".into(),
                    next_layer_index: Some(5),
                    completed_layers: 5,
                    operator_layers: 5,
                    has_attention_path: true,
                    has_ffn_path: false,
                    has_projection_path: true,
                }),
                transient: None,
                carry: Some(merged),
            },
            Some("packed-tail".into()),
        );
        let programs = vec![LayerExecutionProgram {
            layer_index: 5,
            hidden_dim: Some(8),
            q_out_dim: Some(8),
            k_out_dim: Some(8),
            v_out_dim: Some(8),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.5.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionK,
                    vec!["blk.5.attn_k.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionV,
                    vec!["blk.5.attn_v.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.5.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let transfer = frame.to_transfer_frame_for_execution_boundary(&layout, &programs);
        let plan = frame.to_boundary_plan_for_execution_boundary(&layout, &programs);

        plan.validate_against_transfer(&transfer).unwrap();
        assert_eq!(plan.expected_attention_q_distance, Some(1));
        assert_eq!(plan.expected_attention_k_distance, Some(1));
        assert_eq!(plan.expected_attention_v_distance, Some(1));
        assert_eq!(plan.expected_attention_score_distance, Some(1));
        assert_eq!(plan.expected_attention_value_distance, Some(1));
        assert_eq!(plan.resumable_attention_q_max_distance, Some(2));
        assert_eq!(plan.resumable_attention_k_max_distance, Some(2));
        assert_eq!(plan.resumable_attention_v_max_distance, Some(2));
        assert_eq!(plan.resumable_attention_score_max_distance, Some(1));
        assert_eq!(plan.resumable_attention_value_max_distance, Some(1));
    }

    #[test]
    fn freshness_policy_derives_slice_limits_from_resume_semantics_not_op_order() {
        let capabilities = StageResumeCapabilities {
            attention_path: true,
            attention_q: true,
            attention_k: true,
            attention_v: true,
            attention_projection: true,
            attention_mix: true,
            ffn_path: false,
            projection_path: false,
        };
        let programs = vec![LayerExecutionProgram {
            layer_index: 21,
            hidden_dim: Some(8),
            q_out_dim: Some(8),
            k_out_dim: Some(8),
            v_out_dim: Some(8),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.21.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionK,
                    vec!["blk.21.attn_k.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionV,
                    vec!["blk.21.attn_v.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.21.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        }];

        let freshness =
            StageResumeFreshnessPolicy::from_execution_programs(&programs, Some(21), &capabilities);

        assert_eq!(freshness.attention_q_max_distance, Some(2));
        assert_eq!(freshness.attention_k_max_distance, Some(2));
        assert_eq!(freshness.attention_v_max_distance, Some(2));
        assert_eq!(freshness.attention_score_max_distance, Some(1));
        assert_eq!(freshness.attention_value_max_distance, Some(1));
    }

    #[test]
    fn resume_path_without_entry_blend_is_treated_as_overwrite() {
        let direct = StageResumeFreshnessPolicy::direct_resume_path(true, None).unwrap();
        let after_projection =
            StageResumeFreshnessPolicy::after_projection_resume_path(true, true, None).unwrap();

        assert_eq!(direct.recompute_phase, AttentionSliceRecomputePhase::Direct);
        assert_eq!(direct.blend_strength, None);
        assert_eq!(
            StageResumeFreshnessPolicy::max_distance_for_resume_path(Some(direct)),
            Some(0)
        );

        assert_eq!(
            after_projection.recompute_phase,
            AttentionSliceRecomputePhase::AfterProjection
        );
        assert_eq!(after_projection.blend_strength, None);
        assert_eq!(
            StageResumeFreshnessPolicy::max_distance_for_resume_path(Some(after_projection)),
            Some(0)
        );
    }

    #[test]
    fn execution_op_resume_descriptors_drive_attention_paths() {
        let program = LayerExecutionProgram {
            layer_index: 7,
            hidden_dim: Some(8),
            q_out_dim: Some(8),
            k_out_dim: Some(8),
            v_out_dim: Some(8),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.7.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.7.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        };

        let (q_path, k_path, v_path, score_path, value_path) =
            StageResumeFreshnessPolicy::attention_resume_paths(Some(&program));

        assert_eq!(
            q_path,
            Some(AttentionSliceResumePath {
                recompute_phase: AttentionSliceRecomputePhase::Direct,
                blend_strength: Some(AttentionBlendStrength::StrongBlend),
                blend_weight_milli: Some(350),
            })
        );
        assert_eq!(k_path, None);
        assert_eq!(v_path, None);
        assert_eq!(
            score_path,
            Some(AttentionSliceResumePath {
                recompute_phase: AttentionSliceRecomputePhase::AfterProjection,
                blend_strength: Some(AttentionBlendStrength::WeakBlend),
                blend_weight_milli: Some(250),
            })
        );
        assert_eq!(value_path, score_path);
    }

    #[test]
    fn resume_contract_summary_preserves_typed_attention_weights() {
        let program = LayerExecutionProgram {
            layer_index: 7,
            hidden_dim: Some(8),
            q_out_dim: Some(8),
            k_out_dim: Some(8),
            v_out_dim: Some(8),
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![
                ExecutionOp::new(
                    ExecutionOpKind::AttentionQ,
                    vec!["blk.7.attn_q.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionK,
                    vec!["blk.7.attn_k.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionV,
                    vec!["blk.7.attn_v.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
                ExecutionOp::new(
                    ExecutionOpKind::AttentionOut,
                    vec!["blk.7.attn_output.weight".into()],
                    ExecutionBinding::QuantizedMatrix,
                    "quantized matrix tensors",
                ),
            ],
        };
        let capabilities = StageResumeCapabilities {
            attention_path: true,
            attention_q: true,
            attention_k: true,
            attention_v: true,
            attention_projection: true,
            attention_mix: true,
            ffn_path: false,
            projection_path: false,
        };

        let summary =
            StageResumeContractSummary::from_execution_programs(&[program], Some(7), &capabilities);

        let projection = Some(StageAttentionResumeContractSummary {
            phase: StageAttentionResumePhase::Direct,
            blend: StageAttentionResumeBlend::StrongBlend,
            blend_weight_milli: Some(350),
        });
        let mix = Some(StageAttentionResumeContractSummary {
            phase: StageAttentionResumePhase::AfterProjection,
            blend: StageAttentionResumeBlend::WeakBlend,
            blend_weight_milli: Some(250),
        });
        assert_eq!(summary.attention_q, projection);
        assert_eq!(summary.attention_k, projection);
        assert_eq!(summary.attention_v, projection);
        assert_eq!(summary.attention_score, mix);
        assert_eq!(summary.attention_value, mix);
    }

    #[test]
    fn resume_contract_summary_rejects_inconsistent_freshness_limits() {
        let summary = StageResumeContractSummary {
            attention_q: Some(StageAttentionResumeContractSummary {
                phase: StageAttentionResumePhase::Direct,
                blend: StageAttentionResumeBlend::Overwrite,
                blend_weight_milli: None,
            }),
            ..StageResumeContractSummary::default()
        };
        let err = summary
            .validate_against_freshness(StageResumeFreshnessPolicy {
                attention_q_max_distance: Some(2),
                ..StageResumeFreshnessPolicy::default()
            })
            .unwrap_err()
            .to_string();

        assert!(err.contains("attention q contract implies max distance 0"));
    }

    #[test]
    fn resume_contract_summary_rejects_blend_weight_strength_mismatch() {
        let summary = StageResumeContractSummary {
            attention_score: Some(StageAttentionResumeContractSummary {
                phase: StageAttentionResumePhase::AfterProjection,
                blend: StageAttentionResumeBlend::WeakBlend,
                blend_weight_milli: Some(350),
            }),
            ..StageResumeContractSummary::default()
        };
        let err = summary
            .validate_against_freshness(StageResumeFreshnessPolicy {
                attention_score_max_distance: Some(1),
                ..StageResumeFreshnessPolicy::default()
            })
            .unwrap_err()
            .to_string();

        assert!(err.contains("attention score weak blend contract has non-weak weight 350"));
    }

    #[test]
    fn execution_op_stored_resume_contract_can_declare_overwrite_path() {
        let op = ExecutionOp::new(
            ExecutionOpKind::AttentionQ,
            vec!["blk.7.attn_q.weight".into()],
            ExecutionBinding::QuantizedMatrix,
            "quantized matrix tensors",
        )
        .with_resume_contract(ExecutionResumeContract {
            attention: vec![ExecutionAttentionResumeContract {
                slice: AttentionResumeSlice::Q,
                recompute_phase: AttentionSliceRecomputePhase::Direct,
                blend_mode: ExecutionResumeBlendMode::Overwrite,
            }],
        });
        let program = LayerExecutionProgram {
            layer_index: 7,
            hidden_dim: Some(8),
            q_out_dim: Some(8),
            k_out_dim: None,
            v_out_dim: None,
            ffn_inner_dim: None,
            runnable_sketch: true,
            ops: vec![op],
        };

        let (q_path, _, _, _, _) =
            StageResumeFreshnessPolicy::attention_resume_paths(Some(&program));

        assert_eq!(
            q_path,
            Some(AttentionSliceResumePath {
                recompute_phase: AttentionSliceRecomputePhase::Direct,
                blend_strength: None,
                blend_weight_milli: None,
            })
        );
        assert_eq!(
            StageResumeFreshnessPolicy::max_distance_for_resume_path(q_path),
            Some(0)
        );
    }

    #[test]
    fn packed_backend_rejects_attention_carry_that_is_too_stale_for_resume_policy() {
        let tail = PackedResidencySketchBackend {
            index_path: PathBuf::from("synthetic-packed-tail"),
            layout: Some(packed_tail_layout()),
            store: None,
            model_view: Some(StageModelView {
                role: "tail".into(),
                prompt_ingress: Vec::new(),
                positional: Vec::new(),
                shared_auxiliary: Vec::new(),
                layers: Vec::new(),
                operator_layers: Vec::new(),
                execution_layers: Vec::new(),
                execution_programs: vec![LayerExecutionProgram {
                    layer_index: 1,
                    hidden_dim: Some(64),
                    q_out_dim: Some(1),
                    k_out_dim: Some(1),
                    v_out_dim: Some(1),
                    ffn_inner_dim: None,
                    runnable_sketch: true,
                    ops: vec![
                        ExecutionOp::new(
                            ExecutionOpKind::AttentionQ,
                            vec!["blk.1.attn_q.weight".into()],
                            ExecutionBinding::QuantizedMatrix,
                            "quantized matrix tensors",
                        ),
                        ExecutionOp::new(
                            ExecutionOpKind::AttentionK,
                            vec!["blk.1.attn_k.weight".into()],
                            ExecutionBinding::QuantizedMatrix,
                            "quantized matrix tensors",
                        ),
                        ExecutionOp::new(
                            ExecutionOpKind::AttentionV,
                            vec!["blk.1.attn_v.weight".into()],
                            ExecutionBinding::QuantizedMatrix,
                            "quantized matrix tensors",
                        ),
                        ExecutionOp::new(
                            ExecutionOpKind::AttentionOut,
                            vec!["blk.1.attn_output.weight".into()],
                            ExecutionBinding::QuantizedMatrix,
                            "quantized matrix tensors",
                        ),
                    ],
                }],
                tail_only: Vec::new(),
            }),
            debug_layer_cap: None,
        };

        let request = StageResumeRequest {
            version: STAGE_FORWARD_FRAME_VERSION,
            model_id: "Toy".into(),
            target_stage_id: Some("packed-tail".into()),
            boundary: StageBoundaryPlan {
                source_stage_id: "packed-head".into(),
                source_stage_role: "head".into(),
                source_stage_start: 0,
                source_stage_end: 0,
                target_stage_id: Some("packed-tail".into()),
                next_layer_index: Some(1),
                expected_payload_kind: PayloadKind::HiddenState,
                expected_hidden_dim: 64,
                carry_policy: StageCarryPolicy {
                    carry_attention: true,
                    carry_ffn: false,
                },
                expects_attention_carry: true,
                expects_ffn_carry: false,
                expected_attention_width: Some(1),
                expected_attention_lanes: Some(1),
                expects_attention_projection_carry: true,
                expects_attention_mix_carry: true,
                expected_attention_q_lanes: Some(1),
                expected_attention_k_lanes: Some(1),
                expected_attention_v_lanes: Some(1),
                expected_attention_score_lanes: Some(1),
                expected_attention_value_lanes: Some(1),
                expected_attention_q_distance: Some(2),
                expected_attention_k_distance: Some(2),
                expected_attention_v_distance: Some(2),
                expected_attention_score_distance: Some(2),
                expected_attention_value_distance: Some(2),
                expected_attention_projection_lanes: Some(1),
                expected_attention_mix_lanes: Some(1),
                expected_ffn_width: None,
                expected_ffn_lanes: None,
                resumable_attention_path: true,
                resumable_attention_q: true,
                resumable_attention_k: true,
                resumable_attention_v: true,
                resumable_attention_q_lanes: Some(1),
                resumable_attention_k_lanes: Some(1),
                resumable_attention_v_lanes: Some(1),
                resumable_attention_q_max_distance: Some(2),
                resumable_attention_k_max_distance: Some(2),
                resumable_attention_v_max_distance: Some(2),
                resumable_attention_score_max_distance: Some(1),
                resumable_attention_value_max_distance: Some(1),
                resumable_attention_contract: StageResumeContractSummary::default(),
                resumable_attention_projection: true,
                resumable_attention_mix: true,
                resumable_ffn_path: false,
                resumable_projection_path: false,
                operator_layers: 1,
                completed_layers: Some(1),
            },
            transfer: StageTransferFrame {
                version: STAGE_FORWARD_FRAME_VERSION,
                model_id: "Toy".into(),
                route: StageRoute {
                    source_stage_id: "packed-head".into(),
                    source_stage_start: 0,
                    source_stage_end: 0,
                    source_stage_role: "head".into(),
                    target_stage_id: Some("packed-tail".into()),
                },
                payload: StageTransferPayload {
                    request_id: "req-stale".into(),
                    kind: PayloadKind::HiddenState,
                    stage_trace: vec!["packed-head".into()],
                    hidden_dim: 64,
                    bytes: vec![0; 64 * 4],
                    prompt_text: Some("hello".into()),
                    max_tokens: Some(8),
                },
                state: StageTransferStateEnvelope {
                    continuation: Some(StageContinuation {
                        version: 1,
                        stage_role: "head".into(),
                        next_layer_index: Some(1),
                        completed_layers: 1,
                        operator_layers: 1,
                        has_attention_path: true,
                        has_ffn_path: false,
                        has_projection_path: false,
                    }),
                    transient: None,
                    carry: Some(StageCarryState {
                        attention: Some(CarryableAttentionState {
                            contract: StageResumeContractSummary::default(),
                            projection: Some(CarryableAttentionProjectionState {
                                width: 1,
                                q_provenance: Some(AttentionCheckpointProvenance {
                                    layer_index: 3,
                                    operator_kind: "AttentionQ".into(),
                                    layer_distance_to_boundary: 2,
                                }),
                                k_provenance: Some(AttentionCheckpointProvenance {
                                    layer_index: 3,
                                    operator_kind: "AttentionK".into(),
                                    layer_distance_to_boundary: 2,
                                }),
                                v_provenance: Some(AttentionCheckpointProvenance {
                                    layer_index: 3,
                                    operator_kind: "AttentionV".into(),
                                    layer_distance_to_boundary: 2,
                                }),
                                q_lane_indices: vec![0],
                                k_lane_indices: vec![0],
                                v_lane_indices: vec![0],
                                q: vec![0.1],
                                k: vec![0.2],
                                v: vec![0.3],
                            }),
                            mix: Some(CarryableAttentionMixState {
                                width: 1,
                                score_provenance: Some(AttentionCheckpointProvenance {
                                    layer_index: 3,
                                    operator_kind: "AttentionOut".into(),
                                    layer_distance_to_boundary: 2,
                                }),
                                value_provenance: Some(AttentionCheckpointProvenance {
                                    layer_index: 3,
                                    operator_kind: "AttentionOut".into(),
                                    layer_distance_to_boundary: 2,
                                }),
                                score_lane_indices: vec![0],
                                value_lane_indices: vec![0],
                                scores: vec![0.4],
                                values: vec![0.5],
                            }),
                        }),
                        ffn: None,
                    }),
                },
            },
        };

        let mut contract_mismatch_request = request.clone();
        contract_mismatch_request
            .boundary
            .resumable_attention_contract
            .attention_q = Some(StageAttentionResumeContractSummary {
            phase: StageAttentionResumePhase::AfterProjection,
            blend: StageAttentionResumeBlend::StrongBlend,
            blend_weight_milli: Some(350),
        });
        let contract_mismatch = tail.admit_resume_request(&contract_mismatch_request);
        assert!(!contract_mismatch.accepted);
        assert!(
            contract_mismatch
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("attention resume contract does not match target execution program")
        );

        let decision = tail.admit_resume_request(&request);
        assert!(!decision.accepted);
        assert!(
            decision
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("attention score freshness distance 2 exceeds target limit 1")
        );
    }

    #[test]
    fn stage_tensor_bytes_base64_json_roundtrip() {
        let mut raw = vec![0u8; 8 * 1024];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        let tensor = StageTensor {
            request_id: "req".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["head".into()],
            hidden_dim: 2048,
            bytes: raw.clone(),
            prompt_text: None,
            max_tokens: None,
            continuation: None,
            transient: None,
            carry: None,
        };

        let json = serde_json::to_string(&tensor).expect("serialize json");
        assert!(
            !json.contains("[0,1,2,3,4,5,6,7,8,9,10"),
            "bytes must not serialize as json number array"
        );
        assert!(
            json.contains("AAECAwQFBgcICQoL"),
            "expected base64 prefix of 0..11 in json payload"
        );
        assert!(
            json.len() < raw.len() * 2,
            "base64 json ({}) should be < 2x raw ({}); number-array was ~4x",
            json.len(),
            raw.len()
        );

        let back: StageTensor = serde_json::from_str(&json).expect("deserialize json");
        assert_eq!(back.bytes, raw, "json roundtrip must preserve bytes");
    }
}
