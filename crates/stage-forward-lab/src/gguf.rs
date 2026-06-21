use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub file_size_bytes: u64,
    pub tensor_data_offset: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageSplit {
    pub stage_index: u32,
    pub start_layer: u32,
    pub end_layer: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TensorRole {
    Layer { layer_index: u32 },
    PromptIngress,
    Positional,
    SharedAuxiliary,
    TailOnly,
    UnknownGlobal,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlannedTensor {
    pub name: String,
    pub role: TensorRole,
    pub assigned_stage: Option<u32>,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageTensorPlan {
    pub splits: Vec<StageSplit>,
    pub planned_tensors: Vec<PlannedTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TensorGroup {
    pub name: String,
    pub tensor_names: Vec<String>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageManifest {
    pub stage_index: u32,
    pub start_layer: u32,
    pub end_layer: u32,
    pub prompt_ingress: TensorGroup,
    pub positional: TensorGroup,
    pub replicated_aux: TensorGroup,
    pub owned: TensorGroup,
    pub tail_only: TensorGroup,
    pub unknown_global: TensorGroup,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageRuntimePlan {
    pub stage_index: u32,
    pub role: String,
    pub required: TensorGroup,
    pub optional: TensorGroup,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TensorSlice {
    pub name: String,
    pub file_offset: u64,
    pub byte_len: u64,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageRuntimeSliceManifest {
    pub stage_index: u32,
    pub role: String,
    pub required: Vec<TensorSlice>,
    pub optional: Vec<TensorSlice>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelShardManifest {
    pub model_name: String,
    pub architecture: String,
    pub hidden_size: Option<u32>,
    pub feed_forward_size: Option<u32>,
    pub attention_heads: Option<u32>,
    pub total_layers: u32,
    pub stages: Vec<StageManifest>,
    pub runtime_plan: Vec<StageRuntimePlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenShardBundle {
    pub root_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub stage_manifest_paths: Vec<PathBuf>,
    pub runtime_plan_paths: Vec<PathBuf>,
    pub runtime_slice_paths: Vec<PathBuf>,
}

impl GgufFile {
    pub fn parse_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::parse_bytes(&bytes)
    }

    pub fn parse_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            bail!("Not a GGUF file");
        }

        let version = read_u32(&mut cursor)?;
        let tensor_count = match version {
            1 => read_u32(&mut cursor)? as u64,
            _ => read_u64(&mut cursor)?,
        };
        let metadata_count = match version {
            1 => read_u32(&mut cursor)? as u64,
            _ => read_u64(&mut cursor)?,
        };

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut cursor, version)?;
            let value_type = read_u32(&mut cursor)?;
            let value = read_metadata_value(&mut cursor, version, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor, version)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(match version {
                    1 => read_u32(&mut cursor)? as u64,
                    _ => read_u64(&mut cursor)?,
                });
            }
            let ggml_type = read_u32(&mut cursor)?;
            let offset = read_u64(&mut cursor)?;
            tensors.push(TensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|value| match value {
                MetadataValue::Uint32(v) => Some(*v as u64),
                MetadataValue::Uint64(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(32);
        let tensor_data_offset = align_to(cursor.position(), alignment);

        Ok(Self {
            version,
            tensor_count,
            file_size_bytes: bytes.len() as u64,
            tensor_data_offset,
            metadata,
            tensors,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(MetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn block_indices(&self) -> Vec<u32> {
        let mut blocks = BTreeSet::new();
        for tensor in &self.tensors {
            if let Some(rest) = tensor.name.strip_prefix("blk.") {
                if let Some(idx) = rest
                    .split('.')
                    .next()
                    .and_then(|part| part.parse::<u32>().ok())
                {
                    blocks.insert(idx);
                }
            }
        }
        blocks.into_iter().collect()
    }

    pub fn inferred_layer_count(&self) -> Option<u32> {
        if let Some(value) = self.metadata_u32("gemma4.block_count") {
            return Some(value);
        }
        if let Some(value) = self.metadata_u32("gemma.block_count") {
            return Some(value);
        }
        if let Some(value) = self.metadata_u32("llama.block_count") {
            return Some(value);
        }
        let blocks = self.block_indices();
        blocks.last().copied().map(|max| max + 1)
    }

    pub fn metadata_string(&self, key: &str) -> Option<String> {
        self.metadata.get(key).map(metadata_value_to_string)
    }

    pub fn metadata_string_array(&self, key: &str) -> Option<Vec<String>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                Some(arr.iter().map(metadata_value_to_string).collect())
            }
            _ => None,
        }
    }

    pub fn metadata_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => Some(
                arr.iter()
                    .map(|v| match v {
                        MetadataValue::Float32(f) => *f,
                        MetadataValue::Float64(f) => *f as f32,
                        MetadataValue::Int32(i) => *i as f32,
                        MetadataValue::Uint32(u) => *u as f32,
                        _ => 0.0,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Uint32(v)) => Some(*v),
            Some(MetadataValue::Uint64(v)) => (*v).try_into().ok(),
            Some(MetadataValue::Int32(v)) if *v >= 0 => Some(*v as u32),
            Some(MetadataValue::Int64(v)) if *v >= 0 => (*v as u64).try_into().ok(),
            _ => None,
        }
    }

    pub fn hidden_size(&self) -> Option<u32> {
        self.metadata_u32("gemma4.embedding_length")
            .or_else(|| self.metadata_u32("gemma.embedding_length"))
            .or_else(|| self.metadata_u32("llama.embedding_length"))
    }

    pub fn feed_forward_length(&self) -> Option<u32> {
        self.metadata_u32("gemma4.feed_forward_length")
            .or_else(|| self.metadata_u32("gemma.feed_forward_length"))
            .or_else(|| self.metadata_u32("llama.feed_forward_length"))
    }

    pub fn attention_head_count(&self) -> Option<u32> {
        self.metadata_u32("gemma4.attention.head_count")
            .or_else(|| self.metadata_u32("gemma.attention.head_count"))
            .or_else(|| self.metadata_u32("llama.attention.head_count"))
    }

    pub fn suggest_even_stage_split(&self, stages: u32) -> Option<Vec<StageSplit>> {
        let total_layers = self.inferred_layer_count()?;
        if stages == 0 || stages > total_layers {
            return None;
        }

        let base = total_layers / stages;
        let remainder = total_layers % stages;
        let mut start = 0u32;
        let mut result = Vec::with_capacity(stages as usize);
        for stage_index in 0..stages {
            let extra = if stage_index < remainder { 1 } else { 0 };
            let span = base + extra;
            let end = start + span - 1;
            result.push(StageSplit {
                stage_index,
                start_layer: start,
                end_layer: end,
            });
            start = end + 1;
        }
        Some(result)
    }

    pub fn tensor_byte_len(&self, tensor_name: &str) -> Option<u64> {
        let mut ordered = self.tensors.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|tensor| tensor.offset);
        let idx = ordered
            .iter()
            .position(|tensor| tensor.name == tensor_name)?;
        let current = ordered[idx];
        let next_offset = ordered
            .get(idx + 1)
            .map(|tensor| tensor.offset)
            .unwrap_or_else(|| self.file_size_bytes.saturating_sub(self.tensor_data_offset));
        Some(next_offset.saturating_sub(current.offset))
    }

    pub fn tensor_file_offset(&self, tensor_name: &str) -> Option<u64> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .map(|tensor| self.tensor_data_offset + tensor.offset)
    }

    pub fn tensor_slice(&self, tensor_name: &str) -> Option<TensorSlice> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)?;
        Some(TensorSlice {
            name: tensor_name.to_string(),
            file_offset: self.tensor_file_offset(tensor_name)?,
            byte_len: self.tensor_byte_len(tensor_name)?,
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
        })
    }

    pub fn plan_for_splits(&self, splits: &[StageSplit]) -> StageTensorPlan {
        let planned_tensors = self
            .tensors
            .iter()
            .map(|tensor| {
                let role = classify_tensor_role(&tensor.name);
                let assigned_stage = match role {
                    TensorRole::Layer { layer_index } => splits
                        .iter()
                        .find(|split| {
                            layer_index >= split.start_layer && layer_index <= split.end_layer
                        })
                        .map(|split| split.stage_index),
                    TensorRole::TailOnly => splits.last().map(|split| split.stage_index),
                    TensorRole::PromptIngress
                    | TensorRole::Positional
                    | TensorRole::SharedAuxiliary
                    | TensorRole::UnknownGlobal => None,
                };
                PlannedTensor {
                    name: tensor.name.clone(),
                    role,
                    assigned_stage,
                    byte_len: self.tensor_byte_len(&tensor.name).unwrap_or(0),
                }
            })
            .collect();

        StageTensorPlan {
            splits: splits.to_vec(),
            planned_tensors,
        }
    }
}

impl StageTensorPlan {
    pub fn total_bytes(&self) -> u64 {
        self.planned_tensors
            .iter()
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn stage_bytes(&self, stage_index: u32) -> u64 {
        self.planned_tensors
            .iter()
            .filter(|tensor| {
                tensor.assigned_stage == Some(stage_index)
                    && matches!(tensor.role, TensorRole::Layer { .. })
            })
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn prompt_ingress_bytes(&self) -> u64 {
        self.planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::PromptIngress))
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn positional_bytes(&self) -> u64 {
        self.planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::Positional))
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn replicated_aux_bytes(&self) -> u64 {
        self.planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::SharedAuxiliary))
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn tail_only_bytes(&self) -> u64 {
        self.planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::TailOnly))
            .map(|tensor| tensor.byte_len)
            .sum()
    }

    pub fn build_manifest(&self, file: &GgufFile) -> ModelShardManifest {
        let prompt_ingress_tensors = self
            .planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::PromptIngress))
            .collect::<Vec<_>>();
        let positional_tensors = self
            .planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::Positional))
            .collect::<Vec<_>>();
        let replicated_aux_tensors = self
            .planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::SharedAuxiliary))
            .collect::<Vec<_>>();
        let tail_only_tensors = self
            .planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::TailOnly))
            .collect::<Vec<_>>();
        let unknown_global_tensors = self
            .planned_tensors
            .iter()
            .filter(|tensor| matches!(tensor.role, TensorRole::UnknownGlobal))
            .collect::<Vec<_>>();

        let stages: Vec<StageManifest> = self
            .splits
            .iter()
            .map(|split| StageManifest {
                stage_index: split.stage_index,
                start_layer: split.start_layer,
                end_layer: split.end_layer,
                prompt_ingress: TensorGroup {
                    name: "prompt_ingress".into(),
                    tensor_names: prompt_ingress_tensors
                        .iter()
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                    total_bytes: prompt_ingress_tensors
                        .iter()
                        .map(|tensor| tensor.byte_len)
                        .sum(),
                },
                positional: TensorGroup {
                    name: "positional".into(),
                    tensor_names: positional_tensors
                        .iter()
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                    total_bytes: positional_tensors
                        .iter()
                        .map(|tensor| tensor.byte_len)
                        .sum(),
                },
                replicated_aux: TensorGroup {
                    name: "replicated_aux".into(),
                    tensor_names: replicated_aux_tensors
                        .iter()
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                    total_bytes: replicated_aux_tensors
                        .iter()
                        .map(|tensor| tensor.byte_len)
                        .sum(),
                },
                owned: TensorGroup {
                    name: format!("stage{}_owned", split.stage_index + 1),
                    tensor_names: self
                        .planned_tensors
                        .iter()
                        .filter(|tensor| {
                            tensor.assigned_stage == Some(split.stage_index)
                                && matches!(tensor.role, TensorRole::Layer { .. })
                        })
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                    total_bytes: self.stage_bytes(split.stage_index),
                },
                tail_only: TensorGroup {
                    name: "tail_only".into(),
                    tensor_names: if split.stage_index + 1 == self.splits.len() as u32 {
                        tail_only_tensors
                            .iter()
                            .map(|tensor| tensor.name.clone())
                            .collect()
                    } else {
                        Vec::new()
                    },
                    total_bytes: if split.stage_index + 1 == self.splits.len() as u32 {
                        tail_only_tensors.iter().map(|tensor| tensor.byte_len).sum()
                    } else {
                        0
                    },
                },
                unknown_global: TensorGroup {
                    name: "unknown_global".into(),
                    tensor_names: unknown_global_tensors
                        .iter()
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                    total_bytes: unknown_global_tensors
                        .iter()
                        .map(|tensor| tensor.byte_len)
                        .sum(),
                },
            })
            .collect();

        let runtime_plan = stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| {
                let mut required = Vec::new();
                if idx == 0 {
                    required.extend(stage.prompt_ingress.tensor_names.iter().cloned());
                }
                required.extend(stage.positional.tensor_names.iter().cloned());
                required.extend(stage.replicated_aux.tensor_names.iter().cloned());
                required.extend(stage.owned.tensor_names.iter().cloned());
                if idx + 1 == stages.len() {
                    required.extend(stage.tail_only.tensor_names.iter().cloned());
                }

                let optional = if idx == 0 {
                    Vec::new()
                } else {
                    stage.prompt_ingress.tensor_names.clone()
                };

                StageRuntimePlan {
                    stage_index: stage.stage_index,
                    role: if idx == 0 {
                        "head".into()
                    } else if idx + 1 == stages.len() {
                        "tail".into()
                    } else {
                        "middle".into()
                    },
                    required: TensorGroup {
                        name: "required_runtime".into(),
                        total_bytes: required
                            .iter()
                            .filter_map(|name| {
                                self.planned_tensors
                                    .iter()
                                    .find(|tensor| tensor.name == *name)
                                    .map(|tensor| tensor.byte_len)
                            })
                            .sum(),
                        tensor_names: required,
                    },
                    optional: TensorGroup {
                        name: "optional_runtime".into(),
                        total_bytes: optional
                            .iter()
                            .filter_map(|name| {
                                self.planned_tensors
                                    .iter()
                                    .find(|tensor| tensor.name == *name)
                                    .map(|tensor| tensor.byte_len)
                            })
                            .sum(),
                        tensor_names: optional,
                    },
                }
            })
            .collect();

        ModelShardManifest {
            model_name: file
                .metadata_string("general.name")
                .unwrap_or_else(|| "unknown".into()),
            architecture: file.architecture().unwrap_or("unknown").to_string(),
            hidden_size: file.hidden_size(),
            feed_forward_size: file.feed_forward_length(),
            attention_heads: file.attention_head_count(),
            total_layers: file.inferred_layer_count().unwrap_or(0),
            stages,
            runtime_plan,
        }
    }

    pub fn write_bundle(&self, file: &GgufFile, out_dir: &Path) -> Result<WrittenShardBundle> {
        std::fs::create_dir_all(out_dir)?;
        let manifest = self.build_manifest(file);
        let manifest_path = out_dir.join("model-shard-manifest.json");
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

        let mut stage_manifest_paths = Vec::with_capacity(manifest.stages.len());
        let mut runtime_plan_paths = Vec::with_capacity(manifest.runtime_plan.len());
        let mut runtime_slice_paths = Vec::with_capacity(manifest.runtime_plan.len());
        for stage in &manifest.stages {
            let stage_path = out_dir.join(format!("stage-{}.json", stage.stage_index + 1));
            std::fs::write(&stage_path, serde_json::to_vec_pretty(stage)?)?;
            stage_manifest_paths.push(stage_path);

            let prompt_ingress_list = out_dir.join(format!(
                "stage-{}-prompt-ingress.txt",
                stage.stage_index + 1
            ));
            let positional_list =
                out_dir.join(format!("stage-{}-positional.txt", stage.stage_index + 1));
            let replicated_aux_list = out_dir.join(format!(
                "stage-{}-replicated-aux.txt",
                stage.stage_index + 1
            ));
            let owned_list = out_dir.join(format!("stage-{}-owned.txt", stage.stage_index + 1));
            let tail_list = out_dir.join(format!("stage-{}-tail-only.txt", stage.stage_index + 1));
            let unknown_global_list = out_dir.join(format!(
                "stage-{}-unknown-global.txt",
                stage.stage_index + 1
            ));

            std::fs::write(
                prompt_ingress_list,
                stage.prompt_ingress.tensor_names.join("\n") + "\n",
            )?;
            std::fs::write(
                positional_list,
                stage.positional.tensor_names.join("\n") + "\n",
            )?;
            std::fs::write(
                replicated_aux_list,
                stage.replicated_aux.tensor_names.join("\n") + "\n",
            )?;
            std::fs::write(owned_list, stage.owned.tensor_names.join("\n") + "\n")?;
            std::fs::write(tail_list, stage.tail_only.tensor_names.join("\n") + "\n")?;
            std::fs::write(
                unknown_global_list,
                stage.unknown_global.tensor_names.join("\n") + "\n",
            )?;
        }

        for runtime in &manifest.runtime_plan {
            let runtime_path =
                out_dir.join(format!("runtime-stage-{}.json", runtime.stage_index + 1));
            std::fs::write(&runtime_path, serde_json::to_vec_pretty(runtime)?)?;
            runtime_plan_paths.push(runtime_path);

            let required_list = out_dir.join(format!(
                "runtime-stage-{}-required.txt",
                runtime.stage_index + 1
            ));
            let optional_list = out_dir.join(format!(
                "runtime-stage-{}-optional.txt",
                runtime.stage_index + 1
            ));

            std::fs::write(
                required_list,
                runtime.required.tensor_names.join("\n") + "\n",
            )?;
            std::fs::write(
                optional_list,
                runtime.optional.tensor_names.join("\n") + "\n",
            )?;

            let slice_manifest = StageRuntimeSliceManifest {
                stage_index: runtime.stage_index,
                role: runtime.role.clone(),
                required: runtime
                    .required
                    .tensor_names
                    .iter()
                    .map(|name| {
                        file.tensor_slice(name).ok_or_else(|| {
                            anyhow::anyhow!("Missing tensor slice for required tensor {}", name)
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                optional: runtime
                    .optional
                    .tensor_names
                    .iter()
                    .map(|name| {
                        file.tensor_slice(name).ok_or_else(|| {
                            anyhow::anyhow!("Missing tensor slice for optional tensor {}", name)
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let slice_path = out_dir.join(format!(
                "runtime-stage-{}-slices.json",
                runtime.stage_index + 1
            ));
            std::fs::write(&slice_path, serde_json::to_vec_pretty(&slice_manifest)?)?;
            runtime_slice_paths.push(slice_path);
        }

        Ok(WrittenShardBundle {
            root_dir: out_dir.to_path_buf(),
            manifest_path,
            stage_manifest_paths,
            runtime_plan_paths,
            runtime_slice_paths,
        })
    }
}

fn classify_tensor_role(name: &str) -> TensorRole {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(idx) = rest
            .split('.')
            .next()
            .and_then(|part| part.parse::<u32>().ok())
        {
            return TensorRole::Layer { layer_index: idx };
        }
    }

    if matches!(name, "token_embd.weight") {
        return TensorRole::PromptIngress;
    }

    if matches!(name, "rope_freqs.weight") {
        return TensorRole::Positional;
    }

    if matches!(
        name,
        "per_layer_token_embd.weight"
            | "per_layer_model_proj.weight"
            | "per_layer_proj_norm.weight"
    ) {
        return TensorRole::SharedAuxiliary;
    }

    if name.starts_with("output") {
        return TensorRole::TailOnly;
    }

    TensorRole::UnknownGlobal
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn metadata_value_to_string(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Uint8(v) => v.to_string(),
        MetadataValue::Int8(v) => v.to_string(),
        MetadataValue::Uint16(v) => v.to_string(),
        MetadataValue::Int16(v) => v.to_string(),
        MetadataValue::Uint32(v) => v.to_string(),
        MetadataValue::Int32(v) => v.to_string(),
        MetadataValue::Float32(v) => v.to_string(),
        MetadataValue::Bool(v) => v.to_string(),
        MetadataValue::String(v) => v.clone(),
        MetadataValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(metadata_value_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        MetadataValue::Uint64(v) => v.to_string(),
        MetadataValue::Int64(v) => v.to_string(),
        MetadataValue::Float64(v) => v.to_string(),
    }
}

fn read_metadata_value(
    cursor: &mut Cursor<&[u8]>,
    version: u32,
    value_type: u32,
) -> Result<MetadataValue> {
    Ok(match value_type {
        0 => MetadataValue::Uint8(read_u8(cursor)?),
        1 => MetadataValue::Int8(read_i8(cursor)?),
        2 => MetadataValue::Uint16(read_u16(cursor)?),
        3 => MetadataValue::Int16(read_i16(cursor)?),
        4 => MetadataValue::Uint32(read_u32(cursor)?),
        5 => MetadataValue::Int32(read_i32(cursor)?),
        6 => MetadataValue::Float32(read_f32(cursor)?),
        7 => MetadataValue::Bool(read_u8(cursor)? != 0),
        8 => MetadataValue::String(read_string(cursor, version)?),
        9 => {
            let element_type = read_u32(cursor)?;
            let count = match version {
                1 => read_u32(cursor)? as u64,
                _ => read_u64(cursor)?,
            };
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_metadata_value(cursor, version, element_type)?);
            }
            MetadataValue::Array(values)
        }
        10 => MetadataValue::Uint64(read_u64(cursor)?),
        11 => MetadataValue::Int64(read_i64(cursor)?),
        12 => MetadataValue::Float64(read_f64(cursor)?),
        other => bail!("Unsupported GGUF metadata type {}", other),
    })
}

fn read_string(cursor: &mut Cursor<&[u8]>, version: u32) -> Result<String> {
    let len = match version {
        1 => read_u32(cursor)? as usize,
        _ => read_u64(cursor)? as usize,
    };
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
    Ok(read_u8(cursor)? as i8)
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parse_synthetic_v3_gguf() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 2); // tensors
        push_u64(&mut bytes, 2); // metadata

        push_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8);
        push_string(&mut bytes, "gemma");

        push_string(&mut bytes, "gemma.block_count");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 28);

        push_string(&mut bytes, "blk.0.attn_q.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 4);
        push_u64(&mut bytes, 8);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        push_string(&mut bytes, "blk.27.ffn_down.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 4);
        push_u64(&mut bytes, 8);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 128);

        let file = GgufFile::parse_bytes(&bytes).unwrap();
        assert_eq!(file.version, 3);
        assert_eq!(file.tensor_count, 2);
        assert_eq!(file.architecture(), Some("gemma"));
        assert_eq!(file.inferred_layer_count(), Some(28));
        assert_eq!(file.block_indices(), vec![0, 27]);
        assert_eq!(
            file.suggest_even_stage_split(2).unwrap(),
            vec![
                StageSplit {
                    stage_index: 0,
                    start_layer: 0,
                    end_layer: 13
                },
                StageSplit {
                    stage_index: 1,
                    start_layer: 14,
                    end_layer: 27
                },
            ]
        );
    }

    #[test]
    fn classify_tensor_roles() {
        assert_eq!(
            classify_tensor_role("blk.7.attn_q.weight"),
            TensorRole::Layer { layer_index: 7 }
        );
        assert_eq!(
            classify_tensor_role("token_embd.weight"),
            TensorRole::PromptIngress
        );
        assert_eq!(
            classify_tensor_role("output_norm.weight"),
            TensorRole::TailOnly
        );
        assert_eq!(
            classify_tensor_role("rope_freqs.weight"),
            TensorRole::Positional
        );
        assert_eq!(
            classify_tensor_role("per_layer_token_embd.weight"),
            TensorRole::SharedAuxiliary
        );
        assert_eq!(
            classify_tensor_role("general.weird"),
            TensorRole::UnknownGlobal
        );
    }

    #[test]
    fn manifest_builds_replicated_and_owned_groups() {
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

        let splits = vec![
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
        ];
        let plan = file.plan_for_splits(&splits);
        let manifest = plan.build_manifest(&file);

        assert_eq!(manifest.model_name, "Toy");
        assert_eq!(manifest.stages.len(), 2);
        assert_eq!(
            manifest.stages[0].prompt_ingress.tensor_names,
            vec!["token_embd.weight"]
        );
        assert_eq!(
            manifest.stages[0].owned.tensor_names,
            vec!["blk.0.attn_q.weight"]
        );
        assert!(manifest.stages[1].owned.tensor_names.is_empty());
        assert_eq!(
            manifest.stages[1].tail_only.tensor_names,
            vec!["output_norm.weight"]
        );
        assert_eq!(manifest.runtime_plan.len(), 2);
        assert_eq!(manifest.runtime_plan[0].role, "head");
        assert_eq!(manifest.runtime_plan[1].role, "tail");
        assert_eq!(
            manifest.runtime_plan[1].required.tensor_names,
            vec!["output_norm.weight"]
        );
        assert_eq!(
            manifest.runtime_plan[1].optional.tensor_names,
            vec!["token_embd.weight"]
        );
    }

    #[test]
    fn bundle_writer_emits_manifest_files() {
        let temp = tempfile::tempdir().unwrap();
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

        let splits = vec![
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
        ];
        let plan = file.plan_for_splits(&splits);
        let written = plan.write_bundle(&file, temp.path()).unwrap();

        assert!(written.manifest_path.exists());
        assert_eq!(written.stage_manifest_paths.len(), 2);
        assert_eq!(written.runtime_plan_paths.len(), 2);
        assert!(temp.path().join("stage-1-owned.txt").exists());
        assert!(temp.path().join("stage-2-tail-only.txt").exists());
        assert!(temp.path().join("runtime-stage-1.json").exists());
        assert!(temp.path().join("runtime-stage-2-required.txt").exists());
    }
}
