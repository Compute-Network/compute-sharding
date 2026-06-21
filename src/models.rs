use std::fmt;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const MODEL_ID: &str = "gemma-4-e4b-q4";
pub const MODEL_LABEL: &str = "Gemma 4 E4B Q4 two-stage split";
pub const HF_REPO: &str = "ComputeNet-sh/gemma-4-e4b-q4-gguf-stages";
pub const TOTAL_LAYERS: u32 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ShardKind {
    Head,
    Tail,
}

impl fmt::Display for ShardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardKind::Head => write!(f, "head"),
            ShardKind::Tail => write!(f, "tail"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ShardSelection {
    Head,
    Tail,
    Both,
}

#[derive(Debug, Clone, Copy)]
pub struct ShardSpec {
    pub kind: ShardKind,
    pub filename: &'static str,
    pub url: &'static str,
    pub stage_id: &'static str,
    pub start_layer: u32,
    pub end_layer: u32,
    pub approx_gb: f32,
}

pub const HEAD_SHARD: ShardSpec = ShardSpec {
    kind: ShardKind::Head,
    filename: "gemma-4-e4b-q4-head-0-20.gguf",
    url: "https://huggingface.co/ComputeNet-sh/gemma-4-e4b-q4-gguf-stages/resolve/main/gemma-4-e4b-q4-head-0-20.gguf",
    stage_id: "stage-0-20",
    start_layer: 0,
    end_layer: 20,
    approx_gb: 2.68,
};

pub const TAIL_SHARD: ShardSpec = ShardSpec {
    kind: ShardKind::Tail,
    filename: "gemma-4-e4b-q4-tail-21-41.gguf",
    url: "https://huggingface.co/ComputeNet-sh/gemma-4-e4b-q4-gguf-stages/resolve/main/gemma-4-e4b-q4-tail-21-41.gguf",
    stage_id: "stage-21-41",
    start_layer: 21,
    end_layer: 41,
    approx_gb: 2.69,
};

pub fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".compute-sharding")
        .join("models")
        .join(MODEL_ID)
}

pub fn shards_for_selection(selection: ShardSelection) -> Vec<ShardSpec> {
    match selection {
        ShardSelection::Head => vec![HEAD_SHARD],
        ShardSelection::Tail => vec![TAIL_SHARD],
        ShardSelection::Both => vec![HEAD_SHARD, TAIL_SHARD],
    }
}

pub fn shard_for_kind(kind: ShardKind) -> ShardSpec {
    match kind {
        ShardKind::Head => HEAD_SHARD,
        ShardKind::Tail => TAIL_SHARD,
    }
}

pub fn shard_path(root: &std::path::Path, shard: ShardSpec) -> PathBuf {
    root.join(shard.filename)
}

pub fn print_catalog() {
    println!("Compute sharding catalog");
    println!("model: {MODEL_ID} ({MODEL_LABEL})");
    println!("source: https://huggingface.co/{HF_REPO}");
    println!("layers: {TOTAL_LAYERS}");
    println!();
    for shard in [HEAD_SHARD, TAIL_SHARD] {
        println!(
            "{:<4} layers {:>2}-{:>2}  {:>4.2} GB  {}",
            shard.kind, shard.start_layer, shard.end_layer, shard.approx_gb, shard.filename
        );
        println!("     {}", shard.url);
    }
}
