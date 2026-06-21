use anyhow::Result;
use stage_forward_lab::gguf::{GgufFile, TensorRole};
use std::path::PathBuf;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".compute")
                .join("models")
                .join("gemma-4-E4B-it-Q4_K_M.gguf")
        });

    let file = GgufFile::parse_file(&path)?;

    println!("path               : {}", path.display());
    println!("version            : {}", file.version);
    println!("tensor_count       : {}", file.tensor_count);
    println!(
        "architecture       : {}",
        file.architecture().unwrap_or("unknown")
    );
    println!(
        "inferred_layers    : {}",
        file.inferred_layer_count()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "metadata examples  : general.name={}, general.architecture={}",
        file.metadata_string("general.name")
            .unwrap_or_else(|| "n/a".into()),
        file.metadata_string("general.architecture")
            .unwrap_or_else(|| "n/a".into())
    );
    println!(
        "model dims         : hidden_size={} ffn_size={} heads={}",
        file.hidden_size()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into()),
        file.feed_forward_length()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into()),
        file.attention_head_count()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    if let Some(split) = file.suggest_even_stage_split(2) {
        let formatted = split
            .iter()
            .map(|stage| {
                format!(
                    "stage{}={}..{}",
                    stage.stage_index + 1,
                    stage.start_layer,
                    stage.end_layer
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("suggested 2-stage  : {}", formatted);
        let plan = file.plan_for_splits(&split);
        let manifest = plan.build_manifest(&file);
        println!(
            "planned bytes      : total={} stage1={} stage2={} ingress={} positional={} replicated_aux={} tail_only={}",
            plan.total_bytes(),
            plan.stage_bytes(0),
            plan.stage_bytes(1),
            plan.prompt_ingress_bytes(),
            plan.positional_bytes(),
            plan.replicated_aux_bytes(),
            plan.tail_only_bytes(),
        );
        println!("planned tensors:");
        for tensor in plan.planned_tensors.iter().take(20) {
            let role = match &tensor.role {
                TensorRole::Layer { layer_index } => format!("layer({layer_index})"),
                TensorRole::PromptIngress => "prompt_ingress".into(),
                TensorRole::Positional => "positional".into(),
                TensorRole::SharedAuxiliary => "replicated_aux".into(),
                TensorRole::TailOnly => "tail_only".into(),
                TensorRole::UnknownGlobal => "unknown_global".into(),
            };
            println!(
                "  - {} role={} stage={:?} bytes={}",
                tensor.name, role, tensor.assigned_stage, tensor.byte_len
            );
        }
        println!("stage manifests:");
        for stage in &manifest.stages {
            println!(
                "  - stage{} layers {}..{} ingress={} positional={} replicated_aux={} owned={} tail_only={} unknown_global={}",
                stage.stage_index + 1,
                stage.start_layer,
                stage.end_layer,
                stage.prompt_ingress.total_bytes,
                stage.positional.total_bytes,
                stage.replicated_aux.total_bytes,
                stage.owned.total_bytes,
                stage.tail_only.total_bytes,
                stage.unknown_global.total_bytes
            );
        }
        println!("runtime plan:");
        for runtime in &manifest.runtime_plan {
            println!(
                "  - stage{} role={} required={} optional={}",
                runtime.stage_index + 1,
                runtime.role,
                runtime.required.total_bytes,
                runtime.optional.total_bytes
            );
        }
    }
    println!("first tensors:");
    for tensor in file.tensors.iter().take(12) {
        println!(
            "  - {} dims={:?} type={} offset={}",
            tensor.name, tensor.dimensions, tensor.ggml_type, tensor.offset
        );
    }

    Ok(())
}
