use anyhow::Result;
use stage_forward_lab::{ExecutionBinding, ExecutionOpKind, StageTensorStore};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let index_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: execution_program_probe <stage-index.json>"))?;

    let store = StageTensorStore::load(&index_path)?;
    store.validate_offsets()?;
    let view = store.model_view();

    println!("role              : {}", view.role);
    println!("execution_layers  : {}", view.execution_programs.len());
    println!(
        "runnable_layers   : {}",
        view.execution_programs
            .iter()
            .filter(|p| p.runnable_sketch)
            .count()
    );

    for program in view.execution_programs.iter().take(3) {
        println!(
            "layer {:>2}        : hidden={:?} q={:?} k={:?} v={:?} ffn={:?} runnable={}",
            program.layer_index,
            program.hidden_dim,
            program.q_out_dim,
            program.k_out_dim,
            program.v_out_dim,
            program.ffn_inner_dim,
            program.runnable_sketch
        );
        for op in &program.ops {
            println!(
                "  - {:<18} tensors={} binding={} reason={}",
                op_name(&op.kind),
                op.tensor_names.len(),
                binding_name(&op.binding),
                op.binding_reason
            );
        }
    }

    Ok(())
}

fn binding_name(binding: &ExecutionBinding) -> &'static str {
    match binding {
        ExecutionBinding::F32Vector => "f32-vector",
        ExecutionBinding::F32Matrix => "f32-matrix",
        ExecutionBinding::QuantizedMatrix => "quantized-matrix",
        ExecutionBinding::Mixed => "mixed",
        ExecutionBinding::Unsupported => "unsupported",
    }
}

fn op_name(kind: &ExecutionOpKind) -> &'static str {
    match kind {
        ExecutionOpKind::PromptIngress => "prompt_ingress",
        ExecutionOpKind::Positional => "positional",
        ExecutionOpKind::SharedAuxiliary => "shared_auxiliary",
        ExecutionOpKind::AttentionNorm => "attention_norm",
        ExecutionOpKind::AttentionQ => "attention_q",
        ExecutionOpKind::AttentionK => "attention_k",
        ExecutionOpKind::AttentionV => "attention_v",
        ExecutionOpKind::AttentionOut => "attention_out",
        ExecutionOpKind::PostAttentionNorm => "post_attn_norm",
        ExecutionOpKind::FfnNorm => "ffn_norm",
        ExecutionOpKind::FfnGate => "ffn_gate",
        ExecutionOpKind::FfnUp => "ffn_up",
        ExecutionOpKind::FfnDown => "ffn_down",
        ExecutionOpKind::PostFfnNorm => "post_ffn_norm",
        ExecutionOpKind::InputGate => "input_gate",
        ExecutionOpKind::Projection => "projection",
        ExecutionOpKind::LayerOutputScale => "layer_output_scale",
        ExecutionOpKind::TailOnly => "tail_only",
        ExecutionOpKind::Unknown => "unknown",
    }
}
