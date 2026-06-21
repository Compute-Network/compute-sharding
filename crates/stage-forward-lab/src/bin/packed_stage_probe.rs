use anyhow::Result;
use stage_forward_lab::StageTensorStore;
use std::path::PathBuf;

fn main() -> Result<()> {
    let index_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
            )
        });

    let store = StageTensorStore::load(&index_path)?;
    store.validate_offsets()?;
    let view = store.model_view();

    println!("index              : {}", index_path.display());
    println!(
        "pack               : {}",
        store.artifact.pack_path.display()
    );
    println!("role               : {}", store.artifact.index.role);
    println!("tensor count       : {}", store.tensor_count());
    println!("total bytes        : {}", store.total_bytes());
    println!(
        "view               : ingress={} positional={} shared_aux={} layers={} operator_layers={} execution_layers={} tail_only={}",
        view.prompt_ingress.len(),
        view.positional.len(),
        view.shared_auxiliary.len(),
        view.layers.len(),
        view.operator_layers.len(),
        view.execution_layers.len(),
        view.tail_only.len()
    );
    if let Some(layer) = view.operator_layers.first() {
        println!(
            "layer{}            : q={} k={} v={} out={} up={} down={} gate={} proj={} unknown={}",
            layer.layer_index,
            layer.attn_q.is_some(),
            layer.attn_k.is_some(),
            layer.attn_v.is_some(),
            layer.attn_output.is_some(),
            layer.ffn_up.is_some(),
            layer.ffn_down.is_some(),
            layer.ffn_gate.is_some(),
            layer.proj.is_some(),
            layer.unknown.len()
        );
    }
    if let Some(spec) = view.execution_layers.first() {
        println!(
            "exec{}             : hidden={:?} q={:?} k={:?} v={:?} ffn={:?} attn_core={} ffn_core={} proj={} runnable={}",
            spec.layer_index,
            spec.hidden_dim,
            spec.q_out_dim,
            spec.k_out_dim,
            spec.v_out_dim,
            spec.ffn_inner_dim,
            spec.has_attention_core,
            spec.has_ffn_core,
            spec.has_projection_path,
            spec.runnable_sketch
        );
    }
    let runnable_layers = view
        .execution_layers
        .iter()
        .filter(|spec| spec.runnable_sketch)
        .count();
    println!("runnable layers    : {}", runnable_layers);
    for name in store.tensor_names().take(8) {
        let entry = store.entry(name).unwrap();
        println!(
            "  - {} pack_offset={} bytes={} dims={:?} type={}",
            entry.name, entry.pack_offset, entry.byte_len, entry.dimensions, entry.ggml_type
        );
    }

    Ok(())
}
