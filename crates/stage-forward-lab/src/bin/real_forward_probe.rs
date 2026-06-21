use anyhow::Result;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let stage1_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json")
    });
    let prompt = args.get(2).cloned().unwrap_or_else(|| "Hello".to_string());

    println!("=== Real Gemma Forward Probe ===");
    println!("stage artifact : {}", stage1_path.display());
    println!("prompt         : {:?}", prompt);
    println!();

    let t_load = Instant::now();
    let mut head = RealGemmaBackend::new(&stage1_path);
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;
    let load_ms = t_load.elapsed().as_millis();
    println!("layout loaded  : {}ms", load_ms);

    let t_fwd = Instant::now();
    let tensor = head.begin_prompt("probe-req", &prompt, Some(1), 0)?;
    let fwd_ms = t_fwd.elapsed().as_millis();

    println!("forward done   : {}ms", fwd_ms);
    println!("hidden_dim     : {}", tensor.hidden_dim);
    println!("hidden_bytes   : {}", tensor.bytes.len());
    println!("stage_trace    : {:?}", tensor.stage_trace);
    println!("kind           : {:?}", tensor.kind);

    let state: Vec<f32> = tensor
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let finite_count = state.iter().filter(|v| v.is_finite()).count();
    let nan_count = state.iter().filter(|v| v.is_nan()).count();
    let inf_count = state.iter().filter(|v| v.is_infinite()).count();
    let min = state.iter().copied().fold(f32::INFINITY, f32::min);
    let max = state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = state.iter().sum::<f32>() / state.len() as f32;
    let rms = (state.iter().map(|v| v * v).sum::<f32>() / state.len() as f32).sqrt();

    println!();
    println!("=== Hidden State Stats ===");
    println!("elements       : {}", state.len());
    println!("finite         : {}", finite_count);
    println!("nan            : {}", nan_count);
    println!("inf            : {}", inf_count);
    println!("min            : {:.6}", min);
    println!("max            : {:.6}", max);
    println!("mean           : {:.6}", mean);
    println!("rms            : {:.6}", rms);

    let preview_len = 8.min(state.len());
    let preview: Vec<String> = state[..preview_len]
        .iter()
        .map(|v| format!("{:.4}", v))
        .collect();
    println!("preview        : [{}]", preview.join(", "));

    Ok(())
}
