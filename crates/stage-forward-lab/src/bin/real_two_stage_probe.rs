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
    let stage2_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json")
    });
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Hello".to_string());
    let vocab_path = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"));
    let layer_cap = args.get(5).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(6).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(7)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real 2-Stage Gemma Forward ===");
    println!("stage 1  : {}", stage1_path.display());
    println!("stage 2  : {}", stage2_path.display());
    println!("prompt   : {:?}", prompt);
    println!("layer cap: {:?}", layer_cap);
    println!("vocab cap: {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!();

    let t0 = Instant::now();
    let mut head = RealGemmaBackend::new(&stage1_path);
    head.set_debug_layer_cap(layer_cap);
    head.set_debug_vocab_cap(vocab_cap);
    head.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;
    println!("head loaded      : {}ms", t0.elapsed().as_millis());

    let t1 = Instant::now();
    let mut tail = RealGemmaBackend::new(&stage2_path);
    tail.set_debug_layer_cap(layer_cap);
    tail.set_debug_vocab_cap(vocab_cap);
    tail.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
        println!("tokenizer loaded : {}", vocab_path.display());
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;
    println!("tail loaded      : {}ms", t1.elapsed().as_millis());
    println!();

    let t_head = Instant::now();
    let head_output = head.begin_prompt("probe-req", &prompt, Some(1), 0)?;
    let head_ms = t_head.elapsed().as_millis();

    let head_state: Vec<f32> =
        RealGemmaBackend::decode_hidden_states_payload(&head_output.bytes, head_output.hidden_dim)?
            .into_iter()
            .flatten()
            .collect();

    println!("=== Stage 1 (Head) ===");
    println!("forward time     : {}ms", head_ms);
    println!("hidden_dim       : {}", head_output.hidden_dim);
    println!("stage_trace      : {:?}", head_output.stage_trace);
    print_stats("head", &head_state);
    if let Some(profile) = head.last_forward_profile() {
        print_profile("head", &profile);
    }
    println!();

    let t_tail = Instant::now();
    let tail_output = tail.continue_forward(head_output)?;
    let tail_ms = t_tail.elapsed().as_millis();

    let tail_state: Vec<f32> =
        RealGemmaBackend::decode_hidden_states_payload(&tail_output.bytes, tail_output.hidden_dim)?
            .into_iter()
            .flatten()
            .collect();

    println!("=== Stage 2 (Tail Forward) ===");
    println!("forward time     : {}ms", tail_ms);
    println!("hidden_dim       : {}", tail_output.hidden_dim);
    println!("stage_trace      : {:?}", tail_output.stage_trace);
    print_stats("tail_fwd", &tail_state);
    if let Some(profile) = tail.last_forward_profile() {
        print_profile("tail", &profile);
    }
    println!();

    let t_trace_sample = Instant::now();
    let (sample, trace) = tail.sample_tail_with_trace(tail_output, 5)?;
    let trace_sample_ms = t_trace_sample.elapsed().as_millis();

    println!("=== Sampling ===");
    println!("trace+sample time: {}ms", trace_sample_ms);
    println!("text             : {:?}", sample.text);
    println!("tokens           : {}", sample.completion_tokens);
    println!("sample ids       : {:?}", sample.token_ids);
    println!("logits tensor    : {}", trace.projection_tensor);
    println!(
        "selected id      : {} ({:.6})",
        trace.selected_token_id, trace.selected_score
    );
    println!(
        "trace/sample id  : {}",
        sample.token_ids.first().copied() == Some(trace.selected_token_id)
    );
    println!("state rms        : {:.6}", trace.state_rms);
    println!("top logits       : {:?}", trace.top_logits);
    println!("trace json       : {}", serde_json::to_string(&trace)?);
    println!();

    println!("=== Total ===");
    println!("head forward     : {}ms", head_ms);
    println!("tail forward     : {}ms", tail_ms);
    println!("trace+sampling   : {}ms", trace_sample_ms);
    println!(
        "total            : {}ms",
        head_ms + tail_ms + trace_sample_ms
    );

    Ok(())
}

fn print_stats(label: &str, state: &[f32]) {
    let finite = state.iter().filter(|v| v.is_finite()).count();
    let nan = state.iter().filter(|v| v.is_nan()).count();
    let inf = state.iter().filter(|v| v.is_infinite()).count();
    let min = state.iter().copied().fold(f32::INFINITY, f32::min);
    let max = state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = state.iter().sum::<f32>() / state.len() as f32;
    let rms = (state.iter().map(|v| v * v).sum::<f32>() / state.len() as f32).sqrt();
    let preview: Vec<String> = state[..8.min(state.len())]
        .iter()
        .map(|v| format!("{:.4}", v))
        .collect();

    println!("{} finite      : {}/{}", label, finite, state.len());
    println!("{} nan/inf     : {}/{}", label, nan, inf);
    println!("{} range       : [{:.4}, {:.4}]", label, min, max);
    println!("{} mean/rms    : {:.4} / {:.4}", label, mean, rms);
    println!("{} preview     : [{}]", label, preview.join(", "));
}

fn print_profile(label: &str, profile: &stage_forward_lab::real_forward::RealForwardProfile) {
    println!(
        "{} profile     : seq={} layers={} embed={}ms aux={}ms (lookup={}ms project={}ms combine={}ms materialize={}ms) attn={}ms (qkv={}ms core={}ms out={}ms) ffn={}ms (gate+up={}ms down={}ms) ple={}ms (gate={}ms proj={}ms)",
        label,
        profile.seq_len,
        profile.layers,
        profile.embed_micros / 1_000,
        profile.aux_micros / 1_000,
        profile.aux_lookup_micros / 1_000,
        profile.aux_project_micros / 1_000,
        profile.aux_combine_micros / 1_000,
        profile.aux_materialize_micros / 1_000,
        profile.attn_micros / 1_000,
        profile.attn_qkv_micros / 1_000,
        profile.attn_core_micros / 1_000,
        profile.attn_out_micros / 1_000,
        profile.ffn_micros / 1_000,
        profile.ffn_gate_up_micros / 1_000,
        profile.ffn_down_micros / 1_000,
        profile.ple_micros / 1_000,
        profile.ple_gate_micros / 1_000,
        profile.ple_proj_micros / 1_000,
    );
}
