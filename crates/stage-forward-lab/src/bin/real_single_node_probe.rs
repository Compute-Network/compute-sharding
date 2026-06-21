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
    let arg4_is_cap = args
        .get(4)
        .and_then(|value| value.parse::<usize>().ok())
        .is_some();
    let vocab_path = if arg4_is_cap {
        PathBuf::from("out/gemma-e4b-2stage/vocab.json")
    } else {
        args.get(4)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"))
    };
    let layer_cap_arg = if arg4_is_cap { 4 } else { 5 };
    let vocab_cap_arg = if arg4_is_cap { 5 } else { 6 };
    let layer_cap = args
        .get(layer_cap_arg)
        .and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args
        .get(vocab_cap_arg)
        .and_then(|value| value.parse::<usize>().ok());
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Single-Node Reference (Sequential 2-Stage) ===");
    println!("prompt   : {:?}", prompt);
    println!("vocab    : {}", vocab_path.display());
    println!("layer cap: {:?}", layer_cap);
    println!("vocab cap: {:?}", vocab_cap);
    println!();

    let mut head = RealGemmaBackend::new(&stage1_path);
    head.set_debug_layer_cap(layer_cap);
    head.set_debug_vocab_cap(vocab_cap);
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

    let mut tail = RealGemmaBackend::new(&stage2_path);
    tail.set_debug_layer_cap(layer_cap);
    tail.set_debug_vocab_cap(vocab_cap);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    println!("--- Two-stage path ---");
    let t_2stage = Instant::now();

    let head_output = head.begin_prompt("ref-req", &prompt, Some(1), 0)?;
    let head_state: Vec<f32> =
        RealGemmaBackend::decode_hidden_states_payload(&head_output.bytes, head_output.hidden_dim)?
            .into_iter()
            .flatten()
            .collect();

    let tail_output = tail.continue_forward(head_output)?;
    let tail_state: Vec<f32> =
        RealGemmaBackend::decode_hidden_states_payload(&tail_output.bytes, tail_output.hidden_dim)?
            .into_iter()
            .flatten()
            .collect();

    let t_trace_sample = Instant::now();
    let (sample, trace) = tail.sample_tail_with_trace(tail_output, 5)?;
    let trace_sample_ms = t_trace_sample.elapsed().as_millis();
    let two_stage_ms = t_2stage.elapsed().as_millis();

    println!("total time     : {}ms", two_stage_ms);
    println!("sampled text   : {:?}", sample.text);
    println!("sample ids     : {:?}", sample.token_ids);
    println!("trace+sample   : {}ms", trace_sample_ms);
    println!("logits tensor  : {}", trace.projection_tensor);
    println!(
        "selected id    : {} ({:.6})",
        trace.selected_token_id, trace.selected_score
    );
    println!(
        "trace/sample id: {}",
        sample.token_ids.first().copied() == Some(trace.selected_token_id)
    );
    println!("top logits     : {:?}", trace.top_logits);
    println!("trace json     : {}", serde_json::to_string(&trace)?);
    println!("head rms       : {:.4}", rms(&head_state));
    println!("tail rms       : {:.4}", rms(&tail_state));
    println!("head preview   : {:?}", &head_state[..4]);
    println!("tail preview   : {:?}", &tail_state[..4]);
    println!();

    println!("--- Single-node path (same stages, sequential) ---");
    let t_single = Instant::now();

    let head_output2 = head.begin_prompt("ref-req-2", &prompt, Some(1), 0)?;
    let head_state2: Vec<f32> = RealGemmaBackend::decode_hidden_states_payload(
        &head_output2.bytes,
        head_output2.hidden_dim,
    )?
    .into_iter()
    .flatten()
    .collect();

    let tail_output2 = tail.continue_forward(head_output2)?;
    let tail_state2: Vec<f32> = RealGemmaBackend::decode_hidden_states_payload(
        &tail_output2.bytes,
        tail_output2.hidden_dim,
    )?
    .into_iter()
    .flatten()
    .collect();

    let t_trace_sample2 = Instant::now();
    let (sample2, trace2) = tail.sample_tail_with_trace(tail_output2, 5)?;
    let trace_sample2_ms = t_trace_sample2.elapsed().as_millis();
    let single_ms = t_single.elapsed().as_millis();

    println!("total time     : {}ms", single_ms);
    println!("sampled text   : {:?}", sample2.text);
    println!("sample ids     : {:?}", sample2.token_ids);
    println!("trace+sample   : {}ms", trace_sample2_ms);
    println!("logits tensor  : {}", trace2.projection_tensor);
    println!(
        "selected id    : {} ({:.6})",
        trace2.selected_token_id, trace2.selected_score
    );
    println!(
        "trace/sample id: {}",
        sample2.token_ids.first().copied() == Some(trace2.selected_token_id)
    );
    println!("top logits     : {:?}", trace2.top_logits);
    println!("trace json     : {}", serde_json::to_string(&trace2)?);
    println!("head rms       : {:.4}", rms(&head_state2));
    println!("tail rms       : {:.4}", rms(&tail_state2));
    println!("head preview   : {:?}", &head_state2[..4]);
    println!("tail preview   : {:?}", &tail_state2[..4]);
    println!();

    let head_match = head_state == head_state2;
    let tail_match = tail_state == tail_state2;
    let text_match = sample.text == sample2.text;

    println!("=== Determinism Check ===");
    println!("head states match  : {}", head_match);
    println!("tail states match  : {}", tail_match);
    println!("output text match  : {}", text_match);

    if head_match && tail_match && text_match {
        println!("PASS: execution is deterministic");
    } else {
        println!("FAIL: non-deterministic execution detected");
        if !head_match {
            let diff: f32 = head_state
                .iter()
                .zip(head_state2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / head_state.len() as f32;
            println!("  head mean abs diff: {:.8}", diff);
        }
        if !tail_match {
            let diff: f32 = tail_state
                .iter()
                .zip(tail_state2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / tail_state.len() as f32;
            println!("  tail mean abs diff: {:.8}", diff);
        }
    }

    Ok(())
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}
