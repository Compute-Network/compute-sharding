use anyhow::Result;
use stage_forward_lab::prompting::GemmaPromptMode;
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
    let max_tokens = args
        .get(5)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(16)
        .max(1);
    let layer_cap = args.get(6).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(7).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(8)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let stop_sequences = args
        .get(9)
        .map(|value| parse_stop_sequences(value))
        .unwrap_or_default();
    let prompt_mode = args
        .get(10)
        .and_then(|value| GemmaPromptMode::parse(value))
        .unwrap_or_default();
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real 2-Stage Gemma Generate ===");
    println!("stage 1    : {}", stage1_path.display());
    println!("stage 2    : {}", stage2_path.display());
    println!("prompt     : {:?}", prompt);
    println!("max tokens : {}", max_tokens);
    println!("layer cap  : {:?}", layer_cap);
    println!("vocab cap  : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!("stop seqs  : {:?}", stop_sequences);
    println!("prompt mode: {}", prompt_mode.as_str());
    println!();

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
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    let eos_token_id = tail.eos_token_id().or_else(|| head.eos_token_id());
    let mut prompt_token_ids = head.tokenize_prompt_mode(&prompt, prompt_mode);
    let mut generated_token_ids = Vec::with_capacity(max_tokens as usize);
    let mut finish_reason = "length".to_string();
    let mut generated_text = String::new();
    let mut ttft_ms = None;
    let total_start = Instant::now();
    let request_id = "real-generate";

    for step in 0..max_tokens as usize {
        let step_start = Instant::now();
        let step_token_ids: Vec<u32> = if step == 0 {
            prompt_token_ids.clone()
        } else {
            vec![*generated_token_ids.last().unwrap()]
        };
        let head_output = head.begin_token_ids(request_id, &step_token_ids, Some(1), 0)?;
        let tail_output = tail.continue_forward(head_output)?;
        let (sample, trace) = tail.sample_tail_with_trace(tail_output, 5)?;
        let step_ms = step_start.elapsed().as_millis();

        let Some(&next_token_id) = sample.token_ids.first() else {
            println!("step {:>2}: {}ms empty sample", step + 1, step_ms);
            break;
        };

        if ttft_ms.is_none() {
            ttft_ms = Some(total_start.elapsed().as_millis());
        }

        println!(
            "step {:>2}: {}ms id={} text={:?} score={:.6}",
            step + 1,
            step_ms,
            next_token_id,
            sample.text,
            trace.selected_score
        );

        if eos_token_id == Some(next_token_id) {
            finish_reason = "stop".to_string();
            println!("stopped on eos: {}", next_token_id);
            break;
        }

        generated_token_ids.push(next_token_id);
        prompt_token_ids.push(next_token_id);
        generated_text = tail.decode_token_ids(&generated_token_ids);

        if let Some(trimmed) = trim_at_stop_sequence(&generated_text, &stop_sequences) {
            generated_text = trimmed;
            finish_reason = "stop".to_string();
            println!("stopped on stop sequence");
            break;
        }
    }

    let total_ms = total_start.elapsed().as_millis();
    if generated_text.is_empty() && !generated_token_ids.is_empty() {
        generated_text = tail.decode_token_ids(&generated_token_ids);
    }
    let ttft_ms = ttft_ms.unwrap_or(total_ms);
    let continuation_tokens = generated_token_ids.len().saturating_sub(1);
    let continuation_ms = total_ms.saturating_sub(ttft_ms);
    let continuation_tok_s = if continuation_tokens == 0 || continuation_ms == 0 {
        0.0
    } else {
        continuation_tokens as f64 / (continuation_ms as f64 / 1_000.0)
    };

    println!();
    println!("=== Result ===");
    println!("generated text : {:?}", generated_text);
    println!("generated ids  : {:?}", generated_token_ids);
    println!("finish reason  : {}", finish_reason);
    println!("completion toks: {}", generated_token_ids.len());
    println!("ttft           : {}ms", ttft_ms);
    println!("total          : {}ms", total_ms);
    println!("cont tok/s     : {:.2}", continuation_tok_s);

    head.clear_decode_session(request_id);
    tail.clear_decode_session(request_id);

    Ok(())
}

fn parse_stop_sequences(value: &str) -> Vec<String> {
    value
        .split('|')
        .map(str::to_string)
        .filter(|item| !item.is_empty())
        .collect()
}

fn trim_at_stop_sequence(text: &str, stop_sequences: &[String]) -> Option<String> {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()?;
    Some(text[..stop_at].to_string())
}
