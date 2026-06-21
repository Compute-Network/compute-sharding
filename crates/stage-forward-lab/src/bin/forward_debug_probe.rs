use anyhow::Result;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::path::PathBuf;

fn main() -> Result<()> {
    let base = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage".into()
    }));
    let stage1_idx = base.join("packed-stage-1/stage-1-required.index.json");
    let stage2_idx = base.join("packed-stage-2/stage-2-required.index.json");
    let vocab_path = base.join("vocab.json");
    let scores_path = base.join("vocab_scores.json");

    let prompt = "The capital of France is";

    // Test 1: Run head with 0 layers (just embedding), then sample on tail with 0 layers
    // This tests: embedding -> output_norm -> logits projection
    println!("=== Test 1: Zero-layer forward (embedding only) ===");
    {
        let mut head = RealGemmaBackend::new(&stage1_idx);
        head.set_debug_layer_cap(Some(0));
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })?;

        let head_out = head.begin_prompt("req-0", prompt, Some(1), 0)?;
        let state: Vec<f32> =
            RealGemmaBackend::decode_hidden_states_payload(&head_out.bytes, head_out.hidden_dim)?
                .into_iter()
                .flatten()
                .collect();
        println!(
            "head state (0 layers): rms={:.4}, range=[{:.4}, {:.4}]",
            rms(&state),
            state.iter().copied().fold(f32::INFINITY, f32::min),
            state.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );
        println!("  preview: {:?}", &state[..8]);

        let mut tail = RealGemmaBackend::new(&stage2_idx);
        tail.set_debug_layer_cap(Some(0));
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
        tail.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-2".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        })?;

        let tail_out = tail.continue_forward(head_out)?;
        let (sample, trace) = tail.sample_tail_with_trace(tail_out, 10)?;
        println!(
            "  output_norm -> logits -> token: {} (id={})",
            sample.text, trace.selected_token_id
        );
        println!("  top 10: {:?}", trace.top_logits);
        println!("  decoded top 10:");
        for (id, score) in &trace.top_logits {
            println!(
                "    {} ({:.3}) = {:?}",
                id,
                score,
                tail.decode_token_ids(&[*id])
            );
        }
    }

    // Test 2: Run single-node (all 42 layers on one backend) to eliminate stage boundary issues
    println!("\n=== Test 2: Single-stage full model (would need full pack) ===");
    println!("  (skipped - need combined pack)\n");

    // Test 3: Run head=1 layer, tail=0 layers to isolate layer 0 effect
    println!("=== Test 3: Head=1 layer, Tail=0 layers ===");
    {
        let mut head = RealGemmaBackend::new(&stage1_idx);
        head.set_debug_layer_cap(Some(1));
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })?;

        let head_out = head.begin_prompt("req-1", prompt, Some(1), 0)?;
        let state: Vec<f32> =
            RealGemmaBackend::decode_hidden_states_payload(&head_out.bytes, head_out.hidden_dim)?
                .into_iter()
                .flatten()
                .collect();
        println!("head state (1 layer): rms={:.4}", rms(&state));

        let mut tail = RealGemmaBackend::new(&stage2_idx);
        tail.set_debug_layer_cap(Some(0));
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
        tail.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-2".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        })?;

        let tail_out = tail.continue_forward(head_out)?;
        let (sample, trace) = tail.sample_tail_with_trace(tail_out, 10)?;
        println!("  token: {} (id={})", sample.text, trace.selected_token_id);
        println!("  decoded top 5:");
        for (id, score) in trace.top_logits.iter().take(5) {
            println!(
                "    {} ({:.3}) = {:?}",
                id,
                score,
                tail.decode_token_ids(&[*id])
            );
        }
    }

    // Test 4: Head=all 21 layers, Tail=all 21 layers (full pipeline)
    println!("\n=== Test 4: Full pipeline (21+21 layers) ===");
    {
        let mut head = RealGemmaBackend::new(&stage1_idx);
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })?;

        let head_out = head.begin_prompt("req-full", prompt, Some(1), 0)?;
        let state: Vec<f32> =
            RealGemmaBackend::decode_hidden_states_payload(&head_out.bytes, head_out.hidden_dim)?
                .into_iter()
                .flatten()
                .collect();
        println!("head state (21 layers): rms={:.4}", rms(&state));

        let mut tail = RealGemmaBackend::new(&stage2_idx);
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
        tail.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-2".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        })?;

        let tail_out = tail.continue_forward(head_out)?;
        let tail_state: Vec<f32> =
            RealGemmaBackend::decode_hidden_states_payload(&tail_out.bytes, tail_out.hidden_dim)?
                .into_iter()
                .flatten()
                .collect();
        println!("tail state (21 layers): rms={:.4}", rms(&tail_state));

        let (sample, trace) = tail.sample_tail_with_trace(tail_out, 10)?;
        println!("  token: {} (id={})", sample.text, trace.selected_token_id);
        println!("  decoded top 10:");
        for (id, score) in &trace.top_logits {
            println!(
                "    {} ({:.3}) = {:?}",
                id,
                score,
                tail.decode_token_ids(&[*id])
            );
        }
    }

    // Test 5: Check embedding scale factor
    println!("\n=== Test 5: Embedding scale sanity ===");
    {
        let mut head = RealGemmaBackend::new(&stage1_idx);
        head.set_debug_layer_cap(Some(0));
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
        head.load_layout(StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-1".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        })?;

        // Token "The" should embed to hidden_dim=2560
        let out_the = head.begin_prompt("scale-1", "The", Some(1), 0)?;
        let state_the: Vec<f32> =
            RealGemmaBackend::decode_hidden_states_payload(&out_the.bytes, out_the.hidden_dim)?
                .into_iter()
                .flatten()
                .collect();
        println!(
            "  'The' embedding: rms={:.4}, scale=sqrt(2560)={:.4}",
            rms(&state_the),
            (2560.0f32).sqrt()
        );
        println!("  'The' tokens: {:?}", head.tokenize_text("The"));
        println!(
            "  'The capital of France is' tokens: {:?}",
            head.tokenize_text("The capital of France is")
        );
    }

    Ok(())
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}
