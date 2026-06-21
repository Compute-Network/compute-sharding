#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Result, bail};
use llama_stage_backend::{RemoteStagePair, greedy_single_node_completion, resolve_model_arg};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> Result<(PathBuf, String, String, u32, bool, Vec<String>)> {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);

    let mut head_addr = None;
    let mut tail_addr = None;
    let mut max_tokens = 4u32;
    let mut reconnect_after_prompt = false;

    while idx < args.len() {
        match args[idx].as_str() {
            "--head" => {
                head_addr = args.get(idx + 1).cloned();
                idx += 2;
            }
            "--tail" => {
                tail_addr = args.get(idx + 1).cloned();
                idx += 2;
            }
            "--max-tokens" => {
                if let Some(raw) = args.get(idx + 1) {
                    if let Ok(parsed) = raw.parse::<u32>() {
                        max_tokens = parsed.max(1);
                    }
                }
                idx += 2;
            }
            "--reconnect-after-prompt" => {
                reconnect_after_prompt = true;
                idx += 1;
            }
            _ => break,
        }
    }

    let prompts = if args.len() > idx {
        args[idx..].to_vec()
    } else {
        default_prompts()
    };

    let head_addr = head_addr.ok_or_else(|| anyhow::anyhow!("missing --head <addr:port>"))?;
    let tail_addr = tail_addr.ok_or_else(|| anyhow::anyhow!("missing --tail <addr:port>"))?;

    Ok((
        model_path,
        head_addr,
        tail_addr,
        max_tokens,
        reconnect_after_prompt,
        prompts,
    ))
}

fn main() -> Result<()> {
    let (model_path, head_addr, tail_addr, max_tokens, reconnect_after_prompt, prompts) =
        parse_args()?;

    let mut pair = RemoteStagePair::connect(&head_addr, &tail_addr)?;

    for (idx, prompt) in prompts.iter().enumerate() {
        let t_baseline = Instant::now();
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let baseline_ms = t_baseline.elapsed().as_millis();

        let completion = pair.run_greedy_completion(prompt, max_tokens, reconnect_after_prompt)?;

        let matches = baseline.text == completion.text
            && baseline.completion_tokens == completion.completion_tokens
            && baseline.token_ids == completion.token_ids;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("head_addr={head_addr}");
        println!("tail_addr={tail_addr}");
        println!("head_stage={}", pair.head_info.stage_id);
        println!("tail_stage={}", pair.tail_info.stage_id);
        println!(
            "head_layers={}-{}",
            pair.head_info.start_layer, pair.head_info.end_layer
        );
        println!(
            "tail_layers={}-{}",
            pair.tail_info.start_layer, pair.tail_info.end_layer
        );
        println!("max_tokens={max_tokens}");
        println!("reconnect_after_prompt={reconnect_after_prompt}");
        println!("baseline_text={:?}", baseline.text);
        println!("two_stage_text={:?}", completion.text);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("two_stage_token_ids={:?}", completion.token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("two_stage_tokens={}", completion.completion_tokens);
        println!("baseline_ms={baseline_ms}");
        println!("head_prefill_ms={}", completion.timings.head_prefill_ms);
        println!("head_decode_ms={}", completion.timings.head_decode_ms);
        println!("transfer_bytes={}", completion.timings.transfer_bytes);
        println!("tail_decode_ms={}", completion.timings.tail_decode_ms);
        println!("sample_ms={}", completion.timings.sample_ms);
        println!("ttft_ms={}", completion.timings.ttft_ms);
        println!("total_ms={}", completion.timings.total_ms);
        println!("match={matches}");
        println!();

        if !matches {
            bail!("tcp remote output diverged from baseline on case {idx}");
        }
    }

    println!("overall=PASS");
    Ok(())
}
