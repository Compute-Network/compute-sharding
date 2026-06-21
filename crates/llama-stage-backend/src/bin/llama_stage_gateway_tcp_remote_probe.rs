#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Result, bail};
use llama_stage_backend::{
    GatewayStep, GreedyCompletion, StageGatewayRequest, StageGatewayResponse, TcpGatewayClient,
    greedy_single_node_completion, resolve_model_arg,
};
use std::env;
use std::path::PathBuf;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> Result<(PathBuf, String, u32, bool, Vec<String>)> {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);

    let mut gateway_addr = None;
    let mut max_tokens = 4u32;
    let mut interleave = false;

    while idx < args.len() {
        match args[idx].as_str() {
            "--gateway" => {
                gateway_addr = args.get(idx + 1).cloned();
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
            "--interleave" => {
                interleave = true;
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

    let gateway_addr =
        gateway_addr.ok_or_else(|| anyhow::anyhow!("missing --gateway <addr:port>"))?;
    Ok((model_path, gateway_addr, max_tokens, interleave, prompts))
}

fn expect_completion(response: StageGatewayResponse) -> Result<GreedyCompletion> {
    match response {
        StageGatewayResponse::Completion { completion } => Ok(GreedyCompletion {
            text: completion.text,
            completion_tokens: completion.completion_tokens,
            token_ids: completion.token_ids,
        }),
        other => bail!("expected completion response, got {other:?}"),
    }
}

fn main() -> Result<()> {
    let (model_path, gateway_addr, max_tokens, interleave, prompts) = parse_args()?;
    let mut gateway = TcpGatewayClient::connect(&gateway_addr)?;

    let info = match gateway.request(&StageGatewayRequest::Info)? {
        StageGatewayResponse::Info {
            protocol_version: _,
            head_info,
            tail_info,
            reconnect_after_prompt,
        } => (head_info, tail_info, reconnect_after_prompt),
        other => bail!("expected info response, got {other:?}"),
    };

    if interleave {
        if prompts.len() < 2 {
            bail!("need at least two prompts for --interleave");
        }
        let baseline_a = greedy_single_node_completion(&model_path, &prompts[0], max_tokens)?;
        let baseline_b = greedy_single_node_completion(&model_path, &prompts[1], max_tokens)?;

        let _ = gateway.request(&StageGatewayRequest::BeginCompletion {
            request_id: "gw-a".to_string(),
            prompt: prompts[0].clone(),
            max_tokens,
        })?;
        let _ = gateway.request(&StageGatewayRequest::BeginCompletion {
            request_id: "gw-b".to_string(),
            prompt: prompts[1].clone(),
            max_tokens,
        })?;

        let mut done_a = None;
        let mut done_b = None;
        while done_a.is_none() || done_b.is_none() {
            if done_a.is_none() {
                match gateway.request(&StageGatewayRequest::StepCompletion {
                    request_id: "gw-a".to_string(),
                })? {
                    StageGatewayResponse::Step {
                        step: GatewayStep::Complete { completion, .. },
                    } => {
                        done_a = Some(GreedyCompletion {
                            text: completion.text,
                            completion_tokens: completion.completion_tokens,
                            token_ids: completion.token_ids,
                        });
                    }
                    StageGatewayResponse::Step {
                        step: GatewayStep::Token { .. },
                    } => {}
                    other => bail!("unexpected response for gw-a: {other:?}"),
                }
            }
            if done_b.is_none() {
                match gateway.request(&StageGatewayRequest::StepCompletion {
                    request_id: "gw-b".to_string(),
                })? {
                    StageGatewayResponse::Step {
                        step: GatewayStep::Complete { completion, .. },
                    } => {
                        done_b = Some(GreedyCompletion {
                            text: completion.text,
                            completion_tokens: completion.completion_tokens,
                            token_ids: completion.token_ids,
                        });
                    }
                    StageGatewayResponse::Step {
                        step: GatewayStep::Token { .. },
                    } => {}
                    other => bail!("unexpected response for gw-b: {other:?}"),
                }
            }
        }

        let done_a = done_a.expect("gw-a completed");
        let done_b = done_b.expect("gw-b completed");

        println!("gateway_addr={gateway_addr}");
        println!("head_stage={}", info.0.stage_id);
        println!("tail_stage={}", info.1.stage_id);
        println!("head_layers={}-{}", info.0.start_layer, info.0.end_layer);
        println!("tail_layers={}-{}", info.1.start_layer, info.1.end_layer);
        println!("reconnect_after_prompt={}", info.2);
        println!();

        println!("case=0");
        println!("prompt={:?}", prompts[0]);
        println!("baseline_text={:?}", baseline_a.text);
        println!("gateway_text={:?}", done_a.text);
        println!("baseline_token_ids={:?}", baseline_a.token_ids);
        println!("gateway_token_ids={:?}", done_a.token_ids);
        println!(
            "match={}",
            baseline_a.text == done_a.text
                && baseline_a.completion_tokens == done_a.completion_tokens
                && baseline_a.token_ids == done_a.token_ids
        );
        println!();

        println!("case=1");
        println!("prompt={:?}", prompts[1]);
        println!("baseline_text={:?}", baseline_b.text);
        println!("gateway_text={:?}", done_b.text);
        println!("baseline_token_ids={:?}", baseline_b.token_ids);
        println!("gateway_token_ids={:?}", done_b.token_ids);
        println!(
            "match={}",
            baseline_b.text == done_b.text
                && baseline_b.completion_tokens == done_b.completion_tokens
                && baseline_b.token_ids == done_b.token_ids
        );
        println!();

        if baseline_a.text != done_a.text
            || baseline_a.completion_tokens != done_a.completion_tokens
            || baseline_a.token_ids != done_a.token_ids
        {
            bail!("gateway interleave case 0 diverged");
        }
        if baseline_b.text != done_b.text
            || baseline_b.completion_tokens != done_b.completion_tokens
            || baseline_b.token_ids != done_b.token_ids
        {
            bail!("gateway interleave case 1 diverged");
        }

        println!("overall=PASS");
        return Ok(());
    }

    for (idx, prompt) in prompts.iter().enumerate() {
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let completion = expect_completion(gateway.request(&StageGatewayRequest::Complete {
            request_id: format!("gw-{idx}"),
            prompt: prompt.clone(),
            max_tokens,
        })?)?;

        let matches = baseline.text == completion.text
            && baseline.completion_tokens == completion.completion_tokens
            && baseline.token_ids == completion.token_ids;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("gateway_addr={gateway_addr}");
        println!("head_stage={}", info.0.stage_id);
        println!("tail_stage={}", info.1.stage_id);
        println!("head_layers={}-{}", info.0.start_layer, info.0.end_layer);
        println!("tail_layers={}-{}", info.1.start_layer, info.1.end_layer);
        println!("reconnect_after_prompt={}", info.2);
        println!("baseline_text={:?}", baseline.text);
        println!("gateway_text={:?}", completion.text);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("gateway_token_ids={:?}", completion.token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("gateway_tokens={}", completion.completion_tokens);
        println!("match={matches}");
        println!();

        if !matches {
            bail!("gateway remote case {idx} diverged");
        }
    }

    println!("overall=PASS");
    Ok(())
}
