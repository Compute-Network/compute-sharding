#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    GreedyTokenSample, StageNodeRequest, StageNodeResponse, greedy_single_node_completion,
    resolve_model_arg,
};
use stage_forward_lab::{StageSample, StageTensor};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> (PathBuf, u32, Vec<String>) {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 4u32;

    if args.get(idx).map(|s| s.as_str()) == Some("--max-tokens") {
        if let Some(raw) = args.get(idx + 1) {
            if let Ok(parsed) = raw.parse::<u32>() {
                max_tokens = parsed.max(1);
            }
        }
        idx += 2;
    }

    let prompts = if args.len() > idx {
        args[idx..].to_vec()
    } else {
        default_prompts()
    };

    (model_path, max_tokens, prompts)
}

struct StageNodeChild {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StageNodeChild {
    fn spawn(
        model_path: &Path,
        stage_id: &str,
        start_layer: u32,
        end_layer: u32,
        is_head: bool,
        is_tail: bool,
    ) -> Result<Self> {
        let direct = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("llama_stage_stdio_node")))
            .filter(|path| path.exists());

        let mut command = if let Some(path) = direct {
            let mut cmd = Command::new(path);
            cmd.arg("--model")
                .arg(model_path)
                .arg("--stage-id")
                .arg(stage_id)
                .arg("--start-layer")
                .arg(start_layer.to_string())
                .arg("--end-layer")
                .arg(end_layer.to_string());
            if is_head {
                cmd.arg("--head");
            }
            if is_tail {
                cmd.arg("--tail");
            }
            cmd
        } else {
            let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .context("failed to resolve workspace root")?;
            let mut cmd = Command::new("cargo");
            cmd.current_dir(workspace_root)
                .arg("run")
                .arg("-q")
                .arg("-p")
                .arg("llama-stage-backend")
                .arg("--bin")
                .arg("llama_stage_stdio_node")
                .arg("--")
                .arg("--model")
                .arg(model_path)
                .arg("--stage-id")
                .arg(stage_id)
                .arg("--start-layer")
                .arg(start_layer.to_string())
                .arg("--end-layer")
                .arg(end_layer.to_string());
            if is_head {
                cmd.arg("--head");
            }
            if is_tail {
                cmd.arg("--tail");
            }
            cmd
        };

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning stage node")?;

        let stdin = child.stdin.take().context("missing child stdin")?;
        let stdout = child.stdout.take().context("missing child stdout")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        serde_json::to_writer(&mut self.stdin, request)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        if line.trim().is_empty() {
            bail!("child returned empty response");
        }

        let response: StageNodeResponse = serde_json::from_str(line.trim())?;
        if let StageNodeResponse::Error { message } = &response {
            bail!("stage node error: {message}");
        }
        Ok(response)
    }
}

impl Drop for StageNodeChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn expect_token_ids(response: StageNodeResponse) -> Result<Vec<i32>> {
    match response {
        StageNodeResponse::TokenIds { token_ids } => Ok(token_ids),
        other => bail!("expected token_ids response, got {other:?}"),
    }
}

fn expect_tensor(response: StageNodeResponse) -> Result<StageTensor> {
    match response {
        StageNodeResponse::Tensor { tensor, .. } => {
            tensor.context("roundtrip probe expected tensor in response")
        }
        other => bail!("expected tensor response, got {other:?}"),
    }
}

fn expect_token_sample(response: StageNodeResponse) -> Result<GreedyTokenSample> {
    match response {
        StageNodeResponse::TokenSample { sample } => Ok(sample),
        other => bail!("expected token_sample response, got {other:?}"),
    }
}

#[allow(dead_code)]
fn expect_sample(response: StageNodeResponse) -> Result<StageSample> {
    match response {
        StageNodeResponse::Sample { sample } => Ok(sample),
        other => bail!("expected sample response, got {other:?}"),
    }
}

fn main() -> Result<()> {
    let (model_path, max_tokens, prompts) = parse_args();

    let mut head = StageNodeChild::spawn(&model_path, "stage-1", 0, 20, true, false)?;
    let mut tail = StageNodeChild::spawn(&model_path, "stage-2", 21, 41, false, true)?;

    for (idx, prompt) in prompts.iter().enumerate() {
        let request_id = format!("stdio-two-stage-{idx}");

        let t_baseline = Instant::now();
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let baseline_ms = t_baseline.elapsed().as_millis();

        let prompt_tokens = expect_token_ids(head.request(&StageNodeRequest::Tokenize {
            text: prompt.clone(),
        })?)?;

        let t_head = Instant::now();
        let mut head_tensor = expect_tensor(head.request(&StageNodeRequest::BeginPrompt {
            request_id: request_id.clone(),
            prompt: prompt.clone(),
            max_tokens: Some(max_tokens),
        })?)?;
        let head_prefill_ms = t_head.elapsed().as_millis();
        let transfer_bytes = head_tensor.bytes.len();

        let t_tail = Instant::now();
        let mut tail_tensor =
            expect_tensor(tail.request(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor.clone(),
                token_ids: prompt_tokens,
                clear_memory: true,
            })?)?;
        let prompt_tail_ms = t_tail.elapsed().as_millis();

        let mut generated_text = String::new();
        let mut generated_token_ids = Vec::new();
        let mut head_decode_ms_total = 0u128;
        let mut tail_decode_ms_total = prompt_tail_ms;
        let mut sample_ms_total = 0u128;

        for step in 0..max_tokens {
            let t_sample = Instant::now();
            let sampled =
                expect_token_sample(tail.request(&StageNodeRequest::SampleTailToken {
                    tensor: tail_tensor.clone(),
                })?)?;
            sample_ms_total += t_sample.elapsed().as_millis();

            generated_text.push_str(&sampled.piece);
            generated_token_ids.push(sampled.token_id);

            if sampled.is_eog || step + 1 >= max_tokens {
                break;
            }

            let t_head_step = Instant::now();
            head_tensor = expect_tensor(head.request(&StageNodeRequest::ContinueHeadTokens {
                request_id: request_id.clone(),
                token_ids: vec![sampled.token_id],
                max_tokens: Some(max_tokens),
            })?)?;
            head_decode_ms_total += t_head_step.elapsed().as_millis();

            let t_tail_step = Instant::now();
            tail_tensor =
                expect_tensor(tail.request(&StageNodeRequest::ContinueForwardTokens {
                    tensor: head_tensor.clone(),
                    token_ids: vec![sampled.token_id],
                    clear_memory: false,
                })?)?;
            tail_decode_ms_total += t_tail_step.elapsed().as_millis();
        }

        let matches = baseline.text == generated_text
            && baseline.completion_tokens == generated_token_ids.len() as u32
            && baseline.token_ids == generated_token_ids;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("max_tokens={max_tokens}");
        println!("baseline_text={:?}", baseline.text);
        println!("two_stage_text={:?}", generated_text);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("two_stage_token_ids={:?}", generated_token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("two_stage_tokens={}", generated_token_ids.len());
        println!("baseline_ms={baseline_ms}");
        println!("head_prefill_ms={head_prefill_ms}");
        println!("head_decode_ms={head_decode_ms_total}");
        println!("transfer_bytes={transfer_bytes}");
        println!("tail_decode_ms={tail_decode_ms_total}");
        println!("sample_ms={sample_ms_total}");
        println!(
            "ttft_ms={}",
            head_prefill_ms + prompt_tail_ms + sample_ms_total
        );
        println!("match={matches}");
        println!();

        if !matches {
            bail!("stdio two-stage output diverged from baseline on case {idx}");
        }
    }

    println!("overall=PASS");
    Ok(())
}
