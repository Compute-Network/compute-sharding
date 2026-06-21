#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    GreedyCompletion, GreedyTokenSample, StageNodeRequest, StageNodeResponse, TcpStageClient,
    greedy_single_node_completion, resolve_model_arg,
};
use stage_forward_lab::StageTensor;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
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

struct TcpStageChild {
    child: Child,
    addr: String,
    client: TcpStageClient,
}

impl TcpStageChild {
    fn spawn(
        model_path: &Path,
        stage_id: &str,
        start_layer: u32,
        end_layer: u32,
        is_head: bool,
        is_tail: bool,
    ) -> Result<Self> {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .context("failed to resolve workspace root")?;
        let mut command = Command::new("cargo");
        command
            .current_dir(workspace_root)
            .arg("run")
            .arg("-q")
            .arg("-p")
            .arg("llama-stage-backend")
            .arg("--bin")
            .arg("llama_stage_tcp_node")
            .arg("--")
            .arg("--model")
            .arg(model_path)
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--stage-id")
            .arg(stage_id)
            .arg("--start-layer")
            .arg(start_layer.to_string())
            .arg("--end-layer")
            .arg(end_layer.to_string());
        if is_head {
            command.arg("--head");
        }
        if is_tail {
            command.arg("--tail");
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning tcp stage node")?;

        let stderr = child.stderr.take().context("missing child stderr")?;
        let addr = Self::read_listening_addr(stderr)?;
        let client = TcpStageClient::connect(&addr)?;

        Ok(Self {
            child,
            addr,
            client,
        })
    }

    fn read_listening_addr(stderr: ChildStderr) -> Result<String> {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader.read_line(&mut line)?;
            if read == 0 {
                bail!("tcp stage node exited before announcing listening address");
            }
            let trimmed = line.trim();
            if let Some(addr) = trimmed.strip_prefix("listening=") {
                return Ok(addr.to_string());
            }
        }
    }

    fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        self.client.request(request)
    }
}

impl Drop for TcpStageChild {
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
            tensor.context("interleave probe expected tensor in response")
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

struct InflightCase {
    idx: usize,
    prompt: String,
    request_id: String,
    baseline: GreedyCompletion,
    head_tensor: StageTensor,
    tail_tensor: StageTensor,
    generated_text: String,
    generated_token_ids: Vec<i32>,
    head_prefill_ms: u128,
    head_decode_ms: u128,
    tail_decode_ms: u128,
    sample_ms: u128,
    transfer_bytes: usize,
    done: bool,
}

fn main() -> Result<()> {
    let (model_path, max_tokens, prompts) = parse_args();
    if prompts.len() < 2 {
        bail!("need at least two prompts for interleave probe");
    }

    let mut head = TcpStageChild::spawn(&model_path, "stage-1", 0, 20, true, false)?;
    let mut tail = TcpStageChild::spawn(&model_path, "stage-2", 21, 41, false, true)?;

    let mut inflight = Vec::new();
    for (idx, prompt) in prompts.iter().enumerate() {
        let request_id = format!("interleave-{idx}");
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let prompt_tokens = expect_token_ids(head.request(&StageNodeRequest::Tokenize {
            text: prompt.clone(),
        })?)?;

        let t_head = Instant::now();
        let head_tensor = expect_tensor(head.request(&StageNodeRequest::BeginPrompt {
            request_id: request_id.clone(),
            prompt: prompt.clone(),
            max_tokens: Some(max_tokens),
        })?)?;
        let head_prefill_ms = t_head.elapsed().as_millis();
        let transfer_bytes = head_tensor.bytes.len();

        let t_tail = Instant::now();
        let tail_tensor =
            expect_tensor(tail.request(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor.clone(),
                token_ids: prompt_tokens,
                clear_memory: true,
            })?)?;
        let tail_decode_ms = t_tail.elapsed().as_millis();

        inflight.push(InflightCase {
            idx,
            prompt: prompt.clone(),
            request_id,
            baseline,
            head_tensor,
            tail_tensor,
            generated_text: String::new(),
            generated_token_ids: Vec::new(),
            head_prefill_ms,
            head_decode_ms: 0,
            tail_decode_ms,
            sample_ms: 0,
            transfer_bytes,
            done: false,
        });
    }

    while inflight.iter().any(|case| !case.done) {
        for case in inflight.iter_mut().filter(|case| !case.done) {
            let t_sample = Instant::now();
            let sampled =
                expect_token_sample(tail.request(&StageNodeRequest::SampleTailToken {
                    tensor: case.tail_tensor.clone(),
                })?)?;
            case.sample_ms += t_sample.elapsed().as_millis();

            case.generated_text.push_str(&sampled.piece);
            case.generated_token_ids.push(sampled.token_id);

            if sampled.is_eog || case.generated_token_ids.len() as u32 >= max_tokens {
                case.done = true;
                let _ = head.request(&StageNodeRequest::ClearDecodeSession {
                    request_id: case.request_id.clone(),
                })?;
                let _ = tail.request(&StageNodeRequest::ClearDecodeSession {
                    request_id: case.request_id.clone(),
                })?;
                continue;
            }

            let t_head = Instant::now();
            case.head_tensor =
                expect_tensor(head.request(&StageNodeRequest::ContinueHeadTokens {
                    request_id: case.request_id.clone(),
                    token_ids: vec![sampled.token_id],
                    max_tokens: Some(max_tokens),
                })?)?;
            case.head_decode_ms += t_head.elapsed().as_millis();

            let t_tail = Instant::now();
            case.tail_tensor =
                expect_tensor(tail.request(&StageNodeRequest::ContinueForwardTokens {
                    tensor: case.head_tensor.clone(),
                    token_ids: vec![sampled.token_id],
                    clear_memory: false,
                })?)?;
            case.tail_decode_ms += t_tail.elapsed().as_millis();
        }
    }

    for case in &inflight {
        let matches = case.baseline.text == case.generated_text
            && case.baseline.completion_tokens == case.generated_token_ids.len() as u32
            && case.baseline.token_ids == case.generated_token_ids;

        println!("case={}", case.idx);
        println!("prompt={:?}", case.prompt);
        println!("request_id={}", case.request_id);
        println!("head_addr={}", head.addr);
        println!("tail_addr={}", tail.addr);
        println!("max_tokens={max_tokens}");
        println!("baseline_text={:?}", case.baseline.text);
        println!("two_stage_text={:?}", case.generated_text);
        println!("baseline_token_ids={:?}", case.baseline.token_ids);
        println!("two_stage_token_ids={:?}", case.generated_token_ids);
        println!("baseline_tokens={}", case.baseline.completion_tokens);
        println!("two_stage_tokens={}", case.generated_token_ids.len());
        println!("head_prefill_ms={}", case.head_prefill_ms);
        println!("head_decode_ms={}", case.head_decode_ms);
        println!("transfer_bytes={}", case.transfer_bytes);
        println!("tail_decode_ms={}", case.tail_decode_ms);
        println!("sample_ms={}", case.sample_ms);
        println!("match={matches}");
        println!();

        if !matches {
            bail!("tcp interleave output diverged on case {}", case.idx);
        }
    }

    println!("overall=PASS");
    Ok(())
}
