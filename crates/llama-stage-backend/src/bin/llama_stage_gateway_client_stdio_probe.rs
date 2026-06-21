#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    GatewayStep, GreedyCompletion, StageGatewayRequest, StageGatewayResponse,
    greedy_single_node_completion, resolve_model_arg,
};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> (PathBuf, u32, bool, bool, Vec<String>) {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 4u32;
    let mut reconnect_after_prompt = false;
    let mut interleave = false;

    while idx < args.len() {
        match args[idx].as_str() {
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

    (
        model_path,
        max_tokens,
        reconnect_after_prompt,
        interleave,
        prompts,
    )
}

fn workspace_root() -> Result<&'static Path> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .context("failed to resolve workspace root")
}

fn read_listening_addr(stderr: ChildStderr) -> Result<String> {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            bail!("child exited before announcing listening address");
        }
        let trimmed = line.trim();
        if let Some(addr) = trimmed.strip_prefix("listening=") {
            return Ok(addr.to_string());
        }
    }
}

struct TcpStageChild {
    child: Child,
    addr: String,
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
        let mut command = Command::new("cargo");
        command
            .current_dir(workspace_root()?)
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
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }
}

impl Drop for TcpStageChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct TcpGatewayChild {
    child: Child,
    addr: String,
}

impl TcpGatewayChild {
    fn spawn(head_addr: &str, tail_addr: &str, reconnect_after_prompt: bool) -> Result<Self> {
        let mut command = Command::new("cargo");
        command
            .current_dir(workspace_root()?)
            .arg("run")
            .arg("-q")
            .arg("-p")
            .arg("llama-stage-backend")
            .arg("--bin")
            .arg("llama_stage_gateway_tcp_node")
            .arg("--")
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--head")
            .arg(head_addr)
            .arg("--tail")
            .arg(tail_addr);
        if reconnect_after_prompt {
            command.arg("--reconnect-after-prompt");
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning tcp gateway node")?;

        let stderr = child.stderr.take().context("missing gateway stderr")?;
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }
}

impl Drop for TcpGatewayChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct GatewayClientChild {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl GatewayClientChild {
    fn spawn(gateway_addr: &str) -> Result<Self> {
        let mut command = Command::new("cargo");
        command
            .current_dir(workspace_root()?)
            .arg("run")
            .arg("-q")
            .arg("-p")
            .arg("llama-stage-backend")
            .arg("--bin")
            .arg("llama_stage_gateway_client_stdio")
            .arg("--")
            .arg("--gateway")
            .arg(gateway_addr);

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning gateway client stdio process")?;

        let stdin = child.stdin.take().context("missing client stdin")?;
        let stdout = child.stdout.take().context("missing client stdout")?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn request(&mut self, request: &StageGatewayRequest) -> Result<StageGatewayResponse> {
        serde_json::to_writer(&mut self.stdin, request)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        if line.trim().is_empty() {
            bail!("gateway client returned empty response");
        }

        let response: StageGatewayResponse = serde_json::from_str(line.trim())?;
        if let StageGatewayResponse::Error { message } = &response {
            bail!("gateway client error: {message}");
        }
        Ok(response)
    }
}

impl Drop for GatewayClientChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    let (model_path, max_tokens, reconnect_after_prompt, interleave, prompts) = parse_args();
    let _head = TcpStageChild::spawn(&model_path, "stage-1", 0, 20, true, false)?;
    let _tail = TcpStageChild::spawn(&model_path, "stage-2", 21, 41, false, true)?;
    let gateway = TcpGatewayChild::spawn(&_head.addr, &_tail.addr, reconnect_after_prompt)?;
    let mut client = GatewayClientChild::spawn(&gateway.addr)?;

    if interleave {
        if prompts.len() < 2 {
            bail!("need at least two prompts for --interleave");
        }
        let baseline_a = greedy_single_node_completion(&model_path, &prompts[0], max_tokens)?;
        let baseline_b = greedy_single_node_completion(&model_path, &prompts[1], max_tokens)?;

        let _ = client.request(&StageGatewayRequest::BeginCompletion {
            request_id: "gw-a".to_string(),
            prompt: prompts[0].clone(),
            max_tokens,
        })?;
        let _ = client.request(&StageGatewayRequest::BeginCompletion {
            request_id: "gw-b".to_string(),
            prompt: prompts[1].clone(),
            max_tokens,
        })?;

        let mut done_a = None;
        let mut done_b = None;
        while done_a.is_none() || done_b.is_none() {
            if done_a.is_none() {
                match client.request(&StageGatewayRequest::StepCompletion {
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
                match client.request(&StageGatewayRequest::StepCompletion {
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

        println!("case=0");
        println!("prompt={:?}", prompts[0]);
        println!("baseline_text={:?}", baseline_a.text);
        println!("client_text={:?}", done_a.text);
        println!("baseline_token_ids={:?}", baseline_a.token_ids);
        println!("client_token_ids={:?}", done_a.token_ids);
        println!("match={}", baseline_a == done_a);
        println!();

        println!("case=1");
        println!("prompt={:?}", prompts[1]);
        println!("baseline_text={:?}", baseline_b.text);
        println!("client_text={:?}", done_b.text);
        println!("baseline_token_ids={:?}", baseline_b.token_ids);
        println!("client_token_ids={:?}", done_b.token_ids);
        println!("match={}", baseline_b == done_b);
        println!();

        if baseline_a != done_a {
            bail!("gateway client interleave case 0 diverged");
        }
        if baseline_b != done_b {
            bail!("gateway client interleave case 1 diverged");
        }

        println!("overall=PASS");
        return Ok(());
    }

    for (idx, prompt) in prompts.iter().enumerate() {
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let completion = expect_completion(client.request(&StageGatewayRequest::Complete {
            request_id: format!("gw-{idx}"),
            prompt: prompt.clone(),
            max_tokens,
        })?)?;

        let matches = baseline == completion;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("baseline_text={:?}", baseline.text);
        println!("client_text={:?}", completion.text);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("client_token_ids={:?}", completion.token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("client_tokens={}", completion.completion_tokens);
        println!("match={matches}");
        println!();

        if !matches {
            bail!("gateway client sequential case {idx} diverged");
        }
    }

    println!("overall=PASS");
    Ok(())
}
