#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    RemoteStagePair, TcpStageClient, greedy_single_node_completion, resolve_model_arg,
};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> (PathBuf, u32, bool, Vec<String>) {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 4u32;
    let mut reconnect_after_prompt = false;

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
            _ => break,
        }
    }

    let prompts = if args.len() > idx {
        args[idx..].to_vec()
    } else {
        default_prompts()
    };

    (model_path, max_tokens, reconnect_after_prompt, prompts)
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
        let _client = TcpStageClient::connect(&addr)?;

        Ok(Self { child, addr })
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
}

impl Drop for TcpStageChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> Result<()> {
    let (model_path, max_tokens, reconnect_after_prompt, prompts) = parse_args();

    let head = TcpStageChild::spawn(&model_path, "stage-1", 0, 20, true, false)?;
    let tail = TcpStageChild::spawn(&model_path, "stage-2", 21, 41, false, true)?;
    let mut pair = RemoteStagePair::connect(&head.addr, &tail.addr)?;

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
        println!("head_addr={}", head.addr);
        println!("tail_addr={}", tail.addr);
        println!("head_stage={}", pair.head_info.stage_id);
        println!("tail_stage={}", pair.tail_info.stage_id);
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
            bail!("tcp two-stage output diverged from baseline on case {idx}");
        }
    }

    println!("overall=PASS");
    Ok(())
}
