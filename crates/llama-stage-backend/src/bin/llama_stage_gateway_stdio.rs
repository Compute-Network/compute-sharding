#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    RemoteStageGateway, StageGatewayRequest, StageGatewayResponse, handle_stage_gateway_request,
};
use std::env;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct Args {
    head_addr: String,
    tail_addr: String,
    reconnect_after_prompt: bool,
}

fn parse_args() -> Result<Args> {
    let mut head_addr = None;
    let mut tail_addr = None;
    let mut reconnect_after_prompt = false;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--head" => {
                head_addr = Some(it.next().context("missing value for --head")?);
            }
            "--tail" => {
                tail_addr = Some(it.next().context("missing value for --tail")?);
            }
            "--reconnect-after-prompt" => reconnect_after_prompt = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        head_addr: head_addr.context("missing --head")?,
        tail_addr: tail_addr.context("missing --tail")?,
        reconnect_after_prompt,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut gateway = RemoteStageGateway::connect(
        &args.head_addr,
        &args.tail_addr,
        args.reconnect_after_prompt,
    )?;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<StageGatewayRequest>(trimmed) {
            Ok(request) => handle_stage_gateway_request(&mut gateway, request),
            Err(err) => StageGatewayResponse::Error {
                message: format!("invalid request: {err}"),
            },
        };

        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    Ok(())
}
