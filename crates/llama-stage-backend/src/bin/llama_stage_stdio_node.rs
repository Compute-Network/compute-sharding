#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    StageNodeConfig, StageNodeRequest, StageNodeResponse, build_stage_backend,
    default_gemma_model_path, handle_stage_node_request,
};
use std::env;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Args {
    bind_model_path: PathBuf,
    stage_id: String,
    start_layer: u32,
    end_layer: u32,
    is_head: bool,
    is_tail: bool,
}

fn parse_args() -> Result<Args> {
    let mut model_path = default_gemma_model_path();
    let mut stage_id = None;
    let mut start_layer = None;
    let mut end_layer = None;
    let mut is_head = false;
    let mut is_tail = false;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                model_path = PathBuf::from(it.next().context("missing value for --model")?);
            }
            "--stage-id" => {
                stage_id = Some(it.next().context("missing value for --stage-id")?);
            }
            "--start-layer" => {
                start_layer = Some(
                    it.next()
                        .context("missing value for --start-layer")?
                        .parse::<u32>()
                        .context("invalid --start-layer")?,
                );
            }
            "--end-layer" => {
                end_layer = Some(
                    it.next()
                        .context("missing value for --end-layer")?
                        .parse::<u32>()
                        .context("invalid --end-layer")?,
                );
            }
            "--head" => is_head = true,
            "--tail" => is_tail = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        bind_model_path: model_path,
        stage_id: stage_id.context("missing --stage-id")?,
        start_layer: start_layer.context("missing --start-layer")?,
        end_layer: end_layer.context("missing --end-layer")?,
        is_head,
        is_tail,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let backend = build_stage_backend(&StageNodeConfig {
        model_path: args.bind_model_path,
        stage_id: args.stage_id,
        start_layer: args.start_layer,
        end_layer: args.end_layer,
        is_head: args.is_head,
        is_tail: args.is_tail,
    })?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
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

        let response = match serde_json::from_str::<StageNodeRequest>(trimmed) {
            Ok(request) => handle_stage_node_request(&backend, request),
            Err(err) => StageNodeResponse::Error {
                message: format!("invalid request: {err}"),
            },
        };

        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }

    Ok(())
}
