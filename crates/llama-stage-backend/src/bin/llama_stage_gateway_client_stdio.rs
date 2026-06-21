#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    GatewayServiceClient, StageGatewayRequest, StageGatewayResponse,
    handle_gateway_service_client_request,
};
use std::env;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct Args {
    gateway_addr: String,
}

fn parse_args() -> Result<Args> {
    let mut gateway_addr = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--gateway" => {
                gateway_addr = Some(it.next().context("missing value for --gateway")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        gateway_addr: gateway_addr.context("missing --gateway")?,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut client = GatewayServiceClient::connect(&args.gateway_addr)?;

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
            Ok(request) => handle_gateway_service_client_request(&mut client, request),
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
