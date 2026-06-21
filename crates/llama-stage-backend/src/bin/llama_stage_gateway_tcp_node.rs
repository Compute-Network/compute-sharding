#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    RemoteStageGateway, SpecDecodeConfig, StageGatewayRequest, StageGatewayResponse,
    handle_stage_gateway_request,
};
use std::env;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Args {
    bind_addr: String,
    head_addr: String,
    tail_addr: String,
    reconnect_after_prompt: bool,
    draft_model: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut bind_addr = "127.0.0.1:0".to_string();
    let mut head_addr = None;
    let mut tail_addr = None;
    let mut reconnect_after_prompt = false;
    let mut draft_model = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => {
                bind_addr = it.next().context("missing value for --bind")?;
            }
            "--head" => {
                head_addr = Some(it.next().context("missing value for --head")?);
            }
            "--tail" => {
                tail_addr = Some(it.next().context("missing value for --tail")?);
            }
            "--reconnect-after-prompt" => reconnect_after_prompt = true,
            "--draft-model" => {
                draft_model = Some(PathBuf::from(
                    it.next().context("missing value for --draft-model")?,
                ));
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        bind_addr,
        head_addr: head_addr.context("missing --head")?,
        tail_addr: tail_addr.context("missing --tail")?,
        reconnect_after_prompt,
        draft_model,
    })
}

fn handle_stream(stream: TcpStream, gateway: &mut RemoteStageGateway) -> Result<()> {
    stream.set_nodelay(true)?;
    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;

    loop {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }

        let request_len = u32::from_le_bytes(len_buf) as usize;
        if request_len == 0 {
            continue;
        }

        let mut request_bytes = vec![0u8; request_len];
        reader.read_exact(&mut request_bytes)?;

        let response = match rmp_serde::from_slice::<StageGatewayRequest>(&request_bytes) {
            Ok(request) => handle_stage_gateway_request(gateway, request),
            Err(err) => StageGatewayResponse::Error {
                message: format!("invalid request: {err}"),
            },
        };

        let response_bytes = rmp_serde::to_vec_named(&response)?;
        let response_len =
            u32::try_from(response_bytes.len()).context("gateway response too large")?;
        writer.write_all(&response_len.to_le_bytes())?;
        writer.write_all(&response_bytes)?;
        writer.flush()?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut gateway = match args.draft_model.as_ref() {
        Some(path) => {
            if !path.exists() {
                bail!("--draft-model path does not exist: {}", path.display());
            }
            eprintln!("draft_model={}", path.display());
            let spec_config = SpecDecodeConfig::from_env();
            eprintln!("spec_config={spec_config:?}");
            RemoteStageGateway::connect_with_draft(
                &args.head_addr,
                &args.tail_addr,
                args.reconnect_after_prompt,
                path,
                spec_config,
            )?
        }
        None => RemoteStageGateway::connect(
            &args.head_addr,
            &args.tail_addr,
            args.reconnect_after_prompt,
        )?,
    };
    eprintln!("spec_active={}", gateway.spec_active());

    let listener = TcpListener::bind(&args.bind_addr)
        .with_context(|| format!("binding {}", args.bind_addr))?;
    eprintln!("listening={}", listener.local_addr()?);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_stream(stream, &mut gateway) {
                    eprintln!("connection error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }

    Ok(())
}
