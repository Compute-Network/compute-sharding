#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    StageNodeConfig, StageNodeProfile, StageNodeRequest, StageNodeResponse, build_stage_backend,
    default_gemma_model_path, handle_stage_node_request,
};
use std::env;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Args {
    model_path: PathBuf,
    bind_addr: String,
    stage_id: String,
    start_layer: u32,
    end_layer: u32,
    is_head: bool,
    is_tail: bool,
}

fn parse_args() -> Result<Args> {
    let mut model_path = default_gemma_model_path();
    let mut bind_addr = "127.0.0.1:0".to_string();
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
            "--bind" => {
                bind_addr = it.next().context("missing value for --bind")?;
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
        model_path,
        bind_addr,
        stage_id: stage_id.context("missing --stage-id")?,
        start_layer: start_layer.context("missing --start-layer")?,
        end_layer: end_layer.context("missing --end-layer")?,
        is_head,
        is_tail,
    })
}

fn stamp_server_timing(
    response: &mut StageNodeResponse,
    decode_us: u64,
    handle_us: u64,
    prev_encode_us: u64,
    prev_write_us: u64,
) {
    let stamp = |profile: &mut Option<StageNodeProfile>| {
        let p = profile.get_or_insert_with(StageNodeProfile::default);
        p.server_request_json_decode_us = decode_us;
        p.server_request_json_decode_ms = decode_us / 1000;
        p.server_handle_us = handle_us;
        p.server_handle_ms = handle_us / 1000;
        p.server_response_json_encode_us = prev_encode_us;
        p.server_response_json_encode_ms = prev_encode_us / 1000;
        p.server_response_write_us = prev_write_us;
        p.server_response_write_ms = prev_write_us / 1000;
    };
    match response {
        StageNodeResponse::Tensor { profile, .. } => stamp(profile),
        StageNodeResponse::VerifiedBatch { profile, .. } => stamp(profile),
        _ => {}
    }
}

fn handle_stream(
    stream: TcpStream,
    backend: &llama_stage_backend::LlamaStageBackend,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;

    let mut prev_encode_us: u64 = 0;
    let mut prev_write_us: u64 = 0;

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

        let decode_started = Instant::now();
        let parsed = rmp_serde::from_slice::<StageNodeRequest>(&request_bytes);
        let decode_us = decode_started.elapsed().as_micros() as u64;

        let handle_started = Instant::now();
        let mut response = match parsed {
            Ok(request) => handle_stage_node_request(backend, request),
            Err(err) => StageNodeResponse::Error {
                message: format!("invalid request: {err}"),
            },
        };
        let handle_us = handle_started.elapsed().as_micros() as u64;

        stamp_server_timing(
            &mut response,
            decode_us,
            handle_us,
            prev_encode_us,
            prev_write_us,
        );

        let encode_started = Instant::now();
        let response_bytes = rmp_serde::to_vec_named(&response)?;
        prev_encode_us = encode_started.elapsed().as_micros() as u64;

        let response_len =
            u32::try_from(response_bytes.len()).context("stage response too large")?;
        let write_started = Instant::now();
        writer.write_all(&response_len.to_le_bytes())?;
        writer.write_all(&response_bytes)?;
        writer.flush()?;
        prev_write_us = write_started.elapsed().as_micros() as u64;
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let backend = build_stage_backend(&StageNodeConfig {
        model_path: args.model_path,
        stage_id: args.stage_id,
        start_layer: args.start_layer,
        end_layer: args.end_layer,
        is_head: args.is_head,
        is_tail: args.is_tail,
    })?;

    let listener = TcpListener::bind(&args.bind_addr)
        .with_context(|| format!("binding {}", args.bind_addr))?;
    eprintln!("listening={}", listener.local_addr()?);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_stream(stream, &backend) {
                    eprintln!("connection error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }

    Ok(())
}
