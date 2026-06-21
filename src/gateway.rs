#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const LLAMA_STAGE_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageNodeInfo {
    pub protocol_version: u32,
    pub model_id: String,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayServiceInfo {
    pub protocol_version: u32,
    pub head_info: StageNodeInfo,
    pub tail_info: StageNodeInfo,
    pub reconnect_after_prompt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageTimings {
    pub head_prefill_ms: u64,
    pub head_decode_ms: u64,
    pub tail_decode_ms: u64,
    pub sample_ms: u64,
    pub transfer_bytes: usize,
    pub ttft_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageCompletion {
    pub text: String,
    pub completion_tokens: u32,
    pub token_ids: Vec<i32>,
    pub timings: RemoteStageTimings,
}

#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum StageGatewayRequest<'a> {
    Info,
    Complete {
        request_id: &'a str,
        prompt: &'a str,
        max_tokens: u32,
    },
    Tokenize {
        text: &'a str,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StageGatewayResponse {
    Info {
        protocol_version: u32,
        head_info: StageNodeInfo,
        tail_info: StageNodeInfo,
        reconnect_after_prompt: bool,
    },
    Completion {
        completion: RemoteStageCompletion,
    },
    TokenIds {
        token_ids: Vec<i32>,
    },
    Error {
        message: String,
    },
    Started {
        request_id: String,
    },
    Step {
        step: serde_json::Value,
    },
    Ack,
}

pub struct GatewayClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl GatewayClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, Duration::from_secs(3))
    }

    pub fn connect_with_timeout(addr: &str, timeout: Duration) -> Result<Self> {
        let socket_addr = addr
            .to_socket_addrs()
            .with_context(|| format!("resolving {addr}"))?
            .next()
            .with_context(|| format!("no socket addresses for {addr}"))?;
        let stream = TcpStream::connect_timeout(&socket_addr, timeout)
            .with_context(|| format!("connecting to {addr}"))?;
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(timeout)).ok();
        stream.set_write_timeout(Some(timeout)).ok();
        let reader = BufReader::new(stream.try_clone().context("cloning gateway TCP stream")?);
        Ok(Self { stream, reader })
    }

    pub fn info(&mut self) -> Result<GatewayServiceInfo> {
        match self.request(&StageGatewayRequest::Info)? {
            StageGatewayResponse::Info {
                protocol_version,
                head_info,
                tail_info,
                reconnect_after_prompt,
            } => {
                if protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
                    bail!(
                        "gateway protocol mismatch: expected {}, got {}",
                        LLAMA_STAGE_PROTOCOL_VERSION,
                        protocol_version
                    );
                }
                Ok(GatewayServiceInfo {
                    protocol_version,
                    head_info,
                    tail_info,
                    reconnect_after_prompt,
                })
            }
            other => bail!("expected info response, got {other:?}"),
        }
    }

    pub fn complete(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<RemoteStageCompletion> {
        match self.request(&StageGatewayRequest::Complete {
            request_id,
            prompt,
            max_tokens,
        })? {
            StageGatewayResponse::Completion { completion } => Ok(completion),
            other => bail!("expected completion response, got {other:?}"),
        }
    }

    #[allow(dead_code)]
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<i32>> {
        match self.request(&StageGatewayRequest::Tokenize { text })? {
            StageGatewayResponse::TokenIds { token_ids } => Ok(token_ids),
            other => bail!("expected token_ids response, got {other:?}"),
        }
    }

    fn request(&mut self, request: &StageGatewayRequest<'_>) -> Result<StageGatewayResponse> {
        serde_json::to_writer(&mut self.stream, request).context("serializing gateway request")?;
        self.stream
            .write_all(b"\n")
            .context("writing gateway request")?;
        self.stream.flush().context("flushing gateway request")?;

        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .context("reading gateway response")?;
        if line.trim().is_empty() {
            bail!("gateway returned empty response");
        }
        let response: StageGatewayResponse =
            serde_json::from_str(line.trim()).context("parsing gateway response")?;
        if let StageGatewayResponse::Error { message } = &response {
            bail!("gateway error: {message}");
        }
        Ok(response)
    }
}

pub fn complete_prompt(
    gateway_addr: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<RemoteStageCompletion> {
    let mut client = GatewayClient::connect(gateway_addr)?;
    client
        .complete(
            &format!("compute-sharding-{}", uuid::Uuid::new_v4()),
            prompt,
            max_tokens,
        )
        .with_context(|| format!("completing prompt through {gateway_addr}"))
}
