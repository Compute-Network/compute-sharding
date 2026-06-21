use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::download;
use crate::gateway;
use crate::models::{self, ShardKind};
use crate::process::{self, ChildGuard};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Head,
    Tail,
    Auto,
}

impl Role {
    fn counterpart(self) -> Option<Role> {
        match self {
            Role::Head => Some(Role::Tail),
            Role::Tail => Some(Role::Head),
            Role::Auto => None,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::Head => write!(f, "head"),
            Role::Tail => write!(f, "tail"),
            Role::Auto => write!(f, "auto"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub role: Role,
    pub bind: String,
    pub public_addr: Option<String>,
    pub peers: Vec<String>,
    pub stage_bind: Option<String>,
    pub stage_connect_addr: Option<String>,
    pub public_stage_addr: Option<String>,
    pub gateway_bind: String,
    pub sidecar_dir: Option<PathBuf>,
    pub model_dir: PathBuf,
    pub no_spawn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerAdvert {
    pub id: String,
    pub role: Role,
    pub model_id: String,
    pub shard: Option<ShardKind>,
    pub public_addr: String,
    pub stage_addr: Option<String>,
    pub gateway_addr: Option<String>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerRecordView {
    pub advert: PeerAdvert,
    pub latency_ms: Option<u128>,
    pub last_seen_unix: u64,
}

#[derive(Debug, Clone)]
struct PeerRecord {
    advert: PeerAdvert,
    latency_ms: Option<u128>,
    last_seen_unix: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    local: PeerAdvert,
    peers: Vec<PeerRecordView>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegisterResponse {
    local: PeerAdvert,
    peers: Vec<PeerRecordView>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
    prompt: String,
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    text: String,
    completion_tokens: u32,
    ttft_ms: u64,
    total_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub url: String,
    pub latency_ms: u128,
    pub advert: PeerAdvert,
}

struct SharedState {
    local: Mutex<PeerAdvert>,
    peers: Mutex<HashMap<String, PeerRecord>>,
}

pub fn run(config: ServeConfig) -> Result<()> {
    let initial_probes = probe_initial_peers(&config.peers);
    let effective_role = resolve_role(config.role, &initial_probes);
    let shard_kind = match effective_role {
        Role::Head => ShardKind::Head,
        Role::Tail => ShardKind::Tail,
        Role::Auto => unreachable!("auto role must be resolved"),
    };
    let shard = models::shard_for_kind(shard_kind);
    let stage_bind = config
        .stage_bind
        .clone()
        .unwrap_or_else(|| default_stage_bind(effective_role).to_string());
    let stage_connect_addr = config
        .stage_connect_addr
        .clone()
        .unwrap_or_else(|| local_connect_addr(&stage_bind));
    let public_stage_addr = config
        .public_stage_addr
        .clone()
        .unwrap_or_else(|| stage_connect_addr.clone());
    let public_addr = config
        .public_addr
        .clone()
        .unwrap_or_else(|| format!("http://{}", local_connect_addr(&config.bind)));
    let public_addr = normalize_base_url(&public_addr);

    println!("Compute sharding orchestrator");
    println!(
        "role={effective_role} bind={} public={public_addr}",
        config.bind
    );
    println!("model_dir={}", config.model_dir.display());

    let mut children: Vec<ChildGuard> = Vec::new();
    if !config.no_spawn {
        download::ensure_shard(shard, &config.model_dir)?;
        let model_path = models::shard_path(&config.model_dir, shard);
        children.push(process::spawn_stage(
            config.sidecar_dir.as_deref(),
            &model_path,
            shard,
            &stage_bind,
        )?);
        thread::sleep(Duration::from_millis(750));
    } else {
        println!("--no-spawn enabled; assuming stage sidecars are already running");
    }

    let local = PeerAdvert {
        id: uuid::Uuid::new_v4().to_string(),
        role: effective_role,
        model_id: models::MODEL_ID.to_string(),
        shard: Some(shard_kind),
        public_addr: public_addr.clone(),
        stage_addr: Some(if effective_role == Role::Tail {
            public_stage_addr.clone()
        } else {
            stage_connect_addr.clone()
        }),
        gateway_addr: if effective_role == Role::Head {
            Some(config.gateway_bind.clone())
        } else {
            None
        },
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let state = Arc::new(SharedState {
        local: Mutex::new(local),
        peers: Mutex::new(HashMap::new()),
    });
    for probe in initial_probes {
        upsert_peer(&state, probe.advert, Some(probe.latency_ms));
    }

    let server_state = state.clone();
    let bind = config.bind.clone();
    let _server = thread::spawn(move || {
        if let Err(err) = serve_http(server_state, &bind) {
            eprintln!("orchestrator HTTP server stopped: {err:#}");
        }
    });

    let mut gateway_child: Option<ChildGuard> = None;
    loop {
        for peer in &config.peers {
            match register_with_peer(peer, state_local(&state)) {
                Ok((advert, records, latency_ms)) => {
                    upsert_peer(&state, advert, Some(latency_ms));
                    for record in records {
                        if record.advert.id != state_local(&state).id {
                            upsert_peer(&state, record.advert, record.latency_ms);
                        }
                    }
                }
                Err(err) => eprintln!("peer register failed for {peer}: {err:#}"),
            }
        }

        refresh_peer_latencies(&state);
        if effective_role == Role::Head && gateway_child.is_none() {
            if let Some(tail) = lowest_latency_counterpart(&state, effective_role) {
                if let Some(tail_stage_addr) = tail.advert.stage_addr.clone() {
                    println!(
                        "selected tail {} at {} ({} ms)",
                        &tail.advert.id[..8.min(tail.advert.id.len())],
                        tail_stage_addr,
                        tail.latency_ms.unwrap_or_default()
                    );
                    let draft_model = download::ensure_draft_model()?;
                    gateway_child = Some(process::spawn_gateway(
                        config.sidecar_dir.as_deref(),
                        &stage_connect_addr,
                        &tail_stage_addr,
                        &config.gateway_bind,
                        Some(&draft_model),
                    )?);
                    let mut local = state.local.lock().expect("local peer lock poisoned");
                    local.gateway_addr = Some(config.gateway_bind.clone());
                }
            } else {
                println!("waiting for reachable tail peer");
            }
        }

        let peers = peer_views(&state);
        println!("known_peers={} role={effective_role}", peers.len());
        thread::sleep(Duration::from_secs(10));

        // Keep child guards alive.
        let _ = &children;
        let _ = &gateway_child;
    }
}

pub fn probe_peer(peer: &str) -> Result<ProbeResult> {
    let base = normalize_base_url(peer);
    let start = Instant::now();
    let response: HealthResponse = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .context("building HTTP client")?
        .get(format!("{base}/health"))
        .send()
        .with_context(|| format!("GET {base}/health"))?
        .error_for_status()
        .with_context(|| format!("GET {base}/health"))?
        .json()
        .with_context(|| format!("decoding {base}/health"))?;
    Ok(ProbeResult {
        url: base,
        latency_ms: start.elapsed().as_millis(),
        advert: response.local,
    })
}

pub fn fetch_peers(orchestrator_url: &str) -> Result<Vec<PeerRecordView>> {
    let base = normalize_base_url(orchestrator_url);
    let response: HealthResponse = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .context("building HTTP client")?
        .get(format!("{base}/health"))
        .send()
        .with_context(|| format!("GET {base}/health"))?
        .error_for_status()
        .with_context(|| format!("GET {base}/health"))?
        .json()
        .with_context(|| format!("decoding {base}/health"))?;
    let mut peers = response.peers;
    peers.insert(
        0,
        PeerRecordView {
            advert: response.local,
            latency_ms: Some(0),
            last_seen_unix: now_unix(),
        },
    );
    Ok(peers)
}

fn serve_http(state: Arc<SharedState>, bind: &str) -> Result<()> {
    let listener = TcpListener::bind(bind).with_context(|| format!("binding {bind}"))?;
    println!("orchestrator HTTP listening on {bind}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = state.clone();
                thread::spawn(move || {
                    if let Err(err) = handle_http_client(state, stream) {
                        eprintln!("HTTP client error: {err:#}");
                    }
                });
            }
            Err(err) => eprintln!("HTTP accept error: {err}"),
        }
    }
    Ok(())
}

fn handle_http_client(state: Arc<SharedState>, mut stream: TcpStream) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().context("cloning HTTP stream")?);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .context("reading request line")?;
    if first_line.trim().is_empty() {
        return Ok(());
    }
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).context("reading headers")?;
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = line_trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .context("reading request body")?;
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") | ("GET", "/peers") => {
            let body = HealthResponse {
                local: state_local(&state),
                peers: peer_views(&state),
            };
            write_json(&mut stream, 200, &body)
        }
        ("POST", "/register") => {
            let advert: PeerAdvert =
                serde_json::from_slice(&body).context("decoding peer advert")?;
            upsert_peer(&state, advert, None);
            let body = RegisterResponse {
                local: state_local(&state),
                peers: peer_views(&state),
            };
            write_json(&mut stream, 200, &body)
        }
        ("POST", "/chat") => {
            let request: ChatRequest =
                serde_json::from_slice(&body).context("decoding chat request")?;
            let local = state_local(&state);
            let Some(gateway_addr) = local.gateway_addr else {
                bail!("local node does not have a gateway");
            };
            let completion = gateway::complete_prompt(
                &gateway_addr,
                &request.prompt,
                request.max_tokens.unwrap_or(96),
            )?;
            let body = ChatResponse {
                text: completion.text,
                completion_tokens: completion.completion_tokens,
                ttft_ms: completion.timings.ttft_ms,
                total_ms: completion.timings.total_ms,
            };
            write_json(&mut stream, 200, &body)
        }
        _ => write_plain(&mut stream, 404, "not found"),
    }
}

fn write_json<T: Serialize>(stream: &mut TcpStream, status: u16, value: &T) -> Result<()> {
    let body = serde_json::to_vec(value).context("serializing response")?;
    let reason = if status == 200 { "OK" } else { "ERROR" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .context("writing response headers")?;
    stream.write_all(&body).context("writing response body")?;
    Ok(())
}

fn write_plain(stream: &mut TcpStream, status: u16, value: &str) -> Result<()> {
    let reason = if status == 404 { "Not Found" } else { "ERROR" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        value.len(),
        value
    )
    .context("writing response")?;
    Ok(())
}

fn probe_initial_peers(peers: &[String]) -> Vec<ProbeResult> {
    peers
        .iter()
        .filter_map(|peer| match probe_peer(peer) {
            Ok(result) => Some(result),
            Err(err) => {
                eprintln!("initial probe failed for {peer}: {err:#}");
                None
            }
        })
        .collect()
}

fn resolve_role(requested: Role, probes: &[ProbeResult]) -> Role {
    match requested {
        Role::Head | Role::Tail => requested,
        Role::Auto => {
            if probes.iter().any(|probe| probe.advert.role == Role::Tail) {
                Role::Head
            } else {
                Role::Tail
            }
        }
    }
}

fn register_with_peer(
    peer: &str,
    advert: PeerAdvert,
) -> Result<(PeerAdvert, Vec<PeerRecordView>, u128)> {
    let base = normalize_base_url(peer);
    let start = Instant::now();
    let response: RegisterResponse = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .context("building HTTP client")?
        .post(format!("{base}/register"))
        .json(&advert)
        .send()
        .with_context(|| format!("POST {base}/register"))?
        .error_for_status()
        .with_context(|| format!("POST {base}/register"))?
        .json()
        .with_context(|| format!("decoding {base}/register"))?;
    Ok((response.local, response.peers, start.elapsed().as_millis()))
}

fn refresh_peer_latencies(state: &Arc<SharedState>) {
    let peers = peer_views(state);
    for view in peers {
        match probe_peer(&view.advert.public_addr) {
            Ok(result) => upsert_peer(state, result.advert, Some(result.latency_ms)),
            Err(err) => eprintln!(
                "latency probe failed for {}: {err:#}",
                view.advert.public_addr
            ),
        }
    }
}

fn lowest_latency_counterpart(state: &Arc<SharedState>, role: Role) -> Option<PeerRecordView> {
    let counterpart = role.counterpart()?;
    peer_views(state)
        .into_iter()
        .filter(|record| {
            record.advert.role == counterpart
                && record.advert.model_id == models::MODEL_ID
                && record.advert.stage_addr.is_some()
        })
        .min_by_key(|record| record.latency_ms.unwrap_or(u128::MAX))
}

fn upsert_peer(state: &Arc<SharedState>, advert: PeerAdvert, latency_ms: Option<u128>) {
    if advert.id == state_local(state).id {
        return;
    }
    let mut peers = state.peers.lock().expect("peer map lock poisoned");
    let existing_latency = peers.get(&advert.id).and_then(|record| record.latency_ms);
    peers.insert(
        advert.id.clone(),
        PeerRecord {
            advert,
            latency_ms: latency_ms.or(existing_latency),
            last_seen_unix: now_unix(),
        },
    );
}

fn peer_views(state: &Arc<SharedState>) -> Vec<PeerRecordView> {
    let mut peers: Vec<_> = state
        .peers
        .lock()
        .expect("peer map lock poisoned")
        .values()
        .map(|record| PeerRecordView {
            advert: record.advert.clone(),
            latency_ms: record.latency_ms,
            last_seen_unix: record.last_seen_unix,
        })
        .collect();
    peers.sort_by_key(|record| record.latency_ms.unwrap_or(u128::MAX));
    peers
}

fn state_local(state: &Arc<SharedState>) -> PeerAdvert {
    state
        .local
        .lock()
        .expect("local peer lock poisoned")
        .clone()
}

fn default_stage_bind(role: Role) -> &'static str {
    match role {
        Role::Head => "127.0.0.1:9201",
        Role::Tail => "0.0.0.0:9202",
        Role::Auto => "127.0.0.1:9201",
    }
}

fn local_connect_addr(bind: &str) -> String {
    if let Some((host, port)) = bind.rsplit_once(':') {
        if host == "0.0.0.0" || host == "::" || host.is_empty() {
            return format!("127.0.0.1:{port}");
        }
    }
    bind.to_string()
}

fn normalize_base_url(raw: &str) -> String {
    let with_scheme = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    with_scheme.trim_end_matches('/').to_string()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
