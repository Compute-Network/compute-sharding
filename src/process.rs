use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::models::{ShardKind, ShardSpec};

pub struct ChildGuard {
    name: String,
    child: Child,
}

impl ChildGuard {
    fn new(name: impl Into<String>, child: Child) -> Self {
        Self {
            name: name.into(),
            child,
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }
        eprintln!("stopping {}", self.name);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn spawn_stage(
    sidecar_dir: Option<&Path>,
    model_path: &Path,
    shard: ShardSpec,
    bind: &str,
) -> Result<ChildGuard> {
    let bin = resolve_sidecar("llama_stage_tcp_node", sidecar_dir)?;
    let mut command = Command::new(&bin);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--bind")
        .arg(bind)
        .arg("--stage-id")
        .arg(shard.stage_id)
        .arg("--start-layer")
        .arg(shard.start_layer.to_string())
        .arg("--end-layer")
        .arg(shard.end_layer.to_string())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    match shard.kind {
        ShardKind::Head => {
            command.arg("--head");
        }
        ShardKind::Tail => {
            command.arg("--tail");
        }
    }

    println!(
        "starting {} {} on {} with {}",
        shard.kind,
        shard.stage_id,
        bind,
        model_path.display()
    );
    let child = command
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    Ok(ChildGuard::new(format!("{} stage", shard.kind), child))
}

pub fn spawn_gateway(
    sidecar_dir: Option<&Path>,
    head_addr: &str,
    tail_addr: &str,
    bind: &str,
) -> Result<ChildGuard> {
    let bin = resolve_sidecar("llama_stage_gateway_tcp_node", sidecar_dir)?;
    println!("starting gateway on {bind} with head={head_addr} tail={tail_addr}");
    let child = Command::new(&bin)
        .arg("--head")
        .arg(head_addr)
        .arg("--tail")
        .arg(tail_addr)
        .arg("--bind")
        .arg(bind)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    Ok(ChildGuard::new("stage gateway", child))
}

pub fn resolve_sidecar(name: &str, explicit_dir: Option<&Path>) -> Result<PathBuf> {
    for dir in sidecar_search_paths(explicit_dir) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not find {name}; install patched sidecars or pass --sidecar-dir. Run `compute-sharding info` for search paths"
    )
}

pub fn print_sidecar_search_paths() {
    println!();
    println!("sidecar search paths:");
    for path in sidecar_search_paths(None) {
        println!("  {}", path.display());
    }
    println!();
    println!("expected sidecars:");
    println!("  llama_stage_tcp_node");
    println!("  llama_stage_gateway_tcp_node");
}

fn sidecar_search_paths(explicit_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(dir) = explicit_dir {
        paths.push(dir.to_path_buf());
    }
    if let Ok(dir) = std::env::var("COMPUTE_SHARDING_SIDECAR_DIR") {
        paths.push(PathBuf::from(dir));
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".compute").join("bin"));
    }
    if let Ok(current_dir) = std::env::current_dir() {
        if let Some(parent) = current_dir.parent() {
            paths.push(
                parent
                    .join("compute-backend")
                    .join("target")
                    .join("release"),
            );
            paths.push(parent.join("compute-backend").join("target").join("debug"));
        }
    }
    paths
}
