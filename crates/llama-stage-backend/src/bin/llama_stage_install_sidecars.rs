#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::default_compute_bin_dir;
use std::env;
use std::env::consts::EXE_SUFFIX;
use std::fs;
use std::path::{Path, PathBuf};

const SIDECAR_BINS: &[&str] = &["llama_stage_tcp_node", "llama_stage_gateway_tcp_node"];

fn parse_args() -> Result<PathBuf> {
    let mut bin_dir = default_compute_bin_dir().unwrap_or_else(|| PathBuf::from(".compute/bin"));
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bin-dir" => {
                bin_dir = PathBuf::from(it.next().context("missing value for --bin-dir")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(bin_dir)
}

fn current_profile_dir() -> Result<PathBuf> {
    env::current_exe()?
        .parent()
        .map(Path::to_path_buf)
        .context("failed to resolve current executable directory")
}

fn copy_sidecar(src: &Path, dst: &Path) -> Result<()> {
    fs::copy(src, dst)
        .with_context(|| format!("copying {} -> {}", src.display(), dst.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(dst)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(dst, perms)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let bin_dir = parse_args()?;
    let source_dir = current_profile_dir()?;
    fs::create_dir_all(&bin_dir).with_context(|| format!("creating {}", bin_dir.display()))?;

    println!("source_dir={}", source_dir.display());
    println!("target_bin_dir={}", bin_dir.display());

    for name in SIDECAR_BINS {
        let file_name = format!("{name}{EXE_SUFFIX}");
        let src = source_dir.join(&file_name);
        if !src.exists() {
            bail!(
                "missing sidecar binary {}; build with `cargo build -p llama-stage-backend --bins` first",
                src.display()
            );
        }

        let dst = bin_dir.join(&file_name);
        copy_sidecar(&src, &dst)?;
        println!("installed={} from={}", dst.display(), src.display());
    }

    println!("overall=PASS");
    Ok(())
}
