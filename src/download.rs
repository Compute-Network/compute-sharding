use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::models::{self, ShardSelection, ShardSpec};

pub fn download_selection(selection: ShardSelection, root: &Path) -> Result<()> {
    fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    for shard in models::shards_for_selection(selection) {
        download_shard(shard, root)?;
    }
    if models::selection_includes_draft(selection) {
        ensure_draft_model()?;
    }
    Ok(())
}

pub fn ensure_shard(shard: ShardSpec, root: &Path) -> Result<()> {
    let path = models::shard_path(root, shard);
    if path.exists() {
        println!("found {} at {}", shard.kind, path.display());
        return Ok(());
    }
    fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    download_shard(shard, root)
}

fn download_shard(shard: ShardSpec, root: &Path) -> Result<()> {
    let path = models::shard_path(root, shard);
    if path.exists() {
        println!("{} already downloaded: {}", shard.kind, path.display());
        return Ok(());
    }

    let tmp_path = path.with_extension("gguf.part");
    println!("downloading {} shard from Hugging Face", shard.kind);
    println!("{}", shard.url);

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("compute-sharding/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building HTTP client")?;
    let mut response = client
        .get(shard.url)
        .send()
        .with_context(|| format!("requesting {}", shard.url))?
        .error_for_status()
        .with_context(|| format!("downloading {}", shard.url))?;

    let total = response.content_length();
    let mut file =
        File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?;
    let mut downloaded = 0u64;
    let mut last_printed_gb = 0u64;
    let mut buf = [0u8; 1024 * 256];
    loop {
        let read = response.read(&mut buf).context("reading download")?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .with_context(|| format!("writing {}", tmp_path.display()))?;
        downloaded += read as u64;
        let downloaded_gb = downloaded / 1_000_000_000;
        if downloaded_gb > last_printed_gb {
            last_printed_gb = downloaded_gb;
            if let Some(total) = total {
                println!(
                    "  {:.1}/{:.1} GB",
                    downloaded as f64 / 1_000_000_000.0,
                    total as f64 / 1_000_000_000.0
                );
            } else {
                println!("  {:.1} GB", downloaded as f64 / 1_000_000_000.0);
            }
        }
    }
    file.flush().context("flushing shard file")?;
    fs::rename(&tmp_path, &path)
        .with_context(|| format!("moving {} to {}", tmp_path.display(), path.display()))?;
    println!("saved {}", path.display());
    Ok(())
}

pub fn ensure_draft_model() -> Result<std::path::PathBuf> {
    let path = models::default_draft_path();
    if path.exists() {
        println!("found draft model at {}", path.display());
        return Ok(path);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    println!("downloading draft model");
    println!("{}", models::DRAFT_URL);
    download_url(models::DRAFT_URL, &path)?;
    Ok(path)
}

fn download_url(url: &str, path: &Path) -> Result<()> {
    let tmp_path = path.with_extension("part");
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("compute-sharding/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building HTTP client")?;
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("requesting {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    let mut file =
        File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?;
    let mut buf = [0u8; 1024 * 256];
    loop {
        let read = response.read(&mut buf).context("reading download")?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .with_context(|| format!("writing {}", tmp_path.display()))?;
    }
    file.flush().context("flushing download")?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("moving {} to {}", tmp_path.display(), path.display()))?;
    println!("saved {}", path.display());
    Ok(())
}
