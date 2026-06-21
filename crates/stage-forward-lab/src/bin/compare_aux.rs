use anyhow::{Context, Result, bail};
use stage_forward_lab::gguf::GgufFile;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn read_bytes(path: &PathBuf, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn main() -> Result<()> {
    let full = PathBuf::from("/Users/macintosh/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf");
    let tail = PathBuf::from("/Users/macintosh/.compute/stages/gemma-4-e4b-q4/tail-21-41.gguf");

    let full_file = GgufFile::parse_file(&full)?;
    let tail_file = GgufFile::parse_file(&tail)?;

    for name in [
        "token_embd.weight",
        "output_norm.weight",
        "per_layer_proj_norm.weight",
        "rope_freqs.weight",
    ] {
        let f = full_file
            .tensors
            .iter()
            .find(|t| t.name == name)
            .context("full missing")?;
        let t = tail_file
            .tensors
            .iter()
            .find(|t| t.name == name)
            .context("tail missing")?;
        let fb = full_file.tensor_byte_len(name).context("fb")?;
        let tb = tail_file.tensor_byte_len(name).context("tb")?;
        println!(
            "\n{name}: full dims={:?} type={} bytes={}; tail dims={:?} type={} bytes={}",
            f.dimensions, f.ggml_type, fb, t.dimensions, t.ggml_type, tb
        );
        if fb != tb {
            println!("  SIZE MISMATCH");
            continue;
        }
        let fabs = full_file.tensor_data_offset + f.offset;
        let tabs = tail_file.tensor_data_offset + t.offset;
        let cmp = (fb as usize).min(1024);
        let fh = read_bytes(&full, fabs, cmp)?;
        let th = read_bytes(&tail, tabs, cmp)?;
        println!("  first 16 bytes match? {}", fh[..16] == th[..16]);
        println!("  first 1024 bytes match? {}", fh == th);
        if fh != th {
            println!("  full[0..32]: {:02x?}", &fh[..32]);
            println!("  tail[0..32]: {:02x?}", &th[..32]);
        }
    }

    // For the blk.*.rope_freqs check: compare full blk.21..41 with tail blk.0..20.
    // But gemma4 stores rope_freqs as a single shared tensor per-layer duplicated from blk.0.
    // Check a few per-layer tensors sample match.
    let pairs: &[(&str, &str)] = &[
        ("blk.21.attn_norm.weight", "blk.0.attn_norm.weight"),
        ("blk.21.attn_q.weight", "blk.0.attn_q.weight"),
        ("blk.21.attn_k.weight", "blk.0.attn_k.weight"),
        ("blk.21.attn_v.weight", "blk.0.attn_v.weight"),
        ("blk.21.ffn_gate.weight", "blk.0.ffn_gate.weight"),
        ("blk.41.ffn_down.weight", "blk.20.ffn_down.weight"),
        ("blk.41.attn_output.weight", "blk.20.attn_output.weight"),
        ("blk.35.attn_k.weight", "blk.14.attn_k.weight"),
    ];
    for (fname, tname) in pairs {
        let f = full_file.tensors.iter().find(|t| t.name == *fname);
        let t = tail_file.tensors.iter().find(|t| t.name == *tname);
        match (f, t) {
            (Some(f), Some(t)) => {
                let fb = full_file.tensor_byte_len(fname).context("fb")?;
                let tb = tail_file.tensor_byte_len(tname).context("tb")?;
                let fabs = full_file.tensor_data_offset + f.offset;
                let tabs = tail_file.tensor_data_offset + t.offset;
                if fb != tb {
                    println!("\n{fname} vs {tname}: BYTES {} vs {}", fb, tb);
                    continue;
                }
                let cmp = (fb as usize).min(512);
                let fh = read_bytes(&full, fabs, cmp)?;
                let th = read_bytes(&tail, tabs, cmp)?;
                println!(
                    "\n{fname} vs {tname}: dims={:?} type={} match_first_512={}",
                    f.dimensions,
                    f.ggml_type,
                    fh == th
                );
            }
            _ => bail!("missing pair: {fname} or {tname}"),
        }
    }

    Ok(())
}
