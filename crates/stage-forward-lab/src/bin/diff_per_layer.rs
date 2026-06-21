#![allow(clippy::single_element_loop)]

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
    let full =
        PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
            "/Users/macintosh/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf".into()
        }));
    let tail = PathBuf::from(std::env::args().nth(2).unwrap_or_else(|| {
        "/Users/macintosh/.compute/stages/gemma-4-e4b-q4/tail-21-41.gguf".into()
    }));

    let full_file = GgufFile::parse_file(&full)?;
    let tail_file = GgufFile::parse_file(&tail)?;

    let name = "per_layer_model_proj.weight";
    let full_t = full_file
        .tensors
        .iter()
        .find(|t| t.name == name)
        .context("full missing")?;
    let tail_t = tail_file
        .tensors
        .iter()
        .find(|t| t.name == name)
        .context("tail missing")?;

    println!(
        "full:  dims={:?} type={} rel_offset={} abs_offset={}",
        full_t.dimensions,
        full_t.ggml_type,
        full_t.offset,
        full_file.tensor_data_offset + full_t.offset
    );
    println!(
        "tail:  dims={:?} type={} rel_offset={} abs_offset={}",
        tail_t.dimensions,
        tail_t.ggml_type,
        tail_t.offset,
        tail_file.tensor_data_offset + tail_t.offset
    );

    // For dims [2560, 10752] full vs [2560, 5376] tail, ne[0]=2560 is inner.
    // If the shard holds rows 0..5375 of the full tensor's "outer" axis, its
    // bytes match the first half of the full tensor. If it holds rows 5376..10751,
    // its bytes match the second half. We compare the first 512 bytes of the tail
    // against both halves of the full tensor.
    let cmp_len: usize = 512;
    if tail_t.dimensions.len() != 2 || full_t.dimensions.len() != 2 {
        bail!("expected 2D tensors");
    }
    if tail_t.dimensions[0] != full_t.dimensions[0] {
        bail!("inner dim mismatch");
    }
    if full_t.dimensions[1] != tail_t.dimensions[1] * 2 {
        bail!("expected full outer dim == 2x tail outer dim");
    }

    // Byte size per row (along ne[1] axis) = ne[0] * bytes_per_elem.
    // We don't need exact bytes-per-elem: since ne[0]s match, half of full's
    // byte size equals tail's total byte size.
    let full_tensor_bytes = full_file.tensor_byte_len(name).context("full bytes")?;
    let tail_tensor_bytes = tail_file.tensor_byte_len(name).context("tail bytes")?;
    assert_eq!(full_tensor_bytes, tail_tensor_bytes * 2);

    let tail_abs = tail_file.tensor_data_offset + tail_t.offset;
    let full_abs = full_file.tensor_data_offset + full_t.offset;

    let tail_head = read_bytes(&tail, tail_abs, cmp_len)?;
    let full_first_half_head = read_bytes(&full, full_abs, cmp_len)?;
    let full_second_half_head = read_bytes(&full, full_abs + tail_tensor_bytes, cmp_len)?;

    let first_match = tail_head == full_first_half_head;
    let second_match = tail_head == full_second_half_head;

    println!("tail first 16 bytes:      {:02x?}", &tail_head[..16]);
    println!(
        "full rows 0.. first 16:   {:02x?}",
        &full_first_half_head[..16]
    );
    println!(
        "full rows N/2.. first 16: {:02x?}",
        &full_second_half_head[..16]
    );
    println!("tail == full first half?  {}", first_match);
    println!("tail == full second half? {}", second_match);

    if first_match {
        println!(
            "VERDICT: tail shard holds the HEAD half (rows 0..{}). Wrong slice.",
            tail_t.dimensions[1]
        );
    } else if second_match {
        println!(
            "VERDICT: tail shard holds the TAIL half (rows {}..{}). Correct slice.",
            tail_t.dimensions[1], full_t.dimensions[1]
        );
    } else {
        println!("VERDICT: tail shard does not match either half directly.");
    }

    // per_layer_token_embd.weight is Q6_K with inner dim sliced from 10752 → 5376.
    // A Q6_K block covers 256 innermost elements and takes 210 bytes. For each row
    // in the full tensor there are 42 blocks (= 10752/256) laid out contiguously;
    // the tail contains 21 blocks per row. If the packer kept "last 21 blocks" of
    // every row (the tail layer range 21..41), then tail row 0 equals full row 0
    // starting at byte offset 21*210 = 4410. If it kept the first 21 blocks, the
    // tail's bytes match the full tensor's first half of each row.
    {
        let name = "per_layer_token_embd.weight";
        println!("\n--- {name} (Q6_K block-aware) ---");
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
        let full_abs = full_file.tensor_data_offset + f.offset;
        let tail_abs = tail_file.tensor_data_offset + t.offset;
        // type 13 is Q5_K: 256 elements packed into 176 bytes per block.
        // Tail has 21 blocks per row (of ne[1]=262144 rows); full has 42.
        let one_block = 176u64;
        let tail_block0 = read_bytes(&tail, tail_abs, one_block as usize)?;
        let full_block0 = read_bytes(&full, full_abs, one_block as usize)?;
        let full_block21 = read_bytes(&full, full_abs + 21 * one_block, one_block as usize)?;
        println!("tail block0    first 16: {:02x?}", &tail_block0[..16]);
        println!("full block0    first 16: {:02x?}", &full_block0[..16]);
        println!("full block21   first 16: {:02x?}", &full_block21[..16]);
        println!("tail[0] == full block0?   {}", tail_block0 == full_block0);
        println!("tail[0] == full block21?  {}", tail_block0 == full_block21);
        if tail_block0 == full_block21 {
            println!("VERDICT: tail holds last 21 blocks per row (layers 21..41). Correct slice.");
        } else if tail_block0 == full_block0 {
            println!("VERDICT: tail holds first 21 blocks per row (layers 0..20). WRONG slice.");
        } else {
            println!("VERDICT: tail matches neither common slice.");
        }
    }

    // Also compare per_layer_proj_norm.weight
    for name in ["per_layer_proj_norm.weight"] {
        println!("\n--- {name} ---");
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
        println!("full dims={:?} type={} ", f.dimensions, f.ggml_type);
        println!("tail dims={:?} type={} ", t.dimensions, t.ggml_type);
        let fb = full_file.tensor_byte_len(name).context("fb")?;
        let tb = tail_file.tensor_byte_len(name).context("tb")?;
        let tail_abs = tail_file.tensor_data_offset + t.offset;
        let full_abs = full_file.tensor_data_offset + f.offset;
        let tail_head = read_bytes(&tail, tail_abs, cmp_len.min(tb as usize))?;
        let full_first = read_bytes(&full, full_abs, cmp_len.min(tb as usize))?;
        println!("full bytes total = {}, tail bytes total = {}", fb, tb);
        if fb == tb {
            let eq = tail_head == full_first;
            println!(
                "sizes equal; tail first {} bytes == full first? {}",
                cmp_len, eq
            );
        } else if fb == tb * 2 {
            let full_second = read_bytes(&full, full_abs + tb, cmp_len.min(tb as usize))?;
            let first = tail_head == full_first;
            let second = tail_head == full_second;
            println!("tail == full first half?  {}", first);
            println!("tail == full second half? {}", second);
        } else {
            println!("size ratio not 1 or 2; skipping");
        }
    }

    Ok(())
}
