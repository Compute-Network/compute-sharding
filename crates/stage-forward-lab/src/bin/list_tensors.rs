use anyhow::Result;
use stage_forward_lab::gguf::GgufFile;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/macintosh/.compute/stages/gemma-4-e4b-q4/tail-21-41.gguf".into()
    }));
    let mode = std::env::args().nth(2).unwrap_or_else(|| "tensors".into());
    let file = GgufFile::parse_file(&path)?;
    if mode == "meta" {
        for (k, v) in &file.metadata {
            println!("{}\t{:?}", k, v);
        }
    } else {
        for t in &file.tensors {
            println!("{}\tdims={:?}\ttype={}", t.name, t.dimensions, t.ggml_type);
        }
        eprintln!("total={}", file.tensors.len());
    }
    Ok(())
}
