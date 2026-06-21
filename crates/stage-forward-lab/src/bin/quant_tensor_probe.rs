use anyhow::Result;
use stage_forward_lab::{StageTensorStore, quants};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let index_path = env::args_os().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("usage: quant_tensor_probe <stage-index.json> <tensor-name>")
    })?;
    let tensor_name = env::args().nth(2).ok_or_else(|| {
        anyhow::anyhow!("usage: quant_tensor_probe <stage-index.json> <tensor-name>")
    })?;

    let store = StageTensorStore::load(&index_path)?;
    let entry = store
        .entry(&tensor_name)
        .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in stage store", tensor_name))?;
    let bytes = store.read(&tensor_name)?;
    let decoded = quants::dequantize_tensor(entry.ggml_type, &bytes)?;

    println!("tensor         : {}", entry.name);
    println!(
        "ggml_type      : {} ({})",
        entry.ggml_type,
        quants::ggml_type_name(entry.ggml_type)
    );
    println!("dimensions     : {:?}", entry.dimensions);
    println!("byte_len       : {}", entry.byte_len);
    println!("decoded_len    : {}", decoded.len());
    println!(
        "decoded_head   : {:?}",
        decoded.iter().take(16).copied().collect::<Vec<_>>()
    );

    Ok(())
}
