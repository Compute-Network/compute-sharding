use anyhow::Result;
use stage_forward_lab::gguf::GgufFile;
use std::path::PathBuf;

fn main() -> Result<()> {
    let gguf_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".compute")
                .join("models")
                .join("gemma-4-E4B-it-Q4_K_M.gguf")
        });
    let out_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"));

    println!("gguf  : {}", gguf_path.display());
    println!("out   : {}", out_path.display());

    let file = GgufFile::parse_file(&gguf_path)?;

    let tokens = file
        .metadata_string_array("tokenizer.ggml.tokens")
        .ok_or_else(|| anyhow::anyhow!("No tokenizer.ggml.tokens found in GGUF metadata"))?;

    println!("vocab size : {}", tokens.len());
    println!("first 10   :");
    for (i, tok) in tokens.iter().take(10).enumerate() {
        println!("  {}: {:?}", i, tok);
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_vec(&tokens)?)?;
    println!("wrote {} tokens to {}", tokens.len(), out_path.display());

    if let Some(scores) = file.metadata_f32_array("tokenizer.ggml.scores") {
        let scores_path = out_path.with_file_name("vocab_scores.json");
        std::fs::write(&scores_path, serde_json::to_vec(&scores)?)?;
        println!("wrote {} scores to {}", scores.len(), scores_path.display());
    } else {
        println!("no tokenizer.ggml.scores found");
    }

    Ok(())
}
