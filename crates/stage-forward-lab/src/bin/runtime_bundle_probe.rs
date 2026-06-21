use anyhow::Result;
use stage_forward_lab::{LoadedRuntimeBundle, gguf::GgufFile};
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage",
            )
        });
    let gguf_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".compute")
                .join("models")
                .join("gemma-4-E4B-it-Q4_K_M.gguf")
        });

    let bundle = LoadedRuntimeBundle::load(&root)?;
    let gguf = GgufFile::parse_file(&gguf_path)?;
    bundle.validate_against_gguf(&gguf)?;

    println!("bundle root        : {}", root.display());
    println!("gguf               : {}", gguf_path.display());
    println!("model              : {}", bundle.model_name);
    println!("architecture       : {}", bundle.architecture);
    for stage in &bundle.stages {
        println!(
            "stage{} role={} required_tensors={} required_bytes={} required_slices={} optional_tensors={} optional_bytes={} optional_slices={}",
            stage.stage_index + 1,
            stage.role,
            stage.required.len(),
            stage.required_bytes,
            stage.required_slices.len(),
            stage.optional.len(),
            stage.optional_bytes,
            stage.optional_slices.len()
        );
    }

    Ok(())
}
