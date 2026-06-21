use anyhow::Result;
use stage_forward_lab::StageResidencyAdapter;
use std::path::PathBuf;

fn main() -> Result<()> {
    let bundle_root = std::env::args()
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
    let stage_index = std::env::args()
        .nth(3)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(0);
    let include_optional = std::env::args().any(|arg| arg == "--all");

    let adapter = StageResidencyAdapter::load(&bundle_root, &gguf_path, stage_index)?;
    let out_dir = std::env::args()
        .nth(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| bundle_root.join(adapter.default_required_pack_dir_name()));
    let packed = if include_optional {
        adapter.pack_all_tensors(&out_dir)?
    } else {
        adapter.pack_required_tensors(&out_dir)?
    };

    println!("bundle root        : {}", bundle_root.display());
    println!("gguf               : {}", gguf_path.display());
    println!(
        "stage              : {} ({} {}-{})",
        adapter.stage_index() + 1,
        adapter.stage_role(),
        adapter.start_layer(),
        adapter.end_layer()
    );
    println!("out dir            : {}", out_dir.display());
    println!("pack               : {}", packed.pack_path.display());
    println!("index              : {}", packed.index_path.display());
    println!("tensor count       : {}", packed.index.tensor_count);
    println!("packed bytes       : {}", packed.index.total_bytes);

    Ok(())
}
