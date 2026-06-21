use anyhow::{Result, bail};
use stage_forward_lab::gguf::{GgufFile, StageSplit};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".compute")
                .join("models")
                .join("gemma-4-E4B-it-Q4_K_M.gguf")
        });

    let out_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage",
            )
        });
    let split_arg = std::env::args().nth(3);

    let file = GgufFile::parse_file(&model_path)?;
    let splits = parse_splits(&file, split_arg.as_deref())?;
    validate_supported_real_forward_splits(&file, &splits)?;
    let plan = file.plan_for_splits(&splits);
    let written = plan.write_bundle(&file, &out_dir)?;

    println!("bundle root   : {}", written.root_dir.display());
    println!("manifest      : {}", written.manifest_path.display());
    println!("stage count   : {}", splits.len());
    for split in &splits {
        println!(
            "stage         : {} => {}-{}",
            split.stage_index + 1,
            split.start_layer,
            split.end_layer
        );
    }
    for path in written.stage_manifest_paths {
        println!("stage manifest: {}", path.display());
    }

    Ok(())
}

fn parse_splits(file: &GgufFile, value: Option<&str>) -> Result<Vec<StageSplit>> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return file
            .suggest_even_stage_split(2)
            .ok_or_else(|| anyhow::anyhow!("Could not infer a 2-stage split from GGUF"));
    };

    if let Ok(stage_count) = value.parse::<u32>() {
        return file.suggest_even_stage_split(stage_count).ok_or_else(|| {
            anyhow::anyhow!("Could not infer a {stage_count}-stage split from GGUF")
        });
    }

    let mut splits = Vec::new();
    for (stage_index, chunk) in value.split(',').enumerate() {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (start, end) = chunk
            .split_once('-')
            .ok_or_else(|| anyhow::anyhow!("Split must be start-end, got {chunk}"))?;
        let start_layer: u32 = start.trim().parse()?;
        let end_layer: u32 = end.trim().parse()?;
        if end_layer < start_layer {
            anyhow::bail!("Invalid split {chunk}: end before start");
        }
        splits.push(StageSplit {
            stage_index: stage_index as u32,
            start_layer,
            end_layer,
        });
    }
    if splits.is_empty() {
        anyhow::bail!("No valid split ranges were provided");
    }

    if let Some(total_layers) = file.inferred_layer_count() {
        if splits[0].start_layer != 0 {
            anyhow::bail!("Explicit splits must start at layer 0");
        }
        let mut expected_start = 0u32;
        for split in &splits {
            if split.start_layer != expected_start {
                anyhow::bail!(
                    "Explicit splits must be contiguous; expected start {}, got {}",
                    expected_start,
                    split.start_layer
                );
            }
            expected_start = split.end_layer.saturating_add(1);
        }
        if expected_start != total_layers {
            anyhow::bail!(
                "Explicit splits must cover all layers 0..{}; ended at {}",
                total_layers.saturating_sub(1),
                expected_start.saturating_sub(1)
            );
        }
    }

    Ok(splits)
}

fn validate_supported_real_forward_splits(file: &GgufFile, splits: &[StageSplit]) -> Result<()> {
    if !looks_like_gemma_4_e4b(file) {
        return Ok(());
    }

    for split in splits {
        if split.end_layer >= 24 && !(split.start_layer <= 22 && split.end_layer >= 23) {
            bail!(
                "unsupported real_forward split for Gemma E4B: stage {} ({}-{}) places layers 24+ away from shared-KV source layers 22/23; current contract keeps shared-KV caches stage-local",
                split.stage_index + 1,
                split.start_layer,
                split.end_layer
            );
        }
    }

    Ok(())
}

fn looks_like_gemma_4_e4b(file: &GgufFile) -> bool {
    matches!(file.architecture(), Some("gemma4"))
        && file.inferred_layer_count() == Some(42)
        && file.hidden_size() == Some(2560)
}

#[cfg(test)]
mod tests {
    use super::{looks_like_gemma_4_e4b, validate_supported_real_forward_splits};
    use stage_forward_lab::gguf::{GgufFile, MetadataValue, StageSplit};
    use std::collections::BTreeMap;

    fn gemma_4_e4b_stub() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            MetadataValue::String("gemma4".into()),
        );
        metadata.insert("gemma4.block_count".into(), MetadataValue::Uint32(42));
        metadata.insert(
            "gemma4.embedding_length".into(),
            MetadataValue::Uint32(2560),
        );
        GgufFile {
            version: 3,
            tensor_count: 0,
            file_size_bytes: 0,
            tensor_data_offset: 0,
            metadata,
            tensors: Vec::new(),
        }
    }

    #[test]
    fn identifies_gemma_4_e4b_bundle_inputs() {
        assert!(looks_like_gemma_4_e4b(&gemma_4_e4b_stub()));
    }

    #[test]
    fn rejects_invalid_gemma_4_e4b_three_stage_split() {
        let error = validate_supported_real_forward_splits(
            &gemma_4_e4b_stub(),
            &[
                StageSplit {
                    stage_index: 0,
                    start_layer: 0,
                    end_layer: 13,
                },
                StageSplit {
                    stage_index: 1,
                    start_layer: 14,
                    end_layer: 27,
                },
                StageSplit {
                    stage_index: 2,
                    start_layer: 28,
                    end_layer: 41,
                },
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("shared-KV source layers 22/23"));
        assert!(error.contains("stage 3 (28-41)"));
    }

    #[test]
    fn accepts_valid_gemma_4_e4b_three_stage_split() {
        validate_supported_real_forward_splits(
            &gemma_4_e4b_stub(),
            &[
                StageSplit {
                    stage_index: 0,
                    start_layer: 0,
                    end_layer: 10,
                },
                StageSplit {
                    stage_index: 1,
                    start_layer: 11,
                    end_layer: 21,
                },
                StageSplit {
                    stage_index: 2,
                    start_layer: 22,
                    end_layer: 41,
                },
            ],
        )
        .unwrap();
    }
}
