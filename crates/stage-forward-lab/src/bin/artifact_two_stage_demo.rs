use anyhow::Result;
use stage_forward_lab::{
    ArtifactBackedToyBackend, StageForwardBackend, StageLayout, run_artifact_single_node_reference,
    write_sample_toy_artifacts,
};

fn main() -> Result<()> {
    let temp = std::env::temp_dir().join("compute-backend-artifacts");
    let (head_path, tail_path, full_path) = write_sample_toy_artifacts(&temp)?;
    let prompt = "reply exactly STAGE LAB";

    let mut head = ArtifactBackedToyBackend::new(head_path);
    head.load_layout(StageLayout {
        model_id: "toy-linear-4l".into(),
        stage_id: "0-1".into(),
        start_layer: 0,
        end_layer: 1,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = ArtifactBackedToyBackend::new(tail_path);
    tail.load_layout(StageLayout {
        model_id: "toy-linear-4l".into(),
        stage_id: "2-3".into(),
        start_layer: 2,
        end_layer: 3,
        is_head: false,
        is_tail: true,
    })?;

    let stage1 = head.begin_prompt("artifact-demo", prompt, Some(12), 4)?;
    let stage2 = tail.continue_forward(stage1)?;
    let distributed = tail.sample_tail(stage2)?;
    let single = run_artifact_single_node_reference(&full_path, prompt, Some(12))?;

    println!("single-node : {}", single.text);
    println!("two-stage   : {}", distributed.text);
    println!("match       : {}", single.text == distributed.text);

    Ok(())
}
