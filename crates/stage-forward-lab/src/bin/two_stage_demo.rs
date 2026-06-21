use anyhow::Result;
use stage_forward_lab::{
    StageForwardBackend, StageLayout, ToyLinearBackend, run_toy_single_node_reference,
};

fn main() -> Result<()> {
    let prompt = "reply exactly STAGE LAB";

    let mut head = ToyLinearBackend::default();
    head.load_layout(StageLayout {
        model_id: "toy-linear-4l".into(),
        stage_id: "0-1".into(),
        start_layer: 0,
        end_layer: 1,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = ToyLinearBackend::default();
    tail.load_layout(StageLayout {
        model_id: "toy-linear-4l".into(),
        stage_id: "2-3".into(),
        start_layer: 2,
        end_layer: 3,
        is_head: false,
        is_tail: true,
    })?;

    let stage1 = head.begin_prompt("demo-req", prompt, Some(12), 16)?;
    let stage2 = tail.continue_forward(stage1)?;
    let distributed = tail.sample_tail(stage2)?;
    let single = run_toy_single_node_reference(prompt, Some(12));

    println!("single-node : {}", single.text);
    println!("two-stage   : {}", distributed.text);
    println!("match       : {}", single.text == distributed.text);

    Ok(())
}
