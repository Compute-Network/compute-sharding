use anyhow::Result;
use stage_forward_lab::{
    PackedResidencySketchBackend, StageCarryPolicy, StageForwardBackend, StageForwardFrame,
    StageLayout, StageTensorStore,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let index_path = args
        .get(1)
        .map(PathBuf::from)
        .or_else(|| env::args_os().nth(1).map(PathBuf::from))
        .ok_or_else(|| {
            anyhow::anyhow!("usage: transient_probe <stage-index.json> [prompt] [layer-cap]")
        })?;
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "reply exactly: TRANSIENT PROBE".to_string());
    let layer_cap = args.get(3).and_then(|value| value.parse::<usize>().ok());

    let mut backend = PackedResidencySketchBackend::new(index_path.clone());
    backend.set_debug_layer_cap(layer_cap);
    let store = StageTensorStore::load(&index_path)?;
    let model_view = store.model_view();
    let stage_label = index_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("stage")
        .to_string();

    let layout = StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: stage_label.clone(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    };
    backend.load_layout(StageLayout {
        model_id: layout.model_id.clone(),
        stage_id: layout.stage_id.clone(),
        start_layer: layout.start_layer,
        end_layer: layout.end_layer,
        is_head: layout.is_head,
        is_tail: layout.is_tail,
    })?;

    let tensor = backend.begin_prompt("probe-req", &prompt, Some(16), 0)?;
    let frame = StageForwardFrame::from_tensor(
        "gemma-4-e4b-q4",
        &layout,
        tensor.clone(),
        Some("next-stage".into()),
    );
    frame.validate()?;
    let summary = frame.summary();
    let boundary_plan =
        frame.to_boundary_plan_for_execution_boundary(&layout, &model_view.execution_programs);
    let transfer =
        frame.to_transfer_frame_for_execution_boundary(&layout, &model_view.execution_programs);
    let full_transfer = frame.to_transfer_frame_with_policy(&StageCarryPolicy::full());
    let resume_request = boundary_plan.to_resume_request(transfer.clone())?;
    let resume_receipt = resume_request.accept(Some("next-stage".into()));
    transfer.validate()?;
    full_transfer.validate()?;
    boundary_plan.validate_against_transfer(&transfer)?;
    resume_request.validate()?;
    resume_receipt.validate_against_request(&resume_request)?;

    println!("request_id      : {}", tensor.request_id);
    println!("kind            : {:?}", tensor.kind);
    println!("hidden_dim      : {}", tensor.hidden_dim);
    println!("hidden_bytes    : {}", tensor.bytes.len());
    println!("stage_trace     : {:?}", tensor.stage_trace);
    println!("layer_cap       : {:?}", layer_cap);

    if let Some(cont) = &tensor.continuation {
        println!("continuation.v  : {}", cont.version);
        println!("continuation.role: {}", cont.stage_role);
        println!(
            "continuation.layers: {}/{}",
            cont.completed_layers, cont.operator_layers
        );
        println!(
            "continuation.paths: attention={} ffn={} projection={}",
            cont.has_attention_path, cont.has_ffn_path, cont.has_projection_path
        );
    } else {
        println!("continuation    : none");
    }

    if let Some(transient) = &tensor.transient {
        if let Some(attn) = &transient.attention {
            println!("attention.width : {}", attn.width);
            println!("attention.q     : {:?}", attn.q_preview);
            println!("attention.k     : {:?}", attn.k_preview);
            println!("attention.v     : {:?}", attn.v_preview);
            println!("attention.score : {:?}", attn.score_preview);
            println!("attention.value : {:?}", attn.value_preview);
        } else {
            println!("attention       : none");
        }

        if let Some(ffn) = &transient.ffn {
            println!("ffn.width       : {}", ffn.width);
            println!("ffn.gate        : {:?}", ffn.gate_preview);
            println!("ffn.up          : {:?}", ffn.up_preview);
            println!("ffn.activations : {:?}", ffn.activation_preview);
        } else {
            println!("ffn             : none");
        }
    } else {
        println!("transient       : none");
    }

    println!("frame.version   : {}", frame.version);
    println!(
        "frame.route     : {} -> {:?}",
        frame.route.source_stage_id, frame.route.target_stage_id
    );
    println!(
        "boundary.plan   : next_layer={:?} expects(attn={} {}@{:?} proj={} mix={} p_lanes={} m_lanes={} ffn={} {}@{:?}) resumable(attn={} q={}({}) k={}({}) v={}({}) proj={} mix={} ffn={} proj_path={})",
        boundary_plan.next_layer_index,
        boundary_plan.expects_attention_carry,
        boundary_plan.expected_attention_lanes.unwrap_or(0),
        boundary_plan.expected_attention_width,
        boundary_plan.expects_attention_projection_carry,
        boundary_plan.expects_attention_mix_carry,
        boundary_plan
            .expected_attention_projection_lanes
            .unwrap_or(0),
        boundary_plan.expected_attention_mix_lanes.unwrap_or(0),
        boundary_plan.expects_ffn_carry,
        boundary_plan.expected_ffn_lanes.unwrap_or(0),
        boundary_plan.expected_ffn_width,
        boundary_plan.resumable_attention_path,
        boundary_plan.resumable_attention_q,
        boundary_plan.resumable_attention_q_lanes.unwrap_or(0),
        boundary_plan.resumable_attention_k,
        boundary_plan.resumable_attention_k_lanes.unwrap_or(0),
        boundary_plan.resumable_attention_v,
        boundary_plan.resumable_attention_v_lanes.unwrap_or(0),
        boundary_plan.resumable_attention_projection,
        boundary_plan.resumable_attention_mix,
        boundary_plan.resumable_ffn_path,
        boundary_plan.resumable_projection_path
    );
    println!(
        "boundary.fresh  : q={:?} k={:?} v={:?} score={:?} value={:?}",
        boundary_plan.expected_attention_q_distance,
        boundary_plan.expected_attention_k_distance,
        boundary_plan.expected_attention_v_distance,
        boundary_plan.expected_attention_score_distance,
        boundary_plan.expected_attention_value_distance
    );
    println!(
        "boundary.stale  : q<={:?} k<={:?} v<={:?} score<={:?} value<={:?}",
        boundary_plan.resumable_attention_q_max_distance,
        boundary_plan.resumable_attention_k_max_distance,
        boundary_plan.resumable_attention_v_max_distance,
        boundary_plan.resumable_attention_score_max_distance,
        boundary_plan.resumable_attention_value_max_distance
    );
    println!(
        "boundary.contract: q={:?} k={:?} v={:?} score={:?} value={:?}",
        boundary_plan.resumable_attention_contract.attention_q,
        boundary_plan.resumable_attention_contract.attention_k,
        boundary_plan.resumable_attention_contract.attention_v,
        boundary_plan.resumable_attention_contract.attention_score,
        boundary_plan.resumable_attention_contract.attention_value
    );
    println!(
        "resume.request  : target={:?} accepted(attn={},ffn={})",
        resume_request.target_stage_id,
        resume_request.boundary.expects_attention_carry,
        resume_request.boundary.expects_ffn_carry
    );
    println!(
        "resume.receipt  : accepted={} stage={:?} next_layer={:?} attn={} {}@{:?} q={}({}) k={}({}) v={}({}) proj={} mix={} p_lanes={} m_lanes={} ffn={} {}@{:?}",
        resume_receipt.accepted,
        resume_receipt.accepted_stage_id,
        resume_receipt.accepted_next_layer_index,
        resume_receipt.accepted_attention_carry,
        resume_receipt.accepted_attention_lanes.unwrap_or(0),
        resume_receipt.accepted_attention_width,
        resume_receipt.accepted_attention_q_resume,
        resume_receipt.accepted_attention_q_lanes.unwrap_or(0),
        resume_receipt.accepted_attention_k_resume,
        resume_receipt.accepted_attention_k_lanes.unwrap_or(0),
        resume_receipt.accepted_attention_v_resume,
        resume_receipt.accepted_attention_v_lanes.unwrap_or(0),
        resume_receipt.accepted_attention_projection_carry,
        resume_receipt.accepted_attention_mix_carry,
        resume_receipt
            .accepted_attention_projection_lanes
            .unwrap_or(0),
        resume_receipt.accepted_attention_mix_lanes.unwrap_or(0),
        resume_receipt.accepted_ffn_carry,
        resume_receipt.accepted_ffn_lanes.unwrap_or(0),
        resume_receipt.accepted_ffn_width
    );
    println!(
        "receipt.fresh   : q={:?} k={:?} v={:?} score={:?} value={:?}",
        resume_receipt.accepted_attention_q_distance,
        resume_receipt.accepted_attention_k_distance,
        resume_receipt.accepted_attention_v_distance,
        resume_receipt.accepted_attention_score_distance,
        resume_receipt.accepted_attention_value_distance
    );
    println!(
        "receipt.stale   : q<={:?} k<={:?} v<={:?} score<={:?} value<={:?}",
        resume_receipt.accepted_attention_q_max_distance,
        resume_receipt.accepted_attention_k_max_distance,
        resume_receipt.accepted_attention_v_max_distance,
        resume_receipt.accepted_attention_score_max_distance,
        resume_receipt.accepted_attention_value_max_distance
    );
    println!(
        "receipt.contract: q={:?} k={:?} v={:?} score={:?} value={:?}",
        resume_receipt.accepted_attention_contract.attention_q,
        resume_receipt.accepted_attention_contract.attention_k,
        resume_receipt.accepted_attention_contract.attention_v,
        resume_receipt.accepted_attention_contract.attention_score,
        resume_receipt.accepted_attention_contract.attention_value
    );
    println!(
        "frame.summary   : kind={:?} trace_depth={} hidden_bytes={} completed_layers={:?} has_transient={}",
        summary.payload_kind,
        summary.trace_depth,
        summary.hidden_bytes,
        summary.completed_layers,
        summary.has_transient
    );
    if let Some(transient) = &transfer.state.transient {
        if let Some(attn) = &transient.attention {
            println!(
                "transfer.attn  : width={} preview_len={} rms_milli={} checksum={}",
                attn.width, attn.preview_len, attn.rms_milli, attn.checksum
            );
        } else {
            println!("transfer.attn  : none");
        }
        if let Some(ffn) = &transient.ffn {
            println!(
                "transfer.ffn   : width={} preview_len={} rms_milli={} checksum={}",
                ffn.width, ffn.preview_len, ffn.rms_milli, ffn.checksum
            );
        } else {
            println!("transfer.ffn   : none");
        }
    } else {
        println!("transfer.state : none");
    }
    if let Some(carry) = &transfer.state.carry {
        if let Some(attn) = &carry.attention {
            println!(
                "carry.attn     : width={} proj={} mix={} q={:?} score={:?}",
                attn.width(),
                attn.projection_lane_count(),
                attn.mix_lane_count(),
                attn.projection.as_ref().map(|projection| &projection.q),
                attn.mix.as_ref().map(|mix| &mix.scores)
            );
            println!(
                "carry.attn.src : q={:?} k={:?} v={:?} score={:?} value={:?}",
                attn.projection
                    .as_ref()
                    .and_then(|projection| projection.q_provenance.as_ref()),
                attn.projection
                    .as_ref()
                    .and_then(|projection| projection.k_provenance.as_ref()),
                attn.projection
                    .as_ref()
                    .and_then(|projection| projection.v_provenance.as_ref()),
                attn.mix
                    .as_ref()
                    .and_then(|mix| mix.score_provenance.as_ref()),
                attn.mix
                    .as_ref()
                    .and_then(|mix| mix.value_provenance.as_ref())
            );
            println!(
                "carry.attn.lanes: q={:?} k={:?} v={:?} score={:?} value={:?}",
                attn.projection
                    .as_ref()
                    .map(|projection| &projection.q_lane_indices),
                attn.projection
                    .as_ref()
                    .map(|projection| &projection.k_lane_indices),
                attn.projection
                    .as_ref()
                    .map(|projection| &projection.v_lane_indices),
                attn.mix.as_ref().map(|mix| &mix.score_lane_indices),
                attn.mix.as_ref().map(|mix| &mix.value_lane_indices)
            );
            println!(
                "carry.attn.contract: q={:?} k={:?} v={:?} score={:?} value={:?}",
                attn.contract.attention_q,
                attn.contract.attention_k,
                attn.contract.attention_v,
                attn.contract.attention_score,
                attn.contract.attention_value
            );
        } else {
            println!("carry.attn     : none");
        }
    } else {
        println!("carry.state    : none");
    }
    if let Some(carry) = &full_transfer.state.carry {
        if let Some(ffn) = &carry.ffn {
            println!(
                "carry.ffn      : width={} lanes={} gate={:?} act={:?}",
                ffn.width,
                ffn.lane_count(),
                ffn.gate_head,
                ffn.activation_head
            );
        } else {
            println!("carry.ffn      : none");
        }
    } else {
        println!("carry.full     : none");
    }

    Ok(())
}
