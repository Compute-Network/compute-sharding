use anyhow::Result;
use stage_forward_lab::quants;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::real_math::{self, GemmaLayerConfig};
use stage_forward_lab::{PackedTensorEntry, StageForwardBackend, StageLayout, StageTensorStore};
use std::path::PathBuf;

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn preview(x: &[f32], n: usize) -> String {
    x[..n.min(x.len())]
        .iter()
        .map(|v| format!("{:.4}", v))
        .collect::<Vec<_>>()
        .join(", ")
}

fn decode_f32(store: &StageTensorStore, entry: &PackedTensorEntry) -> Vec<f32> {
    let bytes = store.read(&entry.name).unwrap();
    quants::dequantize_f32_tensor(&bytes).unwrap()
}

fn decode_matrix(store: &StageTensorStore, entry: &PackedTensorEntry) -> (usize, usize, Vec<f32>) {
    let in_dim = entry.dimensions[0] as usize;
    let out_dim = entry.dimensions[1] as usize;
    let bytes = store.read(&entry.name).unwrap();
    let matrix = quants::dequantize_tensor(entry.ggml_type, &bytes).unwrap();
    (out_dim, in_dim, matrix)
}

fn main() -> Result<()> {
    let base = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage".into()
    }));
    let stage1_idx = base.join("packed-stage-1/stage-1-required.index.json");
    let vocab_path = base.join("vocab.json");
    let scores_path = base.join("vocab_scores.json");

    let prompt = "The capital of France is";

    // Load the head stage
    let mut head = RealGemmaBackend::new(&stage1_idx);
    head.set_debug_layer_cap(Some(0)); // we'll manually do layers
    let sp = if scores_path.exists() {
        Some(scores_path.as_path())
    } else {
        None
    };
    head.load_tokenizer(&vocab_path, sp)?;
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    // Get the embedding (0 layers)
    let tensor = head.begin_prompt("req", prompt, Some(1), 0)?;
    let mut state: Vec<f32> = tensor
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("prompt tokens: {:?}", head.tokenize_text(prompt));
    println!(
        "embedding: rms={:.6}, preview=[{}]",
        rms(&state),
        preview(&state, 8)
    );
    println!();

    // Now manually load the store and step through layer 0
    let store = StageTensorStore::load(&stage1_idx)?;
    let model_view = store.model_view();

    let layer = &model_view.operator_layers[0];
    println!("=== Layer 0 step-by-step ===");

    // Step 1: attn_norm
    let residual = state.clone();
    if let Some(ref entry) = layer.attn_norm {
        let w = decode_f32(&store, entry);
        println!(
            "attn_norm weight: len={}, rms={:.6}, preview=[{}]",
            w.len(),
            rms(&w),
            preview(&w, 4)
        );
        real_math::rms_norm_inplace(&mut state, &w, 1e-6);
        println!(
            "after attn_norm: rms={:.6}, preview=[{}]",
            rms(&state),
            preview(&state, 8)
        );
    }

    // Step 2: Q, K, V projections
    let q_entry = layer.attn_q.as_ref().unwrap();
    let k_entry = layer.attn_k.as_ref().unwrap();
    let v_entry = layer.attn_v.as_ref().unwrap();
    println!(
        "\nQ dims={:?} type={}",
        q_entry.dimensions,
        quants::ggml_type_name(q_entry.ggml_type)
    );
    println!(
        "K dims={:?} type={}",
        k_entry.dimensions,
        quants::ggml_type_name(k_entry.ggml_type)
    );
    println!(
        "V dims={:?} type={}",
        v_entry.dimensions,
        quants::ggml_type_name(v_entry.ggml_type)
    );

    let (q_out, q_in, q_mat) = decode_matrix(&store, q_entry);
    let (k_out, k_in, k_mat) = decode_matrix(&store, k_entry);
    let (v_out, v_in, v_mat) = decode_matrix(&store, v_entry);
    println!("Q: out={}, in={}, mat_len={}", q_out, q_in, q_mat.len());
    println!("K: out={}, in={}, mat_len={}", k_out, k_in, k_mat.len());
    println!("V: out={}, in={}, mat_len={}", v_out, v_in, v_mat.len());

    let mut q = real_math::matmul(&q_mat, &state, q_out, q_in);
    let mut k = real_math::matmul(&k_mat, &state, k_out, k_in);
    let v = real_math::matmul(&v_mat, &state, v_out, v_in);
    println!(
        "\nQ raw: rms={:.6}, len={}, preview=[{}]",
        rms(&q),
        q.len(),
        preview(&q, 4)
    );
    println!(
        "K raw: rms={:.6}, len={}, preview=[{}]",
        rms(&k),
        k.len(),
        preview(&k, 4)
    );
    println!(
        "V raw: rms={:.6}, len={}, preview=[{}]",
        rms(&v),
        v.len(),
        preview(&v, 4)
    );

    // Step 3: per-head Q/K norms
    let config = GemmaLayerConfig::from_dims(
        q_entry.dimensions[0] as usize,
        q_entry.dimensions[1] as usize,
        k_entry.dimensions[1] as usize,
        layer
            .ffn_up
            .as_ref()
            .map(|e| e.dimensions[1] as usize)
            .unwrap_or(10240),
    );
    println!(
        "\nConfig: hidden={}, n_heads={}, n_kv_heads={}, head_dim={}, ffn_dim={}",
        config.hidden_dim, config.n_heads, config.n_kv_heads, config.head_dim, config.ffn_dim
    );

    if let Some(ref entry) = layer.attn_q_norm {
        let w = decode_f32(&store, entry);
        println!("q_norm weight: len={}, rms={:.6}", w.len(), rms(&w));
        real_math::per_head_rms_norm(&mut q, &w, config.n_heads, config.head_dim);
        println!(
            "Q after norm: rms={:.6}, preview=[{}]",
            rms(&q),
            preview(&q, 4)
        );
    }
    if let Some(ref entry) = layer.attn_k_norm {
        let w = decode_f32(&store, entry);
        println!("k_norm weight: len={}, rms={:.6}", w.len(), rms(&w));
        real_math::per_head_rms_norm(&mut k, &w, config.n_kv_heads, config.head_dim);
        println!(
            "K after norm: rms={:.6}, preview=[{}]",
            rms(&k),
            preview(&k, 4)
        );
    }

    // Step 4: RoPE
    let rope_entry = model_view
        .positional
        .iter()
        .find(|e| e.name == "rope_freqs.weight");
    if let Some(entry) = rope_entry {
        let freqs = decode_f32(&store, entry);
        println!(
            "\nRoPE freqs: len={}, preview=[{}]",
            freqs.len(),
            preview(&freqs, 4)
        );
        real_math::rope_apply(
            &mut q,
            &mut k,
            &freqs,
            0,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
        );
        println!("Q after RoPE: rms={:.6}", rms(&q));
        println!("K after RoPE: rms={:.6}", rms(&k));
    }

    // Step 5: Attention (single-token so it's just softmax(QK^T/sqrt(d)) * V)
    let k_cache = vec![k.clone()];
    let v_cache = vec![v.clone()];
    let attn_out = real_math::gqa_attention_seq(
        &q,
        &k_cache,
        &v_cache,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
    );
    println!(
        "\nattn output: rms={:.6}, len={}, preview=[{}]",
        rms(&attn_out),
        attn_out.len(),
        preview(&attn_out, 8)
    );

    // Step 6: Output projection
    if let Some(ref entry) = layer.attn_output {
        let (o_out, o_in, o_mat) = decode_matrix(&store, entry);
        println!(
            "O dims: out={}, in={}, entry_dims={:?}",
            o_out, o_in, entry.dimensions
        );
        let attn_proj = real_math::matmul(&o_mat, &attn_out, o_out, o_in);
        println!(
            "attn projected: rms={:.6}, preview=[{}]",
            rms(&attn_proj),
            preview(&attn_proj, 8)
        );

        // Step 7: post-attention norm + residual
        if let Some(ref norm_entry) = layer.post_attention_norm {
            let w = decode_f32(&store, norm_entry);
            let normed = real_math::rms_norm(&attn_proj, &w, 1e-6);
            state = real_math::vec_add(&residual, &normed);
            println!(
                "after post_attn_norm + residual: rms={:.6}, preview=[{}]",
                rms(&state),
                preview(&state, 8)
            );
        } else {
            state = real_math::vec_add(&residual, &attn_proj);
            println!(
                "after attn + residual (no post_norm): rms={:.6}",
                rms(&state)
            );
        }
    }

    // Step 8: FFN
    let residual_ffn = state.clone();
    if let Some(ref entry) = layer.ffn_norm {
        let w = decode_f32(&store, entry);
        real_math::rms_norm_inplace(&mut state, &w, 1e-6);
        println!("\nafter ffn_norm: rms={:.6}", rms(&state));
    }

    if let (Some(gate_e), Some(up_e), Some(down_e)) =
        (&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down)
    {
        let (g_out, g_in, g_mat) = decode_matrix(&store, gate_e);
        let (u_out, u_in, u_mat) = decode_matrix(&store, up_e);
        let (d_out, d_in, d_mat) = decode_matrix(&store, down_e);
        println!(
            "gate: out={}, in={}, dims={:?}",
            g_out, g_in, gate_e.dimensions
        );
        println!(
            "up:   out={}, in={}, dims={:?}",
            u_out, u_in, up_e.dimensions
        );
        println!(
            "down: out={}, in={}, dims={:?}",
            d_out, d_in, down_e.dimensions
        );

        let gate = real_math::matmul(&g_mat, &state, g_out, g_in);
        let up = real_math::matmul(&u_mat, &state, u_out, u_in);
        println!("gate: rms={:.6}", rms(&gate));
        println!("up:   rms={:.6}", rms(&up));

        let gate_activated = real_math::silu(&gate);
        let ffn_hidden = real_math::vec_mul(&gate_activated, &up);
        println!("ffn_hidden (gate*up): rms={:.6}", rms(&ffn_hidden));

        let ffn_out = real_math::matmul(&d_mat, &ffn_hidden, d_out, d_in);
        println!("ffn_out (down proj): rms={:.6}", rms(&ffn_out));

        if let Some(ref norm_entry) = layer.post_ffw_norm {
            let w = decode_f32(&store, norm_entry);
            let normed = real_math::rms_norm(&ffn_out, &w, 1e-6);
            state = real_math::vec_add(&residual_ffn, &normed);
            println!("after post_ffn_norm + residual: rms={:.6}", rms(&state));
        } else {
            state = real_math::vec_add(&residual_ffn, &ffn_out);
            println!(
                "after ffn + residual (no post_norm): rms={:.6}",
                rms(&state)
            );
        }
    }

    // Step 9: layer_output_scale
    if let Some(ref entry) = layer.layer_output_scale {
        let bytes = store.read(&entry.name)?;
        let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        println!("\nlayer_output_scale: {:.6}", scale);
        for v in state.iter_mut() {
            *v *= scale;
        }
        println!("after scale: rms={:.6}", rms(&state));
    }

    // Step 10: PLE (if present)
    if layer.inp_gate.is_some() {
        println!("\n*** Layer 0 HAS PLE gate - this may be corrupting state ***");
    } else {
        println!("\n(no PLE on layer 0)");
    }

    // Compare final state after 1 manual layer vs 1 layer from begin_prompt
    println!("\n=== Comparison ===");
    println!(
        "manual layer 0 state: rms={:.6}, preview=[{}]",
        rms(&state),
        preview(&state, 8)
    );

    let mut head2 = RealGemmaBackend::new(&stage1_idx);
    head2.set_debug_layer_cap(Some(1));
    let sp = if scores_path.exists() {
        Some(scores_path.as_path())
    } else {
        None
    };
    head2.load_tokenizer(&vocab_path, sp)?;
    head2.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;
    let auto_out = head2.begin_prompt("req-auto", prompt, Some(1), 0)?;
    let auto_state: Vec<f32> = auto_out
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!(
        "auto 1-layer state:   rms={:.6}, preview=[{}]",
        rms(&auto_state),
        preview(&auto_state, 8)
    );

    let diff: f32 = state
        .iter()
        .zip(auto_state.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / state.len() as f32;
    println!("mean abs diff: {:.8}", diff);
    if diff < 1e-4 {
        println!("MATCH: manual and automatic produce same state");
    } else {
        println!("MISMATCH: manual and automatic differ!");
    }

    Ok(())
}
