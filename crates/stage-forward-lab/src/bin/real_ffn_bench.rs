#![allow(clippy::manual_is_multiple_of, clippy::too_many_arguments)]

use anyhow::{Context, Result, bail};
use stage_forward_lab::{StageTensorStore, quants, real_math};
use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct FfnRun {
    norm_micros: u128,
    gate_up_micros: u128,
    activation_micros: u128,
    down_micros: u128,
    total_micros: u128,
    checksum: f32,
}

#[derive(Debug, Clone)]
struct ChunkedFfnRun {
    norm_micros: u128,
    gate_up_micros: u128,
    activation_micros: u128,
    down_micros: u128,
    total_micros: u128,
    checksum: f32,
}

#[derive(Debug, Clone)]
struct InputChunkedFfnRun {
    norm_micros: u128,
    gate_up_micros: u128,
    activation_micros: u128,
    down_micros: u128,
    total_micros: u128,
    checksum: f32,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        bail!("missing required arguments");
    }

    let index_path = PathBuf::from(&args[1]);
    let layer_idx = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let input_count = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6);
    let repeats = args
        .get(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let chunk_size = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(quants::QK_K);

    let norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
    let gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
    let up_name = format!("blk.{}.ffn_up.weight", layer_idx);
    let down_name = format!("blk.{}.ffn_down.weight", layer_idx);

    let store = StageTensorStore::load(&index_path)
        .with_context(|| format!("failed to load packed stage index {}", index_path.display()))?;
    let norm_entry = store
        .entry(&norm_name)
        .cloned()
        .with_context(|| format!("tensor {} not found in {}", norm_name, index_path.display()))?;
    let gate_entry = store
        .entry(&gate_name)
        .cloned()
        .with_context(|| format!("tensor {} not found in {}", gate_name, index_path.display()))?;
    let up_entry = store
        .entry(&up_name)
        .cloned()
        .with_context(|| format!("tensor {} not found in {}", up_name, index_path.display()))?;
    let down_entry = store
        .entry(&down_name)
        .cloned()
        .with_context(|| format!("tensor {} not found in {}", down_name, index_path.display()))?;

    if gate_entry.dimensions != up_entry.dimensions {
        bail!(
            "gate/up dims differ: {:?} vs {:?}",
            gate_entry.dimensions,
            up_entry.dimensions
        );
    }

    let hidden_dim = gate_entry.dimensions[0] as usize;
    let ffn_dim = gate_entry.dimensions[1] as usize;
    if down_entry.dimensions[0] as usize != ffn_dim
        || down_entry.dimensions[1] as usize != hidden_dim
    {
        bail!(
            "down dims {:?} do not match gate/up hidden={} ffn={}",
            down_entry.dimensions,
            hidden_dim,
            ffn_dim
        );
    }

    let norm_raw = store.read(&norm_entry.name)?;
    let gate_raw = store.read(&gate_entry.name)?;
    let up_raw = store.read(&up_entry.name)?;
    let down_raw = store.read(&down_entry.name)?;
    let norm_weight = quants::dequantize_tensor(norm_entry.ggml_type, &norm_raw)
        .with_context(|| format!("failed to decode {}", norm_entry.name))?;
    if norm_weight.len() != hidden_dim {
        bail!(
            "norm weight len {} does not match hidden dim {}",
            norm_weight.len(),
            hidden_dim
        );
    }

    let inputs = build_inputs(input_count, hidden_dim);

    println!("=== Real FFN Bench ===");
    println!("index      : {}", index_path.display());
    println!("layer      : {}", layer_idx);
    println!("inputs     : {}", input_count);
    println!("repeats    : {}", repeats);
    println!("chunk size : {}", chunk_size);
    println!("hidden dim : {}", hidden_dim);
    println!("ffn dim    : {}", ffn_dim);
    println!(
        "gate/up    : type={} dims={:?}",
        gate_entry.ggml_type, gate_entry.dimensions
    );
    println!(
        "down       : type={} dims={:?}",
        down_entry.ggml_type, down_entry.dimensions
    );
    println!();

    let warm = run_ffn_bench(
        &inputs,
        &norm_weight,
        gate_entry.ggml_type,
        gate_raw.as_slice(),
        up_raw.as_slice(),
        down_entry.ggml_type,
        down_raw.as_slice(),
        hidden_dim,
        ffn_dim,
    )?;
    println!(
        "warmup     : total={}us checksum={:.6}",
        warm.total_micros, warm.checksum
    );
    println!();

    let mut runs = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        runs.push(run_ffn_bench(
            &inputs,
            &norm_weight,
            gate_entry.ggml_type,
            gate_raw.as_slice(),
            up_raw.as_slice(),
            down_entry.ggml_type,
            down_raw.as_slice(),
            hidden_dim,
            ffn_dim,
        )?);
    }

    print_series("norm", runs.iter().map(|run| run.norm_micros).collect());
    print_series(
        "gate+up",
        runs.iter().map(|run| run.gate_up_micros).collect(),
    );
    print_series(
        "activation",
        runs.iter().map(|run| run.activation_micros).collect(),
    );
    print_series("down", runs.iter().map(|run| run.down_micros).collect());
    print_series("total", runs.iter().map(|run| run.total_micros).collect());
    println!("checksum   : {:.6}", runs[0].checksum);

    if chunk_size > 0 && ffn_dim % chunk_size == 0 && chunk_size % quants::QK_K == 0 {
        println!();
        println!("=== Chunked FFN Prototype ===");
        let warm_chunked = run_chunked_ffn_bench(
            &inputs,
            &norm_weight,
            gate_raw.as_slice(),
            up_raw.as_slice(),
            down_entry.ggml_type,
            down_raw.as_slice(),
            hidden_dim,
            ffn_dim,
            chunk_size,
        )?;
        println!(
            "warmup     : total={}us checksum={:.6}",
            warm_chunked.total_micros, warm_chunked.checksum
        );
        println!();

        let mut chunked_runs = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            chunked_runs.push(run_chunked_ffn_bench(
                &inputs,
                &norm_weight,
                gate_raw.as_slice(),
                up_raw.as_slice(),
                down_entry.ggml_type,
                down_raw.as_slice(),
                hidden_dim,
                ffn_dim,
                chunk_size,
            )?);
        }

        print_series(
            "chunk-norm",
            chunked_runs.iter().map(|run| run.norm_micros).collect(),
        );
        print_series(
            "chunk-g+u",
            chunked_runs.iter().map(|run| run.gate_up_micros).collect(),
        );
        print_series(
            "chunk-act",
            chunked_runs
                .iter()
                .map(|run| run.activation_micros)
                .collect(),
        );
        print_series(
            "chunk-down",
            chunked_runs.iter().map(|run| run.down_micros).collect(),
        );
        print_series(
            "chunk-total",
            chunked_runs.iter().map(|run| run.total_micros).collect(),
        );
        println!("chunk-sum  : {:.6}", chunked_runs[0].checksum);
    }

    if input_count > 6 {
        println!();
        println!("=== Input-Chunked FFN Prototype ===");
        let warm_input_chunked = run_input_chunked_ffn_bench(
            &inputs,
            &norm_weight,
            gate_entry.ggml_type,
            gate_raw.as_slice(),
            up_raw.as_slice(),
            down_entry.ggml_type,
            down_raw.as_slice(),
            hidden_dim,
            ffn_dim,
            6,
        )?;
        println!(
            "warmup     : total={}us checksum={:.6}",
            warm_input_chunked.total_micros, warm_input_chunked.checksum
        );
        println!();

        let mut input_chunked_runs = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            input_chunked_runs.push(run_input_chunked_ffn_bench(
                &inputs,
                &norm_weight,
                gate_entry.ggml_type,
                gate_raw.as_slice(),
                up_raw.as_slice(),
                down_entry.ggml_type,
                down_raw.as_slice(),
                hidden_dim,
                ffn_dim,
                6,
            )?);
        }

        print_series(
            "input-norm",
            input_chunked_runs
                .iter()
                .map(|run| run.norm_micros)
                .collect(),
        );
        print_series(
            "input-g+u",
            input_chunked_runs
                .iter()
                .map(|run| run.gate_up_micros)
                .collect(),
        );
        print_series(
            "input-act",
            input_chunked_runs
                .iter()
                .map(|run| run.activation_micros)
                .collect(),
        );
        print_series(
            "input-down",
            input_chunked_runs
                .iter()
                .map(|run| run.down_micros)
                .collect(),
        );
        print_series(
            "input-total",
            input_chunked_runs
                .iter()
                .map(|run| run.total_micros)
                .collect(),
        );
        println!("input-sum  : {:.6}", input_chunked_runs[0].checksum);
    }

    Ok(())
}

fn run_ffn_bench(
    inputs: &[Vec<f32>],
    norm_weight: &[f32],
    gate_type: u32,
    gate_raw: &[u8],
    up_raw: &[u8],
    down_type: u32,
    down_raw: &[u8],
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<FfnRun> {
    let total_started = Instant::now();

    let norm_started = Instant::now();
    let mut normed = inputs.to_vec();
    for input in normed.iter_mut() {
        real_math::rms_norm_inplace(input, norm_weight, 1e-6);
    }
    let norm_micros = norm_started.elapsed().as_micros();

    let norm_refs: Vec<&[f32]> = normed.iter().map(|input| input.as_slice()).collect();

    let gate_up_started = Instant::now();
    let (mut gate_all, up_all) = if gate_type == quants::GGML_TYPE_Q4_K {
        real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
            gate_raw, up_raw, &norm_refs, 0, ffn_dim, hidden_dim,
        )?
    } else {
        bail!("real_ffn_bench currently expects paired Q4_K gate/up tensors");
    };
    let gate_up_micros = gate_up_started.elapsed().as_micros();

    let activation_started = Instant::now();
    for token_idx in 0..normed.len() {
        let offset = token_idx * ffn_dim;
        let gate = &mut gate_all[offset..offset + ffn_dim];
        let up = &up_all[offset..offset + ffn_dim];
        real_math::gelu_pytorch_tanh_mul_inplace(gate, up);
    }
    let activation_micros = activation_started.elapsed().as_micros();

    let gated_refs: Vec<&[f32]> = gate_all.chunks_exact(ffn_dim).collect();

    let down_started = Instant::now();
    let down_all = real_math::matmul_quantized_many_refs_range_token_major(
        down_type,
        down_raw,
        &gated_refs,
        0,
        hidden_dim,
        ffn_dim,
    )?;
    let down_micros = down_started.elapsed().as_micros();

    let total_micros = total_started.elapsed().as_micros();
    let checksum = checksum(&down_all);
    black_box(&down_all);

    Ok(FfnRun {
        norm_micros,
        gate_up_micros,
        activation_micros,
        down_micros,
        total_micros,
        checksum,
    })
}

fn run_chunked_ffn_bench(
    inputs: &[Vec<f32>],
    norm_weight: &[f32],
    gate_raw: &[u8],
    up_raw: &[u8],
    down_type: u32,
    down_raw: &[u8],
    hidden_dim: usize,
    ffn_dim: usize,
    chunk_size: usize,
) -> Result<ChunkedFfnRun> {
    let total_started = Instant::now();

    let norm_started = Instant::now();
    let mut normed = inputs.to_vec();
    for input in normed.iter_mut() {
        real_math::rms_norm_inplace(input, norm_weight, 1e-6);
    }
    let norm_micros = norm_started.elapsed().as_micros();

    let norm_refs: Vec<&[f32]> = normed.iter().map(|input| input.as_slice()).collect();
    let mut down_all = vec![0.0f32; inputs.len() * hidden_dim];
    let mut gate_up_micros = 0u128;
    let mut activation_micros = 0u128;
    let mut down_micros = 0u128;

    for chunk_start in (0..ffn_dim).step_by(chunk_size) {
        let gate_up_started = Instant::now();
        let (mut gate_chunk, up_chunk) =
            real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
                gate_raw,
                up_raw,
                &norm_refs,
                chunk_start,
                chunk_size,
                hidden_dim,
            )?;
        gate_up_micros += gate_up_started.elapsed().as_micros();

        let activation_started = Instant::now();
        for token_idx in 0..normed.len() {
            let offset = token_idx * chunk_size;
            let gate = &mut gate_chunk[offset..offset + chunk_size];
            let up = &up_chunk[offset..offset + chunk_size];
            real_math::gelu_pytorch_tanh_mul_inplace(gate, up);
        }
        activation_micros += activation_started.elapsed().as_micros();

        let chunk_refs: Vec<&[f32]> = gate_chunk.chunks_exact(chunk_size).collect();
        let down_started = Instant::now();
        real_math::matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
            down_type,
            down_raw,
            &chunk_refs,
            0,
            hidden_dim,
            ffn_dim,
            chunk_start,
            chunk_size,
            down_all.as_mut_slice(),
        )?;
        down_micros += down_started.elapsed().as_micros();
    }

    let total_micros = total_started.elapsed().as_micros();
    let checksum = checksum(&down_all);
    black_box(&down_all);

    Ok(ChunkedFfnRun {
        norm_micros,
        gate_up_micros,
        activation_micros,
        down_micros,
        total_micros,
        checksum,
    })
}

fn run_input_chunked_ffn_bench(
    inputs: &[Vec<f32>],
    norm_weight: &[f32],
    gate_type: u32,
    gate_raw: &[u8],
    up_raw: &[u8],
    down_type: u32,
    down_raw: &[u8],
    hidden_dim: usize,
    ffn_dim: usize,
    input_chunk_size: usize,
) -> Result<InputChunkedFfnRun> {
    let total_started = Instant::now();

    let norm_started = Instant::now();
    let mut normed = inputs.to_vec();
    for input in normed.iter_mut() {
        real_math::rms_norm_inplace(input, norm_weight, 1e-6);
    }
    let norm_micros = norm_started.elapsed().as_micros();

    let mut gate_up_micros = 0u128;
    let mut activation_micros = 0u128;
    let mut down_micros = 0u128;
    let mut down_all = vec![0.0f32; inputs.len() * hidden_dim];

    for input_start in (0..normed.len()).step_by(input_chunk_size) {
        let input_end = (input_start + input_chunk_size).min(normed.len());
        let norm_refs: Vec<&[f32]> = normed[input_start..input_end]
            .iter()
            .map(|input| input.as_slice())
            .collect();

        let gate_up_started = Instant::now();
        let (mut gate_chunk, up_chunk) = if gate_type == quants::GGML_TYPE_Q4_K {
            real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
                gate_raw, up_raw, &norm_refs, 0, ffn_dim, hidden_dim,
            )?
        } else {
            bail!("input-chunked FFN bench currently expects paired Q4_K gate/up tensors");
        };
        gate_up_micros += gate_up_started.elapsed().as_micros();

        let activation_started = Instant::now();
        for local_token_idx in 0..norm_refs.len() {
            let offset = local_token_idx * ffn_dim;
            let gate = &mut gate_chunk[offset..offset + ffn_dim];
            let up = &up_chunk[offset..offset + ffn_dim];
            real_math::gelu_pytorch_tanh_mul_inplace(gate, up);
        }
        activation_micros += activation_started.elapsed().as_micros();

        let gated_refs: Vec<&[f32]> = gate_chunk.chunks_exact(ffn_dim).collect();
        let down_started = Instant::now();
        let down_chunk = real_math::matmul_quantized_many_refs_range_token_major(
            down_type,
            down_raw,
            &gated_refs,
            0,
            hidden_dim,
            ffn_dim,
        )?;
        down_micros += down_started.elapsed().as_micros();

        for (local_token_idx, chunk_values) in down_chunk.chunks_exact(hidden_dim).enumerate() {
            let global_token_idx = input_start + local_token_idx;
            let global_offset = global_token_idx * hidden_dim;
            down_all[global_offset..global_offset + hidden_dim].copy_from_slice(chunk_values);
        }
    }

    let total_micros = total_started.elapsed().as_micros();
    let checksum = checksum(&down_all);
    black_box(&down_all);

    Ok(InputChunkedFfnRun {
        norm_micros,
        gate_up_micros,
        activation_micros,
        down_micros,
        total_micros,
        checksum,
    })
}

fn build_inputs(input_count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut inputs = Vec::with_capacity(input_count);
    for input_idx in 0..input_count {
        let mut input = Vec::with_capacity(dim);
        for col in 0..dim {
            let angle = (input_idx as f32 + 1.0) * (col as f32 + 0.5) * 0.013;
            input.push(angle.sin() * 0.5 + angle.cos() * 0.25);
        }
        inputs.push(input);
    }
    inputs
}

fn print_series(label: &str, mut values: Vec<u128>) {
    values.sort_unstable();
    let min = values.first().copied().unwrap_or_default();
    let max = values.last().copied().unwrap_or_default();
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<u128>() / values.len() as u128;
    println!("{label:<11}: min={min}us median={median}us avg={avg}us max={max}us");
}

fn checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum::<f32>()
}

fn print_usage(bin: &str) {
    eprintln!("usage: {bin} <stage-index.json> [layer] [inputs] [repeats] [chunk_size]");
}
