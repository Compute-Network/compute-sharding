#![allow(clippy::manual_is_multiple_of)]

use anyhow::{Context, Result, bail};
use stage_forward_lab::{PackedTensorEntry, StageTensorStore, quants, real_math};
use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct BenchRun {
    micros: u128,
    output_len_a: usize,
    checksum_a: f32,
    output_len_b: Option<usize>,
    checksum_b: Option<f32>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        print_usage(&args[0]);
        bail!("missing required arguments");
    }

    let index_path = PathBuf::from(&args[1]);
    let tensor_a_name = args[2].clone();

    let mut cursor = 3usize;
    let tensor_b_name = if args.get(cursor).is_some() && args[cursor].parse::<usize>().is_err() {
        let name = args[cursor].clone();
        cursor += 1;
        Some(name)
    } else {
        None
    };

    let input_count = args
        .get(cursor)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6);
    let repeats = args
        .get(cursor + 1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let row_count_override = args
        .get(cursor + 2)
        .and_then(|value| value.parse::<usize>().ok());

    let store = StageTensorStore::load(&index_path)
        .with_context(|| format!("failed to load packed stage index {}", index_path.display()))?;
    let entry_a = store.entry(&tensor_a_name).cloned().with_context(|| {
        format!(
            "tensor {} not found in {}",
            tensor_a_name,
            index_path.display()
        )
    })?;
    let entry_b = tensor_b_name
        .as_ref()
        .map(|name| {
            store
                .entry(name)
                .cloned()
                .with_context(|| format!("tensor {} not found in {}", name, index_path.display()))
        })
        .transpose()?;

    if entry_a.dimensions.len() != 2 {
        bail!(
            "tensor {} is not 2D: {:?}",
            entry_a.name,
            entry_a.dimensions
        );
    }
    if let Some(ref b) = entry_b {
        if b.dimensions.len() != 2 {
            bail!("tensor {} is not 2D: {:?}", b.name, b.dimensions);
        }
        if entry_a.dimensions != b.dimensions {
            bail!(
                "paired tensors must have identical dimensions, got {:?} and {:?}",
                entry_a.dimensions,
                b.dimensions
            );
        }
    }

    let in_dim = entry_a.dimensions[0] as usize;
    let out_dim = entry_a.dimensions[1] as usize;
    let row_count = row_count_override.unwrap_or(out_dim);
    if row_count > out_dim {
        bail!("row_count {} exceeds tensor out_dim {}", row_count, out_dim);
    }

    let raw_a = store.read(&entry_a.name)?;
    let raw_b = entry_b
        .as_ref()
        .map(|entry| store.read(&entry.name))
        .transpose()?;
    let dense_a = dense_matrix_if_needed(&entry_a, &raw_a)?;
    let dense_b = match (&entry_b, &raw_b) {
        (Some(entry), Some(raw)) => dense_matrix_if_needed(entry, raw)?,
        _ => None,
    };

    let inputs = build_inputs(input_count, in_dim);
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();

    println!("=== Real Projection Bench ===");
    println!("index      : {}", index_path.display());
    println!(
        "tensor A   : {} (type={} dims={:?})",
        entry_a.name, entry_a.ggml_type, entry_a.dimensions
    );
    if let Some(ref entry) = entry_b {
        println!(
            "tensor B   : {} (type={} dims={:?})",
            entry.name, entry.ggml_type, entry.dimensions
        );
    }
    println!("inputs     : {}", input_count);
    println!("repeats    : {}", repeats);
    println!("row_count  : {}", row_count);
    println!();

    let warm = run_projection_bench(
        &entry_a,
        raw_a.as_slice(),
        dense_a.as_deref(),
        entry_b.as_ref(),
        raw_b.as_deref(),
        dense_b.as_deref(),
        &input_refs,
        row_count,
    )?;
    println!("warmup     : {}us", warm.micros);
    println!();

    let mut runs = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        runs.push(run_projection_bench(
            &entry_a,
            raw_a.as_slice(),
            dense_a.as_deref(),
            entry_b.as_ref(),
            raw_b.as_deref(),
            dense_b.as_deref(),
            &input_refs,
            row_count,
        )?);
    }

    print_series("matmul", runs.iter().map(|run| run.micros).collect());
    let first = runs.first().expect("at least one run");
    println!(
        "output A   : len={} checksum={:.6}",
        first.output_len_a, first.checksum_a
    );
    if let (Some(len), Some(checksum)) = (first.output_len_b, first.checksum_b) {
        println!("output B   : len={} checksum={:.6}", len, checksum);
    }
    if entry_a.ggml_type == quants::GGML_TYPE_Q4_K
        && entry_b
            .as_ref()
            .is_some_and(|entry| entry.ggml_type == quants::GGML_TYPE_Q4_K)
    {
        println!();
        println!("=== Breakdown (paired Q4_K) ===");
        let raw_b = raw_b.as_deref().expect("paired raw tensor present");
        let (row_outputs_a, row_outputs_b) =
            compute_q4_pair_row_outputs(raw_a.as_slice(), raw_b, &input_refs, row_count, in_dim)?;
        print_series(
            "transpose-a",
            bench_transpose(row_outputs_a.as_slice(), input_count, row_count, repeats),
        );
        print_series(
            "transpose-b",
            bench_transpose(row_outputs_b.as_slice(), input_count, row_count, repeats),
        );
    } else if matches!(
        entry_a.ggml_type,
        quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K
    ) {
        println!();
        println!("=== Breakdown (quantized single) ===");
        let row_outputs = compute_quantized_row_outputs(
            entry_a.ggml_type,
            raw_a.as_slice(),
            &input_refs,
            row_count,
            in_dim,
        )?;
        print_series(
            "transpose",
            bench_transpose(row_outputs.as_slice(), input_count, row_count, repeats),
        );
    }

    Ok(())
}

fn run_projection_bench(
    entry_a: &PackedTensorEntry,
    raw_a: &[u8],
    dense_a: Option<&[f32]>,
    entry_b: Option<&PackedTensorEntry>,
    raw_b: Option<&[u8]>,
    dense_b: Option<&[f32]>,
    inputs: &[&[f32]],
    row_count: usize,
) -> Result<BenchRun> {
    let in_dim = entry_a.dimensions[0] as usize;
    let started = Instant::now();

    let (output_a, output_b) = if let (Some(entry_b), Some(raw_b)) = (entry_b, raw_b) {
        if entry_a.ggml_type == quants::GGML_TYPE_Q4_K
            && entry_b.ggml_type == quants::GGML_TYPE_Q4_K
        {
            let (out_a, out_b) = real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
                raw_a, raw_b, inputs, 0, row_count, in_dim,
            )?;
            (out_a, Some(out_b))
        } else {
            let out_a = project_single(entry_a, raw_a, dense_a, inputs, row_count)?;
            let out_b = project_single(entry_b, raw_b, dense_b, inputs, row_count)?;
            (out_a, Some(out_b))
        }
    } else {
        (
            project_single(entry_a, raw_a, dense_a, inputs, row_count)?,
            None,
        )
    };

    let micros = started.elapsed().as_micros();
    let checksum_a = checksum(&output_a);
    let output_len_a = output_a.len();
    black_box(&output_a);
    let (output_len_b, checksum_b) = if let Some(ref output_b) = output_b {
        let len = output_b.len();
        let checksum = checksum(output_b);
        black_box(output_b);
        (Some(len), Some(checksum))
    } else {
        (None, None)
    };

    Ok(BenchRun {
        micros,
        output_len_a,
        checksum_a,
        output_len_b,
        checksum_b,
    })
}

fn project_single(
    entry: &PackedTensorEntry,
    raw: &[u8],
    dense: Option<&[f32]>,
    inputs: &[&[f32]],
    row_count: usize,
) -> Result<Vec<f32>> {
    let in_dim = entry.dimensions[0] as usize;
    match entry.ggml_type {
        quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => {
            real_math::matmul_quantized_many_refs_range_token_major(
                entry.ggml_type,
                raw,
                inputs,
                0,
                row_count,
                in_dim,
            )
        }
        _ => {
            let dense =
                dense.with_context(|| format!("missing dense decode for {}", entry.name))?;
            Ok(real_math::matmul_many_refs_range_token_major(
                dense, inputs, 0, row_count, in_dim,
            ))
        }
    }
}

fn compute_quantized_row_outputs(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_count: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let bytes_per_row = quants::bytes_per_row(ggml_type, in_dim)?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .context("quantized row output byte length overflow")?;
    if byte_len > raw.len() {
        bail!(
            "quantized row output benchmark needs {} bytes but tensor has {}",
            byte_len,
            raw.len()
        );
    }
    let rows = &raw[..byte_len];
    let input_count = inputs.len();
    let mut row_outputs = vec![0.0f32; row_count * input_count];
    if ggml_type == quants::GGML_TYPE_Q4_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let quad_values_len = input_count * 4;
        for quad_idx in 0..quad_count {
            let offset = quad_idx * quad_bytes;
            let row_a_bytes = &rows[offset..offset + bytes_per_row];
            let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
            let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
            let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
            let values_offset = quad_idx * quad_values_len;
            let quad_values = &mut row_outputs[values_offset..values_offset + quad_values_len];
            let (row_a_values, rest) = quad_values.split_at_mut(input_count);
            let (row_b_values, rest) = rest.split_at_mut(input_count);
            let (row_c_values, row_d_values) = rest.split_at_mut(input_count);
            quants::dot_many_q4_k_four_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                row_c_bytes,
                row_d_bytes,
                inputs,
                row_a_values,
                row_b_values,
                row_c_values,
                row_d_values,
            )?;
        }
        let mut next_row = quad_count * 4;
        if row_count.saturating_sub(next_row) >= 2 {
            let offset = next_row * bytes_per_row;
            let row_a_bytes = &rows[offset..offset + bytes_per_row];
            let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
            let values_offset = next_row * input_count;
            let (row_a_values, row_b_values) = row_outputs
                [values_offset..values_offset + input_count * 2]
                .split_at_mut(input_count);
            quants::dot_many_q4_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                inputs,
                row_a_values,
                row_b_values,
            )?;
            next_row += 2;
        }
        if next_row < row_count {
            let offset = next_row * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            quants::dot_many_row_refs_into(
                ggml_type,
                row_bytes,
                inputs,
                &mut row_outputs[next_row * input_count..(next_row + 1) * input_count],
            )?;
        }
    } else if (ggml_type == quants::GGML_TYPE_Q4_K || ggml_type == quants::GGML_TYPE_Q6_K)
        && row_count >= 2
    {
        let pair_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        for pair_idx in 0..pair_count {
            let offset = pair_idx * pair_bytes;
            let row_a_bytes = &rows[offset..offset + bytes_per_row];
            let row_b_bytes = &rows[offset + bytes_per_row..offset + pair_bytes];
            let values_offset = pair_idx * pair_values_len;
            let (row_a_values, row_b_values) = row_outputs
                [values_offset..values_offset + pair_values_len]
                .split_at_mut(input_count);
            match ggml_type {
                quants::GGML_TYPE_Q4_K => quants::dot_many_q4_k_two_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    inputs,
                    row_a_values,
                    row_b_values,
                )?,
                quants::GGML_TYPE_Q6_K => quants::dot_many_q6_k_two_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    inputs,
                    row_a_values,
                    row_b_values,
                )?,
                _ => unreachable!(),
            }
        }
        if row_count % 2 != 0 {
            let row_idx = row_count - 1;
            let offset = row_idx * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            quants::dot_many_row_refs_into(
                ggml_type,
                row_bytes,
                inputs,
                &mut row_outputs[row_idx * input_count..(row_idx + 1) * input_count],
            )?;
        }
    } else {
        for (row_idx, row_values) in row_outputs.chunks_exact_mut(input_count).enumerate() {
            let offset = row_idx * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)?;
        }
    }
    Ok(row_outputs)
}

fn compute_q4_pair_row_outputs(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[&[f32]],
    row_count: usize,
    in_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if inputs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, in_dim)?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .context("paired Q4_K row output byte length overflow")?;
    if byte_len > raw_a.len() || byte_len > raw_b.len() {
        bail!(
            "paired Q4_K row output benchmark needs {} bytes but tensors have {} and {}",
            byte_len,
            raw_a.len(),
            raw_b.len()
        );
    }
    let rows_a = &raw_a[..byte_len];
    let rows_b = &raw_b[..byte_len];
    let input_count = inputs.len();
    let mut row_outputs_a = vec![0.0f32; row_count * input_count];
    let mut row_outputs_b = vec![0.0f32; row_count * input_count];
    if row_count >= 2 {
        let pair_row_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        for pair_idx in 0..pair_row_count {
            let offset = pair_idx * pair_bytes;
            let row_a0_bytes = &rows_a[offset..offset + bytes_per_row];
            let row_a1_bytes = &rows_a[offset + bytes_per_row..offset + pair_bytes];
            let row_b0_bytes = &rows_b[offset..offset + bytes_per_row];
            let row_b1_bytes = &rows_b[offset + bytes_per_row..offset + pair_bytes];
            let values_offset = pair_idx * pair_values_len;
            let row_a_values = &mut row_outputs_a[values_offset..values_offset + pair_values_len];
            let row_b_values = &mut row_outputs_b[values_offset..values_offset + pair_values_len];
            let (row_a0_values, row_a1_values) = row_a_values.split_at_mut(input_count);
            let (row_b0_values, row_b1_values) = row_b_values.split_at_mut(input_count);
            quants::dot_many_q4_k_four_rows_refs_into(
                row_a0_bytes,
                row_a1_bytes,
                row_b0_bytes,
                row_b1_bytes,
                inputs,
                row_a0_values,
                row_a1_values,
                row_b0_values,
                row_b1_values,
            )?;
        }
        if row_count % 2 != 0 {
            let row_idx = row_count - 1;
            let offset = row_idx * bytes_per_row;
            let row_a_bytes = &rows_a[offset..offset + bytes_per_row];
            let row_b_bytes = &rows_b[offset..offset + bytes_per_row];
            quants::dot_many_q4_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                inputs,
                &mut row_outputs_a[row_idx * input_count..(row_idx + 1) * input_count],
                &mut row_outputs_b[row_idx * input_count..(row_idx + 1) * input_count],
            )?;
        }
    } else {
        quants::dot_many_q4_k_two_rows_refs_into(
            &rows_a[..bytes_per_row],
            &rows_b[..bytes_per_row],
            inputs,
            &mut row_outputs_a[..input_count],
            &mut row_outputs_b[..input_count],
        )?;
    }
    Ok((row_outputs_a, row_outputs_b))
}

fn transpose_row_outputs_token_major(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
) -> Vec<f32> {
    let mut outputs = vec![0.0f32; input_count * row_count];
    match input_count {
        0 => {}
        1 => {
            for (row_idx, values) in row_outputs.chunks_exact(1).enumerate() {
                outputs[row_idx] = values[0];
            }
        }
        2 => {
            let (out0, out1) = outputs.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(2).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
            }
        }
        3 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, c) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(3).enumerate() {
                a[row_idx] = values[0];
                b[row_idx] = values[1];
                c[row_idx] = values[2];
            }
        }
        4 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, d) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(4).enumerate() {
                a[row_idx] = values[0];
                b[row_idx] = values[1];
                c[row_idx] = values[2];
                d[row_idx] = values[3];
            }
        }
        5 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, rest) = rest.split_at_mut(row_count);
            let (d, e) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(5).enumerate() {
                a[row_idx] = values[0];
                b[row_idx] = values[1];
                c[row_idx] = values[2];
                d[row_idx] = values[3];
                e[row_idx] = values[4];
            }
        }
        6 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, rest) = rest.split_at_mut(row_count);
            let (d, rest) = rest.split_at_mut(row_count);
            let (e, f) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(6).enumerate() {
                a[row_idx] = values[0];
                b[row_idx] = values[1];
                c[row_idx] = values[2];
                d[row_idx] = values[3];
                e[row_idx] = values[4];
                f[row_idx] = values[5];
            }
        }
        _ => {
            for (row_idx, row_values) in row_outputs.chunks_exact(input_count).enumerate() {
                for (input_idx, value) in row_values.iter().copied().enumerate() {
                    outputs[input_idx * row_count + row_idx] = value;
                }
            }
        }
    }
    outputs
}

fn bench_transpose(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
    repeats: usize,
) -> Vec<u128> {
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        let outputs = transpose_row_outputs_token_major(row_outputs, input_count, row_count);
        let micros = started.elapsed().as_micros();
        black_box(&outputs);
        black_box(checksum(&outputs));
        times.push(micros);
    }
    times
}

fn dense_matrix_if_needed(entry: &PackedTensorEntry, raw: &[u8]) -> Result<Option<Vec<f32>>> {
    match entry.ggml_type {
        quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => Ok(None),
        _ => Ok(Some(quants::dequantize_tensor(entry.ggml_type, raw)?)),
    }
}

fn build_inputs(input_count: usize, in_dim: usize) -> Vec<Vec<f32>> {
    (0..input_count)
        .map(|input_idx| {
            (0..in_dim)
                .map(|dim_idx| {
                    let base = ((input_idx + 1) * 97 + dim_idx * 17) as f32;
                    ((base % 1024.0) / 512.0) - 1.0
                })
                .collect()
        })
        .collect()
}

fn checksum(values: &[f32]) -> f32 {
    values.iter().take(32).copied().sum()
}

fn print_series(label: &str, mut values: Vec<u128>) {
    values.sort_unstable();
    let min = values.first().copied().unwrap_or(0);
    let max = values.last().copied().unwrap_or(0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<u128>() / values.len() as u128;
    println!(
        "{}     : min={}us median={}us avg={}us max={}us",
        label, min, median, avg, max
    );
}

fn print_usage(bin: &str) {
    eprintln!(
        "usage:\n  {bin} <index.json> <tensor_a> [tensor_b] [input_count=6] [repeats=10] [row_count=full]"
    );
}
