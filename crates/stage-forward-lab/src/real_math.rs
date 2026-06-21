use crate::quants;
use anyhow::{Result, bail};
use rayon::prelude::*;

const GEMMA_RMS_NORM_EPS: f32 = 1e-6;
const PARALLEL_MATVEC_MIN_OPS: usize = 1_000_000;
const INPUT_CHUNK_FAST_PATH: usize = 6;

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    if weight.len() == n {
        x.iter()
            .zip(weight.iter())
            .map(|(v, w)| v * scale * w)
            .collect()
    } else {
        let wlen = weight.len();
        x.iter()
            .enumerate()
            .map(|(i, v)| v * scale * weight[i % wlen])
            .collect()
    }
}

pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    if weight.len() == n {
        for (v, w) in x.iter_mut().zip(weight.iter()) {
            *v = *v * scale * *w;
        }
    } else {
        let wlen = weight.len();
        for (i, v) in x.iter_mut().enumerate() {
            *v = *v * scale * weight[i % wlen];
        }
    }
}

pub fn rms_norm_chunked_inplace(x: &mut [f32], chunk_size: usize, weight: &[f32], eps: f32) {
    if x.is_empty() || chunk_size == 0 {
        return;
    }
    assert_eq!(
        x.len() % chunk_size,
        0,
        "rms_norm_chunked_inplace: len {} must be divisible by chunk_size {}",
        x.len(),
        chunk_size
    );
    for chunk in x.chunks_exact_mut(chunk_size) {
        rms_norm_inplace(chunk, weight, eps);
    }
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

pub fn gelu_pytorch_tanh(x: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    x.iter()
        .map(|&v| {
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect()
}

pub fn gelu_pytorch_tanh_mul_inplace(x: &mut [f32], y: &[f32]) {
    for (xv, yv) in x.iter_mut().zip(y.iter()) {
        let v = *xv;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v);
        *xv = 0.5 * v * (1.0 + inner.tanh()) * *yv;
    }
}

pub fn softcap(x: &mut [f32], cap: f32) {
    for v in x.iter_mut() {
        *v = cap * (*v / cap).tanh();
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

pub fn matmul(matrix: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    matmul_range(matrix, input, 0, out_dim, in_dim)
}

pub fn matmul_range(
    matrix: &[f32],
    input: &[f32],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert!(
        matrix.len() >= (row_start + row_count) * in_dim,
        "matmul: matrix.len()={} but needed rows {}..{} with in_dim={} (need={})",
        matrix.len(),
        row_start,
        row_start + row_count,
        in_dim,
        (row_start + row_count) * in_dim
    );
    assert!(
        input.len() >= in_dim,
        "matmul: input.len()={} but in_dim={} (row_count={})",
        input.len(),
        in_dim,
        row_count
    );
    let start = row_start * in_dim;
    let end = start + row_count * in_dim;
    let rows = &matrix[start..end];
    let use_parallel = row_count > 1 && row_count.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_OPS;

    if use_parallel {
        rows.par_chunks_exact(in_dim)
            .map(|row| row.iter().zip(input.iter()).map(|(m, x)| m * x).sum())
            .collect()
    } else {
        let mut output = vec![0.0f32; row_count];
        for (row_idx, row) in rows.chunks_exact(in_dim).enumerate() {
            output[row_idx] = row.iter().zip(input.iter()).map(|(m, x)| m * x).sum();
        }
        output
    }
}

pub fn matmul_many_range(
    matrix: &[f32],
    inputs: &[Vec<f32>],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Vec<Vec<f32>> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    matmul_many_refs_range(matrix, &input_refs, row_start, row_count, in_dim)
}

pub fn matmul_many_refs_range(
    matrix: &[f32],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Vec<Vec<f32>> {
    if inputs.is_empty() {
        return Vec::new();
    }
    assert!(
        inputs.iter().all(|input| input.len() >= in_dim),
        "matmul_many_range: at least one input shorter than in_dim={}",
        in_dim
    );
    let start = row_start * in_dim;
    let end = start + row_count * in_dim;
    let rows = &matrix[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;

    let mut row_outputs = vec![0.0f32; row_count * input_count];
    if row_count >= 2 {
        let pair_count = row_count / 2;
        let pair_values_len = input_count * 2;
        let pair_rows_len = in_dim * 2;
        let pair_output_len = pair_count * pair_values_len;
        if use_parallel {
            row_outputs[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .enumerate()
                .for_each(|(pair_idx, pair_values)| {
                    let row_offset = pair_idx * pair_rows_len;
                    let row_a = &rows[row_offset..row_offset + in_dim];
                    let row_b = &rows[row_offset + in_dim..row_offset + pair_rows_len];
                    let (row_a_values, row_b_values) = pair_values.split_at_mut(input_count);
                    dense_two_rows_many_into(row_a, row_b, inputs, row_a_values, row_b_values);
                });
        } else {
            for pair_idx in 0..pair_count {
                let row_offset = pair_idx * pair_rows_len;
                let row_a = &rows[row_offset..row_offset + in_dim];
                let row_b = &rows[row_offset + in_dim..row_offset + pair_rows_len];
                let values_offset = pair_idx * pair_values_len;
                let (row_a_values, row_b_values) = row_outputs
                    [values_offset..values_offset + pair_values_len]
                    .split_at_mut(input_count);
                dense_two_rows_many_into(row_a, row_b, inputs, row_a_values, row_b_values);
            }
        }

        if row_count % 2 != 0 {
            let row_idx = row_count - 1;
            let row = &rows[row_idx * in_dim..(row_idx + 1) * in_dim];
            dense_one_row_many_into(
                row,
                inputs,
                &mut row_outputs[row_idx * input_count..(row_idx + 1) * input_count],
            );
        }
    } else {
        let row = &rows[..in_dim];
        dense_one_row_many_into(row, inputs, &mut row_outputs[..input_count]);
    }

    transpose_row_outputs(&row_outputs, input_count, row_count)
}

pub fn matmul_many_refs_range_token_major(
    matrix: &[f32],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Vec<f32> {
    if inputs.is_empty() {
        return Vec::new();
    }
    assert!(
        inputs.iter().all(|input| input.len() >= in_dim),
        "matmul_many_range_token_major: at least one input shorter than in_dim={}",
        in_dim
    );
    let start = row_start * in_dim;
    let end = start + row_count * in_dim;
    let rows = &matrix[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;
    let mut row_outputs = vec![0.0f32; row_count * input_count];
    if row_count >= 2 {
        let pair_count = row_count / 2;
        let pair_values_len = input_count * 2;
        let pair_rows_len = in_dim * 2;
        let pair_output_len = pair_count * pair_values_len;
        if use_parallel {
            row_outputs[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .enumerate()
                .for_each(|(pair_idx, pair_values)| {
                    let row_offset = pair_idx * pair_rows_len;
                    let row_a = &rows[row_offset..row_offset + in_dim];
                    let row_b = &rows[row_offset + in_dim..row_offset + pair_rows_len];
                    let (row_a_values, row_b_values) = pair_values.split_at_mut(input_count);
                    dense_two_rows_many_into(row_a, row_b, inputs, row_a_values, row_b_values);
                });
        } else {
            for pair_idx in 0..pair_count {
                let row_offset = pair_idx * pair_rows_len;
                let row_a = &rows[row_offset..row_offset + in_dim];
                let row_b = &rows[row_offset + in_dim..row_offset + pair_rows_len];
                let values_offset = pair_idx * pair_values_len;
                let (row_a_values, row_b_values) = row_outputs
                    [values_offset..values_offset + pair_values_len]
                    .split_at_mut(input_count);
                dense_two_rows_many_into(row_a, row_b, inputs, row_a_values, row_b_values);
            }
        }

        if row_count % 2 != 0 {
            let row_idx = row_count - 1;
            let row = &rows[row_idx * in_dim..(row_idx + 1) * in_dim];
            dense_one_row_many_into(
                row,
                inputs,
                &mut row_outputs[row_idx * input_count..(row_idx + 1) * input_count],
            );
        }
    } else {
        let row = &rows[..in_dim];
        dense_one_row_many_into(row, inputs, &mut row_outputs[..input_count]);
    }

    transpose_row_outputs_token_major(&row_outputs, input_count, row_count)
}

fn dense_two_rows_many_into(
    row_a: &[f32],
    row_b: &[f32],
    inputs: &[&[f32]],
    sums_a: &mut [f32],
    sums_b: &mut [f32],
) {
    for input_idx in 0..inputs.len() {
        let input = inputs[input_idx];
        let mut sum_a = 0.0f32;
        let mut sum_b = 0.0f32;
        for elem_idx in 0..row_a.len() {
            let x = input[elem_idx];
            sum_a += row_a[elem_idx] * x;
            sum_b += row_b[elem_idx] * x;
        }
        sums_a[input_idx] = sum_a;
        sums_b[input_idx] = sum_b;
    }
}

fn dense_one_row_many_into(row: &[f32], inputs: &[&[f32]], sums: &mut [f32]) {
    for input_idx in 0..inputs.len() {
        let input = inputs[input_idx];
        let mut sum = 0.0f32;
        for elem_idx in 0..row.len() {
            sum += row[elem_idx] * input[elem_idx];
        }
        sums[input_idx] = sum;
    }
}

pub fn matmul_quantized_range(
    ggml_type: u32,
    raw: &[u8],
    input: &[f32],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let bytes_per_row = quants::bytes_per_row(ggml_type, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized matmul end overflow"))?;
    if end > raw.len() {
        bail!(
            "quantized matmul needs bytes {}..{} but tensor has {} bytes",
            start,
            end,
            raw.len()
        );
    }
    if input.len() < in_dim {
        bail!(
            "quantized matmul input len {} is smaller than in_dim {}",
            input.len(),
            in_dim
        );
    }

    let rows = &raw[start..end];
    let use_parallel = row_count > 1 && row_count.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_OPS;

    if use_parallel {
        Ok(rows
            .par_chunks_exact(bytes_per_row)
            .map(|row_bytes| {
                quants::dot_row(ggml_type, row_bytes, input)
                    .expect("validated quantized row dot should succeed")
            })
            .collect())
    } else {
        let mut output = vec![0.0f32; row_count];
        for (row_idx, row_bytes) in rows.chunks_exact(bytes_per_row).enumerate() {
            output[row_idx] = quants::dot_row(ggml_type, row_bytes, input)
                .expect("validated quantized row dot should succeed");
        }
        Ok(output)
    }
}

pub fn matmul_quantized_many_range(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[Vec<f32>],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    matmul_quantized_many_refs_range(ggml_type, raw, &input_refs, row_start, row_count, in_dim)
}

pub fn matmul_quantized_many_refs_range(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if inputs.iter().any(|input| input.len() < in_dim) {
        bail!(
            "quantized batched matmul has an input shorter than in_dim {}",
            in_dim
        );
    }
    if should_use_input_chunk_fast_path(ggml_type, inputs.len(), row_count, in_dim) {
        let mut outputs = vec![vec![0.0f32; row_count]; inputs.len()];
        for input_start in (0..inputs.len()).step_by(INPUT_CHUNK_FAST_PATH) {
            let input_end = (input_start + INPUT_CHUNK_FAST_PATH).min(inputs.len());
            let chunk_outputs = matmul_quantized_many_refs_range(
                ggml_type,
                raw,
                &inputs[input_start..input_end],
                row_start,
                row_count,
                in_dim,
            )?;
            copy_nested_output_chunk(&chunk_outputs, input_start, &mut outputs);
        }
        return Ok(outputs);
    }

    let bytes_per_row = quants::bytes_per_row(ggml_type, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul end overflow"))?;
    if end > raw.len() {
        bail!(
            "quantized batched matmul needs bytes {}..{} but tensor has {} bytes",
            start,
            end,
            raw.len()
        );
    }

    let rows = &raw[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;

    let mut row_outputs = vec![0.0f32; row_count * input_count];
    if ggml_type == quants::GGML_TYPE_Q4_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let offset = quad_idx * quad_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                    let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                    let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
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
                    )
                    .expect("validated Q4_K four-row batched row dot should succeed");
                });
        } else {
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
                )
                .expect("validated Q4_K four-row batched row dot should succeed");
            }
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
            )
            .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if ggml_type == quants::GGML_TYPE_Q6_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let offset = quad_idx * quad_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                    let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                    let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
                    let (row_a_values, rest) = quad_values.split_at_mut(input_count);
                    let (row_b_values, rest) = rest.split_at_mut(input_count);
                    let (row_c_values, row_d_values) = rest.split_at_mut(input_count);
                    quants::dot_many_q6_k_four_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        row_c_bytes,
                        row_d_bytes,
                        inputs,
                        row_a_values,
                        row_b_values,
                        row_c_values,
                        row_d_values,
                    )
                    .expect("validated Q6_K four-row batched row dot should succeed");
                });
        } else {
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
                quants::dot_many_q6_k_four_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    row_c_bytes,
                    row_d_bytes,
                    inputs,
                    row_a_values,
                    row_b_values,
                    row_c_values,
                    row_d_values,
                )
                .expect("validated Q6_K four-row batched row dot should succeed");
            }
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
            quants::dot_many_q6_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                inputs,
                row_a_values,
                row_b_values,
            )
            .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if (ggml_type == quants::GGML_TYPE_Q4_K || ggml_type == quants::GGML_TYPE_Q6_K)
        && row_count >= 2
    {
        let pair_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        let pair_output_len = pair_count * pair_values_len;
        if use_parallel {
            row_outputs[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .enumerate()
                .for_each(|(pair_idx, pair_values)| {
                    let offset = pair_idx * pair_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + pair_bytes];
                    let (row_a_values, row_b_values) = pair_values.split_at_mut(input_count);
                    match ggml_type {
                        quants::GGML_TYPE_Q4_K => quants::dot_many_q4_k_two_rows_refs_into(
                            row_a_bytes,
                            row_b_bytes,
                            inputs,
                            row_a_values,
                            row_b_values,
                        ),
                        quants::GGML_TYPE_Q6_K => quants::dot_many_q6_k_two_rows_refs_into(
                            row_a_bytes,
                            row_b_bytes,
                            inputs,
                            row_a_values,
                            row_b_values,
                        ),
                        _ => unreachable!("paired path guarded to Q4_K/Q6_K"),
                    }
                    .expect("validated paired quantized batched row dot should succeed");
                });
        } else {
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
                    ),
                    quants::GGML_TYPE_Q6_K => quants::dot_many_q6_k_two_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        inputs,
                        row_a_values,
                        row_b_values,
                    ),
                    _ => unreachable!("paired path guarded to Q4_K/Q6_K"),
                }
                .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if use_parallel {
        row_outputs
            .par_chunks_mut(input_count)
            .enumerate()
            .for_each(|(row_idx, row_values)| {
                let offset = row_idx * bytes_per_row;
                let row_bytes = &rows[offset..offset + bytes_per_row];
                quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                    .expect("validated quantized batched row dot should succeed");
            });
    } else {
        for (row_idx, row_values) in row_outputs.chunks_exact_mut(input_count).enumerate() {
            let offset = row_idx * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                .expect("validated quantized batched row dot should succeed");
        }
    }

    Ok(transpose_row_outputs(&row_outputs, input_count, row_count))
}

pub fn matmul_quantized_many_refs_range_token_major(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let mut outputs = Vec::new();
    let mut row_outputs = Vec::new();
    matmul_quantized_many_refs_range_token_major_with_scratch(
        ggml_type,
        raw,
        inputs,
        row_start,
        row_count,
        in_dim,
        &mut outputs,
        &mut row_outputs,
    )?;
    Ok(outputs)
}

pub fn matmul_quantized_many_refs_range_token_major_into(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
    outputs: &mut Vec<f32>,
) -> Result<()> {
    let mut row_outputs = Vec::new();
    matmul_quantized_many_refs_range_token_major_with_scratch(
        ggml_type,
        raw,
        inputs,
        row_start,
        row_count,
        in_dim,
        outputs,
        &mut row_outputs,
    )
}

pub fn matmul_quantized_many_refs_range_token_major_with_scratch(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
    outputs: &mut Vec<f32>,
    row_outputs: &mut Vec<f32>,
) -> Result<()> {
    if inputs.is_empty() {
        outputs.clear();
        return Ok(());
    }
    if inputs.iter().any(|input| input.len() < in_dim) {
        bail!(
            "quantized batched matmul has an input shorter than in_dim {}",
            in_dim
        );
    }
    if should_use_input_chunk_fast_path(ggml_type, inputs.len(), row_count, in_dim) {
        outputs.resize(inputs.len() * row_count, 0.0f32);
        let outputs = outputs.as_mut_slice();
        let mut chunk_outputs = Vec::new();
        let mut chunk_row_outputs = Vec::new();
        for input_start in (0..inputs.len()).step_by(INPUT_CHUNK_FAST_PATH) {
            let input_end = (input_start + INPUT_CHUNK_FAST_PATH).min(inputs.len());
            matmul_quantized_many_refs_range_token_major_with_scratch(
                ggml_type,
                raw,
                &inputs[input_start..input_end],
                row_start,
                row_count,
                in_dim,
                &mut chunk_outputs,
                &mut chunk_row_outputs,
            )?;
            copy_token_major_output_chunk(&chunk_outputs, input_start, row_count, outputs);
        }
        return Ok(());
    }

    let bytes_per_row = quants::bytes_per_row(ggml_type, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized batched matmul end overflow"))?;
    if end > raw.len() {
        bail!(
            "quantized batched matmul needs bytes {}..{} but tensor has {} bytes",
            start,
            end,
            raw.len()
        );
    }

    let rows = &raw[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;

    row_outputs.resize(row_count * input_count, 0.0f32);
    let row_outputs = row_outputs.as_mut_slice();
    if ggml_type == quants::GGML_TYPE_Q4_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let offset = quad_idx * quad_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                    let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                    let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
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
                    )
                    .expect("validated Q4_K four-row batched row dot should succeed");
                });
        } else {
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
                )
                .expect("validated Q4_K four-row batched row dot should succeed");
            }
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
            )
            .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if ggml_type == quants::GGML_TYPE_Q6_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let offset = quad_idx * quad_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                    let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                    let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
                    let (row_a_values, rest) = quad_values.split_at_mut(input_count);
                    let (row_b_values, rest) = rest.split_at_mut(input_count);
                    let (row_c_values, row_d_values) = rest.split_at_mut(input_count);
                    quants::dot_many_q6_k_four_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        row_c_bytes,
                        row_d_bytes,
                        inputs,
                        row_a_values,
                        row_b_values,
                        row_c_values,
                        row_d_values,
                    )
                    .expect("validated Q6_K four-row batched row dot should succeed");
                });
        } else {
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
                quants::dot_many_q6_k_four_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    row_c_bytes,
                    row_d_bytes,
                    inputs,
                    row_a_values,
                    row_b_values,
                    row_c_values,
                    row_d_values,
                )
                .expect("validated Q6_K four-row batched row dot should succeed");
            }
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
            quants::dot_many_q6_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                inputs,
                row_a_values,
                row_b_values,
            )
            .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if (ggml_type == quants::GGML_TYPE_Q4_K || ggml_type == quants::GGML_TYPE_Q6_K)
        && row_count >= 2
    {
        let pair_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        let pair_output_len = pair_count * pair_values_len;
        if use_parallel {
            row_outputs[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .enumerate()
                .for_each(|(pair_idx, pair_values)| {
                    let offset = pair_idx * pair_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + pair_bytes];
                    let (row_a_values, row_b_values) = pair_values.split_at_mut(input_count);
                    match ggml_type {
                        quants::GGML_TYPE_Q4_K => quants::dot_many_q4_k_two_rows_refs_into(
                            row_a_bytes,
                            row_b_bytes,
                            inputs,
                            row_a_values,
                            row_b_values,
                        ),
                        quants::GGML_TYPE_Q6_K => quants::dot_many_q6_k_two_rows_refs_into(
                            row_a_bytes,
                            row_b_bytes,
                            inputs,
                            row_a_values,
                            row_b_values,
                        ),
                        _ => unreachable!("paired path guarded to Q4_K/Q6_K"),
                    }
                    .expect("validated paired quantized batched row dot should succeed");
                });
        } else {
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
                    ),
                    quants::GGML_TYPE_Q6_K => quants::dot_many_q6_k_two_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        inputs,
                        row_a_values,
                        row_b_values,
                    ),
                    _ => unreachable!("paired path guarded to Q4_K/Q6_K"),
                }
                .expect("validated paired quantized batched row dot should succeed");
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
            )
            .expect("validated quantized batched row dot should succeed");
        }
    } else if use_parallel {
        row_outputs
            .par_chunks_mut(input_count)
            .enumerate()
            .for_each(|(row_idx, row_values)| {
                let offset = row_idx * bytes_per_row;
                let row_bytes = &rows[offset..offset + bytes_per_row];
                quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                    .expect("validated quantized batched row dot should succeed");
            });
    } else {
        for (row_idx, row_values) in row_outputs.chunks_exact_mut(input_count).enumerate() {
            let offset = row_idx * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                .expect("validated quantized batched row dot should succeed");
        }
    }

    transpose_row_outputs_token_major_into(row_outputs, input_count, row_count, outputs);
    Ok(())
}

pub fn matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
    ggml_type: u32,
    raw: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    full_in_dim: usize,
    input_offset: usize,
    input_len: usize,
    outputs: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if input_offset + input_len > full_in_dim {
        bail!(
            "quantized partial batched matmul input range {}..{} exceeds full_in_dim {}",
            input_offset,
            input_offset + input_len,
            full_in_dim
        );
    }
    if inputs.iter().any(|input| input.len() < input_len) {
        bail!(
            "quantized partial batched matmul has an input shorter than input_len {}",
            input_len
        );
    }

    let input_count = inputs.len();
    if outputs.len() != input_count * row_count {
        bail!(
            "quantized partial batched matmul outputs len {} does not match input_count {} * row_count {}",
            outputs.len(),
            input_count,
            row_count
        );
    }

    let full_bytes_per_row = quants::bytes_per_row(ggml_type, full_in_dim)?;
    let partial_bytes_offset = quants::bytes_per_row(ggml_type, input_offset)?;
    let partial_bytes_per_row = quants::bytes_per_row(ggml_type, input_len)?;
    let start = row_start
        .checked_mul(full_bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized partial batched matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(full_bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized partial batched matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized partial batched matmul end overflow"))?;
    if end > raw.len() {
        bail!(
            "quantized partial batched matmul needs bytes {}..{} but tensor has {} bytes",
            start,
            end,
            raw.len()
        );
    }

    let rows = &raw[start..end];
    let use_parallel = row_count > 1
        && row_count
            .saturating_mul(input_len)
            .saturating_mul(input_count)
            >= PARALLEL_MATVEC_MIN_OPS;
    let mut row_outputs = vec![0.0f32; row_count * input_count];

    if ggml_type == quants::GGML_TYPE_Q4_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let base = quad_idx * full_bytes_per_row * 4;
                    let row_a_bytes = &rows[base + partial_bytes_offset
                        ..base + partial_bytes_offset + partial_bytes_per_row];
                    let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                        ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
                    let row_c_bytes = &rows[base + full_bytes_per_row * 2 + partial_bytes_offset
                        ..base
                            + full_bytes_per_row * 2
                            + partial_bytes_offset
                            + partial_bytes_per_row];
                    let row_d_bytes = &rows[base + full_bytes_per_row * 3 + partial_bytes_offset
                        ..base
                            + full_bytes_per_row * 3
                            + partial_bytes_offset
                            + partial_bytes_per_row];
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
                    )
                    .expect("validated partial Q4_K four-row batched row dot should succeed");
                });
        } else {
            for quad_idx in 0..quad_count {
                let base = quad_idx * full_bytes_per_row * 4;
                let row_a_bytes = &rows[base + partial_bytes_offset
                    ..base + partial_bytes_offset + partial_bytes_per_row];
                let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                    ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
                let row_c_bytes = &rows[base + full_bytes_per_row * 2 + partial_bytes_offset
                    ..base + full_bytes_per_row * 2 + partial_bytes_offset + partial_bytes_per_row];
                let row_d_bytes = &rows[base + full_bytes_per_row * 3 + partial_bytes_offset
                    ..base + full_bytes_per_row * 3 + partial_bytes_offset + partial_bytes_per_row];
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
                )
                .expect("validated partial Q4_K four-row batched row dot should succeed");
            }
        }

        let mut next_row = quad_count * 4;
        if row_count.saturating_sub(next_row) >= 2 {
            let base = next_row * full_bytes_per_row;
            let row_a_bytes = &rows
                [base + partial_bytes_offset..base + partial_bytes_offset + partial_bytes_per_row];
            let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
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
            )
            .expect("validated partial Q4_K paired batched row dot should succeed");
            next_row += 2;
        }

        if next_row < row_count {
            let base = next_row * full_bytes_per_row;
            let row_bytes = &rows
                [base + partial_bytes_offset..base + partial_bytes_offset + partial_bytes_per_row];
            quants::dot_many_row_refs_into(
                ggml_type,
                row_bytes,
                inputs,
                &mut row_outputs[next_row * input_count..(next_row + 1) * input_count],
            )
            .expect("validated partial quantized batched row dot should succeed");
        }
    } else if ggml_type == quants::GGML_TYPE_Q6_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_values_len = input_count * 4;
        let quad_output_len = quad_count * quad_values_len;
        if use_parallel {
            row_outputs[..quad_output_len]
                .par_chunks_mut(quad_values_len)
                .enumerate()
                .for_each(|(quad_idx, quad_values)| {
                    let base = quad_idx * full_bytes_per_row * 4;
                    let row_a_bytes = &rows[base + partial_bytes_offset
                        ..base + partial_bytes_offset + partial_bytes_per_row];
                    let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                        ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
                    let row_c_bytes = &rows[base + full_bytes_per_row * 2 + partial_bytes_offset
                        ..base
                            + full_bytes_per_row * 2
                            + partial_bytes_offset
                            + partial_bytes_per_row];
                    let row_d_bytes = &rows[base + full_bytes_per_row * 3 + partial_bytes_offset
                        ..base
                            + full_bytes_per_row * 3
                            + partial_bytes_offset
                            + partial_bytes_per_row];
                    let (row_a_values, rest) = quad_values.split_at_mut(input_count);
                    let (row_b_values, rest) = rest.split_at_mut(input_count);
                    let (row_c_values, row_d_values) = rest.split_at_mut(input_count);
                    quants::dot_many_q6_k_four_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        row_c_bytes,
                        row_d_bytes,
                        inputs,
                        row_a_values,
                        row_b_values,
                        row_c_values,
                        row_d_values,
                    )
                    .expect("validated partial Q6_K four-row batched row dot should succeed");
                });
        } else {
            for quad_idx in 0..quad_count {
                let base = quad_idx * full_bytes_per_row * 4;
                let row_a_bytes = &rows[base + partial_bytes_offset
                    ..base + partial_bytes_offset + partial_bytes_per_row];
                let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                    ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
                let row_c_bytes = &rows[base + full_bytes_per_row * 2 + partial_bytes_offset
                    ..base + full_bytes_per_row * 2 + partial_bytes_offset + partial_bytes_per_row];
                let row_d_bytes = &rows[base + full_bytes_per_row * 3 + partial_bytes_offset
                    ..base + full_bytes_per_row * 3 + partial_bytes_offset + partial_bytes_per_row];
                let values_offset = quad_idx * quad_values_len;
                let quad_values = &mut row_outputs[values_offset..values_offset + quad_values_len];
                let (row_a_values, rest) = quad_values.split_at_mut(input_count);
                let (row_b_values, rest) = rest.split_at_mut(input_count);
                let (row_c_values, row_d_values) = rest.split_at_mut(input_count);
                quants::dot_many_q6_k_four_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    row_c_bytes,
                    row_d_bytes,
                    inputs,
                    row_a_values,
                    row_b_values,
                    row_c_values,
                    row_d_values,
                )
                .expect("validated partial Q6_K four-row batched row dot should succeed");
            }
        }

        let mut next_row = quad_count * 4;
        if row_count.saturating_sub(next_row) >= 2 {
            let base = next_row * full_bytes_per_row;
            let row_a_bytes = &rows
                [base + partial_bytes_offset..base + partial_bytes_offset + partial_bytes_per_row];
            let row_b_bytes = &rows[base + full_bytes_per_row + partial_bytes_offset
                ..base + full_bytes_per_row + partial_bytes_offset + partial_bytes_per_row];
            let values_offset = next_row * input_count;
            let (row_a_values, row_b_values) = row_outputs
                [values_offset..values_offset + input_count * 2]
                .split_at_mut(input_count);
            quants::dot_many_q6_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                inputs,
                row_a_values,
                row_b_values,
            )
            .expect("validated partial Q6_K paired batched row dot should succeed");
            next_row += 2;
        }

        if next_row < row_count {
            let base = next_row * full_bytes_per_row;
            let row_bytes = &rows
                [base + partial_bytes_offset..base + partial_bytes_offset + partial_bytes_per_row];
            quants::dot_many_row_refs_into(
                ggml_type,
                row_bytes,
                inputs,
                &mut row_outputs[next_row * input_count..(next_row + 1) * input_count],
            )
            .expect("validated partial quantized batched row dot should succeed");
        }
    } else if use_parallel {
        row_outputs
            .par_chunks_mut(input_count)
            .enumerate()
            .for_each(|(row_idx, row_values)| {
                let base = row_idx * full_bytes_per_row;
                let row_bytes = &rows[base + partial_bytes_offset
                    ..base + partial_bytes_offset + partial_bytes_per_row];
                quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                    .expect("validated partial quantized batched row dot should succeed");
            });
    } else {
        for (row_idx, row_values) in row_outputs.chunks_exact_mut(input_count).enumerate() {
            let base = row_idx * full_bytes_per_row;
            let row_bytes = &rows
                [base + partial_bytes_offset..base + partial_bytes_offset + partial_bytes_per_row];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, inputs, row_values)
                .expect("validated partial quantized batched row dot should succeed");
        }
    }

    transpose_row_outputs_token_major_accumulate(&row_outputs, input_count, row_count, outputs);
    Ok(())
}

pub fn matmul_quantized_many_pair_q4_k_range(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[Vec<f32>],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    matmul_quantized_many_pair_q4_k_refs_range(
        raw_a,
        raw_b,
        &input_refs,
        row_start,
        row_count,
        in_dim,
    )
}

pub fn matmul_quantized_many_pair_q4_k_refs_range(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    if inputs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if inputs.iter().any(|input| input.len() < in_dim) {
        bail!(
            "quantized paired batched matmul has an input shorter than in_dim {}",
            in_dim
        );
    }

    let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul end overflow"))?;
    if end > raw_a.len() || end > raw_b.len() {
        bail!(
            "quantized paired batched matmul needs bytes {}..{} but tensors have {} and {} bytes",
            start,
            end,
            raw_a.len(),
            raw_b.len()
        );
    }

    let rows_a = &raw_a[start..end];
    let rows_b = &raw_b[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;

    let mut row_outputs_a = vec![0.0f32; row_count * input_count];
    let mut row_outputs_b = vec![0.0f32; row_count * input_count];
    if row_count >= 2 {
        let pair_row_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        let pair_output_len = pair_row_count * pair_values_len;
        if use_parallel {
            row_outputs_a[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .zip(row_outputs_b[..pair_output_len].par_chunks_mut(pair_values_len))
                .enumerate()
                .for_each(|(pair_idx, (row_values_a, row_values_b))| {
                    let offset = pair_idx * pair_bytes;
                    let row_a0_bytes = &rows_a[offset..offset + bytes_per_row];
                    let row_a1_bytes = &rows_a[offset + bytes_per_row..offset + pair_bytes];
                    let row_b0_bytes = &rows_b[offset..offset + bytes_per_row];
                    let row_b1_bytes = &rows_b[offset + bytes_per_row..offset + pair_bytes];
                    let (row_a0_values, row_a1_values) = row_values_a.split_at_mut(input_count);
                    let (row_b0_values, row_b1_values) = row_values_b.split_at_mut(input_count);
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
                    )
                    .expect("validated paired Q4_K two-row-pair batched row dot should succeed");
                });
        } else {
            for pair_idx in 0..pair_row_count {
                let offset = pair_idx * pair_bytes;
                let row_a0_bytes = &rows_a[offset..offset + bytes_per_row];
                let row_a1_bytes = &rows_a[offset + bytes_per_row..offset + pair_bytes];
                let row_b0_bytes = &rows_b[offset..offset + bytes_per_row];
                let row_b1_bytes = &rows_b[offset + bytes_per_row..offset + pair_bytes];
                let values_offset = pair_idx * pair_values_len;
                let (row_a_values, row_b_values) = (
                    &mut row_outputs_a[values_offset..values_offset + pair_values_len],
                    &mut row_outputs_b[values_offset..values_offset + pair_values_len],
                );
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
                )
                .expect("validated paired Q4_K two-row-pair batched row dot should succeed");
            }
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
            )
            .expect("validated paired quantized batched row dot should succeed");
        }
    } else {
        quants::dot_many_q4_k_two_rows_refs_into(
            &rows_a[..bytes_per_row],
            &rows_b[..bytes_per_row],
            inputs,
            &mut row_outputs_a[..input_count],
            &mut row_outputs_b[..input_count],
        )
        .expect("validated paired quantized batched row dot should succeed");
    }

    Ok((
        transpose_row_outputs(&row_outputs_a, input_count, row_count),
        transpose_row_outputs(&row_outputs_b, input_count, row_count),
    ))
}

pub fn matmul_quantized_many_pair_q4_k_refs_range_token_major(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut outputs_a = Vec::new();
    let mut outputs_b = Vec::new();
    let mut row_outputs_a = Vec::new();
    let mut row_outputs_b = Vec::new();
    matmul_quantized_many_pair_q4_k_refs_range_token_major_with_scratch(
        raw_a,
        raw_b,
        inputs,
        row_start,
        row_count,
        in_dim,
        &mut outputs_a,
        &mut outputs_b,
        &mut row_outputs_a,
        &mut row_outputs_b,
    )?;
    Ok((outputs_a, outputs_b))
}

pub fn matmul_quantized_many_pair_q4_k_refs_range_token_major_into(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
    outputs_a: &mut Vec<f32>,
    outputs_b: &mut Vec<f32>,
) -> Result<()> {
    let mut row_outputs_a = Vec::new();
    let mut row_outputs_b = Vec::new();
    matmul_quantized_many_pair_q4_k_refs_range_token_major_with_scratch(
        raw_a,
        raw_b,
        inputs,
        row_start,
        row_count,
        in_dim,
        outputs_a,
        outputs_b,
        &mut row_outputs_a,
        &mut row_outputs_b,
    )
}

pub fn matmul_quantized_many_pair_q4_k_refs_range_token_major_with_scratch(
    raw_a: &[u8],
    raw_b: &[u8],
    inputs: &[&[f32]],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
    outputs_a: &mut Vec<f32>,
    outputs_b: &mut Vec<f32>,
    row_outputs_a: &mut Vec<f32>,
    row_outputs_b: &mut Vec<f32>,
) -> Result<()> {
    if inputs.is_empty() {
        outputs_a.clear();
        outputs_b.clear();
        return Ok(());
    }
    if inputs.iter().any(|input| input.len() < in_dim) {
        bail!(
            "quantized paired batched matmul has an input shorter than in_dim {}",
            in_dim
        );
    }
    if should_use_input_chunk_fast_path(quants::GGML_TYPE_Q4_K, inputs.len(), row_count, in_dim) {
        outputs_a.resize(inputs.len() * row_count, 0.0f32);
        outputs_b.resize(inputs.len() * row_count, 0.0f32);
        let outputs_a = outputs_a.as_mut_slice();
        let outputs_b = outputs_b.as_mut_slice();
        let mut chunk_outputs_a = Vec::new();
        let mut chunk_outputs_b = Vec::new();
        let mut chunk_row_outputs_a = Vec::new();
        let mut chunk_row_outputs_b = Vec::new();
        for input_start in (0..inputs.len()).step_by(INPUT_CHUNK_FAST_PATH) {
            let input_end = (input_start + INPUT_CHUNK_FAST_PATH).min(inputs.len());
            matmul_quantized_many_pair_q4_k_refs_range_token_major_with_scratch(
                raw_a,
                raw_b,
                &inputs[input_start..input_end],
                row_start,
                row_count,
                in_dim,
                &mut chunk_outputs_a,
                &mut chunk_outputs_b,
                &mut chunk_row_outputs_a,
                &mut chunk_row_outputs_b,
            )?;
            copy_token_major_output_chunk(&chunk_outputs_a, input_start, row_count, outputs_a);
            copy_token_major_output_chunk(&chunk_outputs_b, input_start, row_count, outputs_b);
        }
        return Ok(());
    }

    let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("quantized paired batched matmul end overflow"))?;
    if end > raw_a.len() || end > raw_b.len() {
        bail!(
            "quantized paired batched matmul needs bytes {}..{} but tensors have {} and {} bytes",
            start,
            end,
            raw_a.len(),
            raw_b.len()
        );
    }

    let rows_a = &raw_a[start..end];
    let rows_b = &raw_b[start..end];
    let input_count = inputs.len();
    let use_parallel = row_count > 1
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS;

    row_outputs_a.resize(row_count * input_count, 0.0f32);
    row_outputs_b.resize(row_count * input_count, 0.0f32);
    let row_outputs_a = row_outputs_a.as_mut_slice();
    let row_outputs_b = row_outputs_b.as_mut_slice();
    if row_count >= 2 {
        let pair_row_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let pair_values_len = input_count * 2;
        let pair_output_len = pair_row_count * pair_values_len;
        if use_parallel {
            row_outputs_a[..pair_output_len]
                .par_chunks_mut(pair_values_len)
                .zip(row_outputs_b[..pair_output_len].par_chunks_mut(pair_values_len))
                .enumerate()
                .for_each(|(pair_idx, (row_values_a, row_values_b))| {
                    let offset = pair_idx * pair_bytes;
                    let row_a0_bytes = &rows_a[offset..offset + bytes_per_row];
                    let row_a1_bytes = &rows_a[offset + bytes_per_row..offset + pair_bytes];
                    let row_b0_bytes = &rows_b[offset..offset + bytes_per_row];
                    let row_b1_bytes = &rows_b[offset + bytes_per_row..offset + pair_bytes];
                    let (row_a0_values, row_a1_values) = row_values_a.split_at_mut(input_count);
                    let (row_b0_values, row_b1_values) = row_values_b.split_at_mut(input_count);
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
                    )
                    .expect("validated paired Q4_K two-row-pair batched row dot should succeed");
                });
        } else {
            for pair_idx in 0..pair_row_count {
                let offset = pair_idx * pair_bytes;
                let row_a0_bytes = &rows_a[offset..offset + bytes_per_row];
                let row_a1_bytes = &rows_a[offset + bytes_per_row..offset + pair_bytes];
                let row_b0_bytes = &rows_b[offset..offset + bytes_per_row];
                let row_b1_bytes = &rows_b[offset + bytes_per_row..offset + pair_bytes];
                let values_offset = pair_idx * pair_values_len;
                let (row_a_values, row_b_values) = (
                    &mut row_outputs_a[values_offset..values_offset + pair_values_len],
                    &mut row_outputs_b[values_offset..values_offset + pair_values_len],
                );
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
                )
                .expect("validated paired Q4_K two-row-pair batched row dot should succeed");
            }
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
            )
            .expect("validated paired quantized batched row dot should succeed");
        }
    } else {
        quants::dot_many_q4_k_two_rows_refs_into(
            &rows_a[..bytes_per_row],
            &rows_b[..bytes_per_row],
            inputs,
            &mut row_outputs_a[..input_count],
            &mut row_outputs_b[..input_count],
        )
        .expect("validated paired quantized batched row dot should succeed");
    }

    transpose_row_outputs_token_major_into(row_outputs_a, input_count, row_count, outputs_a);
    transpose_row_outputs_token_major_into(row_outputs_b, input_count, row_count, outputs_b);
    Ok(())
}

fn transpose_row_outputs(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
) -> Vec<Vec<f32>> {
    let mut outputs = vec![vec![0.0f32; row_count]; input_count];
    match input_count {
        0 => {}
        1 => {
            let out0 = &mut outputs[0];
            for (row_idx, values) in row_outputs.chunks_exact(1).enumerate() {
                out0[row_idx] = values[0];
            }
        }
        2 => {
            let (a, rest) = outputs.split_at_mut(1);
            let out0 = &mut a[0];
            let out1 = &mut rest[0];
            for (row_idx, values) in row_outputs.chunks_exact(2).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
            }
        }
        3 => {
            let (a, rest) = outputs.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let out0 = &mut a[0];
            let out1 = &mut b[0];
            let out2 = &mut rest[0];
            for (row_idx, values) in row_outputs.chunks_exact(3).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
                out2[row_idx] = values[2];
            }
        }
        4 => {
            let (a, rest) = outputs.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let (c, rest) = rest.split_at_mut(1);
            let out0 = &mut a[0];
            let out1 = &mut b[0];
            let out2 = &mut c[0];
            let out3 = &mut rest[0];
            for (row_idx, values) in row_outputs.chunks_exact(4).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
                out2[row_idx] = values[2];
                out3[row_idx] = values[3];
            }
        }
        5 => {
            let (a, rest) = outputs.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let (c, rest) = rest.split_at_mut(1);
            let (d, rest) = rest.split_at_mut(1);
            let out0 = &mut a[0];
            let out1 = &mut b[0];
            let out2 = &mut c[0];
            let out3 = &mut d[0];
            let out4 = &mut rest[0];
            for (row_idx, values) in row_outputs.chunks_exact(5).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
                out2[row_idx] = values[2];
                out3[row_idx] = values[3];
                out4[row_idx] = values[4];
            }
        }
        6 => {
            let (a, rest) = outputs.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let (c, rest) = rest.split_at_mut(1);
            let (d, rest) = rest.split_at_mut(1);
            let (e, rest) = rest.split_at_mut(1);
            let out0 = &mut a[0];
            let out1 = &mut b[0];
            let out2 = &mut c[0];
            let out3 = &mut d[0];
            let out4 = &mut e[0];
            let out5 = &mut rest[0];
            for (row_idx, values) in row_outputs.chunks_exact(6).enumerate() {
                out0[row_idx] = values[0];
                out1[row_idx] = values[1];
                out2[row_idx] = values[2];
                out3[row_idx] = values[3];
                out4[row_idx] = values[4];
                out5[row_idx] = values[5];
            }
        }
        _ => {
            for (row_idx, row_values) in row_outputs.chunks_exact(input_count).enumerate() {
                for (input_idx, value) in row_values.iter().copied().enumerate() {
                    outputs[input_idx][row_idx] = value;
                }
            }
        }
    }
    outputs
}

fn transpose_row_outputs_token_major(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
) -> Vec<f32> {
    let mut outputs = Vec::new();
    transpose_row_outputs_token_major_into(row_outputs, input_count, row_count, &mut outputs);
    outputs
}

fn transpose_row_outputs_token_major_into(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
    outputs: &mut Vec<f32>,
) {
    outputs.resize(input_count * row_count, 0.0f32);
    let outputs = outputs.as_mut_slice();
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
}

fn transpose_row_outputs_token_major_accumulate(
    row_outputs: &[f32],
    input_count: usize,
    row_count: usize,
    outputs: &mut [f32],
) {
    match input_count {
        0 => {}
        1 => {
            let out0 = &mut outputs[..row_count];
            for (row_idx, values) in row_outputs.chunks_exact(1).enumerate() {
                out0[row_idx] += values[0];
            }
        }
        2 => {
            let (out0, out1) = outputs.split_at_mut(row_count);
            let out1 = &mut out1[..row_count];
            for (row_idx, values) in row_outputs.chunks_exact(2).enumerate() {
                out0[row_idx] += values[0];
                out1[row_idx] += values[1];
            }
        }
        3 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, c) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(3).enumerate() {
                a[row_idx] += values[0];
                b[row_idx] += values[1];
                c[row_idx] += values[2];
            }
        }
        4 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, d) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(4).enumerate() {
                a[row_idx] += values[0];
                b[row_idx] += values[1];
                c[row_idx] += values[2];
                d[row_idx] += values[3];
            }
        }
        5 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, rest) = rest.split_at_mut(row_count);
            let (d, e) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(5).enumerate() {
                a[row_idx] += values[0];
                b[row_idx] += values[1];
                c[row_idx] += values[2];
                d[row_idx] += values[3];
                e[row_idx] += values[4];
            }
        }
        6 => {
            let (a, rest) = outputs.split_at_mut(row_count);
            let (b, rest) = rest.split_at_mut(row_count);
            let (c, rest) = rest.split_at_mut(row_count);
            let (d, rest) = rest.split_at_mut(row_count);
            let (e, f) = rest.split_at_mut(row_count);
            for (row_idx, values) in row_outputs.chunks_exact(6).enumerate() {
                a[row_idx] += values[0];
                b[row_idx] += values[1];
                c[row_idx] += values[2];
                d[row_idx] += values[3];
                e[row_idx] += values[4];
                f[row_idx] += values[5];
            }
        }
        _ => {
            for (row_idx, row_values) in row_outputs.chunks_exact(input_count).enumerate() {
                for (input_idx, value) in row_values.iter().copied().enumerate() {
                    outputs[input_idx * row_count + row_idx] += value;
                }
            }
        }
    }
}

fn should_use_input_chunk_fast_path(
    ggml_type: u32,
    input_count: usize,
    row_count: usize,
    in_dim: usize,
) -> bool {
    input_count > INPUT_CHUNK_FAST_PATH
        && (ggml_type == quants::GGML_TYPE_Q4_K || ggml_type == quants::GGML_TYPE_Q6_K)
        && row_count.saturating_mul(in_dim).saturating_mul(input_count) >= PARALLEL_MATVEC_MIN_OPS
}

fn copy_token_major_output_chunk(
    chunk_outputs: &[f32],
    input_start: usize,
    row_count: usize,
    outputs: &mut [f32],
) {
    for (local_input_idx, chunk_values) in chunk_outputs.chunks_exact(row_count).enumerate() {
        let global_offset = (input_start + local_input_idx) * row_count;
        outputs[global_offset..global_offset + row_count].copy_from_slice(chunk_values);
    }
}

fn copy_nested_output_chunk(
    chunk_outputs: &[Vec<f32>],
    input_start: usize,
    outputs: &mut [Vec<f32>],
) {
    for (local_input_idx, chunk_values) in chunk_outputs.iter().enumerate() {
        outputs[input_start + local_input_idx].copy_from_slice(chunk_values);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopKLogits {
    pub argmax_idx: usize,
    pub argmax_score: f32,
    pub top: Vec<(usize, f32)>,
}

pub fn matmul_raw_top_k(
    ggml_type: u32,
    raw: &[u8],
    input: &[f32],
    row_start: usize,
    row_count: usize,
    in_dim: usize,
    k: usize,
    softcap: Option<f32>,
) -> Result<TopKLogits> {
    let bytes_per_row = quants::bytes_per_row(ggml_type, in_dim)?;
    let start = row_start
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("raw top-k row_start overflow"))?;
    let byte_len = row_count
        .checked_mul(bytes_per_row)
        .ok_or_else(|| anyhow::anyhow!("raw top-k row_count overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| anyhow::anyhow!("raw top-k end overflow"))?;
    if end > raw.len() {
        bail!(
            "raw top-k needs bytes {}..{} but tensor has {}",
            start,
            end,
            raw.len()
        );
    }
    if input.len() < in_dim {
        bail!(
            "raw top-k input len {} is smaller than in_dim {}",
            input.len(),
            in_dim
        );
    }
    if row_count == 0 {
        bail!("raw top-k requires at least one row");
    }

    let rows = &raw[start..end];
    let use_parallel = row_count > 1 && row_count.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_OPS;
    let input_refs = [input];

    if ggml_type == quants::GGML_TYPE_Q4_K && row_count >= 4 {
        let quad_count = row_count / 4;
        let quad_bytes = bytes_per_row * 4;
        let mut result = if use_parallel {
            (0..quad_count)
                .into_par_iter()
                .map(|quad_idx| {
                    let offset = quad_idx * quad_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                    let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                    let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
                    let (score_a, score_b, score_c, score_d) = quants::dot_q4_k_four_rows(
                        row_a_bytes,
                        row_b_bytes,
                        row_c_bytes,
                        row_d_bytes,
                        input,
                    )
                    .expect("validated Q4_K four-row logits dot should succeed");
                    let row_base = row_start + quad_idx * 4;
                    let mut partial = empty_top_k(row_base, k);
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base,
                        apply_softcap_score(score_a, softcap),
                    );
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base + 1,
                        apply_softcap_score(score_b, softcap),
                    );
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base + 2,
                        apply_softcap_score(score_c, softcap),
                    );
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base + 3,
                        apply_softcap_score(score_d, softcap),
                    );
                    partial
                })
                .reduce(
                    || empty_top_k(row_start, k),
                    |left, right| merge_top_k(left, right, k),
                )
        } else {
            let mut result = empty_top_k(row_start, k);
            for quad_idx in 0..quad_count {
                let offset = quad_idx * quad_bytes;
                let row_a_bytes = &rows[offset..offset + bytes_per_row];
                let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
                let row_c_bytes = &rows[offset + bytes_per_row * 2..offset + bytes_per_row * 3];
                let row_d_bytes = &rows[offset + bytes_per_row * 3..offset + bytes_per_row * 4];
                let (score_a, score_b, score_c, score_d) = quants::dot_q4_k_four_rows(
                    row_a_bytes,
                    row_b_bytes,
                    row_c_bytes,
                    row_d_bytes,
                    input,
                )
                .expect("validated Q4_K four-row logits dot should succeed");
                let row_base = row_start + quad_idx * 4;
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base,
                    apply_softcap_score(score_a, softcap),
                );
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base + 1,
                    apply_softcap_score(score_b, softcap),
                );
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base + 2,
                    apply_softcap_score(score_c, softcap),
                );
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base + 3,
                    apply_softcap_score(score_d, softcap),
                );
            }
            result
        };
        let mut next_row = quad_count * 4;
        if row_count.saturating_sub(next_row) >= 2 {
            let offset = next_row * bytes_per_row;
            let row_a_bytes = &rows[offset..offset + bytes_per_row];
            let row_b_bytes = &rows[offset + bytes_per_row..offset + bytes_per_row * 2];
            let mut score_a = [0.0f32; 1];
            let mut score_b = [0.0f32; 1];
            quants::dot_many_q4_k_two_rows_refs_into(
                row_a_bytes,
                row_b_bytes,
                &input_refs,
                &mut score_a,
                &mut score_b,
            )
            .expect("validated Q4_K paired logits dot should succeed");
            observe_top_k_score(
                &mut result,
                k,
                row_start + next_row,
                apply_softcap_score(score_a[0], softcap),
            );
            observe_top_k_score(
                &mut result,
                k,
                row_start + next_row + 1,
                apply_softcap_score(score_b[0], softcap),
            );
            next_row += 2;
        }
        if next_row < row_count {
            let offset = next_row * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            let mut score = [0.0f32; 1];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, &input_refs, &mut score)
                .expect("validated logits row dot should succeed");
            observe_top_k_score(
                &mut result,
                k,
                row_start + next_row,
                apply_softcap_score(score[0], softcap),
            );
        }
        return Ok(result);
    }

    if ggml_type == quants::GGML_TYPE_Q6_K && row_count >= 2 {
        let pair_count = row_count / 2;
        let pair_bytes = bytes_per_row * 2;
        let mut result = if use_parallel {
            (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let offset = pair_idx * pair_bytes;
                    let row_a_bytes = &rows[offset..offset + bytes_per_row];
                    let row_b_bytes = &rows[offset + bytes_per_row..offset + pair_bytes];
                    let mut score_a = [0.0f32; 1];
                    let mut score_b = [0.0f32; 1];
                    quants::dot_many_q6_k_two_rows_refs_into(
                        row_a_bytes,
                        row_b_bytes,
                        &input_refs,
                        &mut score_a,
                        &mut score_b,
                    )
                    .expect("validated Q6_K paired logits dot should succeed");
                    let row_base = row_start + pair_idx * 2;
                    let mut partial = empty_top_k(row_base, k);
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base,
                        apply_softcap_score(score_a[0], softcap),
                    );
                    observe_top_k_score(
                        &mut partial,
                        k,
                        row_base + 1,
                        apply_softcap_score(score_b[0], softcap),
                    );
                    partial
                })
                .reduce(
                    || empty_top_k(row_start, k),
                    |left, right| merge_top_k(left, right, k),
                )
        } else {
            let mut result = empty_top_k(row_start, k);
            for pair_idx in 0..pair_count {
                let offset = pair_idx * pair_bytes;
                let row_a_bytes = &rows[offset..offset + bytes_per_row];
                let row_b_bytes = &rows[offset + bytes_per_row..offset + pair_bytes];
                let mut score_a = [0.0f32; 1];
                let mut score_b = [0.0f32; 1];
                quants::dot_many_q6_k_two_rows_refs_into(
                    row_a_bytes,
                    row_b_bytes,
                    &input_refs,
                    &mut score_a,
                    &mut score_b,
                )
                .expect("validated Q6_K paired logits dot should succeed");
                let row_base = row_start + pair_idx * 2;
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base,
                    apply_softcap_score(score_a[0], softcap),
                );
                observe_top_k_score(
                    &mut result,
                    k,
                    row_base + 1,
                    apply_softcap_score(score_b[0], softcap),
                );
            }
            result
        };
        if row_count % 2 != 0 {
            let row_idx = row_count - 1;
            let offset = row_idx * bytes_per_row;
            let row_bytes = &rows[offset..offset + bytes_per_row];
            let mut score = [0.0f32; 1];
            quants::dot_many_row_refs_into(ggml_type, row_bytes, &input_refs, &mut score)
                .expect("validated logits row dot should succeed");
            observe_top_k_score(
                &mut result,
                k,
                row_start + row_idx,
                apply_softcap_score(score[0], softcap),
            );
        }
        return Ok(result);
    }

    if use_parallel {
        Ok(rows
            .par_chunks_exact(bytes_per_row)
            .enumerate()
            .map(|(offset, row_bytes)| {
                let idx = row_start + offset;
                let score = apply_softcap_score(
                    quants::dot_row(ggml_type, row_bytes, input)
                        .expect("validated logits row dot should succeed"),
                    softcap,
                );
                let mut result = empty_top_k(idx, k);
                observe_top_k_score(&mut result, k, idx, score);
                result
            })
            .reduce(
                || empty_top_k(row_start, k),
                |left, right| merge_top_k(left, right, k),
            ))
    } else {
        let mut result = empty_top_k(row_start, k);
        for (offset, row_bytes) in rows.chunks_exact(bytes_per_row).enumerate() {
            let idx = row_start + offset;
            let score = apply_softcap_score(
                quants::dot_row(ggml_type, row_bytes, input)
                    .expect("validated logits row dot should succeed"),
                softcap,
            );
            observe_top_k_score(&mut result, k, idx, score);
        }
        Ok(result)
    }
}

fn empty_top_k(default_idx: usize, k: usize) -> TopKLogits {
    TopKLogits {
        argmax_idx: default_idx,
        argmax_score: f32::NEG_INFINITY,
        top: if k == 0 {
            Vec::new()
        } else {
            Vec::with_capacity(k)
        },
    }
}

fn apply_softcap_score(score: f32, softcap: Option<f32>) -> f32 {
    if let Some(cap) = softcap {
        cap * (score / cap).tanh()
    } else {
        score
    }
}

fn observe_top_k_score(result: &mut TopKLogits, k: usize, idx: usize, score: f32) {
    if score > result.argmax_score {
        result.argmax_idx = idx;
        result.argmax_score = score;
    }
    if k > 0 {
        insert_top_k(&mut result.top, k, idx, score);
    }
}

fn merge_top_k(mut left: TopKLogits, right: TopKLogits, k: usize) -> TopKLogits {
    if right.argmax_score > left.argmax_score {
        left.argmax_idx = right.argmax_idx;
        left.argmax_score = right.argmax_score;
    }
    if k > 0 {
        for (idx, score) in right.top {
            insert_top_k(&mut left.top, k, idx, score);
        }
    }
    left
}

fn insert_top_k(top: &mut Vec<(usize, f32)>, k: usize, idx: usize, score: f32) {
    if k == 0 {
        return;
    }
    let insert_at = top
        .iter()
        .position(|(_, existing)| {
            score
                .partial_cmp(existing)
                .unwrap_or(std::cmp::Ordering::Less)
                .is_gt()
        })
        .unwrap_or(top.len());
    if insert_at < k {
        top.insert(insert_at, (idx, score));
        if top.len() > k {
            top.pop();
        }
    }
}

pub fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

pub fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

pub fn vec_scale(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|v| v * s).collect()
}

pub fn rope_apply(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    position: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    rope_apply_with_base_and_rotary_dim(
        q, k, freqs, position, n_heads, n_kv_heads, head_dim, 10000.0, head_dim,
    );
}

pub fn rope_apply_with_base(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    position: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_theta: f32,
) {
    rope_apply_with_base_and_rotary_dim(
        q, k, freqs, position, n_heads, n_kv_heads, head_dim, base_theta, head_dim,
    );
}

pub fn rope_apply_with_base_and_rotary_dim(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    position: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_theta: f32,
    rotary_dim: usize,
) {
    rope_apply_with_base_and_rotary_dim_mode(
        q, k, freqs, position, n_heads, n_kv_heads, head_dim, base_theta, rotary_dim, false,
    );
}

pub fn rope_apply_with_base_and_rotary_dim_mode(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    position: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_theta: f32,
    rotary_dim: usize,
    proportional: bool,
) {
    let angles = rope_angles(
        freqs,
        position,
        head_dim,
        base_theta,
        rotary_dim,
        proportional,
    );
    rope_apply_with_angles(q, k, &angles, n_heads, n_kv_heads, head_dim);
}

pub fn rope_angles(
    freqs: &[f32],
    position: u32,
    head_dim: usize,
    base_theta: f32,
    rotary_dim: usize,
    proportional: bool,
) -> Vec<(f32, f32)> {
    let head_dim = head_dim & !1usize;
    let rotary_dim = rotary_dim.min(head_dim) & !1usize;
    if rotary_dim == 0 || head_dim == 0 {
        return Vec::new();
    }
    let rope_angles = rotary_dim / 2;
    let freq_denominator = if proportional {
        head_dim as f32
    } else {
        rotary_dim as f32
    };
    let pos = position as f32;
    let mut angles = Vec::with_capacity(rope_angles);
    for i in 0..rope_angles {
        let inv_freq = 1.0 / base_theta.powf(2.0 * i as f32 / freq_denominator);
        let freq_scale = if i < freqs.len() { freqs[i] } else { 1.0 };
        let theta = pos * inv_freq * freq_scale;
        angles.push((theta.cos(), theta.sin()));
    }
    angles
}

pub fn rope_apply_with_angles(
    q: &mut [f32],
    k: &mut [f32],
    angles: &[(f32, f32)],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    rope_apply_heads_with_angles(q, angles, n_heads, head_dim);
    rope_apply_heads_with_angles(k, angles, n_kv_heads, head_dim);
}

fn rope_apply_heads_with_angles(
    x: &mut [f32],
    angles: &[(f32, f32)],
    n_heads: usize,
    head_dim: usize,
) {
    let head_dim = head_dim & !1usize;
    if angles.is_empty() || head_dim == 0 {
        return;
    }
    let half = head_dim / 2;

    for h in 0..n_heads {
        let base = h * head_dim;
        for (i, (cos_t, sin_t)) in angles.iter().copied().enumerate() {
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos_t - x1 * sin_t;
            x[base + half + i] = x0 * sin_t + x1 * cos_t;
        }
    }
}

pub fn gqa_attention(
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let groups = n_heads / n_kv_heads;
    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / groups;
        let v_offset = kv_h * head_dim;
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            output[out_offset + d] = v[v_offset + d];
        }
    }
    output
}

pub fn gqa_attention_seq(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    gqa_attention_seq_with_window_and_limit(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, None, None,
    )
}

pub fn gqa_attention_seq_with_window(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
) -> Vec<f32> {
    gqa_attention_seq_with_window_and_limit(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        window_size,
        None,
    )
}

pub fn gqa_attention_seq_with_window_and_limit(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    cache_limit: Option<usize>,
) -> Vec<f32> {
    let groups = n_heads / n_kv_heads;
    let seq_len = cache_limit
        .unwrap_or(k_cache.len())
        .min(k_cache.len())
        .min(v_cache.len());
    let mut output = vec![0.0f32; n_heads * head_dim];
    if seq_len == 0 {
        return output;
    }
    let window_start = window_size
        .map(|size| seq_len.saturating_sub(size))
        .unwrap_or(0);
    for h in 0..n_heads {
        let kv_h = h / groups;
        let q_offset = h * head_dim;
        let k_offset = kv_h * head_dim;

        let mut scores = Vec::with_capacity(seq_len);
        for t in window_start..seq_len {
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[q_offset + d] * k_cache[t][k_offset + d];
            }
            scores.push(score);
        }

        let weights = softmax(&scores);

        let out_offset = h * head_dim;
        let v_offset = kv_h * head_dim;
        for (window_idx, t) in (window_start..seq_len).enumerate() {
            for d in 0..head_dim {
                output[out_offset + d] += weights[window_idx] * v_cache[t][v_offset + d];
            }
        }
    }
    output
}

pub fn per_head_rms_norm(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize) {
    per_head_rms_norm_impl(x, Some(weight), n_heads, head_dim);
}

pub fn per_head_rms_norm_no_scale(x: &mut [f32], n_heads: usize, head_dim: usize) {
    per_head_rms_norm_impl(x, None, n_heads, head_dim);
}

fn per_head_rms_norm_impl(x: &mut [f32], weight: Option<&[f32]>, n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let slice = &x[base..base + head_dim];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let scale = 1.0 / (mean_sq + GEMMA_RMS_NORM_EPS).sqrt();
        if let Some(weight) = weight {
            if weight.len() == head_dim {
                for (value, weight) in x[base..base + head_dim].iter_mut().zip(weight.iter()) {
                    *value *= scale * *weight;
                }
            } else {
                for i in 0..head_dim {
                    x[base + i] *= scale * weight[i % weight.len()];
                }
            }
        } else {
            for value in &mut x[base..base + head_dim] {
                *value *= scale;
            }
        }
    }
}

pub fn embedding_lookup(
    embd_data: &[f32],
    token_id: u32,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let id = token_id as usize;
    if id >= vocab_size {
        bail!("Token ID {} exceeds vocab size {}", token_id, vocab_size);
    }
    let start = id * hidden_dim;
    let end = start + hidden_dim;
    if end > embd_data.len() {
        bail!(
            "Embedding data too short: need {} but have {}",
            end,
            embd_data.len()
        );
    }
    Ok(embd_data[start..end].to_vec())
}

pub fn embedding_lookup_sum(
    embd_data: &[f32],
    token_ids: &[u32],
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        bail!("No token IDs provided");
    }
    let first = embedding_lookup(embd_data, token_ids[0], hidden_dim, vocab_size)?;
    if token_ids.len() == 1 {
        return Ok(first);
    }
    let mut sum = first;
    for &tid in &token_ids[1..] {
        let emb = embedding_lookup(embd_data, tid, hidden_dim, vocab_size)?;
        vec_add_inplace(&mut sum, &emb);
    }
    Ok(sum)
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn top_k_sample(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || logits.is_empty() {
        return Vec::new();
    }

    let mut top = Vec::<(usize, f32)>::with_capacity(k.min(logits.len()));
    for (idx, score) in logits.iter().copied().enumerate() {
        let insert_at = top
            .iter()
            .position(|(_, existing)| {
                score
                    .partial_cmp(existing)
                    .unwrap_or(std::cmp::Ordering::Less)
                    .is_gt()
            })
            .unwrap_or(top.len());
        if insert_at < k {
            top.insert(insert_at, (idx, score));
            if top.len() > k {
                top.pop();
            }
        }
    }
    top
}

#[derive(Debug, Clone)]
pub struct GemmaLayerConfig {
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub eps: f32,
    pub rope_base_theta: f32,
    pub logit_softcap: Option<f32>,
}

impl GemmaLayerConfig {
    pub fn from_dims(hidden_dim: usize, q_dim: usize, k_dim: usize, ffn_dim: usize) -> Self {
        let head_dim = Self::infer_head_dim(q_dim, k_dim);
        let n_heads = if head_dim > 0 { q_dim / head_dim } else { 1 };
        let n_kv_heads = if head_dim > 0 {
            (k_dim / head_dim).max(1)
        } else {
            1
        };
        Self {
            hidden_dim,
            n_heads: n_heads.max(1),
            n_kv_heads,
            head_dim: head_dim.max(1),
            ffn_dim,
            eps: GEMMA_RMS_NORM_EPS,
            rope_base_theta: 10000.0,
            logit_softcap: None,
        }
    }

    fn infer_head_dim(q_dim: usize, k_dim: usize) -> usize {
        if k_dim == 0 || q_dim == 0 {
            return q_dim.max(k_dim).max(1);
        }
        // Prefer standard head_dim values that produce n_heads >= n_kv_heads >= 1
        // and where n_heads is a multiple of n_kv_heads (GQA requirement).
        for &candidate in &[256, 128, 64] {
            if q_dim % candidate == 0 && k_dim % candidate == 0 {
                let n_heads = q_dim / candidate;
                let n_kv_heads = k_dim / candidate;
                if n_kv_heads >= 1 && n_heads >= n_kv_heads && n_heads % n_kv_heads == 0 {
                    return candidate;
                }
            }
        }
        let g = gcd(q_dim, k_dim);
        if g >= 64 { g } else { g.max(1) }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn rms_norm_identity_weights() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let out = rms_norm(&x, &w, 1e-6);
        let expected_scale = 1.0 / ((9.0 + 16.0) / 2.0 + 1e-6f32).sqrt();
        assert!((out[0] - 3.0 * expected_scale).abs() < 1e-5);
        assert!((out[1] - 4.0 * expected_scale).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_chunked_inplace_matches_per_chunk_norm() {
        let weight = vec![1.0f32, 0.5, 1.5];
        let mut chunked = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            -1.0, 0.0, 1.0,
        ];
        let mut manual = chunked.clone();

        rms_norm_chunked_inplace(&mut chunked, 3, &weight, 1e-6);
        for chunk in manual.chunks_exact_mut(3) {
            rms_norm_inplace(chunk, &weight, 1e-6);
        }

        assert_eq!(chunked, manual);
    }

    #[test]
    fn softmax_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let s = softmax(&x);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn silu_zero() {
        let x = vec![0.0];
        let s = silu(&x);
        assert!((s[0]).abs() < 1e-6);
    }

    #[test]
    fn matmul_identity() {
        let mat = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![3.0, 5.0];
        let out = matmul(&mat, &input, 2, 2);
        assert_eq!(out, vec![3.0, 5.0]);
    }

    #[test]
    fn embedding_lookup_basic() {
        let embd = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let v = embedding_lookup(&embd, 1, 3, 2).unwrap();
        assert_eq!(v, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn argmax_basic() {
        let logits = vec![0.1, 0.9, 0.3];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn top_k_sample_returns_descending_bounded_logits() {
        let logits = vec![0.1, 0.9, 0.3, 0.8];
        assert_eq!(top_k_sample(&logits, 2), vec![(1, 0.9), (3, 0.8)]);
        assert_eq!(top_k_sample(&logits, 0), Vec::<(usize, f32)>::new());
        assert_eq!(
            top_k_sample(&logits, 10),
            vec![(1, 0.9), (3, 0.8), (2, 0.3), (0, 0.1)]
        );
    }

    #[test]
    fn rope_preserves_norm() {
        let head_dim = 8;
        let mut q = vec![1.0; head_dim];
        let mut k = vec![1.0; head_dim];
        let freqs = vec![1.0; head_dim / 2];
        let norm_before_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope_apply(&mut q, &mut k, &freqs, 0, 1, 1, head_dim);
        let norm_after_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before_q - norm_after_q).abs() < 1e-4);
    }

    #[test]
    fn proportional_rope_zero_pads_non_rotated_dims_across_full_head() {
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut k = q.clone();
        rope_apply_with_base_and_rotary_dim_mode(
            &mut q,
            &mut k,
            &[],
            1,
            1,
            1,
            8,
            10_000.0,
            4,
            true,
        );

        let theta0 = 1.0f32;
        let theta1 = 0.1f32;
        let expected_q0 = 1.0 * theta0.cos() - 5.0 * theta0.sin();
        let expected_q4 = 1.0 * theta0.sin() + 5.0 * theta0.cos();
        let expected_q1 = 2.0 * theta1.cos() - 6.0 * theta1.sin();
        let expected_q5 = 2.0 * theta1.sin() + 6.0 * theta1.cos();

        assert!((q[0] - expected_q0).abs() < 1e-5);
        assert!((q[4] - expected_q4).abs() < 1e-5);
        assert!((q[1] - expected_q1).abs() < 1e-5);
        assert!((q[5] - expected_q5).abs() < 1e-5);
        assert_eq!(q[2], 3.0);
        assert_eq!(q[3], 4.0);
        assert_eq!(q[6], 7.0);
        assert_eq!(q[7], 8.0);
        assert_eq!(k, q);
    }

    #[test]
    fn rope_apply_with_angles_matches_direct_application() {
        let mut q_direct = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut k_direct = q_direct.clone();
        let mut q_cached = q_direct.clone();
        let mut k_cached = k_direct.clone();
        let freqs = vec![1.0, 0.75];

        rope_apply_with_base_and_rotary_dim_mode(
            &mut q_direct,
            &mut k_direct,
            &freqs,
            7,
            1,
            1,
            8,
            1_000_000.0,
            4,
            true,
        );
        let angles = rope_angles(&freqs, 7, 8, 1_000_000.0, 4, true);
        rope_apply_with_angles(&mut q_cached, &mut k_cached, &angles, 1, 1, 8);

        assert_eq!(q_cached, q_direct);
        assert_eq!(k_cached, k_direct);
    }

    #[test]
    fn gqa_attention_single_token() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 1.0, 0.0];
        let out = gqa_attention(&q, &k, &v, 2, 1, head_dim);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[0..4], &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(&out[4..8], &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn gqa_attention_seq_with_limit_preserves_causality_for_shared_cache() {
        let q = vec![1.0, 0.0];
        let k_cache = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v_cache = vec![vec![10.0, 1.0], vec![100.0, 2.0]];

        let first_only =
            gqa_attention_seq_with_window_and_limit(&q, &k_cache, &v_cache, 1, 1, 2, None, Some(1));
        assert_eq!(first_only, vec![10.0, 1.0]);

        let both =
            gqa_attention_seq_with_window_and_limit(&q, &k_cache, &v_cache, 1, 1, 2, None, Some(2));
        assert!(both[0] > 10.0 && both[0] < 100.0);
        assert!(both[1] > 1.0 && both[1] < 2.0);
    }

    #[test]
    fn gqa_attention_seq_with_window_uses_recent_tokens_only() {
        let q = vec![1.0, 0.0];
        let k_cache = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let v_cache = vec![vec![10.0, 1.0], vec![100.0, 2.0]];

        let full =
            gqa_attention_seq_with_window_and_limit(&q, &k_cache, &v_cache, 1, 1, 2, None, None);
        let last_only =
            gqa_attention_seq_with_window_and_limit(&q, &k_cache, &v_cache, 1, 1, 2, Some(1), None);

        assert_eq!(last_only, vec![100.0, 2.0]);
        assert_eq!(full, vec![55.0, 1.5]);
    }

    #[test]
    fn gqa_attention_seq_matches_manual_softmax_weighting() {
        let q = vec![1.0, 0.0];
        let k_cache = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let v_cache = vec![vec![10.0, 1.0], vec![20.0, 2.0]];

        let out =
            gqa_attention_seq_with_window_and_limit(&q, &k_cache, &v_cache, 1, 1, 2, None, None);
        let w0 = 1.0f32.exp();
        let w1 = 0.0f32.exp();
        let denom = w0 + w1;
        let expected0 = (w0 * 10.0 + w1 * 20.0) / denom;
        let expected1 = (w0 * 1.0 + w1 * 2.0) / denom;

        assert!((out[0] - expected0).abs() < 1e-5);
        assert!((out[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn matmul_quantized_range_matches_constructed_q4_k_row() {
        let mut row =
            vec![0u8; quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap()];
        row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row[4] = 1;
        row[5] = 1;
        row[6] = 1;
        row[7] = 1;
        for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
            *byte = 0x11;
        }

        let input = vec![1.0f32; quants::QK_K];
        let output =
            matmul_quantized_range(quants::GGML_TYPE_Q4_K, &row, &input, 0, 1, quants::QK_K)
                .unwrap();

        assert_eq!(output, vec![128.0]);
    }

    #[test]
    fn matmul_quantized_many_range_matches_repeated_single_input() {
        let mut row =
            vec![0u8; quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap()];
        row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row[4] = 1;
        row[5] = 1;
        row[6] = 1;
        row[7] = 1;
        for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
            *byte = 0x11;
        }

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let outputs =
            matmul_quantized_many_range(quants::GGML_TYPE_Q4_K, &row, &inputs, 0, 1, quants::QK_K)
                .unwrap();

        assert_eq!(outputs, vec![vec![128.0], vec![64.0]]);
    }

    #[test]
    fn matmul_quantized_range_matches_constructed_q6_k_row() {
        let mut row =
            vec![0u8; quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap()];
        let row_len = row.len();
        row[row_len - 2..].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for scale in row[quants::QK_K / 2 + quants::QK_K / 4
            ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
            .iter_mut()
        {
            *scale = 1;
        }

        let input = vec![1.0f32; quants::QK_K];
        let output =
            matmul_quantized_range(quants::GGML_TYPE_Q6_K, &row, &input, 0, 1, quants::QK_K)
                .unwrap();

        assert_eq!(output, vec![-32.0 * quants::QK_K as f32]);
    }

    #[test]
    fn matmul_quantized_many_pair_q4_k_range_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let mut row_a = vec![0u8; bytes_per_row];
        row_a[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row_a[4] = 1;
        row_a[5] = 1;
        row_a[6] = 1;
        row_a[7] = 1;
        for byte in row_a[4 + quants::K_SCALE_SIZE..].iter_mut() {
            *byte = 0x11;
        }

        let mut row_b = row_a.clone();
        row_b[..2].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let separate_a = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let (paired_a, paired_b) =
            matmul_quantized_many_pair_q4_k_range(&row_a, &row_b, &inputs, 0, 1, quants::QK_K)
                .unwrap();

        assert_eq!(paired_a, separate_a);
        assert_eq!(paired_b, separate_b);
    }

    #[test]
    fn matmul_quantized_many_pair_q4_k_range_two_row_pairs_match_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let row_a0 = make_row(1.0);
        let row_a1 = make_row(2.0);
        let row_b0 = make_row(3.0);
        let row_b1 = make_row(4.0);

        let mut raw_a = row_a0.clone();
        raw_a.extend_from_slice(&row_a1);
        let mut raw_b = row_b0.clone();
        raw_b.extend_from_slice(&row_b1);

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let (paired_a, paired_b) =
            matmul_quantized_many_pair_q4_k_range(&raw_a, &raw_b, &inputs, 0, 2, quants::QK_K)
                .unwrap();

        let separate_a0 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a0,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_a1 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a1,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b0 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b0,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b1 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b1,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        assert_eq!(paired_a[0], vec![separate_a0[0][0], separate_a1[0][0]]);
        assert_eq!(paired_a[1], vec![separate_a0[1][0], separate_a1[1][0]]);
        assert_eq!(paired_b[0], vec![separate_b0[0][0], separate_b1[0][0]]);
        assert_eq!(paired_b[1], vec![separate_b0[1][0], separate_b1[1][0]]);
    }

    #[test]
    fn matmul_quantized_many_pair_q4_k_range_six_inputs_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let row_a0 = make_row(1.0);
        let row_a1 = make_row(2.0);
        let row_b0 = make_row(3.0);
        let row_b1 = make_row(4.0);

        let mut raw_a = row_a0.clone();
        raw_a.extend_from_slice(&row_a1);
        let mut raw_b = row_b0.clone();
        raw_b.extend_from_slice(&row_b1);

        let inputs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i + j) % 11) as f32 - 5.0)
                    .collect()
            })
            .collect();
        let (paired_a, paired_b) =
            matmul_quantized_many_pair_q4_k_range(&raw_a, &raw_b, &inputs, 0, 2, quants::QK_K)
                .unwrap();

        let separate_a0 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a0,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_a1 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a1,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b0 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b0,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b1 = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b1,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        for input_idx in 0..inputs.len() {
            assert_eq!(
                paired_a[input_idx],
                vec![separate_a0[input_idx][0], separate_a1[input_idx][0]]
            );
            assert_eq!(
                paired_b[input_idx],
                vec![separate_b0[input_idx][0], separate_b1[input_idx][0]]
            );
        }
    }

    #[test]
    fn matmul_many_range_dense_pairing_matches_manual() {
        let matrix = vec![
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            2.0, 1.0, 0.0, -1.0, //
        ];
        let inputs = vec![vec![1.0f32, 0.0, 1.0, 0.0], vec![0.5f32, 1.0, -1.0, 2.0]];
        let outputs = matmul_many_range(&matrix, &inputs, 0, 3, 4);

        assert_eq!(outputs[0], vec![4.0, 12.0, 2.0]);
        assert_eq!(outputs[1], vec![7.5, 17.5, 0.0]);
    }

    #[test]
    fn matmul_many_range_dense_six_inputs_matches_manual() {
        let matrix = vec![
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            2.0, 1.0, 0.0, -1.0, //
        ];
        let inputs = vec![
            vec![1.0f32, 0.0, 1.0, 0.0],
            vec![0.5f32, 1.0, -1.0, 2.0],
            vec![0.0f32, 1.0, 0.0, 1.0],
            vec![2.0f32, -1.0, 0.5, 0.0],
            vec![-1.0f32, 1.0, 1.0, -1.0],
            vec![0.25f32, 0.5, 0.75, 1.0],
        ];
        let outputs = matmul_many_range(&matrix, &inputs, 0, 3, 4);

        let manual: Vec<Vec<f32>> = inputs
            .iter()
            .map(|input| {
                (0..3)
                    .map(|row_idx| {
                        let row = &matrix[row_idx * 4..(row_idx + 1) * 4];
                        row.iter().zip(input.iter()).map(|(m, x)| m * x).sum()
                    })
                    .collect()
            })
            .collect();

        assert_eq!(outputs, manual);
    }

    #[test]
    fn matmul_many_refs_range_token_major_matches_nested_outputs() {
        let matrix = vec![
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            2.0, 1.0, 0.0, -1.0, //
        ];
        let inputs = vec![
            vec![1.0f32, 0.0, 1.0, 0.0],
            vec![0.5f32, 1.0, -1.0, 2.0],
            vec![0.0f32, 1.0, 0.0, 1.0],
        ];
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();

        let nested = matmul_many_refs_range(&matrix, &input_refs, 0, 3, 4);
        let slab = matmul_many_refs_range_token_major(&matrix, &input_refs, 0, 3, 4);
        let expected: Vec<f32> = nested.into_iter().flatten().collect();

        assert_eq!(slab, expected);
    }

    #[test]
    fn matmul_quantized_many_range_q6_k_pairing_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let mut row_a = vec![0u8; bytes_per_row];
        row_a[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for scale in row_a[quants::QK_K / 2 + quants::QK_K / 4
            ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
            .iter_mut()
        {
            *scale = 1;
        }

        let mut row_b = row_a.clone();
        row_b[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());

        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let combined =
            matmul_quantized_many_range(quants::GGML_TYPE_Q6_K, &raw, &inputs, 0, 2, quants::QK_K)
                .unwrap();
        let separate_a = matmul_quantized_many_range(
            quants::GGML_TYPE_Q6_K,
            &row_a,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b = matmul_quantized_many_range(
            quants::GGML_TYPE_Q6_K,
            &row_b,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        assert_eq!(combined[0], vec![separate_a[0][0], separate_b[0][0]]);
        assert_eq!(combined[1], vec![separate_a[1][0], separate_b[1][0]]);
    }

    #[test]
    fn matmul_quantized_many_range_q6_k_pairing_six_inputs_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let row_a = make_row(1.0);
        let row_b = make_row(2.0);
        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);

        let inputs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 5 + j) % 17) as f32 - 8.0)
                    .collect()
            })
            .collect();
        let combined =
            matmul_quantized_many_range(quants::GGML_TYPE_Q6_K, &raw, &inputs, 0, 2, quants::QK_K)
                .unwrap();
        let separate_a = matmul_quantized_many_range(
            quants::GGML_TYPE_Q6_K,
            &row_a,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b = matmul_quantized_many_range(
            quants::GGML_TYPE_Q6_K,
            &row_b,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        for input_idx in 0..inputs.len() {
            assert_eq!(
                combined[input_idx],
                vec![separate_a[input_idx][0], separate_b[input_idx][0]]
            );
        }
    }

    #[test]
    fn matmul_quantized_many_range_q6_k_four_row_six_inputs_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let row_a = make_row(1.0);
        let row_b = make_row(2.0);
        let row_c = make_row(3.0);
        let row_d = make_row(4.0);
        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);
        raw.extend_from_slice(&row_c);
        raw.extend_from_slice(&row_d);

        let inputs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 7 + j) % 19) as f32 - 9.0)
                    .collect()
            })
            .collect();
        let combined =
            matmul_quantized_many_range(quants::GGML_TYPE_Q6_K, &raw, &inputs, 0, 4, quants::QK_K)
                .unwrap();
        let separate_rows = [&row_a, &row_b, &row_c, &row_d]
            .into_iter()
            .map(|row| {
                matmul_quantized_many_range(
                    quants::GGML_TYPE_Q6_K,
                    row,
                    &inputs,
                    0,
                    1,
                    quants::QK_K,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        for input_idx in 0..inputs.len() {
            assert_eq!(
                combined[input_idx],
                vec![
                    separate_rows[0][input_idx][0],
                    separate_rows[1][input_idx][0],
                    separate_rows[2][input_idx][0],
                    separate_rows[3][input_idx][0],
                ]
            );
        }
    }

    #[test]
    fn matmul_quantized_many_refs_range_token_major_matches_nested_outputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let row_a = make_row(1.0);
        let row_b = make_row(2.0);
        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let nested = matmul_quantized_many_refs_range(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input_refs,
            0,
            2,
            quants::QK_K,
        )
        .unwrap();
        let slab = matmul_quantized_many_refs_range_token_major(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input_refs,
            0,
            2,
            quants::QK_K,
        )
        .unwrap();
        let expected: Vec<f32> = nested.into_iter().flatten().collect();

        assert_eq!(slab, expected);
    }

    #[test]
    fn matmul_quantized_many_refs_partial_input_range_q6_k_matches_full_output() {
        let block_bytes = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_block = |scale: f32| {
            let mut row = vec![0u8; block_bytes];
            row[block_bytes - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let mut row_a = make_block(1.0);
        row_a.extend_from_slice(&make_block(2.0));
        let mut row_b = make_block(3.0);
        row_b.extend_from_slice(&make_block(4.0));
        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);

        let in_dim = quants::QK_K * 2;
        let inputs: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                (0..in_dim)
                    .map(|j| ((i * 5 + j) % 17) as f32 - 8.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range_token_major(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input_refs,
            0,
            2,
            in_dim,
        )
        .unwrap();

        let chunk0: Vec<&[f32]> = inputs.iter().map(|input| &input[..quants::QK_K]).collect();
        let chunk1: Vec<&[f32]> = inputs.iter().map(|input| &input[quants::QK_K..]).collect();
        let mut accumulated = vec![0.0f32; inputs.len() * 2];
        matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &chunk0,
            0,
            2,
            in_dim,
            0,
            quants::QK_K,
            &mut accumulated,
        )
        .unwrap();
        matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &chunk1,
            0,
            2,
            in_dim,
            quants::QK_K,
            quants::QK_K,
            &mut accumulated,
        )
        .unwrap();

        assert_eq!(accumulated, full);
    }

    #[test]
    fn matmul_quantized_many_refs_partial_input_range_q4_k_matches_full_output() {
        let block_bytes = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_block = |scale: f32| {
            let mut row = vec![0u8; block_bytes];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let mut row_a = make_block(1.0);
        row_a.extend_from_slice(&make_block(2.0));
        let mut row_b = make_block(3.0);
        row_b.extend_from_slice(&make_block(4.0));
        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);

        let in_dim = quants::QK_K * 2;
        let inputs: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                (0..in_dim)
                    .map(|j| ((i * 7 + j) % 19) as f32 - 9.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range_token_major(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &input_refs,
            0,
            2,
            in_dim,
        )
        .unwrap();

        let chunk0: Vec<&[f32]> = inputs.iter().map(|input| &input[..quants::QK_K]).collect();
        let chunk1: Vec<&[f32]> = inputs.iter().map(|input| &input[quants::QK_K..]).collect();
        let mut accumulated = vec![0.0f32; inputs.len() * 2];
        matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &chunk0,
            0,
            2,
            in_dim,
            0,
            quants::QK_K,
            &mut accumulated,
        )
        .unwrap();
        matmul_quantized_many_refs_partial_input_range_token_major_accumulate(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &chunk1,
            0,
            2,
            in_dim,
            quants::QK_K,
            quants::QK_K,
            &mut accumulated,
        )
        .unwrap();

        assert_eq!(accumulated, full);
    }

    #[test]
    fn matmul_quantized_many_range_q4_k_four_row_path_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let mut row_a = vec![0u8; bytes_per_row];
        row_a[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row_a[4] = 1;
        row_a[5] = 1;
        row_a[6] = 1;
        row_a[7] = 1;
        for byte in row_a[4 + quants::K_SCALE_SIZE..].iter_mut() {
            *byte = 0x11;
        }

        let mut row_b = row_a.clone();
        row_b[..2].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        let mut row_c = row_a.clone();
        row_c[..2].copy_from_slice(&f16::from_f32(3.0).to_bits().to_le_bytes());
        let mut row_d = row_a.clone();
        row_d[..2].copy_from_slice(&f16::from_f32(4.0).to_bits().to_le_bytes());

        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);
        raw.extend_from_slice(&row_c);
        raw.extend_from_slice(&row_d);

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let combined =
            matmul_quantized_many_range(quants::GGML_TYPE_Q4_K, &raw, &inputs, 0, 4, quants::QK_K)
                .unwrap();
        let separate_a = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_c = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_c,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_d = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_d,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        assert_eq!(
            combined[0],
            vec![
                separate_a[0][0],
                separate_b[0][0],
                separate_c[0][0],
                separate_d[0][0]
            ]
        );
        assert_eq!(
            combined[1],
            vec![
                separate_a[1][0],
                separate_b[1][0],
                separate_c[1][0],
                separate_d[1][0]
            ]
        );
    }

    #[test]
    fn matmul_quantized_many_range_q4_k_four_row_path_six_inputs_matches_separate_calls() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let row_a = make_row(1.0);
        let row_b = make_row(2.0);
        let row_c = make_row(3.0);
        let row_d = make_row(4.0);

        let mut raw = row_a.clone();
        raw.extend_from_slice(&row_b);
        raw.extend_from_slice(&row_c);
        raw.extend_from_slice(&row_d);

        let inputs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 3 + j) % 13) as f32 - 6.0)
                    .collect()
            })
            .collect();
        let combined =
            matmul_quantized_many_range(quants::GGML_TYPE_Q4_K, &raw, &inputs, 0, 4, quants::QK_K)
                .unwrap();
        let separate_a = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_a,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_b = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_b,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_c = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_c,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();
        let separate_d = matmul_quantized_many_range(
            quants::GGML_TYPE_Q4_K,
            &row_d,
            &inputs,
            0,
            1,
            quants::QK_K,
        )
        .unwrap();

        for input_idx in 0..inputs.len() {
            assert_eq!(
                combined[input_idx],
                vec![
                    separate_a[input_idx][0],
                    separate_b[input_idx][0],
                    separate_c[input_idx][0],
                    separate_d[input_idx][0],
                ]
            );
        }
    }

    #[test]
    fn matmul_quantized_many_pair_q4_k_token_major_matches_nested_outputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let row_a0 = make_row(1.0);
        let row_a1 = make_row(2.0);
        let row_b0 = make_row(3.0);
        let row_b1 = make_row(4.0);

        let mut raw_a = row_a0.clone();
        raw_a.extend_from_slice(&row_a1);
        let mut raw_b = row_b0.clone();
        raw_b.extend_from_slice(&row_b1);

        let inputs = vec![vec![1.0f32; quants::QK_K], vec![0.5f32; quants::QK_K]];
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();

        let (nested_a, nested_b) = matmul_quantized_many_pair_q4_k_refs_range(
            &raw_a,
            &raw_b,
            &input_refs,
            0,
            2,
            quants::QK_K,
        )
        .unwrap();
        let (slab_a, slab_b) = matmul_quantized_many_pair_q4_k_refs_range_token_major(
            &raw_a,
            &raw_b,
            &input_refs,
            0,
            2,
            quants::QK_K,
        )
        .unwrap();

        let expected_a: Vec<f32> = nested_a.into_iter().flatten().collect();
        let expected_b: Vec<f32> = nested_b.into_iter().flatten().collect();
        assert_eq!(slab_a, expected_a);
        assert_eq!(slab_b, expected_b);
    }

    #[test]
    fn matmul_quantized_many_refs_range_token_major_q4_k_chunked_inputs_match_single_inputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));
        raw.extend_from_slice(&make_row(3.0));
        raw.extend_from_slice(&make_row(4.0));

        let inputs: Vec<Vec<f32>> = (0..7)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 7 + j) % 19) as f32 - 9.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range_token_major(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &input_refs,
            0,
            4,
            quants::QK_K,
        )
        .unwrap();

        let mut stitched = vec![0.0f32; inputs.len() * 4];
        for (input_idx, input) in inputs.iter().enumerate() {
            let single = matmul_quantized_many_refs_range_token_major(
                quants::GGML_TYPE_Q4_K,
                &raw,
                &[input.as_slice()],
                0,
                4,
                quants::QK_K,
            )
            .unwrap();
            stitched[input_idx * 4..(input_idx + 1) * 4].copy_from_slice(&single);
        }

        assert_close(&full, &stitched, 1e-3);
    }

    #[test]
    fn matmul_quantized_many_refs_range_q4_k_chunked_inputs_match_single_inputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));
        raw.extend_from_slice(&make_row(3.0));
        raw.extend_from_slice(&make_row(4.0));

        let inputs: Vec<Vec<f32>> = (0..7)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 7 + j) % 19) as f32 - 9.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &input_refs,
            0,
            4,
            quants::QK_K,
        )
        .unwrap();

        let mut stitched = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let single = matmul_quantized_many_refs_range(
                quants::GGML_TYPE_Q4_K,
                &raw,
                &[input.as_slice()],
                0,
                4,
                quants::QK_K,
            )
            .unwrap();
            stitched.push(single[0].clone());
        }

        assert_eq!(full.len(), stitched.len());
        for (full_values, single_values) in full.iter().zip(stitched.iter()) {
            assert_close(full_values, single_values, 1e-3);
        }
    }

    #[test]
    fn matmul_quantized_many_refs_range_token_major_q6_k_chunked_inputs_match_single_inputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));
        raw.extend_from_slice(&make_row(3.0));
        raw.extend_from_slice(&make_row(4.0));

        let inputs: Vec<Vec<f32>> = (0..7)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 5 + j) % 17) as f32 - 8.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range_token_major(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input_refs,
            0,
            4,
            quants::QK_K,
        )
        .unwrap();

        let mut stitched = vec![0.0f32; inputs.len() * 4];
        for (input_idx, input) in inputs.iter().enumerate() {
            let single = matmul_quantized_many_refs_range_token_major(
                quants::GGML_TYPE_Q6_K,
                &raw,
                &[input.as_slice()],
                0,
                4,
                quants::QK_K,
            )
            .unwrap();
            stitched[input_idx * 4..(input_idx + 1) * 4].copy_from_slice(&single);
        }

        assert_close(&full, &stitched, 1e-3);
    }

    #[test]
    fn matmul_quantized_many_refs_range_q6_k_chunked_inputs_match_single_inputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));
        raw.extend_from_slice(&make_row(3.0));
        raw.extend_from_slice(&make_row(4.0));

        let inputs: Vec<Vec<f32>> = (0..7)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 5 + j) % 17) as f32 - 8.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let full = matmul_quantized_many_refs_range(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input_refs,
            0,
            4,
            quants::QK_K,
        )
        .unwrap();

        let mut stitched = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let single = matmul_quantized_many_refs_range(
                quants::GGML_TYPE_Q6_K,
                &raw,
                &[input.as_slice()],
                0,
                4,
                quants::QK_K,
            )
            .unwrap();
            stitched.push(single[0].clone());
        }

        assert_eq!(full.len(), stitched.len());
        for (full_values, single_values) in full.iter().zip(stitched.iter()) {
            assert_close(full_values, single_values, 1e-3);
        }
    }

    #[test]
    fn matmul_quantized_many_pair_q4_k_token_major_chunked_inputs_match_single_inputs() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let mut raw_a = make_row(1.0);
        raw_a.extend_from_slice(&make_row(2.0));
        raw_a.extend_from_slice(&make_row(3.0));
        raw_a.extend_from_slice(&make_row(4.0));
        let mut raw_b = make_row(5.0);
        raw_b.extend_from_slice(&make_row(6.0));
        raw_b.extend_from_slice(&make_row(7.0));
        raw_b.extend_from_slice(&make_row(8.0));

        let inputs: Vec<Vec<f32>> = (0..7)
            .map(|i| {
                (0..quants::QK_K)
                    .map(|j| ((i * 3 + j) % 13) as f32 - 6.0)
                    .collect()
            })
            .collect();
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let (full_a, full_b) = matmul_quantized_many_pair_q4_k_refs_range_token_major(
            &raw_a,
            &raw_b,
            &input_refs,
            0,
            4,
            quants::QK_K,
        )
        .unwrap();

        let mut stitched_a = vec![0.0f32; inputs.len() * 4];
        let mut stitched_b = vec![0.0f32; inputs.len() * 4];
        for (input_idx, input) in inputs.iter().enumerate() {
            let (single_a, single_b) = matmul_quantized_many_pair_q4_k_refs_range_token_major(
                &raw_a,
                &raw_b,
                &[input.as_slice()],
                0,
                4,
                quants::QK_K,
            )
            .unwrap();
            stitched_a[input_idx * 4..(input_idx + 1) * 4].copy_from_slice(&single_a);
            stitched_b[input_idx * 4..(input_idx + 1) * 4].copy_from_slice(&single_b);
        }

        assert_close(&full_a, &stitched_a, 1e-3);
        assert_close(&full_b, &stitched_b, 1e-3);
    }

    #[test]
    fn matmul_raw_top_k_tracks_argmax_and_top_scores() {
        let rows = [[1.0f32, 0.0], [0.0, 2.0], [3.0, 0.0]];
        let mut raw = Vec::new();
        for row in rows {
            for value in row {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }

        let result =
            matmul_raw_top_k(quants::GGML_TYPE_F32, &raw, &[1.0, 1.0], 0, 3, 2, 2, None).unwrap();

        assert_eq!(result.argmax_idx, 2);
        assert_eq!(result.argmax_score, 3.0);
        assert_eq!(result.top, vec![(2, 3.0), (1, 2.0)]);
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tol,
                "value mismatch at {idx}: lhs={a} rhs={b} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn matmul_raw_top_k_q4_k_four_row_fast_path_matches_expected_order() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q4_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            row[4] = 1;
            row[5] = 1;
            row[6] = 1;
            row[7] = 1;
            for byte in row[4 + quants::K_SCALE_SIZE..].iter_mut() {
                *byte = 0x11;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));
        raw.extend_from_slice(&make_row(3.0));
        raw.extend_from_slice(&make_row(4.0));

        let input = vec![1.0f32; quants::QK_K];
        let result = matmul_raw_top_k(
            quants::GGML_TYPE_Q4_K,
            &raw,
            &input,
            0,
            4,
            quants::QK_K,
            2,
            None,
        )
        .unwrap();

        assert_eq!(result.argmax_idx, 3);
        assert_eq!(result.top, vec![(3, 512.0), (2, 384.0)]);
    }

    #[test]
    fn matmul_raw_top_k_q6_k_pair_fast_path_matches_expected_order() {
        let bytes_per_row = quants::bytes_per_row(quants::GGML_TYPE_Q6_K, quants::QK_K).unwrap();
        let make_row = |scale: f32| {
            let mut row = vec![0u8; bytes_per_row];
            row[bytes_per_row - 2..].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for scale in row[quants::QK_K / 2 + quants::QK_K / 4
                ..quants::QK_K / 2 + quants::QK_K / 4 + quants::QK_K / 16]
                .iter_mut()
            {
                *scale = 1;
            }
            row
        };

        let mut raw = make_row(1.0);
        raw.extend_from_slice(&make_row(2.0));

        let input = vec![1.0f32; quants::QK_K];
        let result = matmul_raw_top_k(
            quants::GGML_TYPE_Q6_K,
            &raw,
            &input,
            0,
            2,
            quants::QK_K,
            2,
            None,
        )
        .unwrap();

        assert_eq!(result.argmax_idx, 0);
        assert_eq!(
            result.top,
            vec![
                (0, -32.0 * quants::QK_K as f32),
                (1, -64.0 * quants::QK_K as f32)
            ]
        );
    }

    #[test]
    fn gelu_pytorch_tanh_mul_inplace_matches_separate_ops() {
        let mut gate = vec![-1.5f32, 0.0, 2.0];
        let up = vec![2.0f32, 3.0, 4.0];
        let expected = vec_mul(&gelu_pytorch_tanh(&gate), &up);
        gelu_pytorch_tanh_mul_inplace(&mut gate, &up);
        for (actual, expected) in gate.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn per_head_rms_norm_basic() {
        let mut x = vec![3.0, 4.0, 1.0, 2.0];
        let w = vec![1.0, 1.0];
        per_head_rms_norm(&mut x, &w, 2, 2);
        let norm0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let norm1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
        assert!((norm0 - (2.0f32).sqrt()).abs() < 0.01);
        assert!((norm1 - (2.0f32).sqrt()).abs() < 0.01);
    }

    #[test]
    fn per_head_rms_norm_no_scale_normalizes_without_extra_gain() {
        let mut x = vec![3.0, 4.0, 1.0, 2.0];
        per_head_rms_norm_no_scale(&mut x, 2, 2);
        let norm0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let norm1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
        assert!((norm0 - (2.0f32).sqrt()).abs() < 0.01);
        assert!((norm1 - (2.0f32).sqrt()).abs() < 0.01);
    }
}
