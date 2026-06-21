use anyhow::{Result, bail};
use half::f16;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddvq_f32, vand_u8, vcvtq_f32_u32, vdup_n_u8, vdupq_n_f32, vfmaq_f32,
    vget_high_u16, vget_low_u16, vld1_u8, vld1q_f32, vmovl_u8, vmovl_u16, vmulq_n_f32, vshr_n_u8,
    vsubq_f32,
};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_BF16: u32 = 30;

const BLOCK_Q4_K_SIZE: usize = 2 * 2 + K_SCALE_SIZE + QK_K / 2;
const BLOCK_Q5_K_SIZE: usize = 2 * 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
const BLOCK_Q6_K_SIZE: usize = 2 + QK_K / 16 + 3 * QK_K / 4;

pub fn ggml_type_name(ggml_type: u32) -> &'static str {
    match ggml_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

pub fn dequantize_tensor(ggml_type: u32, bytes: &[u8]) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => dequantize_f32_tensor(bytes),
        GGML_TYPE_F16 => dequantize_f16_tensor(bytes),
        GGML_TYPE_Q4_K => dequantize_q4_k_tensor(bytes),
        GGML_TYPE_Q5_K => dequantize_q5_k_tensor(bytes),
        GGML_TYPE_Q6_K => dequantize_q6_k_tensor(bytes),
        GGML_TYPE_BF16 => dequantize_bf16_tensor(bytes),
        _ => bail!(
            "Unsupported GGML type {} ({})",
            ggml_type,
            ggml_type_name(ggml_type)
        ),
    }
}

pub fn dequantize_f32_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("F32 tensor length {} is not divisible by 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn dequantize_f16_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("F16 tensor length {} is not divisible by 2", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(fp16_to_f32([chunk[0], chunk[1]]));
    }
    Ok(out)
}

pub fn dequantize_bf16_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("BF16 tensor length {} is not divisible by 2", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16_to_f32(bits));
    }
    Ok(out)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn dequantize_q4_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q4_K_SIZE != 0 {
        bail!(
            "Q4_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q4_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q4_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
        dequantize_q4_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q5_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q5_K_SIZE != 0 {
        bail!(
            "Q5_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q5_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q5_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q5_K_SIZE).enumerate() {
        dequantize_q5_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q5_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q5_K_SIZE || out.len() != QK_K {
        bail!("Q5_K block decode shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
    let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q_lo = qs[q_offset + l];
            let qh_byte = qh[l];
            let hbit1 = if qh_byte & u1 != 0 { 16 } else { 0 };
            let hbit2 = if qh_byte & u2 != 0 { 16 } else { 0 };
            out[j + l * 2] = d1 * ((q_lo & 0x0F) as f32 + hbit1 as f32) - m1;
            out[j + l * 2 + 1] = d2 * ((q_lo >> 4) as f32 + hbit2 as f32) - m2;
        }
        q_offset += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    Ok(())
}

pub fn dequantize_q6_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q6_K_SIZE != 0 {
        bail!(
            "Q6_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q6_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q6_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
        dequantize_q6_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q4_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q4_K_SIZE || out.len() != QK_K {
        bail!("Q4_K block decode shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q = qs[q_offset + l];
            out[j + l] = d1 * (q & 0x0F) as f32 - m1;
            out[j + 32 + l] = d2 * (q >> 4) as f32 - m2;
        }
        q_offset += 32;
        is += 2;
    }
    Ok(())
}

pub fn dequantize_q6_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q6_K_SIZE || out.len() != QK_K {
        bail!("Q6_K block decode shape mismatch");
    }
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales = &block[scales_start..scales_start + QK_K / 16];
    let d = fp16_to_f32([block[BLOCK_Q6_K_SIZE - 2], block[BLOCK_Q6_K_SIZE - 1]]);

    let mut y_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let qh_byte = qh[qh_offset + l];
            let q1 =
                (((ql[ql_offset + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2 = (((ql[ql_offset + 32 + l] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32
                - 32) as f32;
            let q3 = ((((ql[ql_offset + l] >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4)) as i32
                - 32) as f32;
            let q4 = ((((ql[ql_offset + 32 + l] >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4))
                as i32
                - 32) as f32;

            out[y_offset + l] = d * (scales[sc_offset + is] as i8 as f32) * q1;
            out[y_offset + 32 + l] = d * (scales[sc_offset + is + 2] as i8 as f32) * q2;
            out[y_offset + 64 + l] = d * (scales[sc_offset + is + 4] as i8 as f32) * q3;
            out[y_offset + 96 + l] = d * (scales[sc_offset + is + 6] as i8 as f32) * q4;
        }
        y_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }
    Ok(())
}

#[inline(always)]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[inline(always)]
fn fp16_to_f32(bytes: [u8; 2]) -> f32 {
    f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn decode_q4_coeffs8(
    qs_ptr: *const u8,
    lane_offset: usize,
    d_left: f32,
    m_left: f32,
    d_right: f32,
    m_right: f32,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let q = unsafe { vld1_u8(qs_ptr.add(lane_offset)) };
    let low = unsafe { vand_u8(q, vdup_n_u8(0x0F)) };
    let high = unsafe { vshr_n_u8(q, 4) };

    let low_wide = unsafe { vmovl_u8(low) };
    let high_wide = unsafe { vmovl_u8(high) };

    let low_lo = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(low_wide))) };
    let low_hi = unsafe { vcvtq_f32_u32(vmovl_u16(vget_high_u16(low_wide))) };
    let high_lo = unsafe { vcvtq_f32_u32(vmovl_u16(vget_low_u16(high_wide))) };
    let high_hi = unsafe { vcvtq_f32_u32(vmovl_u16(vget_high_u16(high_wide))) };

    let left_bias = unsafe { vdupq_n_f32(m_left) };
    let right_bias = unsafe { vdupq_n_f32(m_right) };
    (
        unsafe { vsubq_f32(vmulq_n_f32(low_lo, d_left), left_bias) },
        unsafe { vsubq_f32(vmulq_n_f32(low_hi, d_left), left_bias) },
        unsafe { vsubq_f32(vmulq_n_f32(high_lo, d_right), right_bias) },
        unsafe { vsubq_f32(vmulq_n_f32(high_hi, d_right), right_bias) },
    )
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_many_q4_k_four_blocks_with_offset_six_neon(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let d_c = fp16_to_f32([block_c[0], block_c[1]]);
    let dmin_c = fp16_to_f32([block_c[2], block_c[3]]);
    let scales_c = &block_c[4..4 + K_SCALE_SIZE];
    let qs_c = &block_c[4 + K_SCALE_SIZE..];

    let d_d = fp16_to_f32([block_d[0], block_d[1]]);
    let dmin_d = fp16_to_f32([block_d[2], block_d[3]]);
    let scales_d = &block_d[4..4 + K_SCALE_SIZE];
    let qs_d = &block_d[4 + K_SCALE_SIZE..];

    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];
    let mut sum_a0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d5_v = unsafe { vdupq_n_f32(0.0) };
    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let input_base = input_offset + j;
        let input0_ptr = unsafe { input0.as_ptr().add(input_base) };
        let input1_ptr = unsafe { input1.as_ptr().add(input_base) };
        let input2_ptr = unsafe { input2.as_ptr().add(input_base) };
        let input3_ptr = unsafe { input3.as_ptr().add(input_base) };
        let input4_ptr = unsafe { input4.as_ptr().add(input_base) };
        let input5_ptr = unsafe { input5.as_ptr().add(input_base) };
        let qs_a_ptr = unsafe { qs_a.as_ptr().add(q_offset) };
        let qs_b_ptr = unsafe { qs_b.as_ptr().add(q_offset) };
        let qs_c_ptr = unsafe { qs_c.as_ptr().add(q_offset) };
        let qs_d_ptr = unsafe { qs_d.as_ptr().add(q_offset) };

        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        let (sc1_c, m1_c) = get_scale_min_k4(is, scales_c);
        let d1_c = d_c * sc1_c as f32;
        let m1_c = dmin_c * m1_c as f32;
        let (sc2_c, m2_c) = get_scale_min_k4(is + 1, scales_c);
        let d2_c = d_c * sc2_c as f32;
        let m2_c = dmin_c * m2_c as f32;

        let (sc1_d, m1_d) = get_scale_min_k4(is, scales_d);
        let d1_d = d_d * sc1_d as f32;
        let m1_d = dmin_d * m1_d as f32;
        let (sc2_d, m2_d) = get_scale_min_k4(is + 1, scales_d);
        let d2_d = d_d * sc2_d as f32;
        let m2_d = dmin_d * m2_d as f32;

        for l in (0..32).step_by(8) {
            let (coeff_left_a_lo, coeff_left_a_hi, coeff_right_a_lo, coeff_right_a_hi) =
                unsafe { decode_q4_coeffs8(qs_a_ptr, l, d1_a, m1_a, d2_a, m2_a) };
            let (coeff_left_b_lo, coeff_left_b_hi, coeff_right_b_lo, coeff_right_b_hi) =
                unsafe { decode_q4_coeffs8(qs_b_ptr, l, d1_b, m1_b, d2_b, m2_b) };
            let (coeff_left_c_lo, coeff_left_c_hi, coeff_right_c_lo, coeff_right_c_hi) =
                unsafe { decode_q4_coeffs8(qs_c_ptr, l, d1_c, m1_c, d2_c, m2_c) };
            let (coeff_left_d_lo, coeff_left_d_hi, coeff_right_d_lo, coeff_right_d_hi) =
                unsafe { decode_q4_coeffs8(qs_d_ptr, l, d1_d, m1_d, d2_d, m2_d) };

            macro_rules! accum_input {
                ($sum_a:ident, $sum_b:ident, $sum_c:ident, $sum_d:ident, $ptr:ident, $lane:expr,
                 $coeff_left_a:ident, $coeff_right_a:ident, $coeff_left_b:ident, $coeff_right_b:ident,
                 $coeff_left_c:ident, $coeff_right_c:ident, $coeff_left_d:ident, $coeff_right_d:ident) => {{
                    let left = unsafe { vld1q_f32($ptr.add($lane)) };
                    let right = unsafe { vld1q_f32($ptr.add(32 + $lane)) };
                    $sum_a = unsafe {
                        vfmaq_f32(
                            vfmaq_f32($sum_a, $coeff_left_a, left),
                            $coeff_right_a,
                            right,
                        )
                    };
                    $sum_b = unsafe {
                        vfmaq_f32(
                            vfmaq_f32($sum_b, $coeff_left_b, left),
                            $coeff_right_b,
                            right,
                        )
                    };
                    $sum_c = unsafe {
                        vfmaq_f32(
                            vfmaq_f32($sum_c, $coeff_left_c, left),
                            $coeff_right_c,
                            right,
                        )
                    };
                    $sum_d = unsafe {
                        vfmaq_f32(
                            vfmaq_f32($sum_d, $coeff_left_d, left),
                            $coeff_right_d,
                            right,
                        )
                    };
                }};
            }

            accum_input!(
                sum_a0_v,
                sum_b0_v,
                sum_c0_v,
                sum_d0_v,
                input0_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );
            accum_input!(
                sum_a1_v,
                sum_b1_v,
                sum_c1_v,
                sum_d1_v,
                input1_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );
            accum_input!(
                sum_a2_v,
                sum_b2_v,
                sum_c2_v,
                sum_d2_v,
                input2_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );
            accum_input!(
                sum_a3_v,
                sum_b3_v,
                sum_c3_v,
                sum_d3_v,
                input3_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );
            accum_input!(
                sum_a4_v,
                sum_b4_v,
                sum_c4_v,
                sum_d4_v,
                input4_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );
            accum_input!(
                sum_a5_v,
                sum_b5_v,
                sum_c5_v,
                sum_d5_v,
                input5_ptr,
                l,
                coeff_left_a_lo,
                coeff_right_a_lo,
                coeff_left_b_lo,
                coeff_right_b_lo,
                coeff_left_c_lo,
                coeff_right_c_lo,
                coeff_left_d_lo,
                coeff_right_d_lo
            );

            let l_hi = l + 4;
            accum_input!(
                sum_a0_v,
                sum_b0_v,
                sum_c0_v,
                sum_d0_v,
                input0_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
            accum_input!(
                sum_a1_v,
                sum_b1_v,
                sum_c1_v,
                sum_d1_v,
                input1_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
            accum_input!(
                sum_a2_v,
                sum_b2_v,
                sum_c2_v,
                sum_d2_v,
                input2_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
            accum_input!(
                sum_a3_v,
                sum_b3_v,
                sum_c3_v,
                sum_d3_v,
                input3_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
            accum_input!(
                sum_a4_v,
                sum_b4_v,
                sum_c4_v,
                sum_d4_v,
                input4_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
            accum_input!(
                sum_a5_v,
                sum_b5_v,
                sum_c5_v,
                sum_d5_v,
                input5_ptr,
                l_hi,
                coeff_left_a_hi,
                coeff_right_a_hi,
                coeff_left_b_hi,
                coeff_right_b_hi,
                coeff_left_c_hi,
                coeff_right_c_hi,
                coeff_left_d_hi,
                coeff_right_d_hi
            );
        }

        q_offset += 32;
        is += 2;
    }

    sums_a[0] += unsafe { vaddvq_f32(sum_a0_v) };
    sums_a[1] += unsafe { vaddvq_f32(sum_a1_v) };
    sums_a[2] += unsafe { vaddvq_f32(sum_a2_v) };
    sums_a[3] += unsafe { vaddvq_f32(sum_a3_v) };
    sums_a[4] += unsafe { vaddvq_f32(sum_a4_v) };
    sums_a[5] += unsafe { vaddvq_f32(sum_a5_v) };
    sums_b[0] += unsafe { vaddvq_f32(sum_b0_v) };
    sums_b[1] += unsafe { vaddvq_f32(sum_b1_v) };
    sums_b[2] += unsafe { vaddvq_f32(sum_b2_v) };
    sums_b[3] += unsafe { vaddvq_f32(sum_b3_v) };
    sums_b[4] += unsafe { vaddvq_f32(sum_b4_v) };
    sums_b[5] += unsafe { vaddvq_f32(sum_b5_v) };
    sums_c[0] += unsafe { vaddvq_f32(sum_c0_v) };
    sums_c[1] += unsafe { vaddvq_f32(sum_c1_v) };
    sums_c[2] += unsafe { vaddvq_f32(sum_c2_v) };
    sums_c[3] += unsafe { vaddvq_f32(sum_c3_v) };
    sums_c[4] += unsafe { vaddvq_f32(sum_c4_v) };
    sums_c[5] += unsafe { vaddvq_f32(sum_c5_v) };
    sums_d[0] += unsafe { vaddvq_f32(sum_d0_v) };
    sums_d[1] += unsafe { vaddvq_f32(sum_d1_v) };
    sums_d[2] += unsafe { vaddvq_f32(sum_d2_v) };
    sums_d[3] += unsafe { vaddvq_f32(sum_d3_v) };
    sums_d[4] += unsafe { vaddvq_f32(sum_d4_v) };
    sums_d[5] += unsafe { vaddvq_f32(sum_d5_v) };
    Ok(())
}

pub fn bytes_per_row(ggml_type: u32, row_elements: usize) -> Result<usize> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(row_elements * 4),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => Ok(row_elements * 2),
        GGML_TYPE_Q4_K => {
            if row_elements % QK_K != 0 {
                bail!("Q4_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q4_K_SIZE)
        }
        GGML_TYPE_Q5_K => {
            if row_elements % QK_K != 0 {
                bail!("Q5_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q5_K_SIZE)
        }
        GGML_TYPE_Q6_K => {
            if row_elements % QK_K != 0 {
                bail!("Q6_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q6_K_SIZE)
        }
        _ => bail!("Unsupported GGML type {} for row size", ggml_type),
    }
}

pub fn dequantize_row(
    ggml_type: u32,
    raw: &[u8],
    row_idx: usize,
    row_elements: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; row_elements];
    dequantize_row_into(ggml_type, raw, row_idx, row_elements, &mut out)?;
    Ok(out)
}

pub fn dequantize_row_into(
    ggml_type: u32,
    raw: &[u8],
    row_idx: usize,
    row_elements: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != row_elements {
        bail!(
            "dequantize_row_into output len {} does not match row_elements {}",
            out.len(),
            row_elements
        );
    }
    let bpr = bytes_per_row(ggml_type, row_elements)?;
    let start = row_idx * bpr;
    let end = start + bpr;
    if end > raw.len() {
        bail!(
            "Row {} needs bytes {}..{} but tensor has only {} bytes",
            row_idx,
            start,
            end,
            raw.len()
        );
    }
    dequantize_row_bytes_into(ggml_type, &raw[start..end], out)
}

fn dequantize_row_bytes_into(ggml_type: u32, row_bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match ggml_type {
        GGML_TYPE_F32 => {
            if row_bytes.len() != out.len() * 4 {
                bail!(
                    "F32 row decode length {} does not match output len {}",
                    row_bytes.len(),
                    out.len()
                );
            }
            for (chunk, value) in row_bytes.chunks_exact(4).zip(out.iter_mut()) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        GGML_TYPE_F16 => {
            if row_bytes.len() != out.len() * 2 {
                bail!(
                    "F16 row decode length {} does not match output len {}",
                    row_bytes.len(),
                    out.len()
                );
            }
            for (chunk, value) in row_bytes.chunks_exact(2).zip(out.iter_mut()) {
                *value = fp16_to_f32([chunk[0], chunk[1]]);
            }
            Ok(())
        }
        GGML_TYPE_BF16 => {
            if row_bytes.len() != out.len() * 2 {
                bail!(
                    "BF16 row decode length {} does not match output len {}",
                    row_bytes.len(),
                    out.len()
                );
            }
            for (chunk, value) in row_bytes.chunks_exact(2).zip(out.iter_mut()) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = bf16_to_f32(bits);
            }
            Ok(())
        }
        GGML_TYPE_Q4_K => {
            if row_bytes.len() % BLOCK_Q4_K_SIZE != 0
                || out.len() != row_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
            {
                bail!("Q4_K row decode shape mismatch");
            }
            for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
                dequantize_q4_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
            }
            Ok(())
        }
        GGML_TYPE_Q5_K => {
            if row_bytes.len() % BLOCK_Q5_K_SIZE != 0
                || out.len() != row_bytes.len() / BLOCK_Q5_K_SIZE * QK_K
            {
                bail!("Q5_K row decode shape mismatch");
            }
            for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q5_K_SIZE).enumerate() {
                dequantize_q5_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
            }
            Ok(())
        }
        GGML_TYPE_Q6_K => {
            if row_bytes.len() % BLOCK_Q6_K_SIZE != 0
                || out.len() != row_bytes.len() / BLOCK_Q6_K_SIZE * QK_K
            {
                bail!("Q6_K row decode shape mismatch");
            }
            for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
                dequantize_q6_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
            }
            Ok(())
        }
        _ => bail!(
            "Unsupported GGML type {} ({})",
            ggml_type,
            ggml_type_name(ggml_type)
        ),
    }
}

pub fn dot_row(ggml_type: u32, row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    match ggml_type {
        GGML_TYPE_F32 => dot_f32_row(row_bytes, input),
        GGML_TYPE_F16 => dot_f16_row(row_bytes, input),
        GGML_TYPE_BF16 => dot_bf16_row(row_bytes, input),
        GGML_TYPE_Q4_K => dot_q4_k_row(row_bytes, input),
        GGML_TYPE_Q5_K => dot_q5_k_row(row_bytes, input),
        GGML_TYPE_Q6_K => dot_q6_k_row(row_bytes, input),
        _ => {
            let row = dequantize_tensor(ggml_type, row_bytes)?;
            Ok(row.iter().zip(input.iter()).map(|(m, x)| m * x).sum())
        }
    }
}

pub fn dot_many_row(ggml_type: u32, row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    match ggml_type {
        GGML_TYPE_F32 => dot_many_f32_row(row_bytes, inputs),
        GGML_TYPE_F16 => dot_many_f16_row(row_bytes, inputs),
        GGML_TYPE_BF16 => dot_many_bf16_row(row_bytes, inputs),
        GGML_TYPE_Q4_K => dot_many_q4_k_row(row_bytes, inputs),
        GGML_TYPE_Q5_K => dot_many_q5_k_row(row_bytes, inputs),
        GGML_TYPE_Q6_K => dot_many_q6_k_row(row_bytes, inputs),
        _ => {
            let row = dequantize_tensor(ggml_type, row_bytes)?;
            Ok(inputs
                .iter()
                .map(|input| row.iter().zip(input.iter()).map(|(m, x)| m * x).sum())
                .collect())
        }
    }
}

pub fn dot_many_row_into(
    ggml_type: u32,
    row_bytes: &[u8],
    inputs: &[Vec<f32>],
    sums: &mut [f32],
) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_row_refs_into(ggml_type, row_bytes, &input_refs, sums)
}

pub fn dot_many_row_refs_into(
    ggml_type: u32,
    row_bytes: &[u8],
    inputs: &[&[f32]],
    sums: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if sums.len() != inputs.len() {
        bail!(
            "dot_many_row_into sums len {} does not match input count {}",
            sums.len(),
            inputs.len()
        );
    }
    sums.fill(0.0);
    match ggml_type {
        GGML_TYPE_F32 => dot_many_f32_row_refs_into(row_bytes, inputs, sums),
        GGML_TYPE_F16 => dot_many_f16_row_refs_into(row_bytes, inputs, sums),
        GGML_TYPE_BF16 => dot_many_bf16_row_refs_into(row_bytes, inputs, sums),
        GGML_TYPE_Q4_K => dot_many_q4_k_row_refs_into(row_bytes, inputs, sums),
        GGML_TYPE_Q5_K => dot_many_q5_k_row_refs_into(row_bytes, inputs, sums),
        GGML_TYPE_Q6_K => dot_many_q6_k_row_refs_into(row_bytes, inputs, sums),
        _ => {
            let row = dequantize_tensor(ggml_type, row_bytes)?;
            for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
                *sum = row.iter().zip(input.iter()).map(|(m, x)| m * x).sum();
            }
            Ok(())
        }
    }
}

#[inline(always)]
pub fn dot_many_q4_k_two_rows_refs_into(
    row_a_bytes: &[u8],
    row_b_bytes: &[u8],
    inputs: &[&[f32]],
    sums_a: &mut [f32],
    sums_b: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if sums_a.len() != inputs.len() || sums_b.len() != inputs.len() {
        bail!(
            "Q4_K paired batched dot sums lens {} / {} do not match input count {}",
            sums_a.len(),
            sums_b.len(),
            inputs.len()
        );
    }
    sums_a.fill(0.0);
    sums_b.fill(0.0);
    let input_len = inputs[0].len();
    if row_a_bytes.len() != row_b_bytes.len()
        || row_a_bytes.len() % BLOCK_Q4_K_SIZE != 0
        || input_len != row_a_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
    {
        bail!("Q4_K paired batched dot row shape mismatch");
    }
    if inputs.len() == 6 {
        let inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        ];
        let mut acc_a = [0.0f32; 6];
        let mut acc_b = [0.0f32; 6];
        for (block_idx, (block_a, block_b)) in row_a_bytes
            .chunks_exact(BLOCK_Q4_K_SIZE)
            .zip(row_b_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
            .enumerate()
        {
            dot_many_q4_k_two_blocks_with_offset_six(
                block_a,
                block_b,
                &inputs,
                block_idx * QK_K,
                &mut acc_a,
                &mut acc_b,
            )?;
        }
        sums_a.copy_from_slice(&acc_a);
        sums_b.copy_from_slice(&acc_b);
        return Ok(());
    }
    for (block_idx, (block_a, block_b)) in row_a_bytes
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(row_b_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .enumerate()
    {
        dot_many_q4_k_two_blocks_with_offset(
            block_a,
            block_b,
            inputs,
            block_idx * QK_K,
            sums_a,
            sums_b,
        )?;
    }
    Ok(())
}

#[inline(always)]
pub fn dot_many_q4_k_four_rows_refs_into(
    row_a_bytes: &[u8],
    row_b_bytes: &[u8],
    row_c_bytes: &[u8],
    row_d_bytes: &[u8],
    inputs: &[&[f32]],
    sums_a: &mut [f32],
    sums_b: &mut [f32],
    sums_c: &mut [f32],
    sums_d: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if sums_a.len() != inputs.len()
        || sums_b.len() != inputs.len()
        || sums_c.len() != inputs.len()
        || sums_d.len() != inputs.len()
    {
        bail!(
            "Q4_K four-row batched dot sums lens {} / {} / {} / {} do not match input count {}",
            sums_a.len(),
            sums_b.len(),
            sums_c.len(),
            sums_d.len(),
            inputs.len()
        );
    }
    sums_a.fill(0.0);
    sums_b.fill(0.0);
    sums_c.fill(0.0);
    sums_d.fill(0.0);
    let input_len = inputs[0].len();
    if row_a_bytes.len() != row_b_bytes.len()
        || row_a_bytes.len() != row_c_bytes.len()
        || row_a_bytes.len() != row_d_bytes.len()
        || row_a_bytes.len() % BLOCK_Q4_K_SIZE != 0
        || input_len != row_a_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
    {
        bail!("Q4_K four-row batched dot row shape mismatch");
    }
    if inputs.len() == 6 {
        let inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        ];
        let mut acc_a = [0.0f32; 6];
        let mut acc_b = [0.0f32; 6];
        let mut acc_c = [0.0f32; 6];
        let mut acc_d = [0.0f32; 6];
        for (block_idx, (((block_a, block_b), block_c), block_d)) in row_a_bytes
            .chunks_exact(BLOCK_Q4_K_SIZE)
            .zip(row_b_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
            .zip(row_c_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
            .zip(row_d_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
            .enumerate()
        {
            dot_many_q4_k_four_blocks_with_offset_six(
                block_a,
                block_b,
                block_c,
                block_d,
                &inputs,
                block_idx * QK_K,
                &mut acc_a,
                &mut acc_b,
                &mut acc_c,
                &mut acc_d,
            )?;
        }
        sums_a.copy_from_slice(&acc_a);
        sums_b.copy_from_slice(&acc_b);
        sums_c.copy_from_slice(&acc_c);
        sums_d.copy_from_slice(&acc_d);
        return Ok(());
    }
    for (block_idx, (((block_a, block_b), block_c), block_d)) in row_a_bytes
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(row_b_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .zip(row_c_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .zip(row_d_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .enumerate()
    {
        dot_many_q4_k_four_blocks_with_offset(
            block_a,
            block_b,
            block_c,
            block_d,
            inputs,
            block_idx * QK_K,
            sums_a,
            sums_b,
            sums_c,
            sums_d,
        )?;
    }
    Ok(())
}

pub fn dot_q4_k_four_rows(
    row_a_bytes: &[u8],
    row_b_bytes: &[u8],
    row_c_bytes: &[u8],
    row_d_bytes: &[u8],
    input: &[f32],
) -> Result<(f32, f32, f32, f32)> {
    if row_a_bytes.len() != row_b_bytes.len()
        || row_a_bytes.len() != row_c_bytes.len()
        || row_a_bytes.len() != row_d_bytes.len()
        || row_a_bytes.len() % BLOCK_Q4_K_SIZE != 0
        || input.len() != row_a_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
    {
        bail!("Q4_K four-row single-input dot row shape mismatch");
    }

    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_c = 0.0f32;
    let mut sum_d = 0.0f32;
    for (block_idx, (((block_a, block_b), block_c), block_d)) in row_a_bytes
        .chunks_exact(BLOCK_Q4_K_SIZE)
        .zip(row_b_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .zip(row_c_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .zip(row_d_bytes.chunks_exact(BLOCK_Q4_K_SIZE))
        .enumerate()
    {
        let (block_sum_a, block_sum_b, block_sum_c, block_sum_d) =
            dot_q4_k_four_blocks_with_offset_single(
                block_a,
                block_b,
                block_c,
                block_d,
                input,
                block_idx * QK_K,
            )?;
        sum_a += block_sum_a;
        sum_b += block_sum_b;
        sum_c += block_sum_c;
        sum_d += block_sum_d;
    }
    Ok((sum_a, sum_b, sum_c, sum_d))
}

pub fn dot_many_q6_k_two_rows_refs_into(
    row_a_bytes: &[u8],
    row_b_bytes: &[u8],
    inputs: &[&[f32]],
    sums_a: &mut [f32],
    sums_b: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if sums_a.len() != inputs.len() || sums_b.len() != inputs.len() {
        bail!(
            "Q6_K paired batched dot sums lens {} / {} do not match input count {}",
            sums_a.len(),
            sums_b.len(),
            inputs.len()
        );
    }
    sums_a.fill(0.0);
    sums_b.fill(0.0);
    let input_len = inputs[0].len();
    if row_a_bytes.len() != row_b_bytes.len()
        || row_a_bytes.len() % BLOCK_Q6_K_SIZE != 0
        || input_len != row_a_bytes.len() / BLOCK_Q6_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
    {
        bail!("Q6_K paired batched dot row shape mismatch");
    }
    if inputs.len() == 6 {
        let inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        ];
        let mut acc_a = [0.0f32; 6];
        let mut acc_b = [0.0f32; 6];
        for (block_idx, (block_a, block_b)) in row_a_bytes
            .chunks_exact(BLOCK_Q6_K_SIZE)
            .zip(row_b_bytes.chunks_exact(BLOCK_Q6_K_SIZE))
            .enumerate()
        {
            dot_many_q6_k_two_blocks_with_offset_six(
                block_a,
                block_b,
                &inputs,
                block_idx * QK_K,
                &mut acc_a,
                &mut acc_b,
            )?;
        }
        sums_a.copy_from_slice(&acc_a);
        sums_b.copy_from_slice(&acc_b);
        return Ok(());
    }
    for (block_idx, (block_a, block_b)) in row_a_bytes
        .chunks_exact(BLOCK_Q6_K_SIZE)
        .zip(row_b_bytes.chunks_exact(BLOCK_Q6_K_SIZE))
        .enumerate()
    {
        dot_many_q6_k_two_blocks_with_offset(
            block_a,
            block_b,
            inputs,
            block_idx * QK_K,
            sums_a,
            sums_b,
        )?;
    }
    Ok(())
}

pub fn dot_many_q6_k_four_rows_refs_into(
    row_a_bytes: &[u8],
    row_b_bytes: &[u8],
    row_c_bytes: &[u8],
    row_d_bytes: &[u8],
    inputs: &[&[f32]],
    sums_a: &mut [f32],
    sums_b: &mut [f32],
    sums_c: &mut [f32],
    sums_d: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    if sums_a.len() != inputs.len()
        || sums_b.len() != inputs.len()
        || sums_c.len() != inputs.len()
        || sums_d.len() != inputs.len()
    {
        bail!(
            "Q6_K four-row batched dot sums lens {} / {} / {} / {} do not match input count {}",
            sums_a.len(),
            sums_b.len(),
            sums_c.len(),
            sums_d.len(),
            inputs.len()
        );
    }
    sums_a.fill(0.0);
    sums_b.fill(0.0);
    sums_c.fill(0.0);
    sums_d.fill(0.0);
    let input_len = inputs[0].len();
    if row_a_bytes.len() != row_b_bytes.len()
        || row_a_bytes.len() != row_c_bytes.len()
        || row_a_bytes.len() != row_d_bytes.len()
        || row_a_bytes.len() % BLOCK_Q6_K_SIZE != 0
        || input_len != row_a_bytes.len() / BLOCK_Q6_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
    {
        bail!("Q6_K four-row batched dot row shape mismatch");
    }
    if inputs.len() == 6 {
        let inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        ];
        let mut acc_a = [0.0f32; 6];
        let mut acc_b = [0.0f32; 6];
        let mut acc_c = [0.0f32; 6];
        let mut acc_d = [0.0f32; 6];
        for (block_idx, (((block_a, block_b), block_c), block_d)) in row_a_bytes
            .chunks_exact(BLOCK_Q6_K_SIZE)
            .zip(row_b_bytes.chunks_exact(BLOCK_Q6_K_SIZE))
            .zip(row_c_bytes.chunks_exact(BLOCK_Q6_K_SIZE))
            .zip(row_d_bytes.chunks_exact(BLOCK_Q6_K_SIZE))
            .enumerate()
        {
            dot_many_q6_k_four_blocks_with_offset_six(
                block_a,
                block_b,
                block_c,
                block_d,
                &inputs,
                block_idx * QK_K,
                &mut acc_a,
                &mut acc_b,
                &mut acc_c,
                &mut acc_d,
            )?;
        }
        sums_a.copy_from_slice(&acc_a);
        sums_b.copy_from_slice(&acc_b);
        sums_c.copy_from_slice(&acc_c);
        sums_d.copy_from_slice(&acc_d);
        return Ok(());
    }
    dot_many_q6_k_two_rows_refs_into(row_a_bytes, row_b_bytes, inputs, sums_a, sums_b)?;
    dot_many_q6_k_two_rows_refs_into(row_c_bytes, row_d_bytes, inputs, sums_c, sums_d)?;
    Ok(())
}

fn dot_f32_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() != input.len() * 4 {
        bail!(
            "F32 dot row length {} does not match input len {}",
            row_bytes.len(),
            input.len()
        );
    }
    let mut sum = 0.0f32;
    for (chunk, x) in row_bytes.chunks_exact(4).zip(input.iter()) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum += value * x;
    }
    Ok(sum)
}

fn dot_many_f32_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_f32_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_f32_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_f32_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_f32_row_refs_into(row_bytes: &[u8], inputs: &[&[f32]], sums: &mut [f32]) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() != input_len * 4
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("F32 batched dot row shape mismatch");
    }
    for (idx, chunk) in row_bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
            *sum += value * input[idx];
        }
    }
    Ok(())
}

fn dot_f16_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() != input.len() * 2 {
        bail!(
            "F16 dot row length {} does not match input len {}",
            row_bytes.len(),
            input.len()
        );
    }
    let mut sum = 0.0f32;
    for (chunk, x) in row_bytes.chunks_exact(2).zip(input.iter()) {
        sum += fp16_to_f32([chunk[0], chunk[1]]) * x;
    }
    Ok(sum)
}

fn dot_many_f16_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_f16_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_f16_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_f16_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_f16_row_refs_into(row_bytes: &[u8], inputs: &[&[f32]], sums: &mut [f32]) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() != input_len * 2
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("F16 batched dot row shape mismatch");
    }
    for (idx, chunk) in row_bytes.chunks_exact(2).enumerate() {
        let value = fp16_to_f32([chunk[0], chunk[1]]);
        for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
            *sum += value * input[idx];
        }
    }
    Ok(())
}

fn dot_bf16_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() != input.len() * 2 {
        bail!(
            "BF16 dot row length {} does not match input len {}",
            row_bytes.len(),
            input.len()
        );
    }
    let mut sum = 0.0f32;
    for (chunk, x) in row_bytes.chunks_exact(2).zip(input.iter()) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        sum += bf16_to_f32(bits) * x;
    }
    Ok(sum)
}

fn dot_many_bf16_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_bf16_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_bf16_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_bf16_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_bf16_row_refs_into(
    row_bytes: &[u8],
    inputs: &[&[f32]],
    sums: &mut [f32],
) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() != input_len * 2
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("BF16 batched dot row shape mismatch");
    }
    for (idx, chunk) in row_bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let value = bf16_to_f32(bits);
        for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
            *sum += value * input[idx];
        }
    }
    Ok(())
}

fn dot_q4_k_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() % BLOCK_Q4_K_SIZE != 0
        || input.len() != row_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
    {
        bail!("Q4_K dot row shape mismatch");
    }
    let mut sum = 0.0f32;
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
        sum += dot_q4_k_block(block, &input[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(sum)
}

fn dot_many_q4_k_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_q4_k_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_q4_k_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_q4_k_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_q4_k_row_refs_into(
    row_bytes: &[u8],
    inputs: &[&[f32]],
    sums: &mut [f32],
) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() % BLOCK_Q4_K_SIZE != 0
        || input_len != row_bytes.len() / BLOCK_Q4_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("Q4_K batched dot row shape mismatch");
    }
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
        dot_many_q4_k_block_with_offset(block, inputs, block_idx * QK_K, sums)?;
    }
    Ok(())
}

fn dot_q5_k_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() % BLOCK_Q5_K_SIZE != 0
        || input.len() != row_bytes.len() / BLOCK_Q5_K_SIZE * QK_K
    {
        bail!("Q5_K dot row shape mismatch");
    }
    let mut sum = 0.0f32;
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q5_K_SIZE).enumerate() {
        sum += dot_q5_k_block(block, &input[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(sum)
}

fn dot_many_q5_k_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_q5_k_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_q5_k_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_q5_k_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_q5_k_row_refs_into(
    row_bytes: &[u8],
    inputs: &[&[f32]],
    sums: &mut [f32],
) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() % BLOCK_Q5_K_SIZE != 0
        || input_len != row_bytes.len() / BLOCK_Q5_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("Q5_K batched dot row shape mismatch");
    }
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q5_K_SIZE).enumerate() {
        dot_many_q5_k_block_with_offset(block, inputs, block_idx * QK_K, sums)?;
    }
    Ok(())
}

fn dot_q4_k_block(block: &[u8], input: &[f32]) -> Result<f32> {
    if block.len() != BLOCK_Q4_K_SIZE || input.len() != QK_K {
        bail!("Q4_K block dot shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..];

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q = qs[q_offset + l];
            sum += (d1 * (q & 0x0F) as f32 - m1) * input[j + l];
            sum += (d2 * (q >> 4) as f32 - m2) * input[j + 32 + l];
        }
        q_offset += 32;
        is += 2;
    }
    Ok(sum)
}

fn dot_q5_k_block(block: &[u8], input: &[f32]) -> Result<f32> {
    if block.len() != BLOCK_Q5_K_SIZE || input.len() != QK_K {
        bail!("Q5_K block dot shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
    let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut q_offset = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q_lo = qs[q_offset + l];
            let qh_byte = qh[l];
            let hbit1 = if qh_byte & u1 != 0 { 16.0 } else { 0.0 };
            let hbit2 = if qh_byte & u2 != 0 { 16.0 } else { 0.0 };
            let v1 = d1 * ((q_lo & 0x0F) as f32 + hbit1) - m1;
            let v2 = d2 * ((q_lo >> 4) as f32 + hbit2) - m2;
            sum += v1 * input[j + l * 2] + v2 * input[j + l * 2 + 1];
        }
        q_offset += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    Ok(sum)
}

fn dot_many_q4_k_block_with_offset(
    block: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums: &mut [f32],
) -> Result<()> {
    if block.len() != BLOCK_Q4_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums.len() != inputs.len()
    {
        bail!("Q4_K block batched dot shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q = qs[q_offset + l];
            let v1 = d1 * (q & 0x0F) as f32 - m1;
            let v2 = d2 * (q >> 4) as f32 - m2;
            for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
                *sum += v1 * input[input_offset + j + l] + v2 * input[input_offset + j + 32 + l];
            }
        }
        q_offset += 32;
        is += 2;
    }
    Ok(())
}

fn dot_many_q4_k_two_blocks_with_offset(
    block_a: &[u8],
    block_b: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums_a: &mut [f32],
    sums_b: &mut [f32],
) -> Result<()> {
    if block_a.len() != BLOCK_Q4_K_SIZE
        || block_b.len() != BLOCK_Q4_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums_a.len() != inputs.len()
        || sums_b.len() != inputs.len()
    {
        bail!("Q4_K paired block batched dot shape mismatch");
    }

    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        for l in 0..32 {
            let q_a = qs_a[q_offset + l];
            let a1 = d1_a * (q_a & 0x0F) as f32 - m1_a;
            let a2 = d2_a * (q_a >> 4) as f32 - m2_a;

            let q_b = qs_b[q_offset + l];
            let b1 = d1_b * (q_b & 0x0F) as f32 - m1_b;
            let b2 = d2_b * (q_b >> 4) as f32 - m2_b;

            for input_idx in 0..inputs.len() {
                let input = inputs[input_idx];
                let left = input[input_offset + j + l];
                let right = input[input_offset + j + 32 + l];
                sums_a[input_idx] += a1 * left + a2 * right;
                sums_b[input_idx] += b1 * left + b2 * right;
            }
        }
        q_offset += 32;
        is += 2;
    }
    Ok(())
}

#[inline(always)]
fn dot_many_q4_k_two_blocks_with_offset_six(
    block_a: &[u8],
    block_b: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
) -> Result<()> {
    if block_a.len() != BLOCK_Q4_K_SIZE
        || block_b.len() != BLOCK_Q4_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q4_K paired block batched dot six-input fast path shape mismatch");
    }

    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];
    let mut sum_a0 = sums_a[0];
    let mut sum_a1 = sums_a[1];
    let mut sum_a2 = sums_a[2];
    let mut sum_a3 = sums_a[3];
    let mut sum_a4 = sums_a[4];
    let mut sum_a5 = sums_a[5];
    let mut sum_b0 = sums_b[0];
    let mut sum_b1 = sums_b[1];
    let mut sum_b2 = sums_b[2];
    let mut sum_b3 = sums_b[3];
    let mut sum_b4 = sums_b[4];
    let mut sum_b5 = sums_b[5];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let input_base = input_offset + j;
        let input0_ptr = unsafe { input0.as_ptr().add(input_base) };
        let input1_ptr = unsafe { input1.as_ptr().add(input_base) };
        let input2_ptr = unsafe { input2.as_ptr().add(input_base) };
        let input3_ptr = unsafe { input3.as_ptr().add(input_base) };
        let input4_ptr = unsafe { input4.as_ptr().add(input_base) };
        let input5_ptr = unsafe { input5.as_ptr().add(input_base) };
        let qs_a_ptr = unsafe { qs_a.as_ptr().add(q_offset) };
        let qs_b_ptr = unsafe { qs_b.as_ptr().add(q_offset) };

        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        for l in 0..32 {
            let q_a = unsafe { *qs_a_ptr.add(l) };
            let a1 = d1_a.mul_add((q_a & 0x0F) as f32, -m1_a);
            let a2 = d2_a.mul_add((q_a >> 4) as f32, -m2_a);

            let q_b = unsafe { *qs_b_ptr.add(l) };
            let b1 = d1_b.mul_add((q_b & 0x0F) as f32, -m1_b);
            let b2 = d2_b.mul_add((q_b >> 4) as f32, -m2_b);

            let left0 = unsafe { *input0_ptr.add(l) };
            let right0 = unsafe { *input0_ptr.add(32 + l) };
            sum_a0 = a2.mul_add(right0, a1.mul_add(left0, sum_a0));
            sum_b0 = b2.mul_add(right0, b1.mul_add(left0, sum_b0));

            let left1 = unsafe { *input1_ptr.add(l) };
            let right1 = unsafe { *input1_ptr.add(32 + l) };
            sum_a1 = a2.mul_add(right1, a1.mul_add(left1, sum_a1));
            sum_b1 = b2.mul_add(right1, b1.mul_add(left1, sum_b1));

            let left2 = unsafe { *input2_ptr.add(l) };
            let right2 = unsafe { *input2_ptr.add(32 + l) };
            sum_a2 = a2.mul_add(right2, a1.mul_add(left2, sum_a2));
            sum_b2 = b2.mul_add(right2, b1.mul_add(left2, sum_b2));

            let left3 = unsafe { *input3_ptr.add(l) };
            let right3 = unsafe { *input3_ptr.add(32 + l) };
            sum_a3 = a2.mul_add(right3, a1.mul_add(left3, sum_a3));
            sum_b3 = b2.mul_add(right3, b1.mul_add(left3, sum_b3));

            let left4 = unsafe { *input4_ptr.add(l) };
            let right4 = unsafe { *input4_ptr.add(32 + l) };
            sum_a4 = a2.mul_add(right4, a1.mul_add(left4, sum_a4));
            sum_b4 = b2.mul_add(right4, b1.mul_add(left4, sum_b4));

            let left5 = unsafe { *input5_ptr.add(l) };
            let right5 = unsafe { *input5_ptr.add(32 + l) };
            sum_a5 = a2.mul_add(right5, a1.mul_add(left5, sum_a5));
            sum_b5 = b2.mul_add(right5, b1.mul_add(left5, sum_b5));
        }
        q_offset += 32;
        is += 2;
    }

    sums_a[0] = sum_a0;
    sums_a[1] = sum_a1;
    sums_a[2] = sum_a2;
    sums_a[3] = sum_a3;
    sums_a[4] = sum_a4;
    sums_a[5] = sum_a5;
    sums_b[0] = sum_b0;
    sums_b[1] = sum_b1;
    sums_b[2] = sum_b2;
    sums_b[3] = sum_b3;
    sums_b[4] = sum_b4;
    sums_b[5] = sum_b5;
    Ok(())
}

fn dot_many_q4_k_four_blocks_with_offset(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums_a: &mut [f32],
    sums_b: &mut [f32],
    sums_c: &mut [f32],
    sums_d: &mut [f32],
) -> Result<()> {
    if block_a.len() != BLOCK_Q4_K_SIZE
        || block_b.len() != BLOCK_Q4_K_SIZE
        || block_c.len() != BLOCK_Q4_K_SIZE
        || block_d.len() != BLOCK_Q4_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums_a.len() != inputs.len()
        || sums_b.len() != inputs.len()
        || sums_c.len() != inputs.len()
        || sums_d.len() != inputs.len()
    {
        bail!("Q4_K four-row block batched dot shape mismatch");
    }

    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let d_c = fp16_to_f32([block_c[0], block_c[1]]);
    let dmin_c = fp16_to_f32([block_c[2], block_c[3]]);
    let scales_c = &block_c[4..4 + K_SCALE_SIZE];
    let qs_c = &block_c[4 + K_SCALE_SIZE..];

    let d_d = fp16_to_f32([block_d[0], block_d[1]]);
    let dmin_d = fp16_to_f32([block_d[2], block_d[3]]);
    let scales_d = &block_d[4..4 + K_SCALE_SIZE];
    let qs_d = &block_d[4 + K_SCALE_SIZE..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        let (sc1_c, m1_c) = get_scale_min_k4(is, scales_c);
        let d1_c = d_c * sc1_c as f32;
        let m1_c = dmin_c * m1_c as f32;
        let (sc2_c, m2_c) = get_scale_min_k4(is + 1, scales_c);
        let d2_c = d_c * sc2_c as f32;
        let m2_c = dmin_c * m2_c as f32;

        let (sc1_d, m1_d) = get_scale_min_k4(is, scales_d);
        let d1_d = d_d * sc1_d as f32;
        let m1_d = dmin_d * m1_d as f32;
        let (sc2_d, m2_d) = get_scale_min_k4(is + 1, scales_d);
        let d2_d = d_d * sc2_d as f32;
        let m2_d = dmin_d * m2_d as f32;

        for l in 0..32 {
            let q_a = qs_a[q_offset + l];
            let a1 = d1_a * (q_a & 0x0F) as f32 - m1_a;
            let a2 = d2_a * (q_a >> 4) as f32 - m2_a;

            let q_b = qs_b[q_offset + l];
            let b1 = d1_b * (q_b & 0x0F) as f32 - m1_b;
            let b2 = d2_b * (q_b >> 4) as f32 - m2_b;

            let q_c = qs_c[q_offset + l];
            let c1 = d1_c * (q_c & 0x0F) as f32 - m1_c;
            let c2 = d2_c * (q_c >> 4) as f32 - m2_c;

            let q_d = qs_d[q_offset + l];
            let d1v = d1_d * (q_d & 0x0F) as f32 - m1_d;
            let d2v = d2_d * (q_d >> 4) as f32 - m2_d;

            for input_idx in 0..inputs.len() {
                let input = inputs[input_idx];
                let left = input[input_offset + j + l];
                let right = input[input_offset + j + 32 + l];
                sums_a[input_idx] += a1 * left + a2 * right;
                sums_b[input_idx] += b1 * left + b2 * right;
                sums_c[input_idx] += c1 * left + c2 * right;
                sums_d[input_idx] += d1v * left + d2v * right;
            }
        }
        q_offset += 32;
        is += 2;
    }
    Ok(())
}

#[inline(always)]
fn dot_many_q4_k_four_blocks_with_offset_six(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    if block_a.len() != BLOCK_Q4_K_SIZE
        || block_b.len() != BLOCK_Q4_K_SIZE
        || block_c.len() != BLOCK_Q4_K_SIZE
        || block_d.len() != BLOCK_Q4_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q4_K four-row block batched dot six-input fast path shape mismatch");
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_many_q4_k_four_blocks_with_offset_six_neon(
            block_a,
            block_b,
            block_c,
            block_d,
            inputs,
            input_offset,
            sums_a,
            sums_b,
            sums_c,
            sums_d,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        return dot_many_q4_k_four_blocks_with_offset_six_scalar(
            block_a,
            block_b,
            block_c,
            block_d,
            inputs,
            input_offset,
            sums_a,
            sums_b,
            sums_c,
            sums_d,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot_many_q4_k_four_blocks_with_offset_six_scalar(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let d_c = fp16_to_f32([block_c[0], block_c[1]]);
    let dmin_c = fp16_to_f32([block_c[2], block_c[3]]);
    let scales_c = &block_c[4..4 + K_SCALE_SIZE];
    let qs_c = &block_c[4 + K_SCALE_SIZE..];

    let d_d = fp16_to_f32([block_d[0], block_d[1]]);
    let dmin_d = fp16_to_f32([block_d[2], block_d[3]]);
    let scales_d = &block_d[4..4 + K_SCALE_SIZE];
    let qs_d = &block_d[4 + K_SCALE_SIZE..];

    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];
    let mut sum_a0 = sums_a[0];
    let mut sum_a1 = sums_a[1];
    let mut sum_a2 = sums_a[2];
    let mut sum_a3 = sums_a[3];
    let mut sum_a4 = sums_a[4];
    let mut sum_a5 = sums_a[5];
    let mut sum_b0 = sums_b[0];
    let mut sum_b1 = sums_b[1];
    let mut sum_b2 = sums_b[2];
    let mut sum_b3 = sums_b[3];
    let mut sum_b4 = sums_b[4];
    let mut sum_b5 = sums_b[5];
    let mut sum_c0 = sums_c[0];
    let mut sum_c1 = sums_c[1];
    let mut sum_c2 = sums_c[2];
    let mut sum_c3 = sums_c[3];
    let mut sum_c4 = sums_c[4];
    let mut sum_c5 = sums_c[5];
    let mut sum_d0 = sums_d[0];
    let mut sum_d1 = sums_d[1];
    let mut sum_d2 = sums_d[2];
    let mut sum_d3 = sums_d[3];
    let mut sum_d4 = sums_d[4];
    let mut sum_d5 = sums_d[5];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let input_base = input_offset + j;
        let input0_ptr = unsafe { input0.as_ptr().add(input_base) };
        let input1_ptr = unsafe { input1.as_ptr().add(input_base) };
        let input2_ptr = unsafe { input2.as_ptr().add(input_base) };
        let input3_ptr = unsafe { input3.as_ptr().add(input_base) };
        let input4_ptr = unsafe { input4.as_ptr().add(input_base) };
        let input5_ptr = unsafe { input5.as_ptr().add(input_base) };
        let qs_a_ptr = unsafe { qs_a.as_ptr().add(q_offset) };
        let qs_b_ptr = unsafe { qs_b.as_ptr().add(q_offset) };
        let qs_c_ptr = unsafe { qs_c.as_ptr().add(q_offset) };
        let qs_d_ptr = unsafe { qs_d.as_ptr().add(q_offset) };

        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        let (sc1_c, m1_c) = get_scale_min_k4(is, scales_c);
        let d1_c = d_c * sc1_c as f32;
        let m1_c = dmin_c * m1_c as f32;
        let (sc2_c, m2_c) = get_scale_min_k4(is + 1, scales_c);
        let d2_c = d_c * sc2_c as f32;
        let m2_c = dmin_c * m2_c as f32;

        let (sc1_d, m1_d) = get_scale_min_k4(is, scales_d);
        let d1_d = d_d * sc1_d as f32;
        let m1_d = dmin_d * m1_d as f32;
        let (sc2_d, m2_d) = get_scale_min_k4(is + 1, scales_d);
        let d2_d = d_d * sc2_d as f32;
        let m2_d = dmin_d * m2_d as f32;

        for l in 0..32 {
            let q_a = unsafe { *qs_a_ptr.add(l) };
            let a1 = d1_a.mul_add((q_a & 0x0F) as f32, -m1_a);
            let a2 = d2_a.mul_add((q_a >> 4) as f32, -m2_a);

            let q_b = unsafe { *qs_b_ptr.add(l) };
            let b1 = d1_b.mul_add((q_b & 0x0F) as f32, -m1_b);
            let b2 = d2_b.mul_add((q_b >> 4) as f32, -m2_b);

            let q_c = unsafe { *qs_c_ptr.add(l) };
            let c1 = d1_c.mul_add((q_c & 0x0F) as f32, -m1_c);
            let c2 = d2_c.mul_add((q_c >> 4) as f32, -m2_c);

            let q_d = unsafe { *qs_d_ptr.add(l) };
            let d1v = d1_d.mul_add((q_d & 0x0F) as f32, -m1_d);
            let d2v = d2_d.mul_add((q_d >> 4) as f32, -m2_d);

            let left0 = unsafe { *input0_ptr.add(l) };
            let right0 = unsafe { *input0_ptr.add(32 + l) };
            sum_a0 = a2.mul_add(right0, a1.mul_add(left0, sum_a0));
            sum_b0 = b2.mul_add(right0, b1.mul_add(left0, sum_b0));
            sum_c0 = c2.mul_add(right0, c1.mul_add(left0, sum_c0));
            sum_d0 = d2v.mul_add(right0, d1v.mul_add(left0, sum_d0));

            let left1 = unsafe { *input1_ptr.add(l) };
            let right1 = unsafe { *input1_ptr.add(32 + l) };
            sum_a1 = a2.mul_add(right1, a1.mul_add(left1, sum_a1));
            sum_b1 = b2.mul_add(right1, b1.mul_add(left1, sum_b1));
            sum_c1 = c2.mul_add(right1, c1.mul_add(left1, sum_c1));
            sum_d1 = d2v.mul_add(right1, d1v.mul_add(left1, sum_d1));

            let left2 = unsafe { *input2_ptr.add(l) };
            let right2 = unsafe { *input2_ptr.add(32 + l) };
            sum_a2 = a2.mul_add(right2, a1.mul_add(left2, sum_a2));
            sum_b2 = b2.mul_add(right2, b1.mul_add(left2, sum_b2));
            sum_c2 = c2.mul_add(right2, c1.mul_add(left2, sum_c2));
            sum_d2 = d2v.mul_add(right2, d1v.mul_add(left2, sum_d2));

            let left3 = unsafe { *input3_ptr.add(l) };
            let right3 = unsafe { *input3_ptr.add(32 + l) };
            sum_a3 = a2.mul_add(right3, a1.mul_add(left3, sum_a3));
            sum_b3 = b2.mul_add(right3, b1.mul_add(left3, sum_b3));
            sum_c3 = c2.mul_add(right3, c1.mul_add(left3, sum_c3));
            sum_d3 = d2v.mul_add(right3, d1v.mul_add(left3, sum_d3));

            let left4 = unsafe { *input4_ptr.add(l) };
            let right4 = unsafe { *input4_ptr.add(32 + l) };
            sum_a4 = a2.mul_add(right4, a1.mul_add(left4, sum_a4));
            sum_b4 = b2.mul_add(right4, b1.mul_add(left4, sum_b4));
            sum_c4 = c2.mul_add(right4, c1.mul_add(left4, sum_c4));
            sum_d4 = d2v.mul_add(right4, d1v.mul_add(left4, sum_d4));

            let left5 = unsafe { *input5_ptr.add(l) };
            let right5 = unsafe { *input5_ptr.add(32 + l) };
            sum_a5 = a2.mul_add(right5, a1.mul_add(left5, sum_a5));
            sum_b5 = b2.mul_add(right5, b1.mul_add(left5, sum_b5));
            sum_c5 = c2.mul_add(right5, c1.mul_add(left5, sum_c5));
            sum_d5 = d2v.mul_add(right5, d1v.mul_add(left5, sum_d5));
        }
        q_offset += 32;
        is += 2;
    }

    sums_a[0] = sum_a0;
    sums_a[1] = sum_a1;
    sums_a[2] = sum_a2;
    sums_a[3] = sum_a3;
    sums_a[4] = sum_a4;
    sums_a[5] = sum_a5;
    sums_b[0] = sum_b0;
    sums_b[1] = sum_b1;
    sums_b[2] = sum_b2;
    sums_b[3] = sum_b3;
    sums_b[4] = sum_b4;
    sums_b[5] = sum_b5;
    sums_c[0] = sum_c0;
    sums_c[1] = sum_c1;
    sums_c[2] = sum_c2;
    sums_c[3] = sum_c3;
    sums_c[4] = sum_c4;
    sums_c[5] = sum_c5;
    sums_d[0] = sum_d0;
    sums_d[1] = sum_d1;
    sums_d[2] = sum_d2;
    sums_d[3] = sum_d3;
    sums_d[4] = sum_d4;
    sums_d[5] = sum_d5;
    Ok(())
}

fn dot_many_q5_k_block_with_offset(
    block: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums: &mut [f32],
) -> Result<()> {
    if block.len() != BLOCK_Q5_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums.len() != inputs.len()
    {
        bail!("Q5_K block batched dot shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
    let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q_lo = qs[q_offset + l];
            let qh_byte = qh[l];
            let hbit1 = if qh_byte & u1 != 0 { 16.0 } else { 0.0 };
            let hbit2 = if qh_byte & u2 != 0 { 16.0 } else { 0.0 };
            let v1 = d1 * ((q_lo & 0x0F) as f32 + hbit1) - m1;
            let v2 = d2 * ((q_lo >> 4) as f32 + hbit2) - m2;
            for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
                *sum +=
                    v1 * input[input_offset + j + l * 2] + v2 * input[input_offset + j + l * 2 + 1];
            }
        }
        q_offset += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    Ok(())
}

fn dot_q6_k_row(row_bytes: &[u8], input: &[f32]) -> Result<f32> {
    if row_bytes.len() % BLOCK_Q6_K_SIZE != 0
        || input.len() != row_bytes.len() / BLOCK_Q6_K_SIZE * QK_K
    {
        bail!("Q6_K dot row shape mismatch");
    }
    let mut sum = 0.0f32;
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
        sum += dot_q6_k_block(block, &input[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(sum)
}

fn dot_many_q6_k_row(row_bytes: &[u8], inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut sums = vec![0.0f32; inputs.len()];
    dot_many_q6_k_row_into(row_bytes, inputs, &mut sums)?;
    Ok(sums)
}

fn dot_many_q6_k_row_into(row_bytes: &[u8], inputs: &[Vec<f32>], sums: &mut [f32]) -> Result<()> {
    let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    dot_many_q6_k_row_refs_into(row_bytes, &input_refs, sums)
}

fn dot_many_q6_k_row_refs_into(
    row_bytes: &[u8],
    inputs: &[&[f32]],
    sums: &mut [f32],
) -> Result<()> {
    let input_len = inputs[0].len();
    if row_bytes.len() % BLOCK_Q6_K_SIZE != 0
        || input_len != row_bytes.len() / BLOCK_Q6_K_SIZE * QK_K
        || inputs.iter().any(|input| input.len() != input_len)
        || sums.len() != inputs.len()
    {
        bail!("Q6_K batched dot row shape mismatch");
    }
    if inputs.len() == 6 {
        let inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        ];
        let mut acc = [0.0f32; 6];
        for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
            dot_many_q6_k_block_with_offset_six(block, &inputs, block_idx * QK_K, &mut acc)?;
        }
        sums.copy_from_slice(&acc);
        return Ok(());
    }
    for (block_idx, block) in row_bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
        dot_many_q6_k_block_with_offset(block, inputs, block_idx * QK_K, sums)?;
    }
    Ok(())
}

fn dot_q6_k_block(block: &[u8], input: &[f32]) -> Result<f32> {
    if block.len() != BLOCK_Q6_K_SIZE || input.len() != QK_K {
        bail!("Q6_K block dot shape mismatch");
    }
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales = &block[scales_start..scales_start + QK_K / 16];
    let d = fp16_to_f32([block[BLOCK_Q6_K_SIZE - 2], block[BLOCK_Q6_K_SIZE - 1]]);

    let mut sum = 0.0f32;
    let mut input_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let scale_factors = [
            d * (scales[sc_offset] as i8 as f32),
            d * (scales[sc_offset + 1] as i8 as f32),
            d * (scales[sc_offset + 2] as i8 as f32),
            d * (scales[sc_offset + 3] as i8 as f32),
            d * (scales[sc_offset + 4] as i8 as f32),
            d * (scales[sc_offset + 5] as i8 as f32),
            d * (scales[sc_offset + 6] as i8 as f32),
            d * (scales[sc_offset + 7] as i8 as f32),
        ];
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte = qh[qh_offset + l];
            let q1 =
                (((ql[ql_offset + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2 = (((ql[ql_offset + 32 + l] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32
                - 32) as f32;
            let q3 = ((((ql[ql_offset + l] >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4)) as i32
                - 32) as f32;
            let q4 = ((((ql[ql_offset + 32 + l] >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4))
                as i32
                - 32) as f32;

            sum += scale_factors[scale_idx] * q1 * input[input_offset + l];
            sum += scale_factors[scale_idx + 2] * q2 * input[input_offset + 32 + l];
            sum += scale_factors[scale_idx + 4] * q3 * input[input_offset + 64 + l];
            sum += scale_factors[scale_idx + 6] * q4 * input[input_offset + 96 + l];
        }
        input_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }
    Ok(sum)
}

fn dot_many_q6_k_block_with_offset(
    block: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums: &mut [f32],
) -> Result<()> {
    if block.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums.len() != inputs.len()
    {
        bail!("Q6_K block batched dot shape mismatch");
    }
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales = &block[scales_start..scales_start + QK_K / 16];
    let d = fp16_to_f32([block[BLOCK_Q6_K_SIZE - 2], block[BLOCK_Q6_K_SIZE - 1]]);

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let scale_factors = [
            d * (scales[sc_offset] as i8 as f32),
            d * (scales[sc_offset + 1] as i8 as f32),
            d * (scales[sc_offset + 2] as i8 as f32),
            d * (scales[sc_offset + 3] as i8 as f32),
            d * (scales[sc_offset + 4] as i8 as f32),
            d * (scales[sc_offset + 5] as i8 as f32),
            d * (scales[sc_offset + 6] as i8 as f32),
            d * (scales[sc_offset + 7] as i8 as f32),
        ];
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte = qh[qh_offset + l];
            let q1 =
                (((ql[ql_offset + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2 = (((ql[ql_offset + 32 + l] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32
                - 32) as f32;
            let q3 = ((((ql[ql_offset + l] >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4)) as i32
                - 32) as f32;
            let q4 = ((((ql[ql_offset + 32 + l] >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4))
                as i32
                - 32) as f32;

            for (sum, input) in sums.iter_mut().zip(inputs.iter()) {
                *sum += scale_factors[scale_idx] * q1 * input[input_offset + local_offset + l]
                    + scale_factors[scale_idx + 2]
                        * q2
                        * input[input_offset + local_offset + 32 + l]
                    + scale_factors[scale_idx + 4]
                        * q3
                        * input[input_offset + local_offset + 64 + l]
                    + scale_factors[scale_idx + 6]
                        * q4
                        * input[input_offset + local_offset + 96 + l];
            }
        }
        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_k_four_blocks_with_offset_single_neon(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    input: &[f32],
    input_offset: usize,
) -> Result<(f32, f32, f32, f32)> {
    let d_a = fp16_to_f32([block_a[0], block_a[1]]);
    let dmin_a = fp16_to_f32([block_a[2], block_a[3]]);
    let scales_a = &block_a[4..4 + K_SCALE_SIZE];
    let qs_a = &block_a[4 + K_SCALE_SIZE..];

    let d_b = fp16_to_f32([block_b[0], block_b[1]]);
    let dmin_b = fp16_to_f32([block_b[2], block_b[3]]);
    let scales_b = &block_b[4..4 + K_SCALE_SIZE];
    let qs_b = &block_b[4 + K_SCALE_SIZE..];

    let d_c = fp16_to_f32([block_c[0], block_c[1]]);
    let dmin_c = fp16_to_f32([block_c[2], block_c[3]]);
    let scales_c = &block_c[4..4 + K_SCALE_SIZE];
    let qs_c = &block_c[4 + K_SCALE_SIZE..];

    let d_d = fp16_to_f32([block_d[0], block_d[1]]);
    let dmin_d = fp16_to_f32([block_d[2], block_d[3]]);
    let scales_d = &block_d[4..4 + K_SCALE_SIZE];
    let qs_d = &block_d[4 + K_SCALE_SIZE..];

    let mut sum_a_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d_v = unsafe { vdupq_n_f32(0.0) };

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let input_ptr = unsafe { input.as_ptr().add(input_offset + j) };
        let qs_a_ptr = unsafe { qs_a.as_ptr().add(q_offset) };
        let qs_b_ptr = unsafe { qs_b.as_ptr().add(q_offset) };
        let qs_c_ptr = unsafe { qs_c.as_ptr().add(q_offset) };
        let qs_d_ptr = unsafe { qs_d.as_ptr().add(q_offset) };

        let (sc1_a, m1_a) = get_scale_min_k4(is, scales_a);
        let d1_a = d_a * sc1_a as f32;
        let m1_a = dmin_a * m1_a as f32;
        let (sc2_a, m2_a) = get_scale_min_k4(is + 1, scales_a);
        let d2_a = d_a * sc2_a as f32;
        let m2_a = dmin_a * m2_a as f32;

        let (sc1_b, m1_b) = get_scale_min_k4(is, scales_b);
        let d1_b = d_b * sc1_b as f32;
        let m1_b = dmin_b * m1_b as f32;
        let (sc2_b, m2_b) = get_scale_min_k4(is + 1, scales_b);
        let d2_b = d_b * sc2_b as f32;
        let m2_b = dmin_b * m2_b as f32;

        let (sc1_c, m1_c) = get_scale_min_k4(is, scales_c);
        let d1_c = d_c * sc1_c as f32;
        let m1_c = dmin_c * m1_c as f32;
        let (sc2_c, m2_c) = get_scale_min_k4(is + 1, scales_c);
        let d2_c = d_c * sc2_c as f32;
        let m2_c = dmin_c * m2_c as f32;

        let (sc1_d, m1_d) = get_scale_min_k4(is, scales_d);
        let d1_d = d_d * sc1_d as f32;
        let m1_d = dmin_d * m1_d as f32;
        let (sc2_d, m2_d) = get_scale_min_k4(is + 1, scales_d);
        let d2_d = d_d * sc2_d as f32;
        let m2_d = dmin_d * m2_d as f32;

        for l in (0..32).step_by(8) {
            let (coeff_left_a_lo, coeff_left_a_hi, coeff_right_a_lo, coeff_right_a_hi) =
                unsafe { decode_q4_coeffs8(qs_a_ptr, l, d1_a, m1_a, d2_a, m2_a) };
            let (coeff_left_b_lo, coeff_left_b_hi, coeff_right_b_lo, coeff_right_b_hi) =
                unsafe { decode_q4_coeffs8(qs_b_ptr, l, d1_b, m1_b, d2_b, m2_b) };
            let (coeff_left_c_lo, coeff_left_c_hi, coeff_right_c_lo, coeff_right_c_hi) =
                unsafe { decode_q4_coeffs8(qs_c_ptr, l, d1_c, m1_c, d2_c, m2_c) };
            let (coeff_left_d_lo, coeff_left_d_hi, coeff_right_d_lo, coeff_right_d_hi) =
                unsafe { decode_q4_coeffs8(qs_d_ptr, l, d1_d, m1_d, d2_d, m2_d) };

            let left_lo = unsafe { vld1q_f32(input_ptr.add(l)) };
            let right_lo = unsafe { vld1q_f32(input_ptr.add(32 + l)) };
            sum_a_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_a_v, coeff_left_a_lo, left_lo),
                    coeff_right_a_lo,
                    right_lo,
                )
            };
            sum_b_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_b_v, coeff_left_b_lo, left_lo),
                    coeff_right_b_lo,
                    right_lo,
                )
            };
            sum_c_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_c_v, coeff_left_c_lo, left_lo),
                    coeff_right_c_lo,
                    right_lo,
                )
            };
            sum_d_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_d_v, coeff_left_d_lo, left_lo),
                    coeff_right_d_lo,
                    right_lo,
                )
            };

            let left_hi = unsafe { vld1q_f32(input_ptr.add(l + 4)) };
            let right_hi = unsafe { vld1q_f32(input_ptr.add(36 + l)) };
            sum_a_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_a_v, coeff_left_a_hi, left_hi),
                    coeff_right_a_hi,
                    right_hi,
                )
            };
            sum_b_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_b_v, coeff_left_b_hi, left_hi),
                    coeff_right_b_hi,
                    right_hi,
                )
            };
            sum_c_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_c_v, coeff_left_c_hi, left_hi),
                    coeff_right_c_hi,
                    right_hi,
                )
            };
            sum_d_v = unsafe {
                vfmaq_f32(
                    vfmaq_f32(sum_d_v, coeff_left_d_hi, left_hi),
                    coeff_right_d_hi,
                    right_hi,
                )
            };
        }

        q_offset += 32;
        is += 2;
    }

    Ok((
        unsafe { vaddvq_f32(sum_a_v) },
        unsafe { vaddvq_f32(sum_b_v) },
        unsafe { vaddvq_f32(sum_c_v) },
        unsafe { vaddvq_f32(sum_d_v) },
    ))
}

#[inline(always)]
fn dot_q4_k_four_blocks_with_offset_single(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    input: &[f32],
    input_offset: usize,
) -> Result<(f32, f32, f32, f32)> {
    if block_a.len() != BLOCK_Q4_K_SIZE
        || block_b.len() != BLOCK_Q4_K_SIZE
        || block_c.len() != BLOCK_Q4_K_SIZE
        || block_d.len() != BLOCK_Q4_K_SIZE
        || input_offset + QK_K > input.len()
    {
        bail!("Q4_K four-row block single-input dot shape mismatch");
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q4_k_four_blocks_with_offset_single_neon(
            block_a,
            block_b,
            block_c,
            block_d,
            input,
            input_offset,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let inputs = [input];
        let mut sum_a = [0.0f32; 1];
        let mut sum_b = [0.0f32; 1];
        let mut sum_c = [0.0f32; 1];
        let mut sum_d = [0.0f32; 1];
        dot_many_q4_k_four_blocks_with_offset(
            block_a,
            block_b,
            block_c,
            block_d,
            &inputs,
            input_offset,
            &mut sum_a,
            &mut sum_b,
            &mut sum_c,
            &mut sum_d,
        )?;
        return Ok((sum_a[0], sum_b[0], sum_c[0], sum_d[0]));
    }
}

fn dot_many_q6_k_two_blocks_with_offset(
    block_a: &[u8],
    block_b: &[u8],
    inputs: &[&[f32]],
    input_offset: usize,
    sums_a: &mut [f32],
    sums_b: &mut [f32],
) -> Result<()> {
    if block_a.len() != BLOCK_Q6_K_SIZE
        || block_b.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
        || sums_a.len() != inputs.len()
        || sums_b.len() != inputs.len()
    {
        bail!("Q6_K paired block batched dot shape mismatch");
    }
    let ql_a = &block_a[..QK_K / 2];
    let qh_a = &block_a[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales_a = &block_a[scales_start..scales_start + QK_K / 16];
    let d_a = fp16_to_f32([block_a[BLOCK_Q6_K_SIZE - 2], block_a[BLOCK_Q6_K_SIZE - 1]]);

    let ql_b = &block_b[..QK_K / 2];
    let qh_b = &block_b[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_b = &block_b[scales_start..scales_start + QK_K / 16];
    let d_b = fp16_to_f32([block_b[BLOCK_Q6_K_SIZE - 2], block_b[BLOCK_Q6_K_SIZE - 1]]);

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let scale_factors_a = [
            d_a * (scales_a[sc_offset] as i8 as f32),
            d_a * (scales_a[sc_offset + 1] as i8 as f32),
            d_a * (scales_a[sc_offset + 2] as i8 as f32),
            d_a * (scales_a[sc_offset + 3] as i8 as f32),
            d_a * (scales_a[sc_offset + 4] as i8 as f32),
            d_a * (scales_a[sc_offset + 5] as i8 as f32),
            d_a * (scales_a[sc_offset + 6] as i8 as f32),
            d_a * (scales_a[sc_offset + 7] as i8 as f32),
        ];
        let scale_factors_b = [
            d_b * (scales_b[sc_offset] as i8 as f32),
            d_b * (scales_b[sc_offset + 1] as i8 as f32),
            d_b * (scales_b[sc_offset + 2] as i8 as f32),
            d_b * (scales_b[sc_offset + 3] as i8 as f32),
            d_b * (scales_b[sc_offset + 4] as i8 as f32),
            d_b * (scales_b[sc_offset + 5] as i8 as f32),
            d_b * (scales_b[sc_offset + 6] as i8 as f32),
            d_b * (scales_b[sc_offset + 7] as i8 as f32),
        ];
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte_a = qh_a[qh_offset + l];
            let q1_a = (((ql_a[ql_offset + l] & 0x0F) | (((qh_byte_a >> 0) & 0x03) << 4)) as i32
                - 32) as f32;
            let q2_a = (((ql_a[ql_offset + 32 + l] & 0x0F) | (((qh_byte_a >> 2) & 0x03) << 4))
                as i32
                - 32) as f32;
            let q3_a = ((((ql_a[ql_offset + l] >> 4) & 0x0F) | (((qh_byte_a >> 4) & 0x03) << 4))
                as i32
                - 32) as f32;
            let q4_a = ((((ql_a[ql_offset + 32 + l] >> 4) & 0x0F)
                | (((qh_byte_a >> 6) & 0x03) << 4)) as i32
                - 32) as f32;

            let qh_byte_b = qh_b[qh_offset + l];
            let q1_b = (((ql_b[ql_offset + l] & 0x0F) | (((qh_byte_b >> 0) & 0x03) << 4)) as i32
                - 32) as f32;
            let q2_b = (((ql_b[ql_offset + 32 + l] & 0x0F) | (((qh_byte_b >> 2) & 0x03) << 4))
                as i32
                - 32) as f32;
            let q3_b = ((((ql_b[ql_offset + l] >> 4) & 0x0F) | (((qh_byte_b >> 4) & 0x03) << 4))
                as i32
                - 32) as f32;
            let q4_b = ((((ql_b[ql_offset + 32 + l] >> 4) & 0x0F)
                | (((qh_byte_b >> 6) & 0x03) << 4)) as i32
                - 32) as f32;

            for input_idx in 0..inputs.len() {
                let input = inputs[input_idx];
                let x1 = input[input_offset + local_offset + l];
                let x2 = input[input_offset + local_offset + 32 + l];
                let x3 = input[input_offset + local_offset + 64 + l];
                let x4 = input[input_offset + local_offset + 96 + l];
                sums_a[input_idx] += scale_factors_a[scale_idx] * q1_a * x1
                    + scale_factors_a[scale_idx + 2] * q2_a * x2
                    + scale_factors_a[scale_idx + 4] * q3_a * x3
                    + scale_factors_a[scale_idx + 6] * q4_a * x4;
                sums_b[input_idx] += scale_factors_b[scale_idx] * q1_b * x1
                    + scale_factors_b[scale_idx + 2] * q2_b * x2
                    + scale_factors_b[scale_idx + 4] * q3_b * x3
                    + scale_factors_b[scale_idx + 6] * q4_b * x4;
            }
        }
        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }
    Ok(())
}

fn dot_many_q6_k_block_with_offset_six(
    block: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums: &mut [f32; 6],
) -> Result<()> {
    if block.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q6_K block batched dot six-input fast path shape mismatch");
    }
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales = &block[scales_start..scales_start + QK_K / 16];
    let d = fp16_to_f32([block[BLOCK_Q6_K_SIZE - 2], block[BLOCK_Q6_K_SIZE - 1]]);
    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];

    let mut sum0 = sums[0];
    let mut sum1 = sums[1];
    let mut sum2 = sums[2];
    let mut sum3 = sums[3];
    let mut sum4 = sums[4];
    let mut sum5 = sums[5];

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let sf0 = d * (unsafe { *scales.get_unchecked(sc_offset) } as i8 as f32);
        let sf1 = d * (unsafe { *scales.get_unchecked(sc_offset + 1) } as i8 as f32);
        let sf2 = d * (unsafe { *scales.get_unchecked(sc_offset + 2) } as i8 as f32);
        let sf3 = d * (unsafe { *scales.get_unchecked(sc_offset + 3) } as i8 as f32);
        let sf4 = d * (unsafe { *scales.get_unchecked(sc_offset + 4) } as i8 as f32);
        let sf5 = d * (unsafe { *scales.get_unchecked(sc_offset + 5) } as i8 as f32);
        let sf6 = d * (unsafe { *scales.get_unchecked(sc_offset + 6) } as i8 as f32);
        let sf7 = d * (unsafe { *scales.get_unchecked(sc_offset + 7) } as i8 as f32);
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte = unsafe { *qh.get_unchecked(qh_offset + l) };
            let ql_lo = unsafe { *ql.get_unchecked(ql_offset + l) };
            let ql_hi = unsafe { *ql.get_unchecked(ql_offset + 32 + l) };
            let q1 = (((ql_lo & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2 = (((ql_hi & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3 = ((((ql_lo >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4 = ((((ql_hi >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4)) as i32 - 32) as f32;
            let (v1, v2, v3, v4) = if scale_idx == 0 {
                (sf0 * q1, sf2 * q2, sf4 * q3, sf6 * q4)
            } else {
                (sf1 * q1, sf3 * q2, sf5 * q3, sf7 * q4)
            };

            let idx1 = input_offset + local_offset + l;
            let idx2 = input_offset + local_offset + 32 + l;
            let idx3 = input_offset + local_offset + 64 + l;
            let idx4 = input_offset + local_offset + 96 + l;

            let x10 = unsafe { *input0.get_unchecked(idx1) };
            let x20 = unsafe { *input0.get_unchecked(idx2) };
            let x30 = unsafe { *input0.get_unchecked(idx3) };
            let x40 = unsafe { *input0.get_unchecked(idx4) };
            sum0 = v4.mul_add(x40, v3.mul_add(x30, v2.mul_add(x20, v1.mul_add(x10, sum0))));

            let x11 = unsafe { *input1.get_unchecked(idx1) };
            let x21 = unsafe { *input1.get_unchecked(idx2) };
            let x31 = unsafe { *input1.get_unchecked(idx3) };
            let x41 = unsafe { *input1.get_unchecked(idx4) };
            sum1 = v4.mul_add(x41, v3.mul_add(x31, v2.mul_add(x21, v1.mul_add(x11, sum1))));

            let x12 = unsafe { *input2.get_unchecked(idx1) };
            let x22 = unsafe { *input2.get_unchecked(idx2) };
            let x32 = unsafe { *input2.get_unchecked(idx3) };
            let x42 = unsafe { *input2.get_unchecked(idx4) };
            sum2 = v4.mul_add(x42, v3.mul_add(x32, v2.mul_add(x22, v1.mul_add(x12, sum2))));

            let x13 = unsafe { *input3.get_unchecked(idx1) };
            let x23 = unsafe { *input3.get_unchecked(idx2) };
            let x33 = unsafe { *input3.get_unchecked(idx3) };
            let x43 = unsafe { *input3.get_unchecked(idx4) };
            sum3 = v4.mul_add(x43, v3.mul_add(x33, v2.mul_add(x23, v1.mul_add(x13, sum3))));

            let x14 = unsafe { *input4.get_unchecked(idx1) };
            let x24 = unsafe { *input4.get_unchecked(idx2) };
            let x34 = unsafe { *input4.get_unchecked(idx3) };
            let x44 = unsafe { *input4.get_unchecked(idx4) };
            sum4 = v4.mul_add(x44, v3.mul_add(x34, v2.mul_add(x24, v1.mul_add(x14, sum4))));

            let x15 = unsafe { *input5.get_unchecked(idx1) };
            let x25 = unsafe { *input5.get_unchecked(idx2) };
            let x35 = unsafe { *input5.get_unchecked(idx3) };
            let x45 = unsafe { *input5.get_unchecked(idx4) };
            sum5 = v4.mul_add(x45, v3.mul_add(x35, v2.mul_add(x25, v1.mul_add(x15, sum5))));
        }
        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }

    sums[0] = sum0;
    sums[1] = sum1;
    sums[2] = sum2;
    sums[3] = sum3;
    sums[4] = sum4;
    sums[5] = sum5;
    Ok(())
}

fn dot_many_q6_k_two_blocks_with_offset_six(
    block_a: &[u8],
    block_b: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
) -> Result<()> {
    if block_a.len() != BLOCK_Q6_K_SIZE
        || block_b.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q6_K paired block batched dot six-input fast path shape mismatch");
    }

    let ql_a = &block_a[..QK_K / 2];
    let qh_a = &block_a[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales_a = &block_a[scales_start..scales_start + QK_K / 16];
    let d_a = fp16_to_f32([block_a[BLOCK_Q6_K_SIZE - 2], block_a[BLOCK_Q6_K_SIZE - 1]]);

    let ql_b = &block_b[..QK_K / 2];
    let qh_b = &block_b[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_b = &block_b[scales_start..scales_start + QK_K / 16];
    let d_b = fp16_to_f32([block_b[BLOCK_Q6_K_SIZE - 2], block_b[BLOCK_Q6_K_SIZE - 1]]);
    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];

    let mut sum_a0 = sums_a[0];
    let mut sum_a1 = sums_a[1];
    let mut sum_a2 = sums_a[2];
    let mut sum_a3 = sums_a[3];
    let mut sum_a4 = sums_a[4];
    let mut sum_a5 = sums_a[5];
    let mut sum_b0 = sums_b[0];
    let mut sum_b1 = sums_b[1];
    let mut sum_b2 = sums_b[2];
    let mut sum_b3 = sums_b[3];
    let mut sum_b4 = sums_b[4];
    let mut sum_b5 = sums_b[5];

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let a_sf0 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset) } as i8 as f32);
        let a_sf1 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 1) } as i8 as f32);
        let a_sf2 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 2) } as i8 as f32);
        let a_sf3 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 3) } as i8 as f32);
        let a_sf4 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 4) } as i8 as f32);
        let a_sf5 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 5) } as i8 as f32);
        let a_sf6 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 6) } as i8 as f32);
        let a_sf7 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 7) } as i8 as f32);
        let b_sf0 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset) } as i8 as f32);
        let b_sf1 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 1) } as i8 as f32);
        let b_sf2 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 2) } as i8 as f32);
        let b_sf3 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 3) } as i8 as f32);
        let b_sf4 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 4) } as i8 as f32);
        let b_sf5 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 5) } as i8 as f32);
        let b_sf6 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 6) } as i8 as f32);
        let b_sf7 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 7) } as i8 as f32);
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte_a = unsafe { *qh_a.get_unchecked(qh_offset + l) };
            let ql_a_lo = unsafe { *ql_a.get_unchecked(ql_offset + l) };
            let ql_a_hi = unsafe { *ql_a.get_unchecked(ql_offset + 32 + l) };
            let q1_a = (((ql_a_lo & 0x0F) | (((qh_byte_a >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_a = (((ql_a_hi & 0x0F) | (((qh_byte_a >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_a =
                ((((ql_a_lo >> 4) & 0x0F) | (((qh_byte_a >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_a =
                ((((ql_a_hi >> 4) & 0x0F) | (((qh_byte_a >> 6) & 0x03) << 4)) as i32 - 32) as f32;

            let qh_byte_b = unsafe { *qh_b.get_unchecked(qh_offset + l) };
            let ql_b_lo = unsafe { *ql_b.get_unchecked(ql_offset + l) };
            let ql_b_hi = unsafe { *ql_b.get_unchecked(ql_offset + 32 + l) };
            let q1_b = (((ql_b_lo & 0x0F) | (((qh_byte_b >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_b = (((ql_b_hi & 0x0F) | (((qh_byte_b >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_b =
                ((((ql_b_lo >> 4) & 0x0F) | (((qh_byte_b >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_b =
                ((((ql_b_hi >> 4) & 0x0F) | (((qh_byte_b >> 6) & 0x03) << 4)) as i32 - 32) as f32;
            let (a_v1, a_v2, a_v3, a_v4) = if scale_idx == 0 {
                (a_sf0 * q1_a, a_sf2 * q2_a, a_sf4 * q3_a, a_sf6 * q4_a)
            } else {
                (a_sf1 * q1_a, a_sf3 * q2_a, a_sf5 * q3_a, a_sf7 * q4_a)
            };
            let (b_v1, b_v2, b_v3, b_v4) = if scale_idx == 0 {
                (b_sf0 * q1_b, b_sf2 * q2_b, b_sf4 * q3_b, b_sf6 * q4_b)
            } else {
                (b_sf1 * q1_b, b_sf3 * q2_b, b_sf5 * q3_b, b_sf7 * q4_b)
            };

            let idx1 = input_offset + local_offset + l;
            let idx2 = input_offset + local_offset + 32 + l;
            let idx3 = input_offset + local_offset + 64 + l;
            let idx4 = input_offset + local_offset + 96 + l;

            let x10 = unsafe { *input0.get_unchecked(idx1) };
            let x20 = unsafe { *input0.get_unchecked(idx2) };
            let x30 = unsafe { *input0.get_unchecked(idx3) };
            let x40 = unsafe { *input0.get_unchecked(idx4) };
            sum_a0 = a_v4.mul_add(
                x40,
                a_v3.mul_add(x30, a_v2.mul_add(x20, a_v1.mul_add(x10, sum_a0))),
            );
            sum_b0 = b_v4.mul_add(
                x40,
                b_v3.mul_add(x30, b_v2.mul_add(x20, b_v1.mul_add(x10, sum_b0))),
            );

            let x11 = unsafe { *input1.get_unchecked(idx1) };
            let x21 = unsafe { *input1.get_unchecked(idx2) };
            let x31 = unsafe { *input1.get_unchecked(idx3) };
            let x41 = unsafe { *input1.get_unchecked(idx4) };
            sum_a1 = a_v4.mul_add(
                x41,
                a_v3.mul_add(x31, a_v2.mul_add(x21, a_v1.mul_add(x11, sum_a1))),
            );
            sum_b1 = b_v4.mul_add(
                x41,
                b_v3.mul_add(x31, b_v2.mul_add(x21, b_v1.mul_add(x11, sum_b1))),
            );

            let x12 = unsafe { *input2.get_unchecked(idx1) };
            let x22 = unsafe { *input2.get_unchecked(idx2) };
            let x32 = unsafe { *input2.get_unchecked(idx3) };
            let x42 = unsafe { *input2.get_unchecked(idx4) };
            sum_a2 = a_v4.mul_add(
                x42,
                a_v3.mul_add(x32, a_v2.mul_add(x22, a_v1.mul_add(x12, sum_a2))),
            );
            sum_b2 = b_v4.mul_add(
                x42,
                b_v3.mul_add(x32, b_v2.mul_add(x22, b_v1.mul_add(x12, sum_b2))),
            );

            let x13 = unsafe { *input3.get_unchecked(idx1) };
            let x23 = unsafe { *input3.get_unchecked(idx2) };
            let x33 = unsafe { *input3.get_unchecked(idx3) };
            let x43 = unsafe { *input3.get_unchecked(idx4) };
            sum_a3 = a_v4.mul_add(
                x43,
                a_v3.mul_add(x33, a_v2.mul_add(x23, a_v1.mul_add(x13, sum_a3))),
            );
            sum_b3 = b_v4.mul_add(
                x43,
                b_v3.mul_add(x33, b_v2.mul_add(x23, b_v1.mul_add(x13, sum_b3))),
            );

            let x14 = unsafe { *input4.get_unchecked(idx1) };
            let x24 = unsafe { *input4.get_unchecked(idx2) };
            let x34 = unsafe { *input4.get_unchecked(idx3) };
            let x44 = unsafe { *input4.get_unchecked(idx4) };
            sum_a4 = a_v4.mul_add(
                x44,
                a_v3.mul_add(x34, a_v2.mul_add(x24, a_v1.mul_add(x14, sum_a4))),
            );
            sum_b4 = b_v4.mul_add(
                x44,
                b_v3.mul_add(x34, b_v2.mul_add(x24, b_v1.mul_add(x14, sum_b4))),
            );

            let x15 = unsafe { *input5.get_unchecked(idx1) };
            let x25 = unsafe { *input5.get_unchecked(idx2) };
            let x35 = unsafe { *input5.get_unchecked(idx3) };
            let x45 = unsafe { *input5.get_unchecked(idx4) };
            sum_a5 = a_v4.mul_add(
                x45,
                a_v3.mul_add(x35, a_v2.mul_add(x25, a_v1.mul_add(x15, sum_a5))),
            );
            sum_b5 = b_v4.mul_add(
                x45,
                b_v3.mul_add(x35, b_v2.mul_add(x25, b_v1.mul_add(x15, sum_b5))),
            );
        }
        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }

    sums_a[0] = sum_a0;
    sums_a[1] = sum_a1;
    sums_a[2] = sum_a2;
    sums_a[3] = sum_a3;
    sums_a[4] = sum_a4;
    sums_a[5] = sum_a5;
    sums_b[0] = sum_b0;
    sums_b[1] = sum_b1;
    sums_b[2] = sum_b2;
    sums_b[3] = sum_b3;
    sums_b[4] = sum_b4;
    sums_b[5] = sum_b5;
    Ok(())
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_many_q6_k_four_blocks_with_offset_six_neon(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    let ql_a = &block_a[..QK_K / 2];
    let qh_a = &block_a[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales_a = &block_a[scales_start..scales_start + QK_K / 16];
    let d_a = fp16_to_f32([block_a[BLOCK_Q6_K_SIZE - 2], block_a[BLOCK_Q6_K_SIZE - 1]]);

    let ql_b = &block_b[..QK_K / 2];
    let qh_b = &block_b[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_b = &block_b[scales_start..scales_start + QK_K / 16];
    let d_b = fp16_to_f32([block_b[BLOCK_Q6_K_SIZE - 2], block_b[BLOCK_Q6_K_SIZE - 1]]);

    let ql_c = &block_c[..QK_K / 2];
    let qh_c = &block_c[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_c = &block_c[scales_start..scales_start + QK_K / 16];
    let d_c = fp16_to_f32([block_c[BLOCK_Q6_K_SIZE - 2], block_c[BLOCK_Q6_K_SIZE - 1]]);

    let ql_d = &block_d[..QK_K / 2];
    let qh_d = &block_d[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_d = &block_d[scales_start..scales_start + QK_K / 16];
    let d_d = fp16_to_f32([block_d[BLOCK_Q6_K_SIZE - 2], block_d[BLOCK_Q6_K_SIZE - 1]]);

    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];

    let mut sum_a0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_a5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_b5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_c5_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d0_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d1_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d2_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d3_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d4_v = unsafe { vdupq_n_f32(0.0) };
    let mut sum_d5_v = unsafe { vdupq_n_f32(0.0) };

    macro_rules! accum_input_q6 {
        ($sum_a:ident, $sum_b:ident, $sum_c:ident, $sum_d:ident, $ptr:ident, $l:expr,
         $a1:ident, $a2:ident, $a3:ident, $a4:ident,
         $b1:ident, $b2:ident, $b3:ident, $b4:ident,
         $c1:ident, $c2:ident, $c3:ident, $c4:ident,
         $d1:ident, $d2:ident, $d3:ident, $d4:ident) => {{
            let x1 = unsafe { vld1q_f32($ptr.add($l)) };
            let x2 = unsafe { vld1q_f32($ptr.add(32 + $l)) };
            let x3 = unsafe { vld1q_f32($ptr.add(64 + $l)) };
            let x4 = unsafe { vld1q_f32($ptr.add(96 + $l)) };
            $sum_a = unsafe {
                vfmaq_f32(
                    vfmaq_f32(vfmaq_f32(vfmaq_f32($sum_a, $a1, x1), $a2, x2), $a3, x3),
                    $a4,
                    x4,
                )
            };
            $sum_b = unsafe {
                vfmaq_f32(
                    vfmaq_f32(vfmaq_f32(vfmaq_f32($sum_b, $b1, x1), $b2, x2), $b3, x3),
                    $b4,
                    x4,
                )
            };
            $sum_c = unsafe {
                vfmaq_f32(
                    vfmaq_f32(vfmaq_f32(vfmaq_f32($sum_c, $c1, x1), $c2, x2), $c3, x3),
                    $c4,
                    x4,
                )
            };
            $sum_d = unsafe {
                vfmaq_f32(
                    vfmaq_f32(vfmaq_f32(vfmaq_f32($sum_d, $d1, x1), $d2, x2), $d3, x3),
                    $d4,
                    x4,
                )
            };
        }};
    }

    macro_rules! coeffs_q6 {
        ($ql_ptr:ident, $qh_ptr:ident, $l:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr) => {{
            let qh0 = unsafe { *$qh_ptr.add($l) };
            let qh1 = unsafe { *$qh_ptr.add($l + 1) };
            let qh2 = unsafe { *$qh_ptr.add($l + 2) };
            let qh3 = unsafe { *$qh_ptr.add($l + 3) };
            let ql_lo0 = unsafe { *$ql_ptr.add($l) };
            let ql_lo1 = unsafe { *$ql_ptr.add($l + 1) };
            let ql_lo2 = unsafe { *$ql_ptr.add($l + 2) };
            let ql_lo3 = unsafe { *$ql_ptr.add($l + 3) };
            let ql_hi0 = unsafe { *$ql_ptr.add(32 + $l) };
            let ql_hi1 = unsafe { *$ql_ptr.add(32 + $l + 1) };
            let ql_hi2 = unsafe { *$ql_ptr.add(32 + $l + 2) };
            let ql_hi3 = unsafe { *$ql_ptr.add(32 + $l + 3) };

            let v1 = [
                ((((ql_lo0 & 0x0F) | (((qh0 >> 0) & 0x03) << 4)) as i32 - 32) as f32) * $s1,
                ((((ql_lo1 & 0x0F) | (((qh1 >> 0) & 0x03) << 4)) as i32 - 32) as f32) * $s1,
                ((((ql_lo2 & 0x0F) | (((qh2 >> 0) & 0x03) << 4)) as i32 - 32) as f32) * $s1,
                ((((ql_lo3 & 0x0F) | (((qh3 >> 0) & 0x03) << 4)) as i32 - 32) as f32) * $s1,
            ];
            let v2 = [
                ((((ql_hi0 & 0x0F) | (((qh0 >> 2) & 0x03) << 4)) as i32 - 32) as f32) * $s2,
                ((((ql_hi1 & 0x0F) | (((qh1 >> 2) & 0x03) << 4)) as i32 - 32) as f32) * $s2,
                ((((ql_hi2 & 0x0F) | (((qh2 >> 2) & 0x03) << 4)) as i32 - 32) as f32) * $s2,
                ((((ql_hi3 & 0x0F) | (((qh3 >> 2) & 0x03) << 4)) as i32 - 32) as f32) * $s2,
            ];
            let v3 = [
                (((((ql_lo0 >> 4) & 0x0F) | (((qh0 >> 4) & 0x03) << 4)) as i32 - 32) as f32) * $s3,
                (((((ql_lo1 >> 4) & 0x0F) | (((qh1 >> 4) & 0x03) << 4)) as i32 - 32) as f32) * $s3,
                (((((ql_lo2 >> 4) & 0x0F) | (((qh2 >> 4) & 0x03) << 4)) as i32 - 32) as f32) * $s3,
                (((((ql_lo3 >> 4) & 0x0F) | (((qh3 >> 4) & 0x03) << 4)) as i32 - 32) as f32) * $s3,
            ];
            let v4 = [
                (((((ql_hi0 >> 4) & 0x0F) | (((qh0 >> 6) & 0x03) << 4)) as i32 - 32) as f32) * $s4,
                (((((ql_hi1 >> 4) & 0x0F) | (((qh1 >> 6) & 0x03) << 4)) as i32 - 32) as f32) * $s4,
                (((((ql_hi2 >> 4) & 0x0F) | (((qh2 >> 6) & 0x03) << 4)) as i32 - 32) as f32) * $s4,
                (((((ql_hi3 >> 4) & 0x0F) | (((qh3 >> 6) & 0x03) << 4)) as i32 - 32) as f32) * $s4,
            ];
            unsafe {
                (
                    vld1q_f32(v1.as_ptr()),
                    vld1q_f32(v2.as_ptr()),
                    vld1q_f32(v3.as_ptr()),
                    vld1q_f32(v4.as_ptr()),
                )
            }
        }};
    }

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let a_sf0 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset) } as i8 as f32);
        let a_sf1 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 1) } as i8 as f32);
        let a_sf2 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 2) } as i8 as f32);
        let a_sf3 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 3) } as i8 as f32);
        let a_sf4 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 4) } as i8 as f32);
        let a_sf5 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 5) } as i8 as f32);
        let a_sf6 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 6) } as i8 as f32);
        let a_sf7 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 7) } as i8 as f32);
        let b_sf0 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset) } as i8 as f32);
        let b_sf1 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 1) } as i8 as f32);
        let b_sf2 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 2) } as i8 as f32);
        let b_sf3 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 3) } as i8 as f32);
        let b_sf4 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 4) } as i8 as f32);
        let b_sf5 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 5) } as i8 as f32);
        let b_sf6 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 6) } as i8 as f32);
        let b_sf7 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 7) } as i8 as f32);
        let c_sf0 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset) } as i8 as f32);
        let c_sf1 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 1) } as i8 as f32);
        let c_sf2 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 2) } as i8 as f32);
        let c_sf3 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 3) } as i8 as f32);
        let c_sf4 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 4) } as i8 as f32);
        let c_sf5 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 5) } as i8 as f32);
        let c_sf6 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 6) } as i8 as f32);
        let c_sf7 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 7) } as i8 as f32);
        let d_sf0 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset) } as i8 as f32);
        let d_sf1 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 1) } as i8 as f32);
        let d_sf2 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 2) } as i8 as f32);
        let d_sf3 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 3) } as i8 as f32);
        let d_sf4 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 4) } as i8 as f32);
        let d_sf5 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 5) } as i8 as f32);
        let d_sf6 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 6) } as i8 as f32);
        let d_sf7 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 7) } as i8 as f32);

        let input0_ptr = unsafe { input0.as_ptr().add(input_offset + local_offset) };
        let input1_ptr = unsafe { input1.as_ptr().add(input_offset + local_offset) };
        let input2_ptr = unsafe { input2.as_ptr().add(input_offset + local_offset) };
        let input3_ptr = unsafe { input3.as_ptr().add(input_offset + local_offset) };
        let input4_ptr = unsafe { input4.as_ptr().add(input_offset + local_offset) };
        let input5_ptr = unsafe { input5.as_ptr().add(input_offset + local_offset) };
        let ql_a_ptr = unsafe { ql_a.as_ptr().add(ql_offset) };
        let qh_a_ptr = unsafe { qh_a.as_ptr().add(qh_offset) };
        let ql_b_ptr = unsafe { ql_b.as_ptr().add(ql_offset) };
        let qh_b_ptr = unsafe { qh_b.as_ptr().add(qh_offset) };
        let ql_c_ptr = unsafe { ql_c.as_ptr().add(ql_offset) };
        let qh_c_ptr = unsafe { qh_c.as_ptr().add(qh_offset) };
        let ql_d_ptr = unsafe { ql_d.as_ptr().add(ql_offset) };
        let qh_d_ptr = unsafe { qh_d.as_ptr().add(qh_offset) };

        for l in (0..32).step_by(4) {
            let (a_s1, a_s2, a_s3, a_s4) = if l < 16 {
                (a_sf0, a_sf2, a_sf4, a_sf6)
            } else {
                (a_sf1, a_sf3, a_sf5, a_sf7)
            };
            let (b_s1, b_s2, b_s3, b_s4) = if l < 16 {
                (b_sf0, b_sf2, b_sf4, b_sf6)
            } else {
                (b_sf1, b_sf3, b_sf5, b_sf7)
            };
            let (c_s1, c_s2, c_s3, c_s4) = if l < 16 {
                (c_sf0, c_sf2, c_sf4, c_sf6)
            } else {
                (c_sf1, c_sf3, c_sf5, c_sf7)
            };
            let (d_s1, d_s2, d_s3, d_s4) = if l < 16 {
                (d_sf0, d_sf2, d_sf4, d_sf6)
            } else {
                (d_sf1, d_sf3, d_sf5, d_sf7)
            };
            let (a_v1, a_v2, a_v3, a_v4) =
                coeffs_q6!(ql_a_ptr, qh_a_ptr, l, a_s1, a_s2, a_s3, a_s4);
            let (b_v1, b_v2, b_v3, b_v4) =
                coeffs_q6!(ql_b_ptr, qh_b_ptr, l, b_s1, b_s2, b_s3, b_s4);
            let (c_v1, c_v2, c_v3, c_v4) =
                coeffs_q6!(ql_c_ptr, qh_c_ptr, l, c_s1, c_s2, c_s3, c_s4);
            let (d_v1, d_v2, d_v3, d_v4) =
                coeffs_q6!(ql_d_ptr, qh_d_ptr, l, d_s1, d_s2, d_s3, d_s4);

            accum_input_q6!(
                sum_a0_v, sum_b0_v, sum_c0_v, sum_d0_v, input0_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
            accum_input_q6!(
                sum_a1_v, sum_b1_v, sum_c1_v, sum_d1_v, input1_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
            accum_input_q6!(
                sum_a2_v, sum_b2_v, sum_c2_v, sum_d2_v, input2_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
            accum_input_q6!(
                sum_a3_v, sum_b3_v, sum_c3_v, sum_d3_v, input3_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
            accum_input_q6!(
                sum_a4_v, sum_b4_v, sum_c4_v, sum_d4_v, input4_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
            accum_input_q6!(
                sum_a5_v, sum_b5_v, sum_c5_v, sum_d5_v, input5_ptr, l, a_v1, a_v2, a_v3, a_v4,
                b_v1, b_v2, b_v3, b_v4, c_v1, c_v2, c_v3, c_v4, d_v1, d_v2, d_v3, d_v4
            );
        }

        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }

    sums_a[0] += unsafe { vaddvq_f32(sum_a0_v) };
    sums_a[1] += unsafe { vaddvq_f32(sum_a1_v) };
    sums_a[2] += unsafe { vaddvq_f32(sum_a2_v) };
    sums_a[3] += unsafe { vaddvq_f32(sum_a3_v) };
    sums_a[4] += unsafe { vaddvq_f32(sum_a4_v) };
    sums_a[5] += unsafe { vaddvq_f32(sum_a5_v) };
    sums_b[0] += unsafe { vaddvq_f32(sum_b0_v) };
    sums_b[1] += unsafe { vaddvq_f32(sum_b1_v) };
    sums_b[2] += unsafe { vaddvq_f32(sum_b2_v) };
    sums_b[3] += unsafe { vaddvq_f32(sum_b3_v) };
    sums_b[4] += unsafe { vaddvq_f32(sum_b4_v) };
    sums_b[5] += unsafe { vaddvq_f32(sum_b5_v) };
    sums_c[0] += unsafe { vaddvq_f32(sum_c0_v) };
    sums_c[1] += unsafe { vaddvq_f32(sum_c1_v) };
    sums_c[2] += unsafe { vaddvq_f32(sum_c2_v) };
    sums_c[3] += unsafe { vaddvq_f32(sum_c3_v) };
    sums_c[4] += unsafe { vaddvq_f32(sum_c4_v) };
    sums_c[5] += unsafe { vaddvq_f32(sum_c5_v) };
    sums_d[0] += unsafe { vaddvq_f32(sum_d0_v) };
    sums_d[1] += unsafe { vaddvq_f32(sum_d1_v) };
    sums_d[2] += unsafe { vaddvq_f32(sum_d2_v) };
    sums_d[3] += unsafe { vaddvq_f32(sum_d3_v) };
    sums_d[4] += unsafe { vaddvq_f32(sum_d4_v) };
    sums_d[5] += unsafe { vaddvq_f32(sum_d5_v) };
    Ok(())
}

fn dot_many_q6_k_four_blocks_with_offset_six(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    if block_a.len() != BLOCK_Q6_K_SIZE
        || block_b.len() != BLOCK_Q6_K_SIZE
        || block_c.len() != BLOCK_Q6_K_SIZE
        || block_d.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q6_K four-row block batched dot six-input fast path shape mismatch");
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_many_q6_k_four_blocks_with_offset_six_neon(
            block_a,
            block_b,
            block_c,
            block_d,
            inputs,
            input_offset,
            sums_a,
            sums_b,
            sums_c,
            sums_d,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        return dot_many_q6_k_four_blocks_with_offset_six_scalar(
            block_a,
            block_b,
            block_c,
            block_d,
            inputs,
            input_offset,
            sums_a,
            sums_b,
            sums_c,
            sums_d,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_many_q6_k_four_blocks_with_offset_six_scalar(
    block_a: &[u8],
    block_b: &[u8],
    block_c: &[u8],
    block_d: &[u8],
    inputs: &[&[f32]; 6],
    input_offset: usize,
    sums_a: &mut [f32; 6],
    sums_b: &mut [f32; 6],
    sums_c: &mut [f32; 6],
    sums_d: &mut [f32; 6],
) -> Result<()> {
    if block_a.len() != BLOCK_Q6_K_SIZE
        || block_b.len() != BLOCK_Q6_K_SIZE
        || block_c.len() != BLOCK_Q6_K_SIZE
        || block_d.len() != BLOCK_Q6_K_SIZE
        || inputs.iter().any(|input| input_offset + QK_K > input.len())
    {
        bail!("Q6_K four-row block batched dot six-input fast path shape mismatch");
    }

    let ql_a = &block_a[..QK_K / 2];
    let qh_a = &block_a[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales_a = &block_a[scales_start..scales_start + QK_K / 16];
    let d_a = fp16_to_f32([block_a[BLOCK_Q6_K_SIZE - 2], block_a[BLOCK_Q6_K_SIZE - 1]]);

    let ql_b = &block_b[..QK_K / 2];
    let qh_b = &block_b[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_b = &block_b[scales_start..scales_start + QK_K / 16];
    let d_b = fp16_to_f32([block_b[BLOCK_Q6_K_SIZE - 2], block_b[BLOCK_Q6_K_SIZE - 1]]);

    let ql_c = &block_c[..QK_K / 2];
    let qh_c = &block_c[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_c = &block_c[scales_start..scales_start + QK_K / 16];
    let d_c = fp16_to_f32([block_c[BLOCK_Q6_K_SIZE - 2], block_c[BLOCK_Q6_K_SIZE - 1]]);

    let ql_d = &block_d[..QK_K / 2];
    let qh_d = &block_d[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_d = &block_d[scales_start..scales_start + QK_K / 16];
    let d_d = fp16_to_f32([block_d[BLOCK_Q6_K_SIZE - 2], block_d[BLOCK_Q6_K_SIZE - 1]]);
    let input0 = inputs[0];
    let input1 = inputs[1];
    let input2 = inputs[2];
    let input3 = inputs[3];
    let input4 = inputs[4];
    let input5 = inputs[5];

    let mut sum_a0 = sums_a[0];
    let mut sum_a1 = sums_a[1];
    let mut sum_a2 = sums_a[2];
    let mut sum_a3 = sums_a[3];
    let mut sum_a4 = sums_a[4];
    let mut sum_a5 = sums_a[5];
    let mut sum_b0 = sums_b[0];
    let mut sum_b1 = sums_b[1];
    let mut sum_b2 = sums_b[2];
    let mut sum_b3 = sums_b[3];
    let mut sum_b4 = sums_b[4];
    let mut sum_b5 = sums_b[5];
    let mut sum_c0 = sums_c[0];
    let mut sum_c1 = sums_c[1];
    let mut sum_c2 = sums_c[2];
    let mut sum_c3 = sums_c[3];
    let mut sum_c4 = sums_c[4];
    let mut sum_c5 = sums_c[5];
    let mut sum_d0 = sums_d[0];
    let mut sum_d1 = sums_d[1];
    let mut sum_d2 = sums_d[2];
    let mut sum_d3 = sums_d[3];
    let mut sum_d4 = sums_d[4];
    let mut sum_d5 = sums_d[5];

    let mut local_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let a_sf0 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset) } as i8 as f32);
        let a_sf1 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 1) } as i8 as f32);
        let a_sf2 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 2) } as i8 as f32);
        let a_sf3 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 3) } as i8 as f32);
        let a_sf4 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 4) } as i8 as f32);
        let a_sf5 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 5) } as i8 as f32);
        let a_sf6 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 6) } as i8 as f32);
        let a_sf7 = d_a * (unsafe { *scales_a.get_unchecked(sc_offset + 7) } as i8 as f32);
        let b_sf0 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset) } as i8 as f32);
        let b_sf1 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 1) } as i8 as f32);
        let b_sf2 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 2) } as i8 as f32);
        let b_sf3 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 3) } as i8 as f32);
        let b_sf4 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 4) } as i8 as f32);
        let b_sf5 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 5) } as i8 as f32);
        let b_sf6 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 6) } as i8 as f32);
        let b_sf7 = d_b * (unsafe { *scales_b.get_unchecked(sc_offset + 7) } as i8 as f32);
        let c_sf0 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset) } as i8 as f32);
        let c_sf1 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 1) } as i8 as f32);
        let c_sf2 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 2) } as i8 as f32);
        let c_sf3 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 3) } as i8 as f32);
        let c_sf4 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 4) } as i8 as f32);
        let c_sf5 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 5) } as i8 as f32);
        let c_sf6 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 6) } as i8 as f32);
        let c_sf7 = d_c * (unsafe { *scales_c.get_unchecked(sc_offset + 7) } as i8 as f32);
        let d_sf0 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset) } as i8 as f32);
        let d_sf1 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 1) } as i8 as f32);
        let d_sf2 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 2) } as i8 as f32);
        let d_sf3 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 3) } as i8 as f32);
        let d_sf4 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 4) } as i8 as f32);
        let d_sf5 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 5) } as i8 as f32);
        let d_sf6 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 6) } as i8 as f32);
        let d_sf7 = d_d * (unsafe { *scales_d.get_unchecked(sc_offset + 7) } as i8 as f32);
        for l in 0..32 {
            let scale_idx = l / 16;
            let qh_byte_a = unsafe { *qh_a.get_unchecked(qh_offset + l) };
            let ql_a_lo = unsafe { *ql_a.get_unchecked(ql_offset + l) };
            let ql_a_hi = unsafe { *ql_a.get_unchecked(ql_offset + 32 + l) };
            let q1_a = (((ql_a_lo & 0x0F) | (((qh_byte_a >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_a = (((ql_a_hi & 0x0F) | (((qh_byte_a >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_a =
                ((((ql_a_lo >> 4) & 0x0F) | (((qh_byte_a >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_a =
                ((((ql_a_hi >> 4) & 0x0F) | (((qh_byte_a >> 6) & 0x03) << 4)) as i32 - 32) as f32;

            let qh_byte_b = unsafe { *qh_b.get_unchecked(qh_offset + l) };
            let ql_b_lo = unsafe { *ql_b.get_unchecked(ql_offset + l) };
            let ql_b_hi = unsafe { *ql_b.get_unchecked(ql_offset + 32 + l) };
            let q1_b = (((ql_b_lo & 0x0F) | (((qh_byte_b >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_b = (((ql_b_hi & 0x0F) | (((qh_byte_b >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_b =
                ((((ql_b_lo >> 4) & 0x0F) | (((qh_byte_b >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_b =
                ((((ql_b_hi >> 4) & 0x0F) | (((qh_byte_b >> 6) & 0x03) << 4)) as i32 - 32) as f32;

            let qh_byte_c = unsafe { *qh_c.get_unchecked(qh_offset + l) };
            let ql_c_lo = unsafe { *ql_c.get_unchecked(ql_offset + l) };
            let ql_c_hi = unsafe { *ql_c.get_unchecked(ql_offset + 32 + l) };
            let q1_c = (((ql_c_lo & 0x0F) | (((qh_byte_c >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_c = (((ql_c_hi & 0x0F) | (((qh_byte_c >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_c =
                ((((ql_c_lo >> 4) & 0x0F) | (((qh_byte_c >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_c =
                ((((ql_c_hi >> 4) & 0x0F) | (((qh_byte_c >> 6) & 0x03) << 4)) as i32 - 32) as f32;

            let qh_byte_d = unsafe { *qh_d.get_unchecked(qh_offset + l) };
            let ql_d_lo = unsafe { *ql_d.get_unchecked(ql_offset + l) };
            let ql_d_hi = unsafe { *ql_d.get_unchecked(ql_offset + 32 + l) };
            let q1_d = (((ql_d_lo & 0x0F) | (((qh_byte_d >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2_d = (((ql_d_hi & 0x0F) | (((qh_byte_d >> 2) & 0x03) << 4)) as i32 - 32) as f32;
            let q3_d =
                ((((ql_d_lo >> 4) & 0x0F) | (((qh_byte_d >> 4) & 0x03) << 4)) as i32 - 32) as f32;
            let q4_d =
                ((((ql_d_hi >> 4) & 0x0F) | (((qh_byte_d >> 6) & 0x03) << 4)) as i32 - 32) as f32;
            let (a_v1, a_v2, a_v3, a_v4) = if scale_idx == 0 {
                (a_sf0 * q1_a, a_sf2 * q2_a, a_sf4 * q3_a, a_sf6 * q4_a)
            } else {
                (a_sf1 * q1_a, a_sf3 * q2_a, a_sf5 * q3_a, a_sf7 * q4_a)
            };
            let (b_v1, b_v2, b_v3, b_v4) = if scale_idx == 0 {
                (b_sf0 * q1_b, b_sf2 * q2_b, b_sf4 * q3_b, b_sf6 * q4_b)
            } else {
                (b_sf1 * q1_b, b_sf3 * q2_b, b_sf5 * q3_b, b_sf7 * q4_b)
            };
            let (c_v1, c_v2, c_v3, c_v4) = if scale_idx == 0 {
                (c_sf0 * q1_c, c_sf2 * q2_c, c_sf4 * q3_c, c_sf6 * q4_c)
            } else {
                (c_sf1 * q1_c, c_sf3 * q2_c, c_sf5 * q3_c, c_sf7 * q4_c)
            };
            let (d_v1, d_v2, d_v3, d_v4) = if scale_idx == 0 {
                (d_sf0 * q1_d, d_sf2 * q2_d, d_sf4 * q3_d, d_sf6 * q4_d)
            } else {
                (d_sf1 * q1_d, d_sf3 * q2_d, d_sf5 * q3_d, d_sf7 * q4_d)
            };

            let idx1 = input_offset + local_offset + l;
            let idx2 = input_offset + local_offset + 32 + l;
            let idx3 = input_offset + local_offset + 64 + l;
            let idx4 = input_offset + local_offset + 96 + l;

            let x10 = unsafe { *input0.get_unchecked(idx1) };
            let x20 = unsafe { *input0.get_unchecked(idx2) };
            let x30 = unsafe { *input0.get_unchecked(idx3) };
            let x40 = unsafe { *input0.get_unchecked(idx4) };
            sum_a0 = a_v4.mul_add(
                x40,
                a_v3.mul_add(x30, a_v2.mul_add(x20, a_v1.mul_add(x10, sum_a0))),
            );
            sum_b0 = b_v4.mul_add(
                x40,
                b_v3.mul_add(x30, b_v2.mul_add(x20, b_v1.mul_add(x10, sum_b0))),
            );
            sum_c0 = c_v4.mul_add(
                x40,
                c_v3.mul_add(x30, c_v2.mul_add(x20, c_v1.mul_add(x10, sum_c0))),
            );
            sum_d0 = d_v4.mul_add(
                x40,
                d_v3.mul_add(x30, d_v2.mul_add(x20, d_v1.mul_add(x10, sum_d0))),
            );

            let x11 = unsafe { *input1.get_unchecked(idx1) };
            let x21 = unsafe { *input1.get_unchecked(idx2) };
            let x31 = unsafe { *input1.get_unchecked(idx3) };
            let x41 = unsafe { *input1.get_unchecked(idx4) };
            sum_a1 = a_v4.mul_add(
                x41,
                a_v3.mul_add(x31, a_v2.mul_add(x21, a_v1.mul_add(x11, sum_a1))),
            );
            sum_b1 = b_v4.mul_add(
                x41,
                b_v3.mul_add(x31, b_v2.mul_add(x21, b_v1.mul_add(x11, sum_b1))),
            );
            sum_c1 = c_v4.mul_add(
                x41,
                c_v3.mul_add(x31, c_v2.mul_add(x21, c_v1.mul_add(x11, sum_c1))),
            );
            sum_d1 = d_v4.mul_add(
                x41,
                d_v3.mul_add(x31, d_v2.mul_add(x21, d_v1.mul_add(x11, sum_d1))),
            );

            let x12 = unsafe { *input2.get_unchecked(idx1) };
            let x22 = unsafe { *input2.get_unchecked(idx2) };
            let x32 = unsafe { *input2.get_unchecked(idx3) };
            let x42 = unsafe { *input2.get_unchecked(idx4) };
            sum_a2 = a_v4.mul_add(
                x42,
                a_v3.mul_add(x32, a_v2.mul_add(x22, a_v1.mul_add(x12, sum_a2))),
            );
            sum_b2 = b_v4.mul_add(
                x42,
                b_v3.mul_add(x32, b_v2.mul_add(x22, b_v1.mul_add(x12, sum_b2))),
            );
            sum_c2 = c_v4.mul_add(
                x42,
                c_v3.mul_add(x32, c_v2.mul_add(x22, c_v1.mul_add(x12, sum_c2))),
            );
            sum_d2 = d_v4.mul_add(
                x42,
                d_v3.mul_add(x32, d_v2.mul_add(x22, d_v1.mul_add(x12, sum_d2))),
            );

            let x13 = unsafe { *input3.get_unchecked(idx1) };
            let x23 = unsafe { *input3.get_unchecked(idx2) };
            let x33 = unsafe { *input3.get_unchecked(idx3) };
            let x43 = unsafe { *input3.get_unchecked(idx4) };
            sum_a3 = a_v4.mul_add(
                x43,
                a_v3.mul_add(x33, a_v2.mul_add(x23, a_v1.mul_add(x13, sum_a3))),
            );
            sum_b3 = b_v4.mul_add(
                x43,
                b_v3.mul_add(x33, b_v2.mul_add(x23, b_v1.mul_add(x13, sum_b3))),
            );
            sum_c3 = c_v4.mul_add(
                x43,
                c_v3.mul_add(x33, c_v2.mul_add(x23, c_v1.mul_add(x13, sum_c3))),
            );
            sum_d3 = d_v4.mul_add(
                x43,
                d_v3.mul_add(x33, d_v2.mul_add(x23, d_v1.mul_add(x13, sum_d3))),
            );

            let x14 = unsafe { *input4.get_unchecked(idx1) };
            let x24 = unsafe { *input4.get_unchecked(idx2) };
            let x34 = unsafe { *input4.get_unchecked(idx3) };
            let x44 = unsafe { *input4.get_unchecked(idx4) };
            sum_a4 = a_v4.mul_add(
                x44,
                a_v3.mul_add(x34, a_v2.mul_add(x24, a_v1.mul_add(x14, sum_a4))),
            );
            sum_b4 = b_v4.mul_add(
                x44,
                b_v3.mul_add(x34, b_v2.mul_add(x24, b_v1.mul_add(x14, sum_b4))),
            );
            sum_c4 = c_v4.mul_add(
                x44,
                c_v3.mul_add(x34, c_v2.mul_add(x24, c_v1.mul_add(x14, sum_c4))),
            );
            sum_d4 = d_v4.mul_add(
                x44,
                d_v3.mul_add(x34, d_v2.mul_add(x24, d_v1.mul_add(x14, sum_d4))),
            );

            let x15 = unsafe { *input5.get_unchecked(idx1) };
            let x25 = unsafe { *input5.get_unchecked(idx2) };
            let x35 = unsafe { *input5.get_unchecked(idx3) };
            let x45 = unsafe { *input5.get_unchecked(idx4) };
            sum_a5 = a_v4.mul_add(
                x45,
                a_v3.mul_add(x35, a_v2.mul_add(x25, a_v1.mul_add(x15, sum_a5))),
            );
            sum_b5 = b_v4.mul_add(
                x45,
                b_v3.mul_add(x35, b_v2.mul_add(x25, b_v1.mul_add(x15, sum_b5))),
            );
            sum_c5 = c_v4.mul_add(
                x45,
                c_v3.mul_add(x35, c_v2.mul_add(x25, c_v1.mul_add(x15, sum_c5))),
            );
            sum_d5 = d_v4.mul_add(
                x45,
                d_v3.mul_add(x35, d_v2.mul_add(x25, d_v1.mul_add(x15, sum_d5))),
            );
        }
        local_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }

    sums_a[0] = sum_a0;
    sums_a[1] = sum_a1;
    sums_a[2] = sum_a2;
    sums_a[3] = sum_a3;
    sums_a[4] = sum_a4;
    sums_a[5] = sum_a5;
    sums_b[0] = sum_b0;
    sums_b[1] = sum_b1;
    sums_b[2] = sum_b2;
    sums_b[3] = sum_b3;
    sums_b[4] = sum_b4;
    sums_b[5] = sum_b5;
    sums_c[0] = sum_c0;
    sums_c[1] = sum_c1;
    sums_c[2] = sum_c2;
    sums_c[3] = sum_c3;
    sums_c[4] = sum_c4;
    sums_c[5] = sum_c5;
    sums_d[0] = sum_d0;
    sums_d[1] = sum_d1;
    sums_d[2] = sum_d2;
    sums_d[3] = sum_d3;
    sums_d[4] = sum_d4;
    sums_d[5] = sum_d5;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_k_zero_block_dequantizes_to_zeroes() {
        let block = vec![0u8; BLOCK_Q6_K_SIZE];
        let mut out = vec![1.0f32; QK_K];
        dequantize_q6_k_block(&block, &mut out).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_k_zero_block_dequantizes_to_zeroes() {
        let block = vec![0u8; BLOCK_Q4_K_SIZE];
        let mut out = vec![1.0f32; QK_K];
        dequantize_q4_k_block(&block, &mut out).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_k_dot_matches_constructed_row() {
        let mut row = vec![0u8; BLOCK_Q4_K_SIZE];
        row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row[4] = 1;
        row[5] = 1;
        row[6] = 1;
        row[7] = 1;
        for byte in row[4 + K_SCALE_SIZE..].iter_mut() {
            *byte = 0x11;
        }
        let input = vec![1.0f32; QK_K];
        assert_eq!(dot_row(GGML_TYPE_Q4_K, &row, &input).unwrap(), 128.0);
        assert_eq!(
            dot_many_row(GGML_TYPE_Q4_K, &row, &[input.clone(), vec![0.5; QK_K]]).unwrap(),
            vec![128.0, 64.0]
        );
        let mut sums = vec![0.0f32; 2];
        dot_many_row_into(
            GGML_TYPE_Q4_K,
            &row,
            &[input.clone(), vec![0.5; QK_K]],
            &mut sums,
        )
        .unwrap();
        assert_eq!(sums, vec![128.0, 64.0]);
        let half = vec![0.5f32; QK_K];
        let inputs = [input.as_slice(), half.as_slice()];
        dot_many_row_refs_into(GGML_TYPE_Q4_K, &row, &inputs, &mut sums).unwrap();
        assert_eq!(sums, vec![128.0, 64.0]);
    }

    #[test]
    fn q6_k_dot_matches_constructed_row() {
        let mut row = vec![0u8; BLOCK_Q6_K_SIZE];
        row[BLOCK_Q6_K_SIZE - 2..].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for scale in row[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16].iter_mut() {
            *scale = 1;
        }
        let input = vec![1.0f32; QK_K];
        assert_eq!(
            dot_row(GGML_TYPE_Q6_K, &row, &input).unwrap(),
            -32.0 * QK_K as f32
        );
        assert_eq!(
            dot_many_row(GGML_TYPE_Q6_K, &row, &[input.clone(), vec![0.5; QK_K]]).unwrap(),
            vec![-32.0 * QK_K as f32, -16.0 * QK_K as f32]
        );
    }

    #[test]
    fn q5_k_dot_matches_constructed_row() {
        let mut row = vec![0u8; BLOCK_Q5_K_SIZE];
        row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row[4] = 1;
        row[5] = 1;
        row[6] = 1;
        row[7] = 1;
        for byte in row[4 + K_SCALE_SIZE + QK_K / 8..].iter_mut() {
            *byte = 0x11;
        }
        let input = vec![1.0f32; QK_K];
        assert_eq!(dot_row(GGML_TYPE_Q5_K, &row, &input).unwrap(), 128.0);
        assert_eq!(
            dot_many_row(GGML_TYPE_Q5_K, &row, &[input.clone(), vec![0.5; QK_K]]).unwrap(),
            vec![128.0, 64.0]
        );
        let mut sums = vec![0.0f32; 2];
        dot_many_row_into(
            GGML_TYPE_Q5_K,
            &row,
            &[input.clone(), vec![0.5; QK_K]],
            &mut sums,
        )
        .unwrap();
        assert_eq!(sums, vec![128.0, 64.0]);
        let half = vec![0.5f32; QK_K];
        let inputs = [input.as_slice(), half.as_slice()];
        dot_many_row_refs_into(GGML_TYPE_Q5_K, &row, &inputs, &mut sums).unwrap();
        assert_eq!(sums, vec![128.0, 64.0]);
    }

    #[test]
    fn q5_k_row_into_matches_allocating_decode() {
        let mut row = vec![0u8; BLOCK_Q5_K_SIZE];
        row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        row[4] = 1;
        row[5] = 1;
        row[6] = 1;
        row[7] = 1;
        for byte in row[4 + K_SCALE_SIZE + QK_K / 8..].iter_mut() {
            *byte = 0x11;
        }

        let expected = dequantize_row(GGML_TYPE_Q5_K, &row, 0, QK_K).unwrap();
        let mut actual = vec![0.0f32; QK_K];
        dequantize_row_into(GGML_TYPE_Q5_K, &row, 0, QK_K, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}
