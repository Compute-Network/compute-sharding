use anyhow::{Context, Result, bail};
use stage_forward_lab::{
    gguf::{GgufFile, MetadataValue, TensorInfo},
    quants,
};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct ShardSpec {
    role: &'static str,
    start_layer: u32,
    end_layer: u32,
}

#[derive(Debug, Clone)]
struct ShardTensor {
    source_name: String,
    info: TensorInfo,
    data: TensorData,
}

#[derive(Debug, Clone)]
enum TensorData {
    Copy {
        source_file_offset: u64,
        byte_len: u64,
    },
    StridedRows {
        source_file_offset: u64,
        rows: u64,
        full_row_bytes: u64,
        row_start_bytes: u64,
        row_bytes: u64,
    },
}

impl TensorData {
    fn byte_len(&self) -> u64 {
        match self {
            TensorData::Copy { byte_len, .. } => *byte_len,
            TensorData::StridedRows {
                rows, row_bytes, ..
            } => rows.saturating_mul(*row_bytes),
        }
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().context(
        "usage: write_gemma4_stage_ggufs <model.gguf> <out_dir> [head_end tail_start tail_end]",
    )?);
    let out_dir = PathBuf::from(args.next().context(
        "usage: write_gemma4_stage_ggufs <model.gguf> <out_dir> [head_end tail_start tail_end]",
    )?);

    let head_end = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let tail_start = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(head_end + 1);
    let tail_end = args.next().and_then(|s| s.parse().ok()).unwrap_or(34);

    let file = GgufFile::parse_file(&model_path)
        .with_context(|| format!("parse GGUF {}", model_path.display()))?;
    if file.version != 3 {
        bail!("only GGUF v3 is supported, got v{}", file.version);
    }
    if file.architecture() != Some("gemma4") {
        bail!("expected gemma4 GGUF, got {:?}", file.architecture());
    }
    let total_layers = file
        .inferred_layer_count()
        .context("missing Gemma4 block count")?;
    if tail_end >= total_layers {
        bail!("tail_end={tail_end} exceeds total layer count {total_layers}");
    }

    let specs = [
        ShardSpec {
            role: "head",
            start_layer: 0,
            end_layer: head_end,
        },
        ShardSpec {
            role: "tail",
            start_layer: tail_start,
            end_layer: tail_end,
        },
    ];

    fs::create_dir_all(&out_dir)?;
    for spec in specs {
        let out_path = out_dir.join(format!(
            "{}-{}-{}.gguf",
            spec.role, spec.start_layer, spec.end_layer
        ));
        write_shard(&model_path, &file, spec, &out_path)?;
        let written = GgufFile::parse_file(&out_path)
            .with_context(|| format!("parse written shard {}", out_path.display()))?;
        println!(
            "{} {}..{} -> {} (layers={} shared_kv={} tensors={} size={:.2} GiB)",
            spec.role,
            spec.start_layer,
            spec.end_layer,
            out_path.display(),
            written.inferred_layer_count().unwrap_or(0),
            written
                .metadata_u32("gemma4.attention.shared_kv_layers")
                .unwrap_or(0),
            written.tensors.len(),
            fs::metadata(&out_path)?.len() as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }

    Ok(())
}

fn write_shard(model_path: &Path, file: &GgufFile, spec: ShardSpec, out_path: &Path) -> Result<()> {
    let total_layers = file.inferred_layer_count().context("missing layer count")?;
    let local_layers = spec.end_layer - spec.start_layer + 1;
    let full_shared_kv = file
        .metadata_u32("gemma4.attention.shared_kv_layers")
        .unwrap_or(0);
    let full_kv_from_start = total_layers.saturating_sub(full_shared_kv);
    let local_kv_layers = count_local_kv_layers(spec, full_kv_from_start);
    let local_shared_kv = local_layers.saturating_sub(local_kv_layers);

    if local_shared_kv > 0 && local_kv_layers == 0 {
        bail!(
            "{} shard {}..{} has shared-KV layers but no local KV source layers; move tail_start earlier",
            spec.role,
            spec.start_layer,
            spec.end_layer
        );
    }

    let mut metadata = file.metadata.clone();
    metadata.insert(
        "gemma4.block_count".to_string(),
        MetadataValue::Uint32(local_layers),
    );
    if metadata.contains_key("gemma4.attention.shared_kv_layers") {
        metadata.insert(
            "gemma4.attention.shared_kv_layers".to_string(),
            MetadataValue::Uint32(local_shared_kv),
        );
    }
    slice_layer_metadata_arrays(&mut metadata, total_layers, spec)?;

    let mut tensors = collect_shard_tensors(file, spec)?;
    assign_new_offsets(&mut tensors, alignment(&metadata));

    let mut out = File::create(out_path)
        .with_context(|| format!("create shard output {}", out_path.display()))?;
    write_header(&mut out, file.version, &metadata, &tensors)?;

    let data_start = align_to(out.stream_position()?, alignment(&metadata));
    pad_to_abs(&mut out, data_start)?;

    let mut source = File::open(model_path)
        .with_context(|| format!("open source GGUF {}", model_path.display()))?;
    let mut data_written = 0u64;
    for tensor in &tensors {
        pad_to_abs(&mut out, data_start + tensor.info.offset)?;
        data_written = tensor.info.offset;
        write_tensor_data(&mut source, &mut out, &tensor.data)
            .with_context(|| format!("copy tensor {}", tensor.source_name))?;
        data_written += tensor.data.byte_len();
    }
    let final_size = data_start + data_written;
    pad_to_abs(&mut out, final_size)?;
    out.flush()?;
    Ok(())
}

fn count_local_kv_layers(spec: ShardSpec, full_kv_from_start: u32) -> u32 {
    if spec.start_layer >= full_kv_from_start {
        return 0;
    }
    let last_kv = spec.end_layer.min(full_kv_from_start.saturating_sub(1));
    last_kv - spec.start_layer + 1
}

fn slice_layer_metadata_arrays(
    metadata: &mut BTreeMap<String, MetadataValue>,
    total_layers: u32,
    spec: ShardSpec,
) -> Result<()> {
    for value in metadata.values_mut() {
        let MetadataValue::Array(values) = value else {
            continue;
        };
        if values.len() != total_layers as usize {
            continue;
        }
        let start = spec.start_layer as usize;
        let end = spec.end_layer as usize + 1;
        *values = values
            .get(start..end)
            .with_context(|| {
                format!(
                    "slice layer metadata array for {}..{} out of {}",
                    spec.start_layer, spec.end_layer, total_layers
                )
            })?
            .to_vec();
    }
    Ok(())
}

fn collect_shard_tensors(file: &GgufFile, spec: ShardSpec) -> Result<Vec<ShardTensor>> {
    let mut tensors = Vec::new();
    let total_layers = file.inferred_layer_count().context("missing layer count")?;
    let local_layers = spec.end_layer - spec.start_layer + 1;
    let n_embd_per_layer =
        file.metadata_u32("gemma4.embedding_length_per_layer_input")
            .context("missing gemma4.embedding_length_per_layer_input")? as u64;

    for tensor in &file.tensors {
        let (include, new_name) = match parse_layer_tensor_name(&tensor.name) {
            Some((layer, suffix)) => {
                if layer < spec.start_layer || layer > spec.end_layer {
                    (false, String::new())
                } else {
                    let local_layer = layer - spec.start_layer;
                    (true, format!("blk.{local_layer}.{suffix}"))
                }
            }
            None => (true, tensor.name.clone()),
        };
        if !include {
            continue;
        }

        let source_file_offset = file
            .tensor_file_offset(&tensor.name)
            .with_context(|| format!("missing file offset for {}", tensor.name))?;
        let mut info = tensor.clone();
        info.name = new_name;
        let data = match tensor.name.as_str() {
            "per_layer_model_proj.weight" => {
                slice_per_layer_model_proj(tensor, source_file_offset, spec, n_embd_per_layer)?
            }
            "per_layer_token_embd.weight" => slice_per_layer_token_embd(
                tensor,
                source_file_offset,
                spec,
                total_layers,
                local_layers,
                n_embd_per_layer,
            )?,
            _ => TensorData::Copy {
                source_file_offset,
                byte_len: file
                    .tensor_byte_len(&tensor.name)
                    .with_context(|| format!("missing byte length for {}", tensor.name))?,
            },
        };
        if tensor.name == "per_layer_model_proj.weight" {
            info.dimensions[1] = local_layers as u64 * n_embd_per_layer;
        } else if tensor.name == "per_layer_token_embd.weight" {
            info.dimensions[0] = local_layers as u64 * n_embd_per_layer;
        }
        tensors.push(ShardTensor {
            source_name: tensor.name.clone(),
            info,
            data,
        });
    }
    Ok(tensors)
}

fn slice_per_layer_model_proj(
    tensor: &TensorInfo,
    source_file_offset: u64,
    spec: ShardSpec,
    n_embd_per_layer: u64,
) -> Result<TensorData> {
    if tensor.dimensions.len() != 2 {
        bail!(
            "per_layer_model_proj.weight must be 2D, got {:?}",
            tensor.dimensions
        );
    }
    let full_rows = tensor.dimensions[1];
    let local_layers = (spec.end_layer - spec.start_layer + 1) as u64;
    let start_row = spec.start_layer as u64 * n_embd_per_layer;
    let row_count = local_layers * n_embd_per_layer;
    if start_row + row_count > full_rows {
        bail!(
            "per_layer_model_proj slice {}..{} exceeds full rows {}",
            start_row,
            start_row + row_count,
            full_rows
        );
    }

    let row_bytes = row_size(tensor.ggml_type, tensor.dimensions[0])?;
    Ok(TensorData::Copy {
        source_file_offset: checked_add(
            source_file_offset,
            checked_mul(start_row, row_bytes, "per_layer_model_proj start byte")?,
            "per_layer_model_proj source offset",
        )?,
        byte_len: checked_mul(row_count, row_bytes, "per_layer_model_proj byte length")?,
    })
}

fn slice_per_layer_token_embd(
    tensor: &TensorInfo,
    source_file_offset: u64,
    spec: ShardSpec,
    total_layers: u32,
    local_layers: u32,
    n_embd_per_layer: u64,
) -> Result<TensorData> {
    if tensor.dimensions.len() != 2 {
        bail!(
            "per_layer_token_embd.weight must be 2D, got {:?}",
            tensor.dimensions
        );
    }
    let full_inner = tensor.dimensions[0];
    let expected_full_inner = total_layers as u64 * n_embd_per_layer;
    if full_inner != expected_full_inner {
        bail!(
            "per_layer_token_embd inner dim {} does not match total_layers({}) * per_layer({})",
            full_inner,
            total_layers,
            n_embd_per_layer
        );
    }

    let start_elem = spec.start_layer as u64 * n_embd_per_layer;
    let elem_count = local_layers as u64 * n_embd_per_layer;
    if start_elem + elem_count > full_inner {
        bail!(
            "per_layer_token_embd slice {}..{} exceeds full inner dim {}",
            start_elem,
            start_elem + elem_count,
            full_inner
        );
    }

    let row_start_bytes = row_size(tensor.ggml_type, start_elem)?;
    let row_bytes = row_size(tensor.ggml_type, elem_count)?;
    let full_row_bytes = row_size(tensor.ggml_type, full_inner)?;
    Ok(TensorData::StridedRows {
        source_file_offset,
        rows: tensor.dimensions[1],
        full_row_bytes,
        row_start_bytes,
        row_bytes,
    })
}

fn parse_layer_tensor_name(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    Some((layer.parse().ok()?, suffix))
}

fn assign_new_offsets(tensors: &mut [ShardTensor], alignment: u64) {
    let mut cursor = 0u64;
    for tensor in tensors {
        cursor = align_to(cursor, alignment);
        tensor.info.offset = cursor;
        cursor += tensor.data.byte_len();
    }
}

fn write_header<W: Write>(
    mut writer: W,
    version: u32,
    metadata: &BTreeMap<String, MetadataValue>,
    tensors: &[ShardTensor],
) -> Result<()> {
    writer.write_all(b"GGUF")?;
    write_u32(&mut writer, version)?;
    write_u64(&mut writer, tensors.len() as u64)?;
    write_u64(&mut writer, metadata.len() as u64)?;

    for (key, value) in metadata {
        write_string(&mut writer, key)?;
        write_metadata_value(&mut writer, value)?;
    }

    for tensor in tensors {
        write_string(&mut writer, &tensor.info.name)?;
        write_u32(&mut writer, tensor.info.dimensions.len() as u32)?;
        for dim in &tensor.info.dimensions {
            write_u64(&mut writer, *dim)?;
        }
        write_u32(&mut writer, tensor.info.ggml_type)?;
        write_u64(&mut writer, tensor.info.offset)?;
    }

    Ok(())
}

fn write_metadata_value<W: Write>(writer: &mut W, value: &MetadataValue) -> Result<()> {
    write_u32(writer, metadata_type(value)?)?;
    write_metadata_payload(writer, value)
}

fn write_metadata_payload<W: Write>(writer: &mut W, value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::Uint8(v) => writer.write_all(&[*v])?,
        MetadataValue::Int8(v) => writer.write_all(&[*v as u8])?,
        MetadataValue::Uint16(v) => writer.write_all(&v.to_le_bytes())?,
        MetadataValue::Int16(v) => writer.write_all(&v.to_le_bytes())?,
        MetadataValue::Uint32(v) => write_u32(writer, *v)?,
        MetadataValue::Int32(v) => writer.write_all(&v.to_le_bytes())?,
        MetadataValue::Float32(v) => writer.write_all(&v.to_le_bytes())?,
        MetadataValue::Bool(v) => writer.write_all(&[*v as u8])?,
        MetadataValue::String(v) => write_string(writer, v)?,
        MetadataValue::Array(values) => {
            let element_type = array_element_type(values)?;
            write_u32(writer, element_type)?;
            write_u64(writer, values.len() as u64)?;
            for item in values {
                if metadata_type(item)? != element_type {
                    bail!("GGUF metadata arrays must be homogeneous");
                }
                write_metadata_payload(writer, item)?;
            }
        }
        MetadataValue::Uint64(v) => write_u64(writer, *v)?,
        MetadataValue::Int64(v) => writer.write_all(&v.to_le_bytes())?,
        MetadataValue::Float64(v) => writer.write_all(&v.to_le_bytes())?,
    }
    Ok(())
}

fn metadata_type(value: &MetadataValue) -> Result<u32> {
    Ok(match value {
        MetadataValue::Uint8(_) => 0,
        MetadataValue::Int8(_) => 1,
        MetadataValue::Uint16(_) => 2,
        MetadataValue::Int16(_) => 3,
        MetadataValue::Uint32(_) => 4,
        MetadataValue::Int32(_) => 5,
        MetadataValue::Float32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::Array(_) => 9,
        MetadataValue::Uint64(_) => 10,
        MetadataValue::Int64(_) => 11,
        MetadataValue::Float64(_) => 12,
    })
}

fn array_element_type(values: &[MetadataValue]) -> Result<u32> {
    let first = values
        .first()
        .context("cannot write empty GGUF metadata array")?;
    let ty = metadata_type(first)?;
    if ty == 9 {
        bail!("nested GGUF metadata arrays are not supported by this shard writer");
    }
    Ok(ty)
}

fn alignment(metadata: &BTreeMap<String, MetadataValue>) -> u64 {
    match metadata.get("general.alignment") {
        Some(MetadataValue::Uint32(v)) => *v as u64,
        Some(MetadataValue::Uint64(v)) => *v,
        _ => 32,
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn pad_to_abs<W: Write + Seek>(writer: &mut W, target: u64) -> Result<()> {
    let pos = writer.stream_position()?;
    if pos > target {
        bail!("writer already past target offset {target}: {pos}");
    }
    let mut remaining = target - pos;
    const ZEROES: [u8; 4096] = [0; 4096];
    while remaining > 0 {
        let chunk = remaining.min(ZEROES.len() as u64) as usize;
        writer.write_all(&ZEROES[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn write_tensor_data<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    data: &TensorData,
) -> Result<()> {
    match data {
        TensorData::Copy {
            source_file_offset,
            byte_len,
        } => {
            reader.seek(SeekFrom::Start(*source_file_offset))?;
            copy_exact(reader, writer, *byte_len)
        }
        TensorData::StridedRows {
            source_file_offset,
            rows,
            full_row_bytes,
            row_start_bytes,
            row_bytes,
        } => {
            for row in 0..*rows {
                let row_offset = checked_add(
                    *source_file_offset,
                    checked_add(
                        checked_mul(row, *full_row_bytes, "strided row byte offset")?,
                        *row_start_bytes,
                        "strided row slice offset",
                    )?,
                    "strided source offset",
                )?;
                reader.seek(SeekFrom::Start(row_offset))?;
                copy_exact(reader, writer, *row_bytes)?;
            }
            Ok(())
        }
    }
}

fn copy_exact<R: Read, W: Write>(reader: &mut R, writer: &mut W, mut bytes: u64) -> Result<()> {
    let mut buf = vec![0u8; 1024 * 1024];
    while bytes > 0 {
        let want = bytes.min(buf.len() as u64) as usize;
        reader.read_exact(&mut buf[..want])?;
        writer.write_all(&buf[..want])?;
        bytes -= want as u64;
    }
    Ok(())
}

fn row_size(ggml_type: u32, elements: u64) -> Result<u64> {
    let elements: usize = elements
        .try_into()
        .with_context(|| format!("row element count {elements} does not fit usize"))?;
    quants::bytes_per_row(ggml_type, elements)
        .map(|value| value as u64)
        .with_context(|| format!("row size for ggml type {ggml_type} and {elements} elements"))
}

fn checked_mul(lhs: u64, rhs: u64, context: &'static str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("{context} overflow: {lhs} * {rhs}"))
}

fn checked_add(lhs: u64, rhs: u64, context: &'static str) -> Result<u64> {
    lhs.checked_add(rhs)
        .with_context(|| format!("{context} overflow: {lhs} + {rhs}"))
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}
