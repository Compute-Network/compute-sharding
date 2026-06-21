use anyhow::{Context, Result, bail};
use stage_forward_lab::real_forward::RealTailLogitsTrace;
use std::env;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        bail!(
            "usage: {} <left-trace-json-or-log> <right-trace-json-or-log> [score_tolerance]",
            args.first()
                .map(String::as_str)
                .unwrap_or("real_trace_compare")
        );
    }
    let tolerance = args
        .get(3)
        .map(|value| value.parse::<f32>().context("invalid score_tolerance"))
        .transpose()?
        .unwrap_or(1e-4);

    let left = read_trace(Path::new(&args[1]))?;
    let right = read_trace(Path::new(&args[2]))?;
    let report = compare_traces(&left, &right, tolerance);

    println!(
        "projection left/right : {} / {}",
        left.projection_tensor, right.projection_tensor
    );
    println!(
        "hidden_dim left/right  : {} / {}",
        left.hidden_dim, right.hidden_dim
    );
    println!(
        "vocab_size left/right  : {} / {}",
        left.vocab_size, right.vocab_size
    );
    println!(
        "selected left/right    : {} ({:.6}) / {} ({:.6})",
        left.selected_token_id, left.selected_score, right.selected_token_id, right.selected_score
    );
    println!(
        "state_rms left/right   : {:.6} / {:.6}",
        left.state_rms, right.state_rms
    );
    println!("max score delta        : {:.8}", report.max_score_delta);
    println!("top-id mismatches      : {}", report.top_id_mismatches);

    if report.is_match {
        println!("PASS: traces match within tolerance {tolerance}");
        Ok(())
    } else {
        bail!("FAIL: traces diverge beyond tolerance {tolerance}")
    }
}

fn read_trace(path: &Path) -> Result<RealTailLogitsTrace> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read trace file {}", path.display()))?;
    parse_trace_text(&text)
        .with_context(|| format!("failed to parse trace file {}", path.display()))
}

fn parse_trace_text(text: &str) -> Result<RealTailLogitsTrace> {
    if let Ok(trace) = serde_json::from_str::<RealTailLogitsTrace>(text.trim()) {
        return Ok(trace);
    }

    for line in text.lines() {
        if let Some(start) = line.find('{') {
            let candidate = &line[start..];
            if let Ok(trace) = serde_json::from_str::<RealTailLogitsTrace>(candidate) {
                return Ok(trace);
            }
        }
    }

    bail!("no RealTailLogitsTrace JSON object found")
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TraceCompareReport {
    is_match: bool,
    max_score_delta: f32,
    top_id_mismatches: usize,
}

fn compare_traces(
    left: &RealTailLogitsTrace,
    right: &RealTailLogitsTrace,
    tolerance: f32,
) -> TraceCompareReport {
    let mut is_match = left.projection_tensor == right.projection_tensor
        && left.hidden_dim == right.hidden_dim
        && left.vocab_size == right.vocab_size
        && left.selected_token_id == right.selected_token_id;

    let mut max_score_delta = (left.selected_score - right.selected_score).abs();
    max_score_delta = max_score_delta.max((left.state_rms - right.state_rms).abs());

    let mut top_id_mismatches = 0usize;
    for ((left_id, left_score), (right_id, right_score)) in
        left.top_logits.iter().zip(right.top_logits.iter())
    {
        if left_id != right_id {
            top_id_mismatches += 1;
        }
        max_score_delta = max_score_delta.max((left_score - right_score).abs());
    }

    if left.top_logits.len() != right.top_logits.len() {
        top_id_mismatches += left.top_logits.len().abs_diff(right.top_logits.len());
    }

    if top_id_mismatches > 0 || max_score_delta > tolerance {
        is_match = false;
    }

    TraceCompareReport {
        is_match,
        max_score_delta,
        top_id_mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(score: f32) -> RealTailLogitsTrace {
        RealTailLogitsTrace {
            projection_tensor: "output.weight".into(),
            hidden_dim: 2,
            vocab_size: 3,
            selected_token_id: 1,
            selected_score: score,
            top_logits: vec![(1, score), (2, 0.25)],
            state_rms: 0.5,
        }
    }

    #[test]
    fn parses_raw_trace_json() {
        let encoded = serde_json::to_string(&trace(1.0)).unwrap();
        let parsed = parse_trace_text(&encoded).unwrap();
        assert_eq!(parsed.selected_token_id, 1);
    }

    #[test]
    fn parses_probe_trace_json_line() {
        let encoded = serde_json::to_string(&trace(1.0)).unwrap();
        let parsed = parse_trace_text(&format!("trace json       : {encoded}\n")).unwrap();
        assert_eq!(parsed.projection_tensor, "output.weight");
    }

    #[test]
    fn compare_rejects_score_drift() {
        let left = trace(1.0);
        let right = trace(1.25);
        let report = compare_traces(&left, &right, 0.01);
        assert!(!report.is_match);
        assert!(report.max_score_delta > 0.01);
    }
}
