use anyhow::Result;
use stage_forward_lab::real_forward::{RealForwardProfile, RealGemmaBackend, RealTailLogitsTrace};
use stage_forward_lab::{StageForwardBackend, StageLayout, StageSample};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct RunMetrics {
    head_ms: u128,
    tail_ms: u128,
    sample_ms: u128,
    total_ms: u128,
    sample: StageSample,
    trace: RealTailLogitsTrace,
    head_profile: Option<RealForwardProfile>,
    tail_profile: Option<RealForwardProfile>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let stage1_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json")
    });
    let stage2_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json")
    });
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Hello".to_string());
    let vocab_path = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"));
    let warm_runs = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    let layer_cap = args.get(6).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(7).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(8)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real 2-Stage Gemma Bench ===");
    println!("stage 1   : {}", stage1_path.display());
    println!("stage 2   : {}", stage2_path.display());
    println!("prompt    : {:?}", prompt);
    println!("warm runs : {}", warm_runs);
    println!("layer cap : {:?}", layer_cap);
    println!("vocab cap : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!();

    let t0 = Instant::now();
    let mut head = RealGemmaBackend::new(&stage1_path);
    head.set_debug_layer_cap(layer_cap);
    head.set_debug_vocab_cap(vocab_cap);
    head.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;
    let head_load_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    let mut tail = RealGemmaBackend::new(&stage2_path);
    tail.set_debug_layer_cap(layer_cap);
    tail.set_debug_vocab_cap(vocab_cap);
    tail.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;
    let tail_load_ms = t1.elapsed().as_millis();

    println!("head loaded : {}ms", head_load_ms);
    println!("tail loaded : {}ms", tail_load_ms);
    println!();

    let cold = run_once(&head, &tail, &prompt, "bench-cold")?;
    println!("=== Cold Run ===");
    print_run(&cold);
    println!();

    let mut warm = Vec::with_capacity(warm_runs);
    for run_idx in 0..warm_runs {
        warm.push(run_once(
            &head,
            &tail,
            &prompt,
            &format!("bench-warm-{}", run_idx),
        )?);
    }

    if warm.is_empty() {
        return Ok(());
    }

    println!("=== Warm Summary ===");
    print_series("head", warm.iter().map(|m| m.head_ms).collect());
    print_series("tail", warm.iter().map(|m| m.tail_ms).collect());
    print_series("sample", warm.iter().map(|m| m.sample_ms).collect());
    print_series("total", warm.iter().map(|m| m.total_ms).collect());

    let first = &warm[0];
    let deterministic = warm.iter().all(|m| {
        m.sample.token_ids == first.sample.token_ids
            && m.sample.text == first.sample.text
            && m.trace.selected_token_id == first.trace.selected_token_id
    });
    println!(
        "deterministic : {}",
        if deterministic { "PASS" } else { "FAIL" }
    );
    println!("text          : {:?}", first.sample.text);
    println!("token ids     : {:?}", first.sample.token_ids);
    println!("selected id   : {}", first.trace.selected_token_id);
    println!();

    println!("=== Warm Profile Avg ===");
    print_profile_avg(
        "head",
        warm.iter()
            .filter_map(|m| m.head_profile.as_ref())
            .collect(),
    );
    print_profile_avg(
        "tail",
        warm.iter()
            .filter_map(|m| m.tail_profile.as_ref())
            .collect(),
    );

    Ok(())
}

fn run_once(
    head: &RealGemmaBackend,
    tail: &RealGemmaBackend,
    prompt: &str,
    request_id: &str,
) -> Result<RunMetrics> {
    let t_head = Instant::now();
    let head_output = head.begin_prompt(request_id, prompt, Some(1), 0)?;
    let head_ms = t_head.elapsed().as_millis();
    let head_profile = head.last_forward_profile();

    let t_tail = Instant::now();
    let tail_output = tail.continue_forward(head_output)?;
    let tail_ms = t_tail.elapsed().as_millis();
    let tail_profile = tail.last_forward_profile();
    let t_sample = Instant::now();
    let (sample, trace) = tail.sample_tail_with_trace(tail_output, 5)?;
    let sample_ms = t_sample.elapsed().as_millis();

    Ok(RunMetrics {
        head_ms,
        tail_ms,
        sample_ms,
        total_ms: head_ms + tail_ms + sample_ms,
        sample,
        trace,
        head_profile,
        tail_profile,
    })
}

fn print_run(run: &RunMetrics) {
    println!("head    : {}ms", run.head_ms);
    println!("tail    : {}ms", run.tail_ms);
    println!("sample  : {}ms", run.sample_ms);
    println!("total   : {}ms", run.total_ms);
    println!("text    : {:?}", run.sample.text);
    println!("tokens  : {:?}", run.sample.token_ids);
    println!("selected: {}", run.trace.selected_token_id);
}

fn print_series(label: &str, mut values: Vec<u128>) {
    values.sort_unstable();
    let min = values.first().copied().unwrap_or(0);
    let max = values.last().copied().unwrap_or(0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<u128>() / values.len() as u128;
    println!(
        "{}        : min={}ms median={}ms avg={}ms max={}ms",
        label, min, median, avg, max
    );
}

fn print_profile_avg(label: &str, profiles: Vec<&RealForwardProfile>) {
    if profiles.is_empty() {
        println!("{} profile : <none>", label);
        return;
    }
    let len = profiles.len() as u128;
    let avg =
        |f: fn(&RealForwardProfile) -> u128| profiles.iter().map(|p| f(p)).sum::<u128>() / len;
    println!(
        "{} profile : attn={}ms (qkv={}ms core={}ms out={}ms) ffn={}ms (gate+up={}ms down={}ms) ple={}ms (gate={}ms proj={}ms)",
        label,
        avg(|p| p.attn_micros) / 1_000,
        avg(|p| p.attn_qkv_micros) / 1_000,
        avg(|p| p.attn_core_micros) / 1_000,
        avg(|p| p.attn_out_micros) / 1_000,
        avg(|p| p.ffn_micros) / 1_000,
        avg(|p| p.ffn_gate_up_micros) / 1_000,
        avg(|p| p.ffn_down_micros) / 1_000,
        avg(|p| p.ple_micros) / 1_000,
        avg(|p| p.ple_gate_micros) / 1_000,
        avg(|p| p.ple_proj_micros) / 1_000,
    );
}
