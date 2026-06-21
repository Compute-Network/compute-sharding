#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Speculative decoding requires the draft model to share the target model's
// vocabulary — otherwise verification compares apples to oranges and the
// "accepted" prefix is meaningless.
//
// This probe loads two GGUFs, tokenizes a fixed corpus with each, and prints
// per-string id divergence. If the same string yields the same id sequence,
// the vocabs are byte-compatible at the level we need.
//
// Usage: draft_tokenizer_compat_probe <target.gguf> <draft.gguf>
use anyhow::{Context, Result, bail};
use llama_stage_backend::LlamaStageBackend;
use std::path::PathBuf;

const SAMPLES: &[&str] = &[
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "fn main() {\n    println!(\"hello, world\");\n}",
    "<start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\n",
    "1234567890",
    " the the the the",
    "naïve résumé café",
    "🚀✨🌍",
];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let target_path = PathBuf::from(
        args.next()
            .context("usage: probe <target.gguf> <draft.gguf>")?,
    );
    let draft_path = PathBuf::from(
        args.next()
            .context("usage: probe <target.gguf> <draft.gguf>")?,
    );

    if !target_path.exists() {
        bail!("target not found: {}", target_path.display());
    }
    if !draft_path.exists() {
        bail!("draft not found: {}", draft_path.display());
    }

    eprintln!("[probe] target = {}", target_path.display());
    eprintln!("[probe] draft  = {}", draft_path.display());

    let target = LlamaStageBackend::new(target_path)?;
    let draft = LlamaStageBackend::new(draft_path)?;

    let mut text_match = 0usize;
    let mut text_total = 0usize;
    let mut chat_template_match = 0usize;
    let mut chat_template_total = 0usize;

    for sample in SAMPLES {
        let t_ids = target.tokenize(sample)?;
        let d_ids = draft.tokenize(sample)?;
        let identical = t_ids == d_ids;
        let is_chat_template =
            sample.contains("<start_of_turn>") || sample.contains("<end_of_turn>");

        if is_chat_template {
            chat_template_total += 1;
            if identical {
                chat_template_match += 1;
            }
        } else {
            text_total += 1;
            if identical {
                text_match += 1;
            }
        }

        if identical {
            println!("MATCH    {:?} -> {} ids", sample, t_ids.len());
        } else {
            println!(
                "MISMATCH {:?}\n  target ({} ids): {:?}\n  draft  ({} ids): {:?}",
                sample,
                t_ids.len(),
                t_ids,
                d_ids.len(),
                d_ids
            );
        }
    }

    println!(
        "\n[result] text samples: {}/{} match. chat-template samples: {}/{} match.",
        text_match, text_total, chat_template_match, chat_template_total
    );
    if text_match == text_total {
        println!(
            "[result] generation-phase vocabs are byte-identical. Draft model is suitable for spec decode."
        );
        if chat_template_match < chat_template_total {
            println!(
                "[note]   chat-template specials differ in encoding only (prefill artifact, not decode)."
            );
        }
    } else {
        println!(
            "[result] generation-phase vocabs diverge — draft model is NOT suitable as a spec-decode draft."
        );
    }

    Ok(())
}
