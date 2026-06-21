#[derive(Debug, Clone, Copy)]
pub enum PromptExpectation {
    Exact(&'static str),
    ContainsAny(&'static [&'static str]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPromptTier {
    Core,
    Extended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPromptSuiteMode {
    Core,
    All,
}

impl ValidationPromptSuiteMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "core" | "default" => Some(Self::Core),
            "all" | "extended" | "full" => Some(Self::All),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ValidationPromptCase {
    pub tier: ValidationPromptTier,
    pub name: &'static str,
    pub prompt: &'static str,
    pub max_tokens: u32,
    pub stop_sequences: &'static [&'static str],
    pub first_token_expectation: PromptExpectation,
    pub continuation_expectation: PromptExpectation,
}

pub const GEMMA_VALIDATION_PROMPT_CASES: &[ValidationPromptCase] = &[
    ValidationPromptCase {
        tier: ValidationPromptTier::Core,
        name: "france_one_word",
        prompt: "Reply with one word. What is the capital of France?",
        max_tokens: 4,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["Paris"]),
        continuation_expectation: PromptExpectation::ContainsAny(&["Paris"]),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Core,
        name: "france_exact_output",
        prompt: "Reply with exactly 'Paris' and nothing else.",
        max_tokens: 4,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["Paris"]),
        continuation_expectation: PromptExpectation::Exact("Paris"),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Core,
        name: "sky_blue_sentence",
        prompt: "In one sentence, explain why the sky looks blue during the day.",
        max_tokens: 6,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["As", "Because", "Rayleigh"]),
        continuation_expectation: PromptExpectation::ContainsAny(&["Rayleigh", "scattering"]),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Core,
        name: "sky_red_sentence",
        prompt: "In one sentence, explain why the sky looks red at sunset.",
        max_tokens: 6,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["As", "Because", "When"]),
        continuation_expectation: PromptExpectation::ContainsAny(&["sun", "low", "sunset"]),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Core,
        name: "cache_reason_sentence",
        prompt: "Answer in one concise sentence. Why can continuation tokens be faster than the first generated token when per-request decode caches are reused across steps?",
        max_tokens: 8,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&[
            "Because",
            "The",
            "Continuation",
            "Cached",
        ]),
        continuation_expectation: PromptExpectation::ContainsAny(&[
            "cache",
            "reuse",
            "reused",
            "prefill",
            "initial token",
            "full",
        ]),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Extended,
        name: "yes_exact_output",
        prompt: "Answer with exactly 'Yes' and nothing else. Is Paris in France?",
        max_tokens: 4,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["Yes"]),
        continuation_expectation: PromptExpectation::Exact("Yes"),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Extended,
        name: "no_exact_output",
        prompt: "Answer with exactly 'No' and nothing else. Is Paris in Germany?",
        max_tokens: 4,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["No"]),
        continuation_expectation: PromptExpectation::Exact("No"),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Extended,
        name: "kv_cache_exact_output",
        prompt: "Reply with exactly 'KV cache' and nothing else. What allows continuation tokens to be faster after the first generated token?",
        max_tokens: 4,
        stop_sequences: &[],
        first_token_expectation: PromptExpectation::ContainsAny(&["KV"]),
        continuation_expectation: PromptExpectation::Exact("KV cache"),
    },
    ValidationPromptCase {
        tier: ValidationPromptTier::Extended,
        name: "paris_comma_stop",
        prompt: "Reply with exactly 'Paris,' and nothing else.",
        max_tokens: 4,
        stop_sequences: &[","],
        first_token_expectation: PromptExpectation::ContainsAny(&["Paris"]),
        continuation_expectation: PromptExpectation::Exact("Paris"),
    },
];

pub fn validation_prompt_cases(
    mode: ValidationPromptSuiteMode,
) -> impl Iterator<Item = &'static ValidationPromptCase> {
    GEMMA_VALIDATION_PROMPT_CASES.iter().filter(move |case| {
        matches!(mode, ValidationPromptSuiteMode::All) || case.tier == ValidationPromptTier::Core
    })
}

pub fn expectation_matches(expectation: PromptExpectation, text: &str) -> bool {
    match expectation {
        PromptExpectation::Exact(expected) => text.trim() == expected,
        PromptExpectation::ContainsAny(candidates) => {
            let haystack = text.to_ascii_lowercase();
            candidates
                .iter()
                .any(|candidate| haystack.contains(&candidate.to_ascii_lowercase()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_expectation_trims_outer_whitespace() {
        assert!(expectation_matches(
            PromptExpectation::Exact("Paris"),
            " Paris "
        ));
    }

    #[test]
    fn contains_any_is_case_insensitive() {
        assert!(expectation_matches(
            PromptExpectation::ContainsAny(&["rayleigh"]),
            "As Rayleigh scattering explains it."
        ));
    }

    #[test]
    fn core_mode_excludes_extended_cases() {
        assert_eq!(
            validation_prompt_cases(ValidationPromptSuiteMode::Core).count(),
            5
        );
    }

    #[test]
    fn all_mode_includes_extended_cases() {
        assert_eq!(
            validation_prompt_cases(ValidationPromptSuiteMode::All).count(),
            9
        );
    }
}
