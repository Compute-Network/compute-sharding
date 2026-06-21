use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaPromptMode {
    Raw,
    GemmaInstruct,
}

impl Default for GemmaPromptMode {
    fn default() -> Self {
        Self::GemmaInstruct
    }
}

impl GemmaPromptMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "raw" | "plain" | "text" => Some(Self::Raw),
            "gemma" | "gemma_instruct" | "gemma-instruct" | "instruct" | "chat" => {
                Some(Self::GemmaInstruct)
            }
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::GemmaInstruct => "gemma_instruct",
        }
    }
}

pub fn format_gemma_prompt(mode: GemmaPromptMode, prompt: &str) -> String {
    match mode {
        GemmaPromptMode::Raw => prompt.to_string(),
        GemmaPromptMode::GemmaInstruct => {
            let prompt = prompt.trim();
            format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_instruct_prompt_wraps_single_turn_user_content() {
        assert_eq!(
            format_gemma_prompt(GemmaPromptMode::GemmaInstruct, "Hello"),
            "<|turn>user\nHello<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn gemma_instruct_prompt_trims_outer_whitespace() {
        assert_eq!(
            format_gemma_prompt(GemmaPromptMode::GemmaInstruct, "  Hello  "),
            "<|turn>user\nHello<turn|>\n<|turn>model\n"
        );
    }
}
