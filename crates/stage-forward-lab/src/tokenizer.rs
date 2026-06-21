use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub struct GemmaTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    scores: Vec<f32>,
    bos_id: u32,
    eos_id: u32,
}

impl GemmaTokenizer {
    pub fn load(vocab_path: &Path, scores_path: Option<&Path>) -> Result<Self> {
        let vocab_data = std::fs::read(vocab_path)?;
        let tokens: Vec<String> = serde_json::from_slice(&vocab_data)?;

        let scores = if let Some(sp) = scores_path {
            let data = std::fs::read(sp)?;
            serde_json::from_slice(&data)?
        } else {
            vec![0.0f32; tokens.len()]
        };

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let bos_id = token_to_id.get("<bos>").copied().unwrap_or(2);
        let eos_id = token_to_id
            .get("<turn|>")
            .copied()
            .or_else(|| token_to_id.get("<eos>").copied())
            .unwrap_or(1);

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            scores,
            bos_id,
            eos_id,
        })
    }

    pub fn id_to_token(&self) -> &[String] {
        &self.id_to_token
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn decode(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("<?>")
    }

    pub fn decode_ids(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.decode(id))
            .collect::<Vec<_>>()
            .join("")
            .replace('▁', " ")
            .trim_start()
            .to_string()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = format!("▁{}", text.replace(' ', "▁"));
        self.bpe_encode(&normalized)
    }

    pub fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.bos_id];
        ids.extend(self.encode(text));
        ids
    }

    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        if let Some(&id) = self.token_to_id.get(text) {
            return vec![id];
        }

        let chars: Vec<char> = text.chars().collect();
        let mut pieces: Vec<String> = Vec::new();

        let mut i = 0;
        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;

            for end in (i + 1..=chars.len()).rev() {
                let candidate: String = chars[i..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    if end - i > best_len {
                        best_len = end - i;
                        best_id = Some(id);
                    }
                    break;
                }
            }

            if let Some(_id) = best_id {
                let piece: String = chars[i..i + best_len].iter().collect();
                pieces.push(piece);
                i += best_len;
            } else {
                let ch: String = chars[i..i + 1].iter().collect();
                pieces.push(ch);
                i += 1;
            }
        }

        loop {
            if pieces.len() <= 1 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = None;

            for j in 0..pieces.len() - 1 {
                let merged = format!("{}{}", pieces[j], pieces[j + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(j);
                    }
                }
            }

            if let Some(j) = best_idx {
                let merged = format!("{}{}", pieces[j], pieces[j + 1]);
                pieces[j] = merged;
                pieces.remove(j + 1);
            } else {
                break;
            }
        }

        pieces
            .iter()
            .map(|piece| self.token_to_id.get(piece).copied().unwrap_or(3))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_test_vocab(dir: &Path) -> std::path::PathBuf {
        let vocab = vec![
            "<pad>".to_string(),
            "<eos>".to_string(),
            "<bos>".to_string(),
            "<unk>".to_string(),
            "▁".to_string(),
            "H".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "▁H".to_string(),
            "el".to_string(),
            "lo".to_string(),
            "▁He".to_string(),
            "llo".to_string(),
            "▁Hel".to_string(),
            "▁Hello".to_string(),
        ];
        let path = dir.join("vocab.json");
        std::fs::write(&path, serde_json::to_vec(&vocab).unwrap()).unwrap();
        path
    }

    fn write_test_scores(dir: &Path) -> std::path::PathBuf {
        let scores: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -2.0, -2.0, -0.5, -1.5, -1.5, -0.3, -1.0, -0.2,
            -0.1,
        ];
        let path = dir.join("scores.json");
        std::fs::write(&path, serde_json::to_vec(&scores).unwrap()).unwrap();
        path
    }

    #[test]
    fn tokenizer_encodes_known_word() {
        let temp = tempdir().unwrap();
        let vocab_path = write_test_vocab(temp.path());
        let scores_path = write_test_scores(temp.path());
        let tok = GemmaTokenizer::load(&vocab_path, Some(&scores_path)).unwrap();

        let ids = tok.encode("Hello");
        assert_eq!(ids, vec![15]);

        let decoded = tok.decode_ids(&ids);
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn tokenizer_encodes_with_bos() {
        let temp = tempdir().unwrap();
        let vocab_path = write_test_vocab(temp.path());
        let tok = GemmaTokenizer::load(&vocab_path, None).unwrap();

        let ids = tok.encode_with_bos("Hello");
        assert_eq!(ids[0], 2);
    }

    #[test]
    fn tokenizer_prefers_turn_token_as_eos_when_present() {
        let temp = tempdir().unwrap();
        let vocab = vec![
            "<pad>".to_string(),
            "<eos>".to_string(),
            "<bos>".to_string(),
            "<unk>".to_string(),
            "<turn|>".to_string(),
        ];
        let vocab_path = temp.path().join("vocab.json");
        std::fs::write(&vocab_path, serde_json::to_vec(&vocab).unwrap()).unwrap();

        let tok = GemmaTokenizer::load(&vocab_path, None).unwrap();
        assert_eq!(tok.eos_id(), 4);
    }

    #[test]
    fn tokenizer_falls_back_to_eos_when_turn_token_is_absent() {
        let temp = tempdir().unwrap();
        let vocab_path = write_test_vocab(temp.path());
        let tok = GemmaTokenizer::load(&vocab_path, None).unwrap();

        assert_eq!(tok.eos_id(), 1);
    }
}
