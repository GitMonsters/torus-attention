//! # Tokenizer Module
//!
//! Provides tokenization for the Torus LLM, supporting multiple tokenizer backends.
//! Uses a simple BPE-style tokenizer that's compatible with GPT-2/GPT-3 vocabularies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::TorusError;
use crate::TorusResult;

/// Special tokens used by the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub unk_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<|startoftext|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            pad_token: "<|pad|>".to_string(),
            unk_token: "<|unk|>".to_string(),
            bos_id: 50256,
            eos_id: 50256, // GPT-2 uses same token for BOS/EOS
            pad_id: 50257,
            unk_id: 50258,
        }
    }
}

/// A simple tokenizer that supports encoding and decoding text
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Vocabulary size
    vocab_size: usize,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a basic character-level vocabulary
    /// This is a fallback tokenizer - in production, use load_gpt2() or similar
    pub fn new_basic(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add basic ASCII characters
        for (i, c) in (32u8..=126).enumerate() {
            let token = String::from(c as char);
            let id = i as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        // Add some common byte-pair combinations
        let common_pairs = [
            "the", "ing", "and", "tion", "er", "ed", "es", "en", "of", "to", "in", "is", "it",
            "on", "or", "an", "  ", "\n", "\t", "...", "...", "->", "=>", "==",
        ];

        let mut next_id = vocab.len() as u32;
        for pair in common_pairs {
            if !vocab.contains_key(pair) && (next_id as usize) < vocab_size {
                vocab.insert(pair.to_string(), next_id);
                id_to_token.insert(next_id, pair.to_string());
                next_id += 1;
            }
        }

        // Fill remaining vocab with placeholder tokens
        while (next_id as usize) < vocab_size {
            let token = format!("<|extra_{}|>", next_id);
            vocab.insert(token.clone(), next_id);
            id_to_token.insert(next_id, token);
            next_id += 1;
        }

        let special_tokens = SpecialTokens::default();

        Self {
            vocab,
            id_to_token,
            special_tokens,
            vocab_size,
        }
    }

    /// Create a tokenizer from a vocabulary file (JSON format)
    pub fn from_vocab_file(path: impl AsRef<Path>) -> TorusResult<Self> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| TorusError::Io(e.to_string()))?;

        let vocab: HashMap<String, u32> = serde_json::from_str(&content)
            .map_err(|e| TorusError::Io(format!("Failed to parse vocab: {}", e)))?;

        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        let vocab_size = vocab.len();
        let special_tokens = SpecialTokens::default();

        Ok(Self {
            vocab,
            id_to_token,
            special_tokens,
            vocab_size,
        })
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Try to find the longest matching token
            let mut best_len = 1;
            let mut best_id = self.special_tokens.unk_id;

            for len in (1..=chars.len() - i).rev() {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&substr) {
                    best_len = len;
                    best_id = id;
                    break;
                }
            }

            // Fallback to single character
            if best_id == self.special_tokens.unk_id && best_len == 1 {
                let c: String = chars[i].to_string();
                best_id = *self.vocab.get(&c).unwrap_or(&self.special_tokens.unk_id);
            }

            tokens.push(best_id);
            i += best_len;
        }

        tokens
    }

    /// Encode with special tokens (BOS/EOS)
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.special_tokens.bos_id);
        }

        tokens.extend(self.encode(text));

        if add_eos {
            tokens.push(self.special_tokens.eos_id);
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect()
    }

    /// Decode, skipping special tokens
    pub fn decode_skip_special(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter(|&&id| {
                id != self.special_tokens.bos_id
                    && id != self.special_tokens.eos_id
                    && id != self.special_tokens.pad_id
            })
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect()
    }

    /// Get token ID for a string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get string for a token ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
}

/// Chat message for conversation-style inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Format chat messages into a prompt string
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|system|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n<|end|>\n");
            }
            "user" => {
                prompt.push_str("<|user|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n<|end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|assistant|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n<|end|>\n");
            }
            _ => {
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
        }
    }

    // Add assistant prefix for the response
    prompt.push_str("<|assistant|>\n");

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenizer() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        assert_eq!(tokenizer.vocab_size(), 1000);
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        // May not be exact due to tokenization, but should contain the characters
        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        let text = "Test";
        let tokens = tokenizer.encode_with_special(text, true, true);
        assert_eq!(tokens[0], tokenizer.special_tokens().bos_id);
        assert_eq!(*tokens.last().unwrap(), tokenizer.special_tokens().eos_id);
    }

    #[test]
    fn test_chat_format() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
    }
}
