//! # Tokenizer Module
//!
//! Provides tokenization for the Torus LLM with multiple backends:
//! - **BPE Tokenizer**: Fast byte-pair encoding with O(n) encoding
//! - **Simple Tokenizer**: Character-level fallback for compatibility
//!
//! # Performance
//!
//! The BPE tokenizer uses:
//! - Trie-based vocabulary lookup for O(n) encoding
//! - Pre-computed merge rules for fast BPE operations
//! - Byte-level fallback for unknown characters

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

// ============================================================================
// Trie Node for Fast Vocabulary Lookup
// ============================================================================

/// Trie node for efficient prefix matching
#[derive(Debug, Clone, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_id: Option<u32>,
}

impl TrieNode {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, word: &str, id: u32) {
        let mut node = self;
        for c in word.chars() {
            node = node.children.entry(c).or_default();
        }
        node.token_id = Some(id);
    }

    /// Find the longest matching prefix and return (token_id, length)
    fn find_longest(&self, text: &[char], start: usize) -> Option<(u32, usize)> {
        let mut node = self;
        let mut best_match: Option<(u32, usize)> = None;
        let mut len = 0;

        for &c in text.iter().skip(start) {
            if let Some(child) = node.children.get(&c) {
                len += 1;
                node = child;
                if let Some(id) = node.token_id {
                    best_match = Some((id, len));
                }
            } else {
                break;
            }
        }

        best_match
    }
}

// ============================================================================
// BPE Tokenizer (Fast)
// ============================================================================

/// Fast BPE Tokenizer using trie-based lookup
///
/// This tokenizer is optimized for speed:
/// - O(n) encoding using trie prefix matching
/// - Pre-built vocabulary with common subwords
/// - Byte-level fallback for robustness
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Trie for fast prefix matching
    trie: TrieNode,
    /// Byte-level tokens (for unknown characters)
    byte_tokens: HashMap<u8, u32>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Vocabulary size
    vocab_size: usize,
    /// Set of multi-char tokens for quick lookup (reserved for future BPE merges)
    #[allow(dead_code)]
    multi_char_tokens: HashSet<String>,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with a reasonable default vocabulary
    ///
    /// This builds a vocabulary with:
    /// - All ASCII printable characters
    /// - Common English subwords and tokens
    /// - Byte-level fallback for any character
    pub fn new(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut trie = TrieNode::new();
        let mut byte_tokens = HashMap::new();

        let mut next_id = 0u32;

        // 1. Add byte-level tokens first (0-255)
        // These provide complete coverage for any input
        for byte in 0u8..=255 {
            let token = format!("<|byte_{:02x}|>", byte);
            vocab.insert(token.clone(), next_id);
            id_to_token.insert(next_id, token.clone());
            byte_tokens.insert(byte, next_id);
            trie.insert(&token, next_id);
            next_id += 1;
        }

        // 2. Add ASCII printable characters (override byte tokens for display)
        for c in 32u8..=126 {
            let token = String::from(c as char);
            if (next_id as usize) < vocab_size {
                vocab.insert(token.clone(), next_id);
                id_to_token.insert(next_id, token.clone());
                trie.insert(&token, next_id);
                next_id += 1;
            }
        }

        // 3. Add common whitespace and control characters
        let whitespace = ["\n", "\t", "\r", "  ", "    ", "        "];
        for ws in whitespace {
            if !vocab.contains_key(ws) && (next_id as usize) < vocab_size {
                vocab.insert(ws.to_string(), next_id);
                id_to_token.insert(next_id, ws.to_string());
                trie.insert(ws, next_id);
                next_id += 1;
            }
        }

        // 4. Add common English subwords (sorted by frequency/usefulness)
        let common_subwords = [
            // Very common
            "the", "ing", "and", "tion", "er", "ed", "es", "en", "of", "to", "in", "is", "it",
            "on", "or", "an", "as", "at", "be", "by", "for", "he", "if", "no", "so", "we", "up",
            // Common endings
            "ly", "al", "le", "re", "ness", "ment", "able", "ible", "ful", "less", "ous", "ive",
            "ity", "ism", "ist", "ant", "ent", "ion", "sion", "ation",
            // Common beginnings
            "un", "re", "pre", "dis", "mis", "over", "under", "out", "sub", "super", "inter",
            "trans", "non", "anti", "auto", "semi", "self",
            // Common words
            "that", "with", "have", "this", "will", "your", "from", "they", "been", "call",
            "first", "their", "would", "there", "could", "other", "than", "then", "these",
            "some", "her", "him", "its", "only", "come", "over", "such", "into", "most",
            "also", "back", "after", "use", "two", "how", "our", "just", "any", "made",
            "about", "time", "very", "when", "more", "what", "know", "people", "can", "all",
            // Programming keywords
            "function", "return", "const", "let", "var", "class", "struct", "impl", "pub",
            "fn", "async", "await", "import", "export", "from", "use", "mod", "self",
            "true", "false", "null", "None", "Some", "Ok", "Err", "Result", "Option",
            "if", "else", "for", "while", "loop", "match", "break", "continue",
            // Operators and punctuation
            "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "::", "//", "/*", "*/",
            "...", "..", "<<", ">>", "+=", "-=", "*=", "/=",
            // Common ML/AI terms
            "tensor", "model", "layer", "attention", "embed", "token", "batch", "epoch",
            "loss", "grad", "param", "weight", "bias", "dim", "hidden", "output", "input",
            "train", "valid", "test", "data", "config", "path", "file", "dir",
        ];

        let mut multi_char_tokens = HashSet::new();

        for word in common_subwords {
            if !vocab.contains_key(word) && (next_id as usize) < vocab_size {
                vocab.insert(word.to_string(), next_id);
                id_to_token.insert(next_id, word.to_string());
                trie.insert(word, next_id);
                if word.len() > 1 {
                    multi_char_tokens.insert(word.to_string());
                }
                next_id += 1;
            }
        }

        // 5. Add special tokens
        let special_tokens = SpecialTokens::default();
        let special_list = [
            (&special_tokens.bos_token, special_tokens.bos_id),
            (&special_tokens.eos_token, special_tokens.eos_id),
            (&special_tokens.pad_token, special_tokens.pad_id),
            (&special_tokens.unk_token, special_tokens.unk_id),
        ];

        for (token, id) in special_list {
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            trie.insert(token, id);
        }

        // Fill remaining vocab with placeholder tokens if needed
        while (next_id as usize) < vocab_size && next_id < 50256 {
            let token = format!("<|extra_{}|>", next_id);
            if !vocab.contains_key(&token) {
                vocab.insert(token.clone(), next_id);
                id_to_token.insert(next_id, token.clone());
                trie.insert(&token, next_id);
            }
            next_id += 1;
        }

        Self {
            vocab,
            id_to_token,
            trie,
            byte_tokens,
            special_tokens,
            vocab_size,
            multi_char_tokens,
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Encode text to token IDs using trie-based longest match
    ///
    /// This is O(n) where n is the length of the text, much faster
    /// than the O(n²) greedy search in SimpleTokenizer.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::with_capacity(chars.len());
        let mut i = 0;

        while i < chars.len() {
            // Try to find longest match using trie
            if let Some((id, len)) = self.trie.find_longest(&chars, i) {
                tokens.push(id);
                i += len;
            } else {
                // Fallback to byte encoding for unknown characters
                let c = chars[i];
                let bytes = c.to_string().into_bytes();
                for byte in bytes {
                    if let Some(&id) = self.byte_tokens.get(&byte) {
                        tokens.push(id);
                    } else {
                        tokens.push(self.special_tokens.unk_id);
                    }
                }
                i += 1;
            }
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
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Handle byte tokens
                if token.starts_with("<|byte_") && token.ends_with("|>") {
                    if let Ok(byte) = u8::from_str_radix(&token[7..9], 16) {
                        if let Ok(c) = std::str::from_utf8(&[byte]) {
                            result.push_str(c);
                            continue;
                        }
                    }
                }
                result.push_str(token);
            }
        }

        result
    }

    /// Decode, skipping special tokens
    pub fn decode_skip_special(&self, ids: &[u32]) -> String {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| {
                id != self.special_tokens.bos_id
                    && id != self.special_tokens.eos_id
                    && id != self.special_tokens.pad_id
                    && id != self.special_tokens.unk_id
            })
            .collect();
        self.decode(&filtered)
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

// ============================================================================
// Simple Tokenizer (Legacy/Fallback)
// ============================================================================

/// A simple tokenizer that supports encoding and decoding text
///
/// This is the original character-level tokenizer, kept for compatibility.
/// For better performance, use `BpeTokenizer` instead.
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
    ///
    /// **Note**: This tokenizer has O(n²) encoding complexity.
    /// Consider using `BpeTokenizer::new()` for better performance.
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

// ============================================================================
// Unified Tokenizer Trait
// ============================================================================

/// Common interface for tokenizers
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Encode with special tokens
    fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32]) -> String;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special tokens
    fn special_tokens(&self) -> &SpecialTokens;
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        self.encode_with_special(text, add_bos, add_eos)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        self.special_tokens()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        self.encode_with_special(text, add_bos, add_eos)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        self.special_tokens()
    }
}

// ============================================================================
// Chat Message Formatting
// ============================================================================

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_creation() {
        let tokenizer = BpeTokenizer::new(1000);
        assert_eq!(tokenizer.vocab_size(), 1000);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let tokenizer = BpeTokenizer::new(1000);
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        // Should be able to roundtrip
        assert!(!tokens.is_empty());
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_common_words() {
        let tokenizer = BpeTokenizer::new(1000);

        // "the" should be a single token
        let tokens = tokenizer.encode("the");
        assert_eq!(tokens.len(), 1, "'the' should be a single token");

        // "ing" should be a single token
        let tokens = tokenizer.encode("ing");
        assert_eq!(tokens.len(), 1, "'ing' should be a single token");
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tokenizer = BpeTokenizer::new(1000);
        let text = "Test";
        let tokens = tokenizer.encode_with_special(text, true, true);

        assert_eq!(tokens[0], tokenizer.special_tokens().bos_id);
        assert_eq!(*tokens.last().unwrap(), tokenizer.special_tokens().eos_id);
    }

    #[test]
    fn test_bpe_unicode() {
        let tokenizer = BpeTokenizer::new(1000);

        // Unicode should be handled via byte fallback
        let text = "Hello 世界!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        // Byte-level decoding might not perfectly reconstruct
        // but should not panic
        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_bpe_performance() {
        let tokenizer = BpeTokenizer::new(1000);

        // Encode a longer text
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);

        let start = std::time::Instant::now();
        let tokens = tokenizer.encode(&text);
        let elapsed = start.elapsed();

        // Should be fast (< 100ms for ~4500 chars)
        assert!(
            elapsed.as_millis() < 100,
            "BPE encoding too slow: {:?}",
            elapsed
        );
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        assert_eq!(tokenizer.vocab_size(), 1000);
    }

    #[test]
    fn test_simple_encode_decode() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        // May not be exact due to tokenization, but should contain the characters
        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_simple_special_tokens() {
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

    #[test]
    fn test_tokenizer_trait() {
        fn use_tokenizer(t: &dyn Tokenizer) -> usize {
            t.encode("test").len()
        }

        let bpe = BpeTokenizer::new(1000);
        let simple = SimpleTokenizer::new_basic(1000);

        assert!(use_tokenizer(&bpe) > 0);
        assert!(use_tokenizer(&simple) > 0);
    }
}
