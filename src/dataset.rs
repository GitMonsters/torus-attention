//! # Dataset Loading for LLM Training
//!
//! Supports loading text data from various formats:
//! - Plain text files (one document per file or line-delimited)
//! - JSON Lines format (each line is a JSON object with a "text" field)
//! - Directory of text files
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::dataset::{TextDataset, DataLoader};
//!
//! // Load from a text file
//! let dataset = TextDataset::from_file("data/corpus.txt", &tokenizer)?;
//!
//! // Create batches
//! let loader = DataLoader::new(dataset, 32, true);
//! for batch in loader {
//!     // batch.input_ids: [batch_size, seq_len]
//!     // batch.labels: [batch_size, seq_len]
//! }
//! ```

use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use std::path::Path;

use crate::error::TorusError;
use crate::tokenizer::SimpleTokenizer;
use crate::TorusResult;

/// A single training example
#[derive(Debug, Clone)]
pub struct Example {
    /// Token IDs
    pub token_ids: Vec<u32>,
}

/// Dataset of tokenized text examples
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// All tokenized examples
    examples: Vec<Example>,
    /// Sequence length for training
    seq_len: usize,
    /// Vocabulary size
    vocab_size: usize,
}

impl TextDataset {
    /// Create a new empty dataset
    pub fn new(seq_len: usize, vocab_size: usize) -> Self {
        Self {
            examples: Vec::new(),
            seq_len,
            vocab_size,
        }
    }

    /// Load dataset from a plain text file
    ///
    /// The file is split into chunks of `seq_len` tokens.
    pub fn from_file(
        path: impl AsRef<Path>,
        tokenizer: &SimpleTokenizer,
        seq_len: usize,
    ) -> TorusResult<Self> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| TorusError::Io(format!("Failed to read file: {}", e)))?;

        Self::from_text(&content, tokenizer, seq_len)
    }

    /// Load dataset from a string
    pub fn from_text(text: &str, tokenizer: &SimpleTokenizer, seq_len: usize) -> TorusResult<Self> {
        let tokens = tokenizer.encode(text);
        let vocab_size = tokenizer.vocab_size();

        let mut dataset = Self::new(seq_len, vocab_size);

        // Split into chunks of seq_len + 1 (for next-token prediction)
        let chunk_size = seq_len + 1;
        for chunk in tokens.chunks(chunk_size) {
            if chunk.len() >= 2 {
                dataset.examples.push(Example {
                    token_ids: chunk.to_vec(),
                });
            }
        }

        log::info!(
            "Loaded {} examples from text ({} tokens)",
            dataset.len(),
            tokens.len()
        );
        Ok(dataset)
    }

    /// Load dataset from JSON Lines file
    ///
    /// Each line should be a JSON object with a "text" field.
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        tokenizer: &SimpleTokenizer,
        seq_len: usize,
    ) -> TorusResult<Self> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| TorusError::Io(format!("Failed to read file: {}", e)))?;

        let vocab_size = tokenizer.vocab_size();
        let mut dataset = Self::new(seq_len, vocab_size);

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    let tokens = tokenizer.encode_with_special(text, true, true);
                    if tokens.len() >= 2 {
                        // Chunk if needed
                        let chunk_size = seq_len + 1;
                        for chunk in tokens.chunks(chunk_size) {
                            if chunk.len() >= 2 {
                                dataset.examples.push(Example {
                                    token_ids: chunk.to_vec(),
                                });
                            }
                        }
                    }
                }
            }
        }

        log::info!("Loaded {} examples from JSONL", dataset.len());
        Ok(dataset)
    }

    /// Load from a directory of text files
    pub fn from_directory(
        dir: impl AsRef<Path>,
        tokenizer: &SimpleTokenizer,
        seq_len: usize,
    ) -> TorusResult<Self> {
        let dir = dir.as_ref();
        let vocab_size = tokenizer.vocab_size();
        let mut dataset = Self::new(seq_len, vocab_size);

        for entry in fs::read_dir(dir).map_err(|e| TorusError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| TorusError::Io(e.to_string()))?;
            let path = entry.path();

            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

                match ext {
                    "txt" => {
                        if let Ok(sub_dataset) = Self::from_file(&path, tokenizer, seq_len) {
                            dataset.examples.extend(sub_dataset.examples);
                        }
                    }
                    "jsonl" => {
                        if let Ok(sub_dataset) = Self::from_jsonl(&path, tokenizer, seq_len) {
                            dataset.examples.extend(sub_dataset.examples);
                        }
                    }
                    _ => {}
                }
            }
        }

        log::info!("Loaded {} examples from directory", dataset.len());
        Ok(dataset)
    }

    /// Create a synthetic dataset for testing
    pub fn synthetic(num_examples: usize, seq_len: usize, vocab_size: usize) -> Self {
        use rand::Rng;
        let mut rng = thread_rng();

        let examples: Vec<Example> = (0..num_examples)
            .map(|_| {
                let token_ids: Vec<u32> = (0..seq_len + 1)
                    .map(|_| rng.gen_range(0..vocab_size as u32))
                    .collect();
                Example { token_ids }
            })
            .collect();

        Self {
            examples,
            seq_len,
            vocab_size,
        }
    }

    /// Number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.examples.shuffle(&mut rng);
    }

    /// Get a single example
    pub fn get(&self, idx: usize) -> Option<&Example> {
        self.examples.get(idx)
    }

    /// Split into train and validation sets
    pub fn train_val_split(mut self, val_ratio: f64) -> (Self, Self) {
        self.shuffle();

        let val_size = (self.examples.len() as f64 * val_ratio) as usize;
        let val_examples = self.examples.split_off(self.examples.len() - val_size);

        let train = Self {
            examples: self.examples,
            seq_len: self.seq_len,
            vocab_size: self.vocab_size,
        };

        let val = Self {
            examples: val_examples,
            seq_len: self.seq_len,
            vocab_size: self.vocab_size,
        };

        (train, val)
    }
}

/// A batch of training data
#[derive(Debug)]
pub struct Batch {
    /// Input token IDs [batch_size, seq_len]
    pub input_ids: Tensor,
    /// Labels for next-token prediction [batch_size, seq_len]
    pub labels: Tensor,
}

/// DataLoader for batching and iterating over a dataset
pub struct DataLoader {
    dataset: TextDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    position: usize,
    device: Device,
}

impl DataLoader {
    /// Create a new DataLoader
    pub fn new(dataset: TextDataset, batch_size: usize, shuffle: bool, device: Device) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            position: 0,
            device,
        }
    }

    /// Reset the iterator and optionally shuffle
    pub fn reset(&mut self) {
        self.position = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Number of batches
    pub fn num_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }

    /// Get the next batch
    pub fn next_batch(&mut self) -> Option<TorusResult<Batch>> {
        if self.position >= self.dataset.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.position..end];
        self.position = end;

        Some(self.create_batch(batch_indices))
    }

    fn create_batch(&self, indices: &[usize]) -> TorusResult<Batch> {
        let seq_len = self.dataset.seq_len();
        let batch_size = indices.len();

        let mut input_data = vec![0u32; batch_size * seq_len];
        let mut label_data = vec![0u32; batch_size * seq_len];

        for (batch_idx, &example_idx) in indices.iter().enumerate() {
            let example = &self.dataset.examples[example_idx];
            let tokens = &example.token_ids;

            // Input is tokens[:-1], labels is tokens[1:]
            let input_len = tokens.len().saturating_sub(1).min(seq_len);
            let offset = batch_idx * seq_len;

            input_data[offset..(input_len + offset)].copy_from_slice(&tokens[..input_len]);
            label_data[offset..(input_len + offset)].copy_from_slice(&tokens[1..(input_len + 1)]);

            // Pad with zeros if needed
            for i in input_len..seq_len {
                input_data[offset + i] = 0;
                label_data[offset + i] = 0; // Will be masked in loss
            }
        }

        let input_ids = Tensor::from_vec(input_data, (batch_size, seq_len), &self.device)?;
        let labels = Tensor::from_vec(label_data, (batch_size, seq_len), &self.device)?;

        Ok(Batch { input_ids, labels })
    }
}

impl Iterator for DataLoader {
    type Item = TorusResult<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let dataset = TextDataset::synthetic(100, 64, 1000);
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.seq_len(), 64);
        assert_eq!(dataset.vocab_size(), 1000);
    }

    #[test]
    fn test_from_text() {
        let tokenizer = SimpleTokenizer::new_basic(1000);
        let text = "Hello world! This is a test of the tokenizer. It should work well.";
        let dataset = TextDataset::from_text(text, &tokenizer, 16).unwrap();
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_train_val_split() {
        let dataset = TextDataset::synthetic(100, 64, 1000);
        let (train, val) = dataset.train_val_split(0.2);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_dataloader() {
        let device = Device::Cpu;
        let dataset = TextDataset::synthetic(100, 64, 1000);
        let mut loader = DataLoader::new(dataset, 32, true, device);

        let mut batch_count = 0;
        while let Some(batch_result) = loader.next_batch() {
            let batch = batch_result.unwrap();
            assert!(batch.input_ids.dims()[0] <= 32);
            batch_count += 1;
        }

        assert_eq!(batch_count, 4); // 100 / 32 = 4 batches (rounded up)
    }
}
