//! # LLM Integration for AGI Core
//!
//! This module provides integration between the TorusLLM language model and
//! the AGI Core system, enabling:
//!
//! 1. **Language Understanding**: Parse natural language inputs into symbolic representations
//! 2. **Language Generation**: Generate natural language from internal states
//! 3. **Semantic Grounding**: Connect language to sensorimotor experience
//! 4. **Reasoning Integration**: Use LLM for complex reasoning tasks
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     LLM Integration Layer                           │
//! │                                                                     │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
//! │  │  Language   │    │  Semantic   │    │    Reasoning Engine     │ │
//! │  │ Processor   │───▶│  Grounding  │───▶│   (Chain-of-Thought)    │ │
//! │  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
//! │         │                  │                       │               │
//! │         ▼                  ▼                       ▼               │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                    AGI Core Interface                        │   │
//! │  │  • Symbol System ◀──────────────────────────────────────────│   │
//! │  │  • World Model   ◀──────────────────────────────────────────│   │
//! │  │  • Goal Hierarchy◀──────────────────────────────────────────│   │
//! │  │  • Memory System ◀──────────────────────────────────────────│   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for LLM integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMIntegrationConfig {
    /// Maximum tokens for generation
    pub max_generation_tokens: usize,
    /// Maximum tokens for context
    pub max_context_tokens: usize,
    /// Temperature for generation
    pub temperature: f64,
    /// Top-p sampling parameter
    pub top_p: f64,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Whether to use chain-of-thought prompting
    pub use_chain_of_thought: bool,
    /// Number of reasoning steps for CoT
    pub cot_steps: usize,
    /// Embedding dimension for semantic grounding
    pub embedding_dim: usize,
    /// Confidence threshold for grounding
    pub grounding_threshold: f64,
    /// Enable self-critique of generated responses
    pub enable_self_critique: bool,
}

impl Default for LLMIntegrationConfig {
    fn default() -> Self {
        Self {
            max_generation_tokens: 512,
            max_context_tokens: 2048,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            use_chain_of_thought: true,
            cot_steps: 3,
            embedding_dim: 256,
            grounding_threshold: 0.5,
            enable_self_critique: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LANGUAGE UNDERSTANDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsed semantic structure from natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSemantics {
    /// Intent detected in the input
    pub intent: Intent,
    /// Entities extracted
    pub entities: Vec<Entity>,
    /// Relations between entities
    pub relations: Vec<Relation>,
    /// Sentiment/emotion
    pub sentiment: Sentiment,
    /// Confidence in the parse
    pub confidence: f64,
    /// Raw embedding vector
    pub embedding: Vec<f64>,
}

/// Types of intents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
    Query,         // Asking for information
    Command,       // Requesting action
    Statement,     // Providing information
    Greeting,      // Social interaction
    Clarification, // Asking for clarification
    Confirmation,  // Confirming something
    Negation,      // Denying something
    Unknown,
}

/// Extracted entity from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
    /// Grounded symbol ID (if grounded)
    pub symbol_id: Option<usize>,
}

/// Types of entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Object,
    Action,
    Location,
    Time,
    Quantity,
    Property,
    Agent,
    Goal,
    Concept,
    Unknown,
}

/// Relation between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity index
    pub source: usize,
    /// Target entity index
    pub target: usize,
    /// Relation type
    pub relation_type: String,
    /// Confidence
    pub confidence: f64,
}

/// Sentiment analysis result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sentiment {
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
}

impl Sentiment {
    pub fn dominant(&self) -> &'static str {
        if self.positive > self.negative && self.positive > self.neutral {
            "positive"
        } else if self.negative > self.positive && self.negative > self.neutral {
            "negative"
        } else {
            "neutral"
        }
    }
}

/// Language processor for parsing and understanding
#[derive(Debug, Clone)]
pub struct LanguageProcessor {
    /// Vocabulary for tokenization
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary
    reverse_vocab: HashMap<u32, String>,
    /// Next token ID
    next_token_id: u32,
    /// Known entity patterns
    entity_patterns: Vec<(String, EntityType)>,
    /// Configuration
    config: LLMIntegrationConfig,
}

impl LanguageProcessor {
    pub fn new(config: LLMIntegrationConfig) -> Self {
        let mut processor = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_token_id: 0,
            entity_patterns: Vec::new(),
            config,
        };

        // Initialize basic vocabulary
        processor.add_special_tokens();
        processor.add_entity_patterns();

        processor
    }

    fn add_special_tokens(&mut self) {
        let special = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"];
        for token in special {
            self.add_token(token.to_string());
        }
    }

    fn add_entity_patterns(&mut self) {
        // Basic entity patterns (would be learned in production)
        self.entity_patterns.extend([
            ("go".to_string(), EntityType::Action),
            ("move".to_string(), EntityType::Action),
            ("find".to_string(), EntityType::Action),
            ("get".to_string(), EntityType::Action),
            ("put".to_string(), EntityType::Action),
            ("take".to_string(), EntityType::Action),
            ("north".to_string(), EntityType::Location),
            ("south".to_string(), EntityType::Location),
            ("east".to_string(), EntityType::Location),
            ("west".to_string(), EntityType::Location),
            ("goal".to_string(), EntityType::Goal),
            ("target".to_string(), EntityType::Goal),
        ]);
    }

    fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.vocab.get(&token) {
            return id;
        }
        let id = self.next_token_id;
        self.vocab.insert(token.clone(), id);
        self.reverse_vocab.insert(id, token);
        self.next_token_id += 1;
        id
    }

    /// Tokenize input text
    pub fn tokenize(&mut self, text: &str) -> Vec<u32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        words
            .iter()
            .map(|w| {
                let lower = w.to_lowercase();
                if let Some(&id) = self.vocab.get(&lower) {
                    id
                } else {
                    self.add_token(lower)
                }
            })
            .collect()
    }

    /// Detokenize back to text
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Parse natural language input into semantic structure
    pub fn parse(&mut self, text: &str) -> ParsedSemantics {
        let tokens = self.tokenize(text);
        let words: Vec<&str> = text.split_whitespace().collect();

        // Detect intent (simplified rule-based)
        let intent = self.detect_intent(text);

        // Extract entities
        let entities = self.extract_entities(&words);

        // Extract relations (simplified)
        let relations = self.extract_relations(&entities);

        // Analyze sentiment (simplified)
        let sentiment = self.analyze_sentiment(text);

        // Create embedding
        let embedding = self.create_embedding(&tokens);

        // Calculate confidence
        let confidence = if entities.is_empty() { 0.3 } else { 0.7 };

        ParsedSemantics {
            intent,
            entities,
            relations,
            sentiment,
            confidence,
            embedding,
        }
    }

    fn detect_intent(&self, text: &str) -> Intent {
        let lower = text.to_lowercase();

        if lower.contains('?')
            || lower.starts_with("what")
            || lower.starts_with("where")
            || lower.starts_with("how")
            || lower.starts_with("why")
            || lower.starts_with("when")
        {
            Intent::Query
        } else if lower.starts_with("go")
            || lower.starts_with("move")
            || lower.starts_with("find")
            || lower.starts_with("get")
            || lower.starts_with("put")
            || lower.starts_with("do")
        {
            Intent::Command
        } else if lower.starts_with("hello") || lower.starts_with("hi") || lower.starts_with("hey")
        {
            Intent::Greeting
        } else if lower.starts_with("yes") || lower.starts_with("ok") || lower.starts_with("sure") {
            Intent::Confirmation
        } else if lower.starts_with("no") || lower.starts_with("don't") || lower.starts_with("not")
        {
            Intent::Negation
        } else {
            Intent::Statement
        }
    }

    fn extract_entities(&self, words: &[&str]) -> Vec<Entity> {
        let mut entities = Vec::new();
        let mut pos = 0;

        for (i, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();

            for (pattern, entity_type) in &self.entity_patterns {
                if lower.contains(pattern) {
                    entities.push(Entity {
                        text: word.to_string(),
                        entity_type: *entity_type,
                        start: pos,
                        end: pos + word.len(),
                        confidence: 0.8,
                        symbol_id: None,
                    });
                    break;
                }
            }

            pos += word.len() + 1; // +1 for space
        }

        entities
    }

    fn extract_relations(&self, entities: &[Entity]) -> Vec<Relation> {
        let mut relations = Vec::new();

        // Simple adjacency-based relation extraction
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                if entities[i].entity_type == EntityType::Action {
                    relations.push(Relation {
                        source: i,
                        target: j,
                        relation_type: "acts_on".to_string(),
                        confidence: 0.6,
                    });
                }
            }
        }

        relations
    }

    fn analyze_sentiment(&self, text: &str) -> Sentiment {
        let lower = text.to_lowercase();

        // Simple keyword-based sentiment (would use model in production)
        let positive_words = ["good", "great", "excellent", "happy", "yes", "success"];
        let negative_words = ["bad", "terrible", "fail", "no", "error", "wrong"];

        let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count();
        let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count();
        let total = (pos_count + neg_count).max(1) as f64;

        let positive = pos_count as f64 / total * 0.5 + 0.25;
        let negative = neg_count as f64 / total * 0.5 + 0.25;
        let neutral = 1.0 - positive - negative;

        Sentiment {
            positive,
            negative,
            neutral,
        }
    }

    fn create_embedding(&self, tokens: &[u32]) -> Vec<f64> {
        let dim = self.config.embedding_dim;
        let mut embedding = vec![0.0; dim];

        // Simple bag-of-words style embedding
        for &token in tokens {
            let idx = (token as usize) % dim;
            embedding[idx] += 1.0;
        }

        // Normalize
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for e in &mut embedding {
                *e /= norm;
            }
        }

        embedding
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC GROUNDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Grounded symbol linking language to experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedSymbol {
    /// Symbol ID
    pub id: usize,
    /// Linguistic form
    pub word: String,
    /// Perceptual signature (from sensorimotor experience)
    pub perceptual_signature: Vec<f64>,
    /// Action signature (motor patterns associated)
    pub action_signature: Vec<f64>,
    /// Contexts where this symbol was used
    pub usage_contexts: Vec<usize>,
    /// Grounding strength (0-1)
    pub grounding_strength: f64,
    /// Number of times grounded
    pub grounding_count: usize,
}

/// Semantic grounding system
#[derive(Debug, Clone)]
pub struct SemanticGrounding {
    /// Grounded symbols
    symbols: Vec<GroundedSymbol>,
    /// Word to symbol ID mapping
    word_to_symbol: HashMap<String, usize>,
    /// Next symbol ID
    next_id: usize,
    /// Configuration
    config: LLMIntegrationConfig,
}

impl SemanticGrounding {
    pub fn new(config: LLMIntegrationConfig) -> Self {
        Self {
            symbols: Vec::new(),
            word_to_symbol: HashMap::new(),
            next_id: 0,
            config,
        }
    }

    /// Ground a word to perceptual/motor experience
    pub fn ground(
        &mut self,
        word: &str,
        perceptual: Vec<f64>,
        action: Vec<f64>,
        context_id: usize,
    ) -> usize {
        let lower = word.to_lowercase();

        if let Some(&id) = self.word_to_symbol.get(&lower) {
            // Update existing symbol
            if let Some(symbol) = self.symbols.iter_mut().find(|s| s.id == id) {
                // Blend perceptual signatures
                for (i, p) in perceptual.iter().enumerate() {
                    if i < symbol.perceptual_signature.len() {
                        symbol.perceptual_signature[i] =
                            0.9 * symbol.perceptual_signature[i] + 0.1 * p;
                    }
                }
                // Blend action signatures
                for (i, a) in action.iter().enumerate() {
                    if i < symbol.action_signature.len() {
                        symbol.action_signature[i] = 0.9 * symbol.action_signature[i] + 0.1 * a;
                    }
                }
                symbol.usage_contexts.push(context_id);
                symbol.grounding_count += 1;
                symbol.grounding_strength = (symbol.grounding_strength + 0.1).min(1.0);
            }
            return id;
        }

        // Create new grounded symbol
        let symbol = GroundedSymbol {
            id: self.next_id,
            word: lower.clone(),
            perceptual_signature: perceptual,
            action_signature: action,
            usage_contexts: vec![context_id],
            grounding_strength: 0.5,
            grounding_count: 1,
        };

        let id = symbol.id;
        self.symbols.push(symbol);
        self.word_to_symbol.insert(lower, id);
        self.next_id += 1;

        id
    }

    /// Look up grounding for a word
    pub fn lookup(&self, word: &str) -> Option<&GroundedSymbol> {
        let lower = word.to_lowercase();
        self.word_to_symbol
            .get(&lower)
            .and_then(|&id| self.symbols.iter().find(|s| s.id == id))
    }

    /// Find words similar to a perceptual pattern
    pub fn find_by_perception(&self, pattern: &[f64], top_k: usize) -> Vec<&GroundedSymbol> {
        let mut scored: Vec<(usize, f64)> = self
            .symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let sim = cosine_similarity(&s.perceptual_signature, pattern);
                (i, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(top_k)
            .filter(|(_, sim)| *sim >= self.config.grounding_threshold)
            .map(|(i, _)| &self.symbols[*i])
            .collect()
    }

    /// Get all well-grounded symbols
    pub fn well_grounded(&self) -> Vec<&GroundedSymbol> {
        self.symbols
            .iter()
            .filter(|s| s.grounding_strength >= self.config.grounding_threshold)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// A reasoning step in chain-of-thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step: usize,
    /// The reasoning content
    pub content: String,
    /// Type of reasoning
    pub reasoning_type: ReasoningType,
    /// Confidence in this step
    pub confidence: f64,
    /// Supporting evidence (symbol IDs)
    pub evidence: Vec<usize>,
}

/// Types of reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningType {
    Observation,    // What we observe
    Inference,      // What we infer
    Hypothesis,     // What we hypothesize
    Deduction,      // Logical deduction
    Analogy,        // Reasoning by analogy
    Counterfactual, // What-if reasoning
    Conclusion,     // Final conclusion
}

/// Reasoning chain (chain-of-thought)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    /// Query that started the reasoning
    pub query: String,
    /// Steps in the chain
    pub steps: Vec<ReasoningStep>,
    /// Final answer/conclusion
    pub conclusion: String,
    /// Overall confidence
    pub confidence: f64,
    /// Whether reasoning was successful
    pub success: bool,
}

/// Reasoning engine using LLM
#[derive(Debug, Clone)]
pub struct ReasoningEngine {
    /// Reasoning history
    history: VecDeque<ReasoningChain>,
    /// Configuration
    config: LLMIntegrationConfig,
    /// Known facts for reasoning
    knowledge_base: Vec<String>,
}

impl ReasoningEngine {
    pub fn new(config: LLMIntegrationConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(100),
            config,
            knowledge_base: Vec::new(),
        }
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&mut self, fact: String) {
        if !self.knowledge_base.contains(&fact) {
            self.knowledge_base.push(fact);
        }
    }

    /// Perform chain-of-thought reasoning
    pub fn reason(&mut self, query: &str, context: &[String]) -> ReasoningChain {
        let mut steps = Vec::new();
        let mut confidence = 1.0;

        // Step 1: Observation - what do we know?
        let observations: Vec<String> = context.iter().take(3).cloned().collect();

        if !observations.is_empty() {
            steps.push(ReasoningStep {
                step: 1,
                content: format!("Given: {}", observations.join("; ")),
                reasoning_type: ReasoningType::Observation,
                confidence: 0.9,
                evidence: vec![],
            });
        }

        // Step 2: Inference - what can we infer?
        let relevant_facts: Vec<&String> = self
            .knowledge_base
            .iter()
            .filter(|f| {
                query
                    .split_whitespace()
                    .any(|w| f.to_lowercase().contains(&w.to_lowercase()))
            })
            .take(3)
            .collect();

        if !relevant_facts.is_empty() {
            steps.push(ReasoningStep {
                step: 2,
                content: format!(
                    "Relevant knowledge: {}",
                    relevant_facts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                reasoning_type: ReasoningType::Inference,
                confidence: 0.8,
                evidence: vec![],
            });
            confidence *= 0.9;
        } else {
            confidence *= 0.7;
        }

        // Step 3: Hypothesis
        steps.push(ReasoningStep {
            step: 3,
            content: format!("Analyzing: {}", query),
            reasoning_type: ReasoningType::Hypothesis,
            confidence: 0.7,
            evidence: vec![],
        });

        // Step 4: Conclusion
        let conclusion = if relevant_facts.is_empty() && observations.is_empty() {
            format!("Insufficient information to answer: {}", query)
        } else {
            format!("Based on available information regarding: {}", query)
        };

        steps.push(ReasoningStep {
            step: 4,
            content: conclusion.clone(),
            reasoning_type: ReasoningType::Conclusion,
            confidence,
            evidence: vec![],
        });

        let chain = ReasoningChain {
            query: query.to_string(),
            steps,
            conclusion,
            confidence,
            success: confidence >= 0.5,
        };

        // Store in history
        self.history.push_back(chain.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        chain
    }

    /// Get recent reasoning chains
    pub fn recent_chains(&self, n: usize) -> Vec<&ReasoningChain> {
        self.history.iter().rev().take(n).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED LLM INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of language processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageResult {
    /// Parsed semantics
    pub semantics: ParsedSemantics,
    /// Grounded symbols used
    pub grounded_symbols: Vec<String>,
    /// Reasoning chain (if applicable)
    pub reasoning: Option<ReasoningChain>,
    /// Generated response (if applicable)
    pub response: Option<String>,
}

/// Unified LLM Integration system
#[derive(Debug, Clone)]
pub struct LLMIntegration {
    /// Language processor
    pub processor: LanguageProcessor,
    /// Semantic grounding
    pub grounding: SemanticGrounding,
    /// Reasoning engine
    pub reasoning: ReasoningEngine,
    /// Configuration
    config: LLMIntegrationConfig,
    /// Conversation history
    conversation_history: VecDeque<(String, String)>,
    /// Current step
    current_step: usize,
}

impl LLMIntegration {
    pub fn new(config: LLMIntegrationConfig) -> Self {
        Self {
            processor: LanguageProcessor::new(config.clone()),
            grounding: SemanticGrounding::new(config.clone()),
            reasoning: ReasoningEngine::new(config.clone()),
            config,
            conversation_history: VecDeque::with_capacity(50),
            current_step: 0,
        }
    }

    /// Process natural language input
    pub fn process_input(&mut self, text: &str) -> LanguageResult {
        // Parse the input
        let semantics = self.processor.parse(text);

        // Find grounded symbols
        let mut grounded_symbols = Vec::new();
        for entity in &semantics.entities {
            if let Some(symbol) = self.grounding.lookup(&entity.text) {
                grounded_symbols.push(symbol.word.clone());
            }
        }

        // Perform reasoning if it's a query
        let reasoning = if semantics.intent == Intent::Query && self.config.use_chain_of_thought {
            let context: Vec<String> = self
                .conversation_history
                .iter()
                .rev()
                .take(5)
                .map(|(q, _)| q.clone())
                .collect();
            Some(self.reasoning.reason(text, &context))
        } else {
            None
        };

        // Generate response based on intent
        let response = self.generate_response(&semantics, &reasoning);

        // Store in conversation history
        if let Some(ref resp) = response {
            self.conversation_history
                .push_back((text.to_string(), resp.clone()));
            if self.conversation_history.len() > 50 {
                self.conversation_history.pop_front();
            }
        }

        self.current_step += 1;

        LanguageResult {
            semantics,
            grounded_symbols,
            reasoning,
            response,
        }
    }

    /// Generate a response based on parsed semantics
    fn generate_response(
        &self,
        semantics: &ParsedSemantics,
        reasoning: &Option<ReasoningChain>,
    ) -> Option<String> {
        match semantics.intent {
            Intent::Query => {
                if let Some(chain) = reasoning {
                    Some(chain.conclusion.clone())
                } else {
                    Some("I don't have enough information to answer that.".to_string())
                }
            }
            Intent::Command => {
                if semantics.entities.is_empty() {
                    Some(
                        "I understand you want me to do something, but I need more details."
                            .to_string(),
                    )
                } else {
                    let actions: Vec<&str> = semantics
                        .entities
                        .iter()
                        .filter(|e| e.entity_type == EntityType::Action)
                        .map(|e| e.text.as_str())
                        .collect();
                    if actions.is_empty() {
                        Some("I'll try to help with that.".to_string())
                    } else {
                        Some(format!("I'll attempt to: {}", actions.join(", ")))
                    }
                }
            }
            Intent::Greeting => Some("Hello! How can I help you?".to_string()),
            Intent::Confirmation => Some("Understood, proceeding.".to_string()),
            Intent::Negation => Some("Alright, I'll stop or reconsider.".to_string()),
            Intent::Statement => Some("I've noted that information.".to_string()),
            _ => None,
        }
    }

    /// Ground language to experience
    pub fn ground_experience(&mut self, text: &str, perceptual: Vec<f64>, action: Vec<f64>) {
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            self.grounding
                .ground(word, perceptual.clone(), action.clone(), self.current_step);
        }
    }

    /// Add knowledge to reasoning engine
    pub fn add_knowledge(&mut self, fact: String) {
        self.reasoning.add_fact(fact);
    }

    /// Get summary of integration state
    pub fn summary(&self) -> LLMIntegrationSummary {
        LLMIntegrationSummary {
            vocab_size: self.processor.vocab.len(),
            grounded_symbols: self.grounding.len(),
            well_grounded: self.grounding.well_grounded().len(),
            knowledge_facts: self.reasoning.knowledge_base.len(),
            conversation_turns: self.conversation_history.len(),
            total_steps: self.current_step,
        }
    }
}

/// Summary of LLM integration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMIntegrationSummary {
    pub vocab_size: usize,
    pub grounded_symbols: usize,
    pub well_grounded: usize,
    pub knowledge_facts: usize,
    pub conversation_turns: usize,
    pub total_steps: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-10 {
        dot / denom
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_integration_creation() {
        let config = LLMIntegrationConfig::default();
        let integration = LLMIntegration::new(config);

        assert!(integration.processor.vocab.len() > 0);
        assert!(integration.grounding.is_empty());
    }

    #[test]
    fn test_language_processor_tokenize() {
        let config = LLMIntegrationConfig::default();
        let mut processor = LanguageProcessor::new(config);

        let tokens = processor.tokenize("hello world");
        assert_eq!(tokens.len(), 2);

        let text = processor.detokenize(&tokens);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_language_processor_parse() {
        let config = LLMIntegrationConfig::default();
        let mut processor = LanguageProcessor::new(config);

        let semantics = processor.parse("go north to find the goal");

        assert_eq!(semantics.intent, Intent::Command);
        assert!(!semantics.entities.is_empty());
    }

    #[test]
    fn test_intent_detection() {
        let config = LLMIntegrationConfig::default();
        let mut processor = LanguageProcessor::new(config);

        assert_eq!(processor.parse("what is this?").intent, Intent::Query);
        assert_eq!(processor.parse("go north").intent, Intent::Command);
        assert_eq!(processor.parse("hello").intent, Intent::Greeting);
        assert_eq!(processor.parse("yes").intent, Intent::Confirmation);
        assert_eq!(processor.parse("no").intent, Intent::Negation);
    }

    #[test]
    fn test_semantic_grounding() {
        let config = LLMIntegrationConfig::default();
        let mut grounding = SemanticGrounding::new(config);

        let perceptual = vec![0.5, 0.3, 0.2];
        let action = vec![0.1, 0.8, 0.1];

        let id = grounding.ground("north", perceptual.clone(), action.clone(), 0);
        assert_eq!(id, 0);

        let symbol = grounding.lookup("north").unwrap();
        assert_eq!(symbol.word, "north");
        assert_eq!(symbol.grounding_count, 1);

        // Ground again to strengthen
        grounding.ground("north", perceptual, action, 1);
        let symbol = grounding.lookup("north").unwrap();
        assert_eq!(symbol.grounding_count, 2);
    }

    #[test]
    fn test_reasoning_engine() {
        let config = LLMIntegrationConfig::default();
        let mut engine = ReasoningEngine::new(config);

        engine.add_fact("The goal is to the north".to_string());
        engine.add_fact("Moving north increases y coordinate".to_string());

        let chain = engine.reason(
            "Where is the goal?",
            &["Current position: (0, 0)".to_string()],
        );

        assert!(!chain.steps.is_empty());
        assert!(chain.confidence > 0.0);
    }

    #[test]
    fn test_full_integration() {
        let config = LLMIntegrationConfig::default();
        let mut integration = LLMIntegration::new(config);

        // Add some knowledge
        integration.add_knowledge("Goals are typically found in corners.".to_string());

        // Process a query
        let result = integration.process_input("Where should I go?");

        assert_eq!(result.semantics.intent, Intent::Query);
        assert!(result.reasoning.is_some());
        assert!(result.response.is_some());
    }

    #[test]
    fn test_ground_experience() {
        let config = LLMIntegrationConfig::default();
        let mut integration = LLMIntegration::new(config);

        let perceptual = vec![0.1, 0.2, 0.3];
        let action = vec![0.4, 0.5, 0.6];

        integration.ground_experience("go north quickly", perceptual, action);

        assert!(integration.grounding.lookup("go").is_some());
        assert!(integration.grounding.lookup("north").is_some());
        assert!(integration.grounding.lookup("quickly").is_some());
    }

    #[test]
    fn test_conversation_history() {
        let config = LLMIntegrationConfig::default();
        let mut integration = LLMIntegration::new(config);

        integration.process_input("hello");
        integration.process_input("how are you?");

        let summary = integration.summary();
        assert_eq!(summary.conversation_turns, 2);
    }

    #[test]
    fn test_sentiment_analysis() {
        let config = LLMIntegrationConfig::default();
        let mut processor = LanguageProcessor::new(config);

        let positive = processor.parse("This is great and excellent!");
        assert!(positive.sentiment.positive > positive.sentiment.negative);

        let negative = processor.parse("This is bad and terrible.");
        assert!(negative.sentiment.negative > negative.sentiment.positive);
    }
}
