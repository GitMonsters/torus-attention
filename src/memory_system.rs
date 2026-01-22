//! # Long-term Memory System for AGI
//!
//! This module implements episodic, semantic, and procedural memory systems
//! with retrieval-augmented recall for persistent learning across sessions.
//!
//! ## Memory Types
//!
//! 1. **Episodic Memory**: Autobiographical experiences with temporal context
//! 2. **Semantic Memory**: Factual knowledge and concept relationships
//! 3. **Procedural Memory**: Learned skills and action sequences
//! 4. **Working Memory**: Short-term active information buffer
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Memory System                                │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
//! │  │  Episodic   │  │  Semantic   │  │ Procedural  │  │  Working   │ │
//! │  │   Memory    │  │   Memory    │  │   Memory    │  │   Memory   │ │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
//! │         │                │                │               │        │
//! │         └────────────────┼────────────────┴───────────────┘        │
//! │                          ▼                                         │
//! │                 ┌─────────────────┐                                │
//! │                 │ Memory Index    │ (Vector similarity search)     │
//! │                 │ + Retrieval     │                                │
//! │                 └────────┬────────┘                                │
//! │                          │                                         │
//! │                          ▼                                         │
//! │                 ┌─────────────────┐                                │
//! │                 │  Consolidation  │ (Sleep-like replay)            │
//! │                 │    Engine       │                                │
//! │                 └─────────────────┘                                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystemConfig {
    /// Maximum episodic memories to store
    pub max_episodic_memories: usize,
    /// Maximum semantic facts to store
    pub max_semantic_facts: usize,
    /// Maximum procedural skills to store
    pub max_procedural_skills: usize,
    /// Working memory capacity (Miller's 7±2)
    pub working_memory_capacity: usize,
    /// Embedding dimension for memory vectors
    pub embedding_dim: usize,
    /// Similarity threshold for retrieval
    pub retrieval_threshold: f64,
    /// Number of memories to retrieve per query
    pub top_k_retrieval: usize,
    /// Memory decay rate per consolidation cycle
    pub decay_rate: f64,
    /// Consolidation interval (steps)
    pub consolidation_interval: usize,
    /// Minimum importance to prevent decay
    pub importance_floor: f64,
    /// Enable temporal context in retrieval
    pub use_temporal_context: bool,
    /// Enable emotional salience weighting
    pub use_emotional_salience: bool,
}

impl Default for MemorySystemConfig {
    fn default() -> Self {
        Self {
            max_episodic_memories: 10000,
            max_semantic_facts: 50000,
            max_procedural_skills: 1000,
            working_memory_capacity: 7,
            embedding_dim: 256,
            retrieval_threshold: 0.5,
            top_k_retrieval: 10,
            decay_rate: 0.01,
            consolidation_interval: 100,
            importance_floor: 0.1,
            use_temporal_context: true,
            use_emotional_salience: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY - Autobiographical experiences
// ═══════════════════════════════════════════════════════════════════════════════

/// Emotional valence of a memory
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmotionalValence {
    Positive(f64), // 0.0 to 1.0 intensity
    Negative(f64), // 0.0 to 1.0 intensity
    Neutral,
}

impl EmotionalValence {
    pub fn salience(&self) -> f64 {
        match self {
            EmotionalValence::Positive(v) | EmotionalValence::Negative(v) => *v,
            EmotionalValence::Neutral => 0.1,
        }
    }
}

/// A single episodic memory (autobiographical experience)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Unique identifier
    pub id: u64,
    /// When this occurred (step number)
    pub timestamp: usize,
    /// Session ID for cross-session tracking
    pub session_id: u64,
    /// Embedding vector for similarity search
    pub embedding: Vec<f64>,
    /// State features at this moment
    pub state: Vec<f64>,
    /// Action taken (if any)
    pub action: Option<usize>,
    /// Outcome/reward
    pub outcome: f64,
    /// What happened next (next state summary)
    pub consequence: Vec<f64>,
    /// Emotional valence
    pub emotion: EmotionalValence,
    /// Related concepts/symbols
    pub related_concepts: Vec<String>,
    /// Causal variables involved
    pub causal_context: Vec<usize>,
    /// Importance score (determines retention)
    pub importance: f64,
    /// Number of times retrieved
    pub retrieval_count: usize,
    /// Last retrieval timestamp
    pub last_retrieved: usize,
}

/// Episodic Memory Store with temporal indexing
#[derive(Debug, Clone)]
pub struct EpisodicMemoryStore {
    /// All episodic memories
    memories: Vec<EpisodicMemory>,
    /// Index by timestamp for temporal queries
    temporal_index: BTreeMap<usize, Vec<usize>>,
    /// Index by session for cross-session retrieval
    session_index: HashMap<u64, Vec<usize>>,
    /// Next memory ID
    next_id: u64,
    /// Current session
    current_session: u64,
    /// Configuration
    config: MemorySystemConfig,
}

impl EpisodicMemoryStore {
    pub fn new(config: MemorySystemConfig) -> Self {
        Self {
            memories: Vec::new(),
            temporal_index: BTreeMap::new(),
            session_index: HashMap::new(),
            next_id: 0,
            current_session: 0,
            config,
        }
    }

    /// Start a new session
    pub fn new_session(&mut self) {
        self.current_session += 1;
    }

    /// Store an episodic memory
    pub fn store(
        &mut self,
        embedding: Vec<f64>,
        state: Vec<f64>,
        action: Option<usize>,
        outcome: f64,
        consequence: Vec<f64>,
        emotion: EmotionalValence,
        related_concepts: Vec<String>,
        causal_context: Vec<usize>,
        timestamp: usize,
    ) -> u64 {
        // Calculate importance based on emotion, outcome magnitude, and novelty
        let importance = self.calculate_importance(&embedding, outcome, &emotion);

        let memory = EpisodicMemory {
            id: self.next_id,
            timestamp,
            session_id: self.current_session,
            embedding,
            state,
            action,
            outcome,
            consequence,
            emotion,
            related_concepts,
            causal_context,
            importance,
            retrieval_count: 0,
            last_retrieved: timestamp,
        };

        let id = memory.id;
        let idx = self.memories.len();

        // Update indices
        self.temporal_index
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(idx);

        self.session_index
            .entry(self.current_session)
            .or_insert_with(Vec::new)
            .push(idx);

        self.memories.push(memory);
        self.next_id += 1;

        // Prune if over capacity
        self.prune_if_needed();

        id
    }

    /// Calculate memory importance
    fn calculate_importance(
        &self,
        embedding: &[f64],
        outcome: f64,
        emotion: &EmotionalValence,
    ) -> f64 {
        let outcome_salience = outcome.abs().min(1.0);
        let emotional_salience = emotion.salience();

        // Novelty: how different from existing memories
        let novelty = if self.memories.is_empty() {
            1.0
        } else {
            let avg_similarity: f64 = self
                .memories
                .iter()
                .map(|m| cosine_similarity(&m.embedding, embedding))
                .sum::<f64>()
                / self.memories.len() as f64;
            1.0 - avg_similarity
        };

        // Combined importance
        0.3 * outcome_salience + 0.3 * emotional_salience + 0.4 * novelty
    }

    /// Retrieve similar memories
    pub fn retrieve(
        &mut self,
        query: &[f64],
        top_k: Option<usize>,
        current_time: usize,
    ) -> Vec<&EpisodicMemory> {
        let k = top_k.unwrap_or(self.config.top_k_retrieval);

        // Score all memories
        let mut scored: Vec<(usize, f64)> = self
            .memories
            .iter()
            .enumerate()
            .filter_map(|(idx, m)| {
                let sim = cosine_similarity(&m.embedding, query);
                if sim >= self.config.retrieval_threshold {
                    // Boost by importance and recency
                    let recency = 1.0 / (1.0 + (current_time - m.timestamp) as f64 / 1000.0);
                    let score = sim * 0.6 + m.importance * 0.3 + recency * 0.1;
                    Some((idx, score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update retrieval counts for top results
        let top_indices: Vec<usize> = scored.iter().take(k).map(|(idx, _)| *idx).collect();
        for &idx in &top_indices {
            self.memories[idx].retrieval_count += 1;
            self.memories[idx].last_retrieved = current_time;
        }

        // Return references
        top_indices.iter().map(|&idx| &self.memories[idx]).collect()
    }

    /// Retrieve memories from a specific time range
    pub fn retrieve_temporal(&self, start: usize, end: usize) -> Vec<&EpisodicMemory> {
        self.temporal_index
            .range(start..=end)
            .flat_map(|(_, indices)| indices.iter().map(|&idx| &self.memories[idx]))
            .collect()
    }

    /// Retrieve memories from a specific session
    pub fn retrieve_session(&self, session_id: u64) -> Vec<&EpisodicMemory> {
        self.session_index
            .get(&session_id)
            .map(|indices| indices.iter().map(|&idx| &self.memories[idx]).collect())
            .unwrap_or_default()
    }

    /// Decay memories and prune unimportant ones
    pub fn consolidate(&mut self, current_time: usize) {
        for memory in &mut self.memories {
            // Decay based on time since last retrieval
            let time_since_retrieval = current_time.saturating_sub(memory.last_retrieved) as f64;
            let decay_factor = (-self.config.decay_rate * time_since_retrieval / 100.0).exp();

            // Importance can only decay, not increase (unless retrieved)
            memory.importance *= decay_factor;

            // Apply floor
            memory.importance = memory.importance.max(self.config.importance_floor);

            // But boost if frequently retrieved
            let retrieval_boost = (memory.retrieval_count as f64 * 0.01).min(0.3);
            memory.importance = (memory.importance + retrieval_boost).min(1.0);
        }
    }

    /// Remove low-importance memories if over capacity
    fn prune_if_needed(&mut self) {
        if self.memories.len() <= self.config.max_episodic_memories {
            return;
        }

        // Sort by importance (ascending) and remove lowest
        let mut indexed: Vec<(usize, f64)> = self
            .memories
            .iter()
            .enumerate()
            .map(|(i, m)| (i, m.importance))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove bottom 10%
        let to_remove = self.memories.len() / 10;
        let remove_indices: HashSet<usize> =
            indexed.iter().take(to_remove).map(|(i, _)| *i).collect();

        // Rebuild without removed indices
        let remaining: Vec<EpisodicMemory> = self
            .memories
            .drain(..)
            .enumerate()
            .filter(|(i, _)| !remove_indices.contains(i))
            .map(|(_, m)| m)
            .collect();

        self.memories = remaining;
        self.rebuild_indices();
    }

    /// Rebuild temporal and session indices after pruning
    fn rebuild_indices(&mut self) {
        self.temporal_index.clear();
        self.session_index.clear();

        for (idx, memory) in self.memories.iter().enumerate() {
            self.temporal_index
                .entry(memory.timestamp)
                .or_insert_with(Vec::new)
                .push(idx);
            self.session_index
                .entry(memory.session_id)
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    pub fn current_session(&self) -> u64 {
        self.current_session
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC MEMORY - Factual knowledge and concepts
// ═══════════════════════════════════════════════════════════════════════════════

/// A semantic fact or concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique identifier
    pub id: u64,
    /// The concept or fact name
    pub name: String,
    /// Embedding vector
    pub embedding: Vec<f64>,
    /// Definition/description
    pub description: String,
    /// Related concepts (edges in knowledge graph)
    pub relations: Vec<SemanticRelation>,
    /// Source episodic memories this was derived from
    pub source_episodes: Vec<u64>,
    /// Confidence score (how well established)
    pub confidence: f64,
    /// When this was learned
    pub learned_at: usize,
    /// When this was last accessed
    pub last_accessed: usize,
}

/// Relation between semantic concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelation {
    /// Target concept ID
    pub target_id: u64,
    /// Relation type
    pub relation_type: RelationType,
    /// Strength of relation
    pub strength: f64,
}

/// Types of semantic relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    IsA,            // Hyponymy (dog is-a animal)
    HasA,           // Meronymy (car has-a wheel)
    PartOf,         // Holonymy (wheel part-of car)
    Causes,         // Causal relation
    CausedBy,       // Inverse causal
    SimilarTo,      // Similarity
    OppositeTo,     // Antonymy
    UsedFor,        // Functional relation
    LocatedIn,      // Spatial relation
    TemporalBefore, // Temporal relation
    TemporalAfter,
    AssociatedWith, // General association
}

/// Semantic Memory Store (Knowledge Graph)
#[derive(Debug, Clone)]
pub struct SemanticMemoryStore {
    /// All semantic facts
    facts: Vec<SemanticFact>,
    /// Name to ID lookup
    name_index: HashMap<String, u64>,
    /// Next fact ID
    next_id: u64,
    /// Configuration
    config: MemorySystemConfig,
}

impl SemanticMemoryStore {
    pub fn new(config: MemorySystemConfig) -> Self {
        Self {
            facts: Vec::new(),
            name_index: HashMap::new(),
            next_id: 0,
            config,
        }
    }

    /// Store or update a semantic fact
    pub fn store(
        &mut self,
        name: String,
        embedding: Vec<f64>,
        description: String,
        source_episode: Option<u64>,
        timestamp: usize,
    ) -> u64 {
        // Check if fact already exists
        if let Some(&existing_id) = self.name_index.get(&name) {
            // Update existing fact
            if let Some(fact) = self.facts.iter_mut().find(|f| f.id == existing_id) {
                // Blend embeddings
                for (i, e) in embedding.iter().enumerate() {
                    if i < fact.embedding.len() {
                        fact.embedding[i] = 0.9 * fact.embedding[i] + 0.1 * e;
                    }
                }
                // Increase confidence with repetition
                fact.confidence = (fact.confidence + 0.1).min(1.0);
                fact.last_accessed = timestamp;
                if let Some(ep_id) = source_episode {
                    if !fact.source_episodes.contains(&ep_id) {
                        fact.source_episodes.push(ep_id);
                    }
                }
            }
            return existing_id;
        }

        // Create new fact
        let fact = SemanticFact {
            id: self.next_id,
            name: name.clone(),
            embedding,
            description,
            relations: Vec::new(),
            source_episodes: source_episode.into_iter().collect(),
            confidence: 0.5,
            learned_at: timestamp,
            last_accessed: timestamp,
        };

        let id = fact.id;
        self.facts.push(fact);
        self.name_index.insert(name, id);
        self.next_id += 1;

        id
    }

    /// Add a relation between two concepts
    pub fn add_relation(
        &mut self,
        source_id: u64,
        target_id: u64,
        relation_type: RelationType,
        strength: f64,
    ) {
        if let Some(fact) = self.facts.iter_mut().find(|f| f.id == source_id) {
            // Check if relation already exists
            if let Some(existing) = fact
                .relations
                .iter_mut()
                .find(|r| r.target_id == target_id && r.relation_type == relation_type)
            {
                // Strengthen existing relation
                existing.strength = (existing.strength + strength * 0.2).min(1.0);
            } else {
                // Add new relation
                fact.relations.push(SemanticRelation {
                    target_id,
                    relation_type,
                    strength,
                });
            }
        }
    }

    /// Retrieve by name
    pub fn get_by_name(&self, name: &str) -> Option<&SemanticFact> {
        self.name_index
            .get(name)
            .and_then(|&id| self.facts.iter().find(|f| f.id == id))
    }

    /// Retrieve similar concepts
    pub fn retrieve_similar(
        &mut self,
        query: &[f64],
        top_k: usize,
        current_time: usize,
    ) -> Vec<&SemanticFact> {
        let mut scored: Vec<(usize, f64)> = self
            .facts
            .iter()
            .enumerate()
            .map(|(idx, f)| {
                let sim = cosine_similarity(&f.embedding, query);
                // Weight by confidence
                let score = sim * 0.7 + f.confidence * 0.3;
                (idx, score)
            })
            .filter(|(_, score)| *score >= self.config.retrieval_threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update access times
        for &(idx, _) in scored.iter().take(top_k) {
            self.facts[idx].last_accessed = current_time;
        }

        scored
            .iter()
            .take(top_k)
            .map(|(idx, _)| &self.facts[*idx])
            .collect()
    }

    /// Get related concepts via graph traversal
    pub fn get_related(&self, concept_id: u64, max_depth: usize) -> Vec<&SemanticFact> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((concept_id, 0));
        visited.insert(concept_id);

        while let Some((id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(fact) = self.facts.iter().find(|f| f.id == id) {
                if depth > 0 {
                    result.push(fact);
                }

                for relation in &fact.relations {
                    if !visited.contains(&relation.target_id) {
                        visited.insert(relation.target_id);
                        queue.push_back((relation.target_id, depth + 1));
                    }
                }
            }
        }

        result
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROCEDURAL MEMORY - Skills and action sequences
// ═══════════════════════════════════════════════════════════════════════════════

/// A learned skill or procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralSkill {
    /// Unique identifier
    pub id: u64,
    /// Skill name
    pub name: String,
    /// Embedding for retrieval
    pub embedding: Vec<f64>,
    /// Preconditions for skill applicability
    pub preconditions: Vec<f64>,
    /// Action sequence (sequence of action indices)
    pub action_sequence: Vec<usize>,
    /// Expected outcome embedding
    pub expected_outcome: Vec<f64>,
    /// Success rate
    pub success_rate: f64,
    /// Times executed
    pub execution_count: usize,
    /// Times succeeded
    pub success_count: usize,
    /// Average reward when executed
    pub average_reward: f64,
    /// When learned
    pub learned_at: usize,
    /// Last executed
    pub last_executed: usize,
}

/// Procedural Memory Store
#[derive(Debug, Clone)]
pub struct ProceduralMemoryStore {
    /// All skills
    skills: Vec<ProceduralSkill>,
    /// Name to ID lookup
    name_index: HashMap<String, u64>,
    /// Next skill ID
    next_id: u64,
    /// Configuration
    config: MemorySystemConfig,
}

impl ProceduralMemoryStore {
    pub fn new(config: MemorySystemConfig) -> Self {
        Self {
            skills: Vec::new(),
            name_index: HashMap::new(),
            next_id: 0,
            config,
        }
    }

    /// Store a new skill
    pub fn store(
        &mut self,
        name: String,
        embedding: Vec<f64>,
        preconditions: Vec<f64>,
        action_sequence: Vec<usize>,
        expected_outcome: Vec<f64>,
        timestamp: usize,
    ) -> u64 {
        // Check for existing skill with same name
        if let Some(&existing_id) = self.name_index.get(&name) {
            return existing_id;
        }

        let skill = ProceduralSkill {
            id: self.next_id,
            name: name.clone(),
            embedding,
            preconditions,
            action_sequence,
            expected_outcome,
            success_rate: 0.5,
            execution_count: 0,
            success_count: 0,
            average_reward: 0.0,
            learned_at: timestamp,
            last_executed: timestamp,
        };

        let id = skill.id;
        self.skills.push(skill);
        self.name_index.insert(name, id);
        self.next_id += 1;

        id
    }

    /// Record skill execution result
    pub fn record_execution(
        &mut self,
        skill_id: u64,
        success: bool,
        reward: f64,
        timestamp: usize,
    ) {
        if let Some(skill) = self.skills.iter_mut().find(|s| s.id == skill_id) {
            skill.execution_count += 1;
            if success {
                skill.success_count += 1;
            }
            skill.success_rate = skill.success_count as f64 / skill.execution_count as f64;

            // Update running average reward
            let n = skill.execution_count as f64;
            skill.average_reward = ((n - 1.0) * skill.average_reward + reward) / n;

            skill.last_executed = timestamp;
        }
    }

    /// Find applicable skills for current state
    pub fn find_applicable(&self, current_state: &[f64], threshold: f64) -> Vec<&ProceduralSkill> {
        self.skills
            .iter()
            .filter(|s| cosine_similarity(&s.preconditions, current_state) >= threshold)
            .collect()
    }

    /// Retrieve similar skills by embedding
    pub fn retrieve_similar(&self, query: &[f64], top_k: usize) -> Vec<&ProceduralSkill> {
        let mut scored: Vec<(usize, f64)> = self
            .skills
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                let sim = cosine_similarity(&s.embedding, query);
                // Weight by success rate
                let score = sim * 0.6 + s.success_rate * 0.4;
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(top_k)
            .map(|(idx, _)| &self.skills[*idx])
            .collect()
    }

    /// Get skill by name
    pub fn get_by_name(&self, name: &str) -> Option<&ProceduralSkill> {
        self.name_index
            .get(name)
            .and_then(|&id| self.skills.iter().find(|s| s.id == id))
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKING MEMORY - Active short-term buffer
// ═══════════════════════════════════════════════════════════════════════════════

/// Item in working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryItem {
    /// Content embedding
    pub embedding: Vec<f64>,
    /// Item type
    pub item_type: WorkingMemoryItemType,
    /// Activation level (decays over time)
    pub activation: f64,
    /// When added
    pub added_at: usize,
    /// Source ID (episode, fact, or skill)
    pub source_id: Option<u64>,
    /// Reference to content (for display/debugging)
    pub label: String,
}

/// Types of working memory items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkingMemoryItemType {
    Episodic,
    Semantic,
    Procedural,
    Perception,
    Goal,
}

/// Working Memory - Active information buffer
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    /// Items currently in working memory
    items: VecDeque<WorkingMemoryItem>,
    /// Capacity limit
    capacity: usize,
    /// Activation decay rate
    decay_rate: f64,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            decay_rate: 0.1,
        }
    }

    /// Add item to working memory
    pub fn add(&mut self, item: WorkingMemoryItem) {
        // If at capacity, remove lowest activation item
        if self.items.len() >= self.capacity {
            let min_idx = self
                .items
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.activation
                        .partial_cmp(&b.1.activation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(idx) = min_idx {
                self.items.remove(idx);
            }
        }

        self.items.push_back(item);
    }

    /// Update activations (decay over time)
    pub fn update(&mut self, current_time: usize) {
        for item in &mut self.items {
            let time_since_added = (current_time - item.added_at) as f64;
            item.activation *= (-self.decay_rate * time_since_added).exp();
        }

        // Remove items with very low activation
        self.items.retain(|item| item.activation > 0.01);
    }

    /// Boost activation of similar items (rehearsal)
    pub fn rehearse(&mut self, query: &[f64]) {
        for item in &mut self.items {
            let similarity = cosine_similarity(&item.embedding, query);
            if similarity > 0.5 {
                item.activation = (item.activation + 0.3 * similarity).min(1.0);
            }
        }
    }

    /// Get current working memory contents
    pub fn contents(&self) -> &VecDeque<WorkingMemoryItem> {
        &self.items
    }

    /// Get combined embedding (attention-weighted average)
    pub fn get_context(&self) -> Vec<f64> {
        if self.items.is_empty() {
            return vec![0.0; 256]; // Default dimension
        }

        let dim = self.items.front().map(|i| i.embedding.len()).unwrap_or(256);
        let mut context = vec![0.0; dim];
        let mut total_activation = 0.0;

        for item in &self.items {
            for (i, &e) in item.embedding.iter().enumerate() {
                if i < context.len() {
                    context[i] += e * item.activation;
                }
            }
            total_activation += item.activation;
        }

        if total_activation > 0.0 {
            for c in &mut context {
                *c /= total_activation;
            }
        }

        context
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED MEMORY SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified Memory System combining all memory types
#[derive(Debug, Clone)]
pub struct MemorySystem {
    /// Episodic memory store
    pub episodic: EpisodicMemoryStore,
    /// Semantic memory store
    pub semantic: SemanticMemoryStore,
    /// Procedural memory store
    pub procedural: ProceduralMemoryStore,
    /// Working memory
    pub working: WorkingMemory,
    /// Configuration
    config: MemorySystemConfig,
    /// Current step
    current_step: usize,
    /// Steps since last consolidation
    steps_since_consolidation: usize,
}

impl MemorySystem {
    pub fn new(config: MemorySystemConfig) -> Self {
        let working_capacity = config.working_memory_capacity;
        Self {
            episodic: EpisodicMemoryStore::new(config.clone()),
            semantic: SemanticMemoryStore::new(config.clone()),
            procedural: ProceduralMemoryStore::new(config.clone()),
            working: WorkingMemory::new(working_capacity),
            config,
            current_step: 0,
            steps_since_consolidation: 0,
        }
    }

    /// Process a new experience
    pub fn process_experience(
        &mut self,
        state: Vec<f64>,
        action: Option<usize>,
        outcome: f64,
        next_state: Vec<f64>,
        concepts: Vec<String>,
        causal_context: Vec<usize>,
    ) {
        // Create embedding from state (simplified - in production would use encoder)
        let embedding = self.create_embedding(&state, &next_state, outcome);

        // Determine emotional valence from outcome
        let emotion = if outcome > 0.5 {
            EmotionalValence::Positive(outcome.min(1.0))
        } else if outcome < -0.5 {
            EmotionalValence::Negative((-outcome).min(1.0))
        } else {
            EmotionalValence::Neutral
        };

        // Store episodic memory
        let episode_id = self.episodic.store(
            embedding.clone(),
            state.clone(),
            action,
            outcome,
            next_state.clone(),
            emotion,
            concepts.clone(),
            causal_context,
            self.current_step,
        );

        // Extract and store semantic knowledge
        for concept in concepts {
            self.semantic.store(
                concept,
                embedding.clone(),
                format!("Concept observed at step {}", self.current_step),
                Some(episode_id),
                self.current_step,
            );
        }

        // Update working memory
        self.working.update(self.current_step);
        self.working.add(WorkingMemoryItem {
            embedding: embedding.clone(),
            item_type: WorkingMemoryItemType::Episodic,
            activation: 1.0,
            added_at: self.current_step,
            source_id: Some(episode_id),
            label: format!("Episode {}", episode_id),
        });

        self.current_step += 1;
        self.steps_since_consolidation += 1;

        // Consolidate periodically
        if self.steps_since_consolidation >= self.config.consolidation_interval {
            self.consolidate();
        }
    }

    /// Create embedding from experience (simplified version)
    fn create_embedding(&self, state: &[f64], next_state: &[f64], outcome: f64) -> Vec<f64> {
        let target_dim = self.config.embedding_dim;
        let mut embedding = Vec::with_capacity(target_dim);

        // Combine state and next_state features
        let state_len = state.len().min(target_dim / 3);
        let next_len = next_state.len().min(target_dim / 3);

        for i in 0..target_dim {
            let value = if i < state_len {
                state[i]
            } else if i < state_len + next_len {
                next_state[i - state_len]
            } else if i == state_len + next_len {
                outcome
            } else {
                // Pad with combinations
                let idx1 = i % state_len.max(1);
                let idx2 = i % next_len.max(1);
                (state.get(idx1).copied().unwrap_or(0.0)
                    + next_state.get(idx2).copied().unwrap_or(0.0))
                    / 2.0
            };
            embedding.push(value);
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

    /// Retrieve relevant memories for current context
    pub fn retrieve(&mut self, query: &[f64]) -> MemoryRetrievalResult {
        // Retrieve from all stores
        let episodes =
            self.episodic
                .retrieve(query, Some(self.config.top_k_retrieval), self.current_step);
        let facts =
            self.semantic
                .retrieve_similar(query, self.config.top_k_retrieval, self.current_step);
        let skills = self
            .procedural
            .retrieve_similar(query, self.config.top_k_retrieval);

        // Boost working memory with retrieved items
        self.working.rehearse(query);

        // Add retrieved items to working memory
        for episode in &episodes {
            self.working.add(WorkingMemoryItem {
                embedding: episode.embedding.clone(),
                item_type: WorkingMemoryItemType::Episodic,
                activation: 0.7,
                added_at: self.current_step,
                source_id: Some(episode.id),
                label: format!("Retrieved Episode {}", episode.id),
            });
        }

        for fact in &facts {
            self.working.add(WorkingMemoryItem {
                embedding: fact.embedding.clone(),
                item_type: WorkingMemoryItemType::Semantic,
                activation: 0.6,
                added_at: self.current_step,
                source_id: Some(fact.id),
                label: fact.name.clone(),
            });
        }

        // Get working memory context
        let working_context = self.working.get_context();

        MemoryRetrievalResult {
            episode_count: episodes.len(),
            fact_count: facts.len(),
            skill_count: skills.len(),
            working_context,
            top_episode_ids: episodes.iter().map(|e| e.id).collect(),
            top_fact_names: facts.iter().map(|f| f.name.clone()).collect(),
            top_skill_names: skills.iter().map(|s| s.name.clone()).collect(),
        }
    }

    /// Memory consolidation (like sleep)
    pub fn consolidate(&mut self) {
        // Consolidate episodic memories
        self.episodic.consolidate(self.current_step);

        // Extract patterns from episodic to semantic
        self.extract_semantic_patterns();

        // Extract skills from repeated action sequences
        self.extract_procedural_patterns();

        self.steps_since_consolidation = 0;
    }

    /// Extract semantic patterns from episodic memories
    fn extract_semantic_patterns(&mut self) {
        // Find frequently co-occurring concepts
        let mut concept_cooccurrence: HashMap<(String, String), usize> = HashMap::new();

        for memory in &self.episodic.memories {
            let concepts = &memory.related_concepts;
            for i in 0..concepts.len() {
                for j in (i + 1)..concepts.len() {
                    let pair = if concepts[i] < concepts[j] {
                        (concepts[i].clone(), concepts[j].clone())
                    } else {
                        (concepts[j].clone(), concepts[i].clone())
                    };
                    *concept_cooccurrence.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Add relations for frequent co-occurrences
        for ((c1, c2), count) in concept_cooccurrence {
            if count >= 3 {
                if let (Some(&id1), Some(&id2)) = (
                    self.semantic.name_index.get(&c1),
                    self.semantic.name_index.get(&c2),
                ) {
                    let strength = (count as f64 / 10.0).min(1.0);
                    self.semantic
                        .add_relation(id1, id2, RelationType::AssociatedWith, strength);
                    self.semantic
                        .add_relation(id2, id1, RelationType::AssociatedWith, strength);
                }
            }
        }
    }

    /// Extract procedural patterns from repeated action sequences
    fn extract_procedural_patterns(&mut self) {
        // Find repeated action sequences that led to positive outcomes
        let positive_episodes: Vec<&EpisodicMemory> = self
            .episodic
            .memories
            .iter()
            .filter(|m| m.outcome > 0.3)
            .collect();

        // Group by similar states and actions
        // (Simplified - in production would use more sophisticated sequence mining)
        if positive_episodes.len() < 3 {
            return;
        }

        // Create a skill from the most recent successful sequence
        if let Some(episode) = positive_episodes.last() {
            if let Some(action) = episode.action {
                let skill_name = format!("skill_from_episode_{}", episode.id);
                if self.procedural.get_by_name(&skill_name).is_none() {
                    self.procedural.store(
                        skill_name,
                        episode.embedding.clone(),
                        episode.state.clone(),
                        vec![action],
                        episode.consequence.clone(),
                        self.current_step,
                    );
                }
            }
        }
    }

    /// Start a new session (for cross-session persistence)
    pub fn new_session(&mut self) {
        self.episodic.new_session();
        self.working.clear();
    }

    /// Get summary statistics
    pub fn summary(&self) -> MemorySystemSummary {
        MemorySystemSummary {
            episodic_count: self.episodic.len(),
            semantic_count: self.semantic.len(),
            procedural_count: self.procedural.len(),
            working_memory_size: self.working.len(),
            current_session: self.episodic.current_session(),
            total_steps: self.current_step,
        }
    }
}

/// Result of memory retrieval
#[derive(Debug, Clone)]
pub struct MemoryRetrievalResult {
    pub episode_count: usize,
    pub fact_count: usize,
    pub skill_count: usize,
    pub working_context: Vec<f64>,
    pub top_episode_ids: Vec<u64>,
    pub top_fact_names: Vec<String>,
    pub top_skill_names: Vec<String>,
}

/// Summary statistics for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystemSummary {
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub procedural_count: usize,
    pub working_memory_size: usize,
    pub current_session: u64,
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
    fn test_memory_system_creation() {
        let config = MemorySystemConfig::default();
        let memory = MemorySystem::new(config);

        assert!(memory.episodic.is_empty());
        assert!(memory.semantic.is_empty());
        assert!(memory.procedural.is_empty());
        assert!(memory.working.is_empty());
    }

    #[test]
    fn test_episodic_memory_store() {
        let config = MemorySystemConfig::default();
        let mut store = EpisodicMemoryStore::new(config);

        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let state = vec![1.0, 2.0, 3.0];
        let consequence = vec![1.5, 2.5, 3.5];

        let id = store.store(
            embedding.clone(),
            state,
            Some(0),
            0.8,
            consequence,
            EmotionalValence::Positive(0.7),
            vec!["concept1".to_string()],
            vec![0, 1],
            0,
        );

        assert_eq!(id, 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_episodic_retrieval() {
        let config = MemorySystemConfig::default();
        let mut store = EpisodicMemoryStore::new(config);

        // Store multiple memories
        for i in 0..5 {
            let embedding = vec![i as f64 * 0.1, i as f64 * 0.2, 0.5, 0.5];
            store.store(
                embedding,
                vec![i as f64],
                Some(i),
                0.5,
                vec![i as f64 + 1.0],
                EmotionalValence::Neutral,
                vec![],
                vec![],
                i,
            );
        }

        let query = vec![0.3, 0.6, 0.5, 0.5];
        let results = store.retrieve(&query, Some(3), 10);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_memory_store() {
        let config = MemorySystemConfig::default();
        let mut store = SemanticMemoryStore::new(config);

        let id = store.store(
            "dog".to_string(),
            vec![0.1, 0.2, 0.3],
            "A domestic animal".to_string(),
            None,
            0,
        );

        assert_eq!(id, 0);

        let fact = store.get_by_name("dog");
        assert!(fact.is_some());
        assert_eq!(fact.unwrap().name, "dog");
    }

    #[test]
    fn test_semantic_relations() {
        let config = MemorySystemConfig::default();
        let mut store = SemanticMemoryStore::new(config);

        let dog_id = store.store("dog".to_string(), vec![0.1, 0.2], "".to_string(), None, 0);
        let animal_id = store.store(
            "animal".to_string(),
            vec![0.2, 0.3],
            "".to_string(),
            None,
            0,
        );

        store.add_relation(dog_id, animal_id, RelationType::IsA, 0.9);

        let dog = store.get_by_name("dog").unwrap();
        assert_eq!(dog.relations.len(), 1);
        assert_eq!(dog.relations[0].target_id, animal_id);
    }

    #[test]
    fn test_procedural_memory_store() {
        let config = MemorySystemConfig::default();
        let mut store = ProceduralMemoryStore::new(config);

        let id = store.store(
            "jump".to_string(),
            vec![0.1, 0.2, 0.3],
            vec![0.5, 0.5, 0.5],
            vec![0, 1, 2],
            vec![0.8, 0.8, 0.8],
            0,
        );

        assert_eq!(id, 0);

        store.record_execution(id, true, 0.9, 1);

        let skill = store.get_by_name("jump").unwrap();
        assert_eq!(skill.execution_count, 1);
        assert_eq!(skill.success_rate, 1.0);
    }

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new(5);

        for i in 0..3 {
            wm.add(WorkingMemoryItem {
                embedding: vec![i as f64 * 0.1, 0.5],
                item_type: WorkingMemoryItemType::Perception,
                activation: 1.0 - i as f64 * 0.1,
                added_at: i,
                source_id: None,
                label: format!("item_{}", i),
            });
        }

        assert_eq!(wm.len(), 3);

        // Get context
        let context = wm.get_context();
        assert!(!context.is_empty());
    }

    #[test]
    fn test_memory_system_process_experience() {
        let config = MemorySystemConfig::default();
        let mut memory = MemorySystem::new(config);

        memory.process_experience(
            vec![1.0, 2.0, 3.0],
            Some(0),
            0.7,
            vec![1.5, 2.5, 3.5],
            vec!["goal_reached".to_string()],
            vec![0],
        );

        assert_eq!(memory.episodic.len(), 1);
        assert_eq!(memory.semantic.len(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_memory_consolidation() {
        let mut config = MemorySystemConfig::default();
        config.consolidation_interval = 5;
        let mut memory = MemorySystem::new(config);

        // Add some experiences
        for i in 0..10 {
            memory.process_experience(
                vec![i as f64 * 0.1, 0.5, 0.5],
                Some(i % 3),
                if i % 2 == 0 { 0.8 } else { 0.2 },
                vec![(i + 1) as f64 * 0.1, 0.5, 0.5],
                vec![format!("concept_{}", i % 3)],
                vec![],
            );
        }

        // Consolidation should have happened
        let summary = memory.summary();
        assert!(summary.episodic_count > 0);
        assert!(summary.semantic_count > 0);
    }

    #[test]
    fn test_session_management() {
        let config = MemorySystemConfig::default();
        let mut memory = MemorySystem::new(config);

        assert_eq!(memory.episodic.current_session(), 0);

        memory.new_session();
        assert_eq!(memory.episodic.current_session(), 1);

        memory.new_session();
        assert_eq!(memory.episodic.current_session(), 2);
    }
}
