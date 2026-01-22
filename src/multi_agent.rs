//! # Multi-Agent System for AGI
//!
//! This module implements a multi-agent system where AGI agents can:
//!
//! 1. **Communicate**: Exchange information via messages
//! 2. **Collaborate**: Work together on shared goals
//! 3. **Teach**: Share knowledge and skills with peers
//! 4. **Negotiate**: Resolve conflicts and coordinate actions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Multi-Agent System                              │
//! │                                                                     │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
//! │  │   Agent 1   │◀──▶│  Message    │◀──▶│       Agent 2           │ │
//! │  │             │    │   Router    │    │                         │ │
//! │  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
//! │         │                  │                       │               │
//! │         ▼                  ▼                       ▼               │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                  Shared Environment                          │   │
//! │  │  • World State    • Shared Goals    • Collective Memory     │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the multi-agent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Message queue capacity per agent
    pub message_queue_capacity: usize,
    /// Maximum message history to keep
    pub max_message_history: usize,
    /// Communication range (if using spatial topology)
    pub communication_range: f64,
    /// Enable knowledge sharing
    pub enable_knowledge_sharing: bool,
    /// Enable skill teaching
    pub enable_skill_teaching: bool,
    /// Trust decay rate
    pub trust_decay_rate: f64,
    /// Cooperation bonus
    pub cooperation_bonus: f64,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            max_agents: 10,
            message_queue_capacity: 100,
            max_message_history: 1000,
            communication_range: 10.0,
            enable_knowledge_sharing: true,
            enable_skill_teaching: true,
            trust_decay_rate: 0.01,
            cooperation_bonus: 0.2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT IDENTITY AND STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an agent
pub type AgentId = u64;

/// Agent role in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// General-purpose learner
    Learner,
    /// Specialist in a domain
    Specialist(u64), // Domain ID
    /// Teacher that helps others
    Teacher,
    /// Explorer that discovers new knowledge
    Explorer,
    /// Coordinator that manages group activities
    Coordinator,
    /// Critic that evaluates and provides feedback
    Critic,
}

/// State of an agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is idle
    Idle,
    /// Agent is working on a task
    Working,
    /// Agent is communicating
    Communicating,
    /// Agent is learning
    Learning,
    /// Agent is teaching
    Teaching,
    /// Agent is waiting for response
    Waiting,
    /// Agent is offline
    Offline,
}

/// Agent profile and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Unique ID
    pub id: AgentId,
    /// Agent name
    pub name: String,
    /// Agent role
    pub role: AgentRole,
    /// Current state
    pub state: AgentState,
    /// Expertise areas (domain -> proficiency)
    pub expertise: HashMap<String, f64>,
    /// Skills the agent can teach
    pub teachable_skills: Vec<String>,
    /// Learning interests
    pub learning_interests: Vec<String>,
    /// Position (if using spatial topology)
    pub position: Option<(f64, f64, f64)>,
    /// Creation timestamp
    pub created_at: usize,
    /// Last active timestamp
    pub last_active: usize,
}

impl AgentProfile {
    pub fn new(id: AgentId, name: String, role: AgentRole, timestamp: usize) -> Self {
        Self {
            id,
            name,
            role,
            state: AgentState::Idle,
            expertise: HashMap::new(),
            teachable_skills: Vec::new(),
            learning_interests: Vec::new(),
            position: None,
            created_at: timestamp,
            last_active: timestamp,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESSAGING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of message
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// General information sharing
    Information,
    /// Request for help or information
    Request,
    /// Response to a request
    Response,
    /// Teaching/knowledge transfer
    Teaching,
    /// Query/question
    Query,
    /// Proposal for collaboration
    Proposal,
    /// Accept proposal
    Accept,
    /// Reject proposal
    Reject,
    /// Acknowledgment
    Ack,
    /// Broadcast to all
    Broadcast,
}

/// Priority of a message
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Urgent = 3,
}

/// A message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Unique message ID
    pub id: u64,
    /// Sender agent ID
    pub sender: AgentId,
    /// Recipient agent ID (None for broadcast)
    pub recipient: Option<AgentId>,
    /// Message type
    pub message_type: MessageType,
    /// Priority
    pub priority: MessagePriority,
    /// Message content
    pub content: MessageContent,
    /// Timestamp
    pub timestamp: usize,
    /// In reply to (message ID)
    pub in_reply_to: Option<u64>,
    /// Requires acknowledgment
    pub requires_ack: bool,
    /// Expiration (step when message becomes invalid)
    pub expires_at: Option<usize>,
}

/// Content of a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    /// Text message
    Text(String),
    /// Knowledge/fact sharing
    Knowledge {
        topic: String,
        facts: Vec<String>,
        confidence: f64,
    },
    /// Skill sharing
    Skill {
        name: String,
        description: String,
        preconditions: Vec<String>,
        steps: Vec<String>,
    },
    /// Task proposal
    Task {
        description: String,
        required_skills: Vec<String>,
        estimated_reward: f64,
    },
    /// Query
    Query {
        question: String,
        context: Vec<String>,
    },
    /// State update
    StateUpdate {
        position: Option<(f64, f64, f64)>,
        state: AgentState,
        current_goal: Option<String>,
    },
    /// Feedback
    Feedback {
        about_message: u64,
        helpful: bool,
        comment: String,
    },
}

/// Message router for agent communication
#[derive(Debug, Clone)]
pub struct MessageRouter {
    /// Message queues per agent
    queues: HashMap<AgentId, VecDeque<AgentMessage>>,
    /// Message history
    history: VecDeque<AgentMessage>,
    /// Next message ID
    next_id: u64,
    /// Configuration
    config: MultiAgentConfig,
}

impl MessageRouter {
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            queues: HashMap::new(),
            history: VecDeque::with_capacity(config.max_message_history),
            next_id: 0,
            config,
        }
    }

    /// Register an agent with the router
    pub fn register_agent(&mut self, agent_id: AgentId) {
        self.queues
            .entry(agent_id)
            .or_insert_with(|| VecDeque::with_capacity(self.config.message_queue_capacity));
    }

    /// Unregister an agent
    pub fn unregister_agent(&mut self, agent_id: AgentId) {
        self.queues.remove(&agent_id);
    }

    /// Send a message
    pub fn send(&mut self, mut message: AgentMessage) -> u64 {
        message.id = self.next_id;
        self.next_id += 1;

        // Store in history
        self.history.push_back(message.clone());
        if self.history.len() > self.config.max_message_history {
            self.history.pop_front();
        }

        // Route the message
        if let Some(recipient) = message.recipient {
            // Direct message
            if let Some(queue) = self.queues.get_mut(&recipient) {
                // Insert based on priority
                let pos = queue
                    .iter()
                    .position(|m| m.priority < message.priority)
                    .unwrap_or(queue.len());
                queue.insert(pos, message);
            }
        } else {
            // Broadcast
            let sender = message.sender;
            for (&agent_id, queue) in &mut self.queues {
                if agent_id != sender {
                    let pos = queue
                        .iter()
                        .position(|m| m.priority < message.priority)
                        .unwrap_or(queue.len());
                    queue.insert(pos, message.clone());
                }
            }
        }

        self.next_id - 1
    }

    /// Receive messages for an agent
    pub fn receive(&mut self, agent_id: AgentId, max_messages: usize) -> Vec<AgentMessage> {
        if let Some(queue) = self.queues.get_mut(&agent_id) {
            let count = max_messages.min(queue.len());
            queue.drain(..count).collect()
        } else {
            vec![]
        }
    }

    /// Check if an agent has pending messages
    pub fn has_messages(&self, agent_id: AgentId) -> bool {
        self.queues
            .get(&agent_id)
            .map(|q| !q.is_empty())
            .unwrap_or(false)
    }

    /// Get pending message count for an agent
    pub fn pending_count(&self, agent_id: AgentId) -> usize {
        self.queues.get(&agent_id).map(|q| q.len()).unwrap_or(0)
    }

    /// Get message history
    pub fn get_history(&self, limit: usize) -> Vec<&AgentMessage> {
        self.history.iter().rev().take(limit).collect()
    }

    /// Get conversation between two agents
    pub fn get_conversation(&self, agent1: AgentId, agent2: AgentId) -> Vec<&AgentMessage> {
        self.history
            .iter()
            .filter(|m| {
                (m.sender == agent1 && m.recipient == Some(agent2))
                    || (m.sender == agent2 && m.recipient == Some(agent1))
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST AND REPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Trust relationship between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRelation {
    /// Target agent
    pub agent_id: AgentId,
    /// Trust level (0-1)
    pub trust: f64,
    /// Number of positive interactions
    pub positive_interactions: usize,
    /// Number of negative interactions
    pub negative_interactions: usize,
    /// Last interaction timestamp
    pub last_interaction: usize,
}

/// Trust manager for agent relationships
#[derive(Debug, Clone)]
pub struct TrustManager {
    /// Trust relations per agent
    relations: HashMap<AgentId, HashMap<AgentId, TrustRelation>>,
    /// Configuration
    config: MultiAgentConfig,
}

impl TrustManager {
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            relations: HashMap::new(),
            config,
        }
    }

    /// Get or create trust relation
    fn get_or_create(
        &mut self,
        from: AgentId,
        to: AgentId,
        timestamp: usize,
    ) -> &mut TrustRelation {
        self.relations
            .entry(from)
            .or_insert_with(HashMap::new)
            .entry(to)
            .or_insert_with(|| TrustRelation {
                agent_id: to,
                trust: 0.5, // Neutral starting trust
                positive_interactions: 0,
                negative_interactions: 0,
                last_interaction: timestamp,
            })
    }

    /// Record a positive interaction
    pub fn record_positive(&mut self, from: AgentId, to: AgentId, timestamp: usize) {
        let relation = self.get_or_create(from, to, timestamp);
        relation.positive_interactions += 1;
        relation.trust = (relation.trust + 0.1).min(1.0);
        relation.last_interaction = timestamp;
    }

    /// Record a negative interaction
    pub fn record_negative(&mut self, from: AgentId, to: AgentId, timestamp: usize) {
        let relation = self.get_or_create(from, to, timestamp);
        relation.negative_interactions += 1;
        relation.trust = (relation.trust - 0.15).max(0.0);
        relation.last_interaction = timestamp;
    }

    /// Get trust level
    pub fn get_trust(&self, from: AgentId, to: AgentId) -> f64 {
        self.relations
            .get(&from)
            .and_then(|r| r.get(&to))
            .map(|r| r.trust)
            .unwrap_or(0.5)
    }

    /// Decay trust over time
    pub fn decay_trust(&mut self, current_time: usize) {
        for relations in self.relations.values_mut() {
            for relation in relations.values_mut() {
                let time_since = (current_time - relation.last_interaction) as f64;
                let decay = (-self.config.trust_decay_rate * time_since / 100.0).exp();
                // Trust decays towards neutral (0.5)
                relation.trust = 0.5 + (relation.trust - 0.5) * decay;
            }
        }
    }

    /// Get most trusted agents for a given agent
    pub fn most_trusted(&self, agent_id: AgentId, n: usize) -> Vec<(AgentId, f64)> {
        let mut trusted: Vec<(AgentId, f64)> = self
            .relations
            .get(&agent_id)
            .map(|r| r.iter().map(|(&id, rel)| (id, rel.trust)).collect())
            .unwrap_or_default();

        trusted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        trusted.into_iter().take(n).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLABORATION
// ═══════════════════════════════════════════════════════════════════════════════

/// A collaborative task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeTask {
    /// Task ID
    pub id: u64,
    /// Description
    pub description: String,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Participating agents
    pub participants: HashSet<AgentId>,
    /// Task leader (coordinator)
    pub leader: Option<AgentId>,
    /// Current status
    pub status: TaskStatus,
    /// Progress (0-1)
    pub progress: f64,
    /// Estimated reward
    pub estimated_reward: f64,
    /// Actual reward (when completed)
    pub actual_reward: Option<f64>,
    /// Created timestamp
    pub created_at: usize,
    /// Completed timestamp
    pub completed_at: Option<usize>,
}

/// Status of a collaborative task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is proposed, waiting for participants
    Proposed,
    /// Task is active
    Active,
    /// Task is paused
    Paused,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// Collaboration manager
#[derive(Debug, Clone)]
pub struct CollaborationManager {
    /// Active tasks
    tasks: HashMap<u64, CollaborativeTask>,
    /// Agent task assignments
    agent_tasks: HashMap<AgentId, HashSet<u64>>,
    /// Next task ID
    next_id: u64,
    /// Configuration
    config: MultiAgentConfig,
}

impl CollaborationManager {
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            tasks: HashMap::new(),
            agent_tasks: HashMap::new(),
            next_id: 0,
            config,
        }
    }

    /// Propose a new task
    pub fn propose_task(
        &mut self,
        proposer: AgentId,
        description: String,
        required_skills: Vec<String>,
        estimated_reward: f64,
        timestamp: usize,
    ) -> u64 {
        let task = CollaborativeTask {
            id: self.next_id,
            description,
            required_skills,
            participants: HashSet::from([proposer]),
            leader: Some(proposer),
            status: TaskStatus::Proposed,
            progress: 0.0,
            estimated_reward,
            actual_reward: None,
            created_at: timestamp,
            completed_at: None,
        };

        let id = task.id;
        self.tasks.insert(id, task);
        self.agent_tasks
            .entry(proposer)
            .or_insert_with(HashSet::new)
            .insert(id);

        self.next_id += 1;
        id
    }

    /// Join a task
    pub fn join_task(&mut self, agent_id: AgentId, task_id: u64) -> bool {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            if task.status == TaskStatus::Proposed || task.status == TaskStatus::Active {
                task.participants.insert(agent_id);
                self.agent_tasks
                    .entry(agent_id)
                    .or_insert_with(HashSet::new)
                    .insert(task_id);
                return true;
            }
        }
        false
    }

    /// Leave a task
    pub fn leave_task(&mut self, agent_id: AgentId, task_id: u64) -> bool {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.participants.remove(&agent_id);
            if let Some(tasks) = self.agent_tasks.get_mut(&agent_id) {
                tasks.remove(&task_id);
            }
            // If leader left, assign new leader
            if task.leader == Some(agent_id) {
                task.leader = task.participants.iter().next().copied();
            }
            return true;
        }
        false
    }

    /// Start a task
    pub fn start_task(&mut self, task_id: u64) -> bool {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            if task.status == TaskStatus::Proposed && !task.participants.is_empty() {
                task.status = TaskStatus::Active;
                return true;
            }
        }
        false
    }

    /// Update task progress
    pub fn update_progress(&mut self, task_id: u64, progress: f64) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.progress = progress.min(1.0).max(0.0);
        }
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: u64, actual_reward: f64, timestamp: usize) -> bool {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            if task.status == TaskStatus::Active {
                task.status = TaskStatus::Completed;
                task.progress = 1.0;
                task.actual_reward = Some(actual_reward);
                task.completed_at = Some(timestamp);
                return true;
            }
        }
        false
    }

    /// Fail a task
    pub fn fail_task(&mut self, task_id: u64, timestamp: usize) -> bool {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            if task.status == TaskStatus::Active {
                task.status = TaskStatus::Failed;
                task.completed_at = Some(timestamp);
                return true;
            }
        }
        false
    }

    /// Get tasks for an agent
    pub fn get_agent_tasks(&self, agent_id: AgentId) -> Vec<&CollaborativeTask> {
        self.agent_tasks
            .get(&agent_id)
            .map(|ids| ids.iter().filter_map(|id| self.tasks.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get active tasks
    pub fn get_active_tasks(&self) -> Vec<&CollaborativeTask> {
        self.tasks
            .values()
            .filter(|t| t.status == TaskStatus::Active)
            .collect()
    }

    /// Get proposed tasks (looking for participants)
    pub fn get_proposed_tasks(&self) -> Vec<&CollaborativeTask> {
        self.tasks
            .values()
            .filter(|t| t.status == TaskStatus::Proposed)
            .collect()
    }

    /// Calculate reward share for participants
    pub fn calculate_reward_share(&self, task_id: u64) -> HashMap<AgentId, f64> {
        if let Some(task) = self.tasks.get(&task_id) {
            if let Some(reward) = task.actual_reward {
                let n = task.participants.len() as f64;
                let share = reward / n;
                // Add cooperation bonus
                let bonus = self.config.cooperation_bonus * share;
                return task
                    .participants
                    .iter()
                    .map(|&id| (id, share + bonus))
                    .collect();
            }
        }
        HashMap::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED MULTI-AGENT SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified multi-agent system
#[derive(Debug, Clone)]
pub struct MultiAgentSystem {
    /// Registered agents
    pub agents: HashMap<AgentId, AgentProfile>,
    /// Message router
    pub router: MessageRouter,
    /// Trust manager
    pub trust: TrustManager,
    /// Collaboration manager
    pub collaboration: CollaborationManager,
    /// Configuration
    config: MultiAgentConfig,
    /// Next agent ID
    next_agent_id: AgentId,
    /// Current step
    current_step: usize,
}

impl MultiAgentSystem {
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            agents: HashMap::new(),
            router: MessageRouter::new(config.clone()),
            trust: TrustManager::new(config.clone()),
            collaboration: CollaborationManager::new(config.clone()),
            config,
            next_agent_id: 0,
            current_step: 0,
        }
    }

    /// Register a new agent
    pub fn register_agent(&mut self, name: String, role: AgentRole) -> AgentId {
        let id = self.next_agent_id;
        let profile = AgentProfile::new(id, name, role, self.current_step);

        self.agents.insert(id, profile);
        self.router.register_agent(id);
        self.next_agent_id += 1;

        id
    }

    /// Unregister an agent
    pub fn unregister_agent(&mut self, agent_id: AgentId) {
        self.agents.remove(&agent_id);
        self.router.unregister_agent(agent_id);
    }

    /// Update agent state
    pub fn update_agent_state(&mut self, agent_id: AgentId, state: AgentState) {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.state = state;
            agent.last_active = self.current_step;
        }
    }

    /// Add expertise to an agent
    pub fn add_expertise(&mut self, agent_id: AgentId, domain: String, proficiency: f64) {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.expertise.insert(domain, proficiency);
        }
    }

    /// Add teachable skill to an agent
    pub fn add_teachable_skill(&mut self, agent_id: AgentId, skill: String) {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            if !agent.teachable_skills.contains(&skill) {
                agent.teachable_skills.push(skill);
            }
        }
    }

    /// Send a message between agents
    pub fn send_message(
        &mut self,
        sender: AgentId,
        recipient: Option<AgentId>,
        message_type: MessageType,
        content: MessageContent,
        priority: MessagePriority,
    ) -> u64 {
        let message = AgentMessage {
            id: 0, // Will be set by router
            sender,
            recipient,
            message_type,
            priority,
            content,
            timestamp: self.current_step,
            in_reply_to: None,
            requires_ack: false,
            expires_at: None,
        };

        // Update agent states
        if let Some(agent) = self.agents.get_mut(&sender) {
            agent.state = AgentState::Communicating;
            agent.last_active = self.current_step;
        }

        self.router.send(message)
    }

    /// Process messages for an agent
    pub fn receive_messages(&mut self, agent_id: AgentId, max: usize) -> Vec<AgentMessage> {
        let messages = self.router.receive(agent_id, max);

        // Update trust based on message reception
        for msg in &messages {
            if msg.message_type == MessageType::Teaching
                || msg.message_type == MessageType::Information
            {
                // Receiving helpful info increases trust slightly
                self.trust
                    .record_positive(agent_id, msg.sender, self.current_step);
            }
        }

        messages
    }

    /// Find agents with specific expertise
    pub fn find_experts(&self, domain: &str, min_proficiency: f64) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| {
                a.expertise
                    .get(domain)
                    .map(|&p| p >= min_proficiency)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Find agents who can teach a skill
    pub fn find_teachers(&self, skill: &str) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| a.teachable_skills.contains(&skill.to_string()))
            .collect()
    }

    /// Find agents interested in learning something
    pub fn find_learners(&self, topic: &str) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| a.learning_interests.contains(&topic.to_string()))
            .collect()
    }

    /// Get agents within communication range (if using spatial topology)
    pub fn agents_in_range(&self, position: (f64, f64, f64)) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| {
                if let Some(pos) = a.position {
                    let dist = ((pos.0 - position.0).powi(2)
                        + (pos.1 - position.1).powi(2)
                        + (pos.2 - position.2).powi(2))
                    .sqrt();
                    dist <= self.config.communication_range
                } else {
                    true // No position means global communication
                }
            })
            .collect()
    }

    /// Step the system forward
    pub fn step(&mut self) {
        self.current_step += 1;

        // Decay trust periodically
        if self.current_step % 100 == 0 {
            self.trust.decay_trust(self.current_step);
        }
    }

    /// Get summary statistics
    pub fn summary(&self) -> MultiAgentSummary {
        let agents_by_state: HashMap<AgentState, usize> =
            self.agents.values().fold(HashMap::new(), |mut acc, a| {
                *acc.entry(a.state).or_insert(0) += 1;
                acc
            });

        let active_tasks = self.collaboration.get_active_tasks().len();
        let proposed_tasks = self.collaboration.get_proposed_tasks().len();

        MultiAgentSummary {
            total_agents: self.agents.len(),
            idle_agents: *agents_by_state.get(&AgentState::Idle).unwrap_or(&0),
            working_agents: *agents_by_state.get(&AgentState::Working).unwrap_or(&0),
            communicating_agents: *agents_by_state
                .get(&AgentState::Communicating)
                .unwrap_or(&0),
            active_tasks,
            proposed_tasks,
            total_messages: self.router.history.len(),
            current_step: self.current_step,
        }
    }
}

/// Summary of multi-agent system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentSummary {
    pub total_agents: usize,
    pub idle_agents: usize,
    pub working_agents: usize,
    pub communicating_agents: usize,
    pub active_tasks: usize,
    pub proposed_tasks: usize,
    pub total_messages: usize,
    pub current_step: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_agent_system_creation() {
        let config = MultiAgentConfig::default();
        let system = MultiAgentSystem::new(config);

        assert!(system.agents.is_empty());
    }

    #[test]
    fn test_register_agent() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let id = system.register_agent("Agent1".to_string(), AgentRole::Learner);

        assert_eq!(id, 0);
        assert_eq!(system.agents.len(), 1);
        assert_eq!(system.agents.get(&id).unwrap().name, "Agent1");
    }

    #[test]
    fn test_send_message() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Learner);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Teacher);

        let msg_id = system.send_message(
            agent1,
            Some(agent2),
            MessageType::Query,
            MessageContent::Text("Hello!".to_string()),
            MessagePriority::Normal,
        );

        assert_eq!(msg_id, 0);
        assert!(system.router.has_messages(agent2));
        assert!(!system.router.has_messages(agent1));
    }

    #[test]
    fn test_receive_messages() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Learner);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Teacher);

        system.send_message(
            agent1,
            Some(agent2),
            MessageType::Query,
            MessageContent::Text("Hello!".to_string()),
            MessagePriority::Normal,
        );

        let messages = system.receive_messages(agent2, 10);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].sender, agent1);
    }

    #[test]
    fn test_broadcast_message() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Coordinator);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Learner);
        let agent3 = system.register_agent("Agent3".to_string(), AgentRole::Learner);

        system.send_message(
            agent1,
            None, // Broadcast
            MessageType::Broadcast,
            MessageContent::Text("Attention everyone!".to_string()),
            MessagePriority::High,
        );

        assert!(system.router.has_messages(agent2));
        assert!(system.router.has_messages(agent3));
        assert!(!system.router.has_messages(agent1)); // Sender doesn't receive own broadcast
    }

    #[test]
    fn test_trust_management() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Learner);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Teacher);

        // Initial trust is neutral
        assert!((system.trust.get_trust(agent1, agent2) - 0.5).abs() < 0.01);

        // Record positive interaction
        system.trust.record_positive(agent1, agent2, 0);
        assert!(system.trust.get_trust(agent1, agent2) > 0.5);

        // Record negative interaction
        system.trust.record_negative(agent1, agent2, 1);
        // Trust decreases but stays above initial due to prior positive
    }

    #[test]
    fn test_collaborative_task() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Coordinator);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Learner);

        // Propose task
        let task_id = system.collaboration.propose_task(
            agent1,
            "Build a model".to_string(),
            vec!["programming".to_string()],
            10.0,
            0,
        );

        // Agent2 joins
        assert!(system.collaboration.join_task(agent2, task_id));

        // Start task
        assert!(system.collaboration.start_task(task_id));

        // Update progress
        system.collaboration.update_progress(task_id, 0.5);

        // Complete task
        assert!(system.collaboration.complete_task(task_id, 12.0, 1));

        // Calculate rewards
        let rewards = system.collaboration.calculate_reward_share(task_id);
        assert_eq!(rewards.len(), 2);
    }

    #[test]
    fn test_find_experts() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Expert1".to_string(), AgentRole::Specialist(1));
        let agent2 = system.register_agent("Novice".to_string(), AgentRole::Learner);

        system.add_expertise(agent1, "machine_learning".to_string(), 0.9);
        system.add_expertise(agent2, "machine_learning".to_string(), 0.3);

        let experts = system.find_experts("machine_learning", 0.7);

        assert_eq!(experts.len(), 1);
        assert_eq!(experts[0].id, agent1);
    }

    #[test]
    fn test_find_teachers() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Teacher1".to_string(), AgentRole::Teacher);
        let _agent2 = system.register_agent("Student".to_string(), AgentRole::Learner);

        system.add_teachable_skill(agent1, "python".to_string());

        let teachers = system.find_teachers("python");

        assert_eq!(teachers.len(), 1);
        assert_eq!(teachers[0].id, agent1);
    }

    #[test]
    fn test_message_priority() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent1 = system.register_agent("Agent1".to_string(), AgentRole::Learner);
        let agent2 = system.register_agent("Agent2".to_string(), AgentRole::Teacher);

        // Send low priority message first
        system.send_message(
            agent1,
            Some(agent2),
            MessageType::Information,
            MessageContent::Text("Low priority".to_string()),
            MessagePriority::Low,
        );

        // Send high priority message second
        system.send_message(
            agent1,
            Some(agent2),
            MessageType::Request,
            MessageContent::Text("High priority".to_string()),
            MessagePriority::High,
        );

        // High priority should be received first
        let messages = system.receive_messages(agent2, 2);
        assert_eq!(messages[0].priority, MessagePriority::High);
        assert_eq!(messages[1].priority, MessagePriority::Low);
    }

    #[test]
    fn test_summary() {
        let config = MultiAgentConfig::default();
        let mut system = MultiAgentSystem::new(config);

        system.register_agent("Agent1".to_string(), AgentRole::Learner);
        system.register_agent("Agent2".to_string(), AgentRole::Teacher);

        let summary = system.summary();

        assert_eq!(summary.total_agents, 2);
        assert_eq!(summary.idle_agents, 2);
    }
}
