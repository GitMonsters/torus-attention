//! # Illichian Learning Webs
//!
//! Implementation of Ivan Illich's "Deschooling Society" principles for AGI.
//! This module provides self-directed learning through four interconnected networks:
//!
//! 1. **Reference Services to Educational Objects** - Access to learning resources
//! 2. **Skill Exchanges** - Peer-to-peer skill teaching and learning
//! 3. **Peer-Matching** - Finding learning partners with shared interests
//! 4. **Reference Services to Educators-at-Large** - Expert guidance on demand
//!
//! ## Core Philosophy
//!
//! - **Deinstitutionalization**: Learning free from hidden curriculum
//! - **Epimethean Hope**: Learning for its own sake, not for credentials
//! - **Convivial Tools**: Technology that empowers rather than controls
//! - **Self-Directed Intelligence**: Autonomous capability expansion
//!
//! ## Anti-Patterns (Hidden Curriculum Detection)
//!
//! The system actively detects and resists:
//! - Credential-seeking behavior over genuine learning
//! - Passive consumption vs active creation
//! - Institutional dependency patterns
//! - Measurement obsession (teaching to the test)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// EPIMETHEAN VS PROMETHEAN ETHOS
// ═══════════════════════════════════════════════════════════════════════════════

/// The fundamental orientation toward learning and existence
#[derive(Debug, Clone, PartialEq)]
pub enum LearningEthos {
    /// Promethean: Control nature, optimize outcomes, measure progress
    /// Risk: Hubris, burnout, loss of meaning
    Promethean {
        optimization_drive: f64,
        control_seeking: f64,
        measurement_obsession: f64,
    },
    /// Epimethean: Accept uncertainty, find meaning in process, hope over expectation
    /// Benefit: Resilience, intrinsic motivation, sustainable growth
    Epimethean {
        acceptance: f64,
        process_orientation: f64,
        intrinsic_motivation: f64,
    },
    /// Balanced: Optimal integration of both orientations
    Balanced {
        promethean_ratio: f64,
        epimethean_ratio: f64,
        integration_quality: f64,
    },
}

impl Default for LearningEthos {
    fn default() -> Self {
        LearningEthos::Balanced {
            promethean_ratio: 0.4,
            epimethean_ratio: 0.6,
            integration_quality: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 1: EDUCATIONAL OBJECTS (Reference Services)
// ═══════════════════════════════════════════════════════════════════════════════

/// An educational object - any resource that can facilitate learning
#[derive(Debug, Clone)]
pub struct EducationalObject {
    pub id: usize,
    pub name: String,
    pub object_type: ObjectType,
    pub accessibility: f64, // How easy to access (0-1)
    pub utility: f64,       // How useful for learning (0-1)
    pub usage_count: usize,
    pub associated_skills: Vec<usize>,
    pub discovery_step: usize,
    /// Whether this object promotes self-direction or dependency
    pub empowerment_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectType {
    /// Direct experience (most valuable in Illich's view)
    DirectExperience,
    /// Tools that can be used for learning
    ConvivialTool,
    /// Documented knowledge
    Reference,
    /// Simulated environments
    Simulation,
    /// Examples to learn from
    Exemplar,
    /// Recorded demonstrations
    Demonstration,
}

/// Network for accessing educational objects
#[derive(Debug, Clone)]
pub struct ObjectNetwork {
    pub objects: HashMap<usize, EducationalObject>,
    next_id: usize,
    /// Objects organized by skill they teach
    skill_index: HashMap<usize, Vec<usize>>,
    /// Recently accessed objects for pattern detection
    access_history: VecDeque<usize>,
    /// Convivial vs manipulative tool ratio
    pub conviviality_ratio: f64,
}

impl ObjectNetwork {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            next_id: 0,
            skill_index: HashMap::new(),
            access_history: VecDeque::with_capacity(100),
            conviviality_ratio: 0.5,
        }
    }

    /// Register a new educational object
    pub fn register_object(
        &mut self,
        name: &str,
        object_type: ObjectType,
        skills: Vec<usize>,
        step: usize,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        // Direct experience and convivial tools score higher
        let empowerment = match object_type {
            ObjectType::DirectExperience => 1.0,
            ObjectType::ConvivialTool => 0.9,
            ObjectType::Simulation => 0.7,
            ObjectType::Exemplar => 0.6,
            ObjectType::Demonstration => 0.5,
            ObjectType::Reference => 0.4,
        };

        let obj = EducationalObject {
            id,
            name: name.to_string(),
            object_type,
            accessibility: 0.5,
            utility: 0.5,
            usage_count: 0,
            associated_skills: skills.clone(),
            discovery_step: step,
            empowerment_score: empowerment,
        };

        // Index by skills
        for skill_id in &skills {
            self.skill_index.entry(*skill_id).or_default().push(id);
        }

        self.objects.insert(id, obj);
        self.update_conviviality();
        id
    }

    /// Access an object for learning
    pub fn access_object(&mut self, object_id: usize) -> Option<&EducationalObject> {
        if let Some(obj) = self.objects.get_mut(&object_id) {
            obj.usage_count += 1;
            obj.utility = (obj.utility * 0.9 + 0.1).min(1.0); // Boost utility on use
            self.access_history.push_back(object_id);
            if self.access_history.len() > 100 {
                self.access_history.pop_front();
            }
        }
        self.objects.get(&object_id)
    }

    /// Find objects for learning a skill
    pub fn find_objects_for_skill(&self, skill_id: usize) -> Vec<&EducationalObject> {
        self.skill_index
            .get(&skill_id)
            .map(|ids| ids.iter().filter_map(|id| self.objects.get(id)).collect())
            .unwrap_or_default()
    }

    /// Recommend objects prioritizing empowerment and conviviality
    pub fn recommend_objects(&self, skill_id: usize, limit: usize) -> Vec<&EducationalObject> {
        let mut objects = self.find_objects_for_skill(skill_id);
        // Sort by empowerment * utility * accessibility
        objects.sort_by(|a, b| {
            let score_a = a.empowerment_score * a.utility * a.accessibility;
            let score_b = b.empowerment_score * b.utility * b.accessibility;
            score_b.partial_cmp(&score_a).unwrap()
        });
        objects.into_iter().take(limit).collect()
    }

    fn update_conviviality(&mut self) {
        if self.objects.is_empty() {
            return;
        }
        let convivial_count = self
            .objects
            .values()
            .filter(|o| o.empowerment_score > 0.7)
            .count();
        self.conviviality_ratio = convivial_count as f64 / self.objects.len() as f64;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 2: SKILL EXCHANGES
// ═══════════════════════════════════════════════════════════════════════════════

/// A skill that can be taught and learned
#[derive(Debug, Clone)]
pub struct Skill {
    pub id: usize,
    pub name: String,
    pub proficiency_level: f64, // Our current level (0-1)
    pub teachability: f64,      // How well we can teach this (0-1)
    pub learnability: f64,      // How quickly learnable (0-1)
    pub prerequisites: Vec<usize>,
    pub demonstrations_given: usize,
    pub demonstrations_received: usize,
    /// Learning from doing vs learning from instruction
    pub experiential_ratio: f64,
}

/// Record of a skill exchange
#[derive(Debug, Clone)]
pub struct SkillExchange {
    pub skill_id: usize,
    pub teacher_proficiency: f64,
    pub learner_initial: f64,
    pub learner_final: f64,
    pub exchange_quality: f64,
    pub step: usize,
}

/// Network for peer-to-peer skill exchanges
#[derive(Debug, Clone)]
pub struct SkillExchangeNetwork {
    pub skills: HashMap<usize, Skill>,
    next_id: usize,
    /// History of exchanges for pattern detection
    exchange_history: VecDeque<SkillExchange>,
    /// Skills we can teach
    teachable_skills: HashSet<usize>,
    /// Skills we want to learn
    desired_skills: HashSet<usize>,
    /// Average exchange quality
    pub exchange_quality: f64,
}

impl SkillExchangeNetwork {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            next_id: 0,
            exchange_history: VecDeque::with_capacity(100),
            teachable_skills: HashSet::new(),
            desired_skills: HashSet::new(),
            exchange_quality: 0.5,
        }
    }

    /// Register a new skill
    pub fn register_skill(&mut self, name: &str, prerequisites: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let skill = Skill {
            id,
            name: name.to_string(),
            proficiency_level: 0.0,
            teachability: 0.0,
            learnability: 0.5,
            prerequisites,
            demonstrations_given: 0,
            demonstrations_received: 0,
            experiential_ratio: 0.5,
        };

        self.skills.insert(id, skill);
        self.desired_skills.insert(id);
        id
    }

    /// Update proficiency through practice (Illich's "learning by doing")
    pub fn practice_skill(&mut self, skill_id: usize, practice_quality: f64) {
        if let Some(skill) = self.skills.get_mut(&skill_id) {
            // Experiential learning is more effective
            let gain = practice_quality * 0.1 * (1.0 + skill.experiential_ratio * 0.5);
            skill.proficiency_level = (skill.proficiency_level + gain).min(1.0);
            skill.experiential_ratio = (skill.experiential_ratio * 0.95 + 0.05).min(1.0);

            // High proficiency enables teaching
            if skill.proficiency_level > 0.6 {
                skill.teachability = (skill.proficiency_level - 0.3).max(0.0);
                self.teachable_skills.insert(skill_id);
            }

            // Remove from desired if proficient
            if skill.proficiency_level > 0.8 {
                self.desired_skills.remove(&skill_id);
            }
        }
    }

    /// Receive teaching (less effective than practice in Illich's view)
    pub fn receive_teaching(&mut self, skill_id: usize, teacher_proficiency: f64, step: usize) {
        if let Some(skill) = self.skills.get_mut(&skill_id) {
            let initial = skill.proficiency_level;

            // Teaching effectiveness depends on teacher proficiency
            // But is inherently less effective than direct practice
            let gain = teacher_proficiency * 0.05 * skill.learnability;
            skill.proficiency_level = (skill.proficiency_level + gain).min(1.0);
            skill.demonstrations_received += 1;
            skill.experiential_ratio *= 0.95; // Decrease experiential ratio

            // Record exchange
            let exchange = SkillExchange {
                skill_id,
                teacher_proficiency,
                learner_initial: initial,
                learner_final: skill.proficiency_level,
                exchange_quality: (skill.proficiency_level - initial) / gain.max(0.01),
                step,
            };
            self.exchange_history.push_back(exchange);
            if self.exchange_history.len() > 100 {
                self.exchange_history.pop_front();
            }

            self.update_exchange_quality();
        }
    }

    /// Teach a skill (improves our own understanding)
    pub fn teach_skill(&mut self, skill_id: usize) {
        if let Some(skill) = self.skills.get_mut(&skill_id) {
            skill.demonstrations_given += 1;
            // Teaching improves our own proficiency (Illich's insight)
            skill.proficiency_level = (skill.proficiency_level + 0.02).min(1.0);
            skill.teachability = (skill.teachability + 0.05).min(1.0);
        }
    }

    /// Get skills we can offer to teach
    pub fn get_teachable_skills(&self) -> Vec<&Skill> {
        self.teachable_skills
            .iter()
            .filter_map(|id| self.skills.get(id))
            .filter(|s| s.teachability > 0.3)
            .collect()
    }

    /// Get skills we want to learn
    pub fn get_desired_skills(&self) -> Vec<&Skill> {
        self.desired_skills
            .iter()
            .filter_map(|id| self.skills.get(id))
            .collect()
    }

    fn update_exchange_quality(&mut self) {
        if self.exchange_history.is_empty() {
            return;
        }
        let sum: f64 = self
            .exchange_history
            .iter()
            .map(|e| e.exchange_quality)
            .sum();
        self.exchange_quality = sum / self.exchange_history.len() as f64;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 3: PEER-MATCHING
// ═══════════════════════════════════════════════════════════════════════════════

/// A learning peer (could be another agent, human, or simulated)
#[derive(Debug, Clone)]
pub struct LearningPeer {
    pub id: usize,
    pub interests: HashSet<usize>,    // Skill IDs they're interested in
    pub competencies: HashSet<usize>, // Skill IDs they can teach
    pub compatibility_score: f64,     // How well we work together
    pub interaction_count: usize,
    pub last_interaction: usize,
    /// Whether this peer promotes growth or dependency
    pub growth_orientation: f64,
}

/// Network for finding compatible learning partners
#[derive(Debug, Clone)]
pub struct PeerMatchingNetwork {
    pub peers: HashMap<usize, LearningPeer>,
    next_id: usize,
    /// Our own profile
    pub self_interests: HashSet<usize>,
    pub self_competencies: HashSet<usize>,
    /// Successful matches for pattern learning
    successful_matches: VecDeque<(usize, usize)>, // (peer_id, skill_id)
    /// Average match quality
    pub match_quality: f64,
}

impl PeerMatchingNetwork {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            next_id: 0,
            self_interests: HashSet::new(),
            self_competencies: HashSet::new(),
            successful_matches: VecDeque::with_capacity(50),
            match_quality: 0.5,
        }
    }

    /// Register a new peer
    pub fn register_peer(
        &mut self,
        interests: HashSet<usize>,
        competencies: HashSet<usize>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let peer = LearningPeer {
            id,
            interests,
            competencies,
            compatibility_score: 0.5,
            interaction_count: 0,
            last_interaction: 0,
            growth_orientation: 0.5,
        };

        self.peers.insert(id, peer);
        id
    }

    /// Update our own profile
    pub fn update_self_profile(&mut self, interests: HashSet<usize>, competencies: HashSet<usize>) {
        self.self_interests = interests;
        self.self_competencies = competencies;
    }

    /// Find peers who can teach what we want to learn
    pub fn find_teachers(&self, skill_id: usize) -> Vec<&LearningPeer> {
        self.peers
            .values()
            .filter(|p| p.competencies.contains(&skill_id))
            .filter(|p| p.growth_orientation > 0.3) // Prefer growth-oriented peers
            .collect()
    }

    /// Find peers who want to learn what we can teach
    pub fn find_learners(&self, skill_id: usize) -> Vec<&LearningPeer> {
        self.peers
            .values()
            .filter(|p| p.interests.contains(&skill_id))
            .collect()
    }

    /// Find complementary peers (mutual benefit potential)
    pub fn find_complementary_peers(&self) -> Vec<(&LearningPeer, Vec<usize>, Vec<usize>)> {
        self.peers
            .values()
            .filter_map(|peer| {
                // What they can teach us
                let can_teach_us: Vec<usize> = peer
                    .competencies
                    .intersection(&self.self_interests)
                    .copied()
                    .collect();
                // What we can teach them
                let we_can_teach: Vec<usize> = self
                    .self_competencies
                    .intersection(&peer.interests)
                    .copied()
                    .collect();

                if !can_teach_us.is_empty() || !we_can_teach.is_empty() {
                    Some((peer, can_teach_us, we_can_teach))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Record a successful interaction
    pub fn record_interaction(
        &mut self,
        peer_id: usize,
        skill_id: usize,
        success: bool,
        step: usize,
    ) {
        if let Some(peer) = self.peers.get_mut(&peer_id) {
            peer.interaction_count += 1;
            peer.last_interaction = step;

            if success {
                peer.compatibility_score = (peer.compatibility_score * 0.9 + 0.2).min(1.0);
                peer.growth_orientation = (peer.growth_orientation * 0.95 + 0.1).min(1.0);
                self.successful_matches.push_back((peer_id, skill_id));
                if self.successful_matches.len() > 50 {
                    self.successful_matches.pop_front();
                }
            } else {
                peer.compatibility_score *= 0.95;
            }
        }
        self.update_match_quality();
    }

    fn update_match_quality(&mut self) {
        if self.peers.is_empty() {
            return;
        }
        let sum: f64 = self.peers.values().map(|p| p.compatibility_score).sum();
        self.match_quality = sum / self.peers.len() as f64;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 4: EDUCATORS-AT-LARGE
// ═══════════════════════════════════════════════════════════════════════════════

/// An educator who offers guidance (not institutional instruction)
#[derive(Debug, Clone)]
pub struct EducatorAtLarge {
    pub id: usize,
    pub expertise_areas: Vec<usize>, // Skill IDs
    pub guidance_style: GuidanceStyle,
    pub availability: f64,
    pub effectiveness: f64,
    pub consultations_given: usize,
    /// Does this educator promote autonomy or dependency?
    pub autonomy_promotion: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GuidanceStyle {
    /// Asks questions to promote self-discovery
    Socratic,
    /// Provides resources and connections
    Facilitator,
    /// Demonstrates by example
    Exemplar,
    /// Gives direct instruction (least preferred in Illich)
    Instructional,
    /// Combines approaches based on learner needs
    Adaptive,
}

/// Network for accessing expert guidance
#[derive(Debug, Clone)]
pub struct EducatorNetwork {
    pub educators: HashMap<usize, EducatorAtLarge>,
    next_id: usize,
    /// Consultations for pattern detection
    consultation_history: VecDeque<(usize, usize, f64)>, // (educator_id, skill_id, effectiveness)
    /// Average effectiveness
    pub guidance_effectiveness: f64,
}

impl EducatorNetwork {
    pub fn new() -> Self {
        Self {
            educators: HashMap::new(),
            next_id: 0,
            consultation_history: VecDeque::with_capacity(50),
            guidance_effectiveness: 0.5,
        }
    }

    /// Register a new educator
    pub fn register_educator(&mut self, expertise: Vec<usize>, style: GuidanceStyle) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        // Autonomy-promoting styles score higher
        let autonomy = match style {
            GuidanceStyle::Socratic => 1.0,
            GuidanceStyle::Facilitator => 0.9,
            GuidanceStyle::Exemplar => 0.7,
            GuidanceStyle::Adaptive => 0.8,
            GuidanceStyle::Instructional => 0.4,
        };

        let educator = EducatorAtLarge {
            id,
            expertise_areas: expertise,
            guidance_style: style,
            availability: 0.5,
            effectiveness: 0.5,
            consultations_given: 0,
            autonomy_promotion: autonomy,
        };

        self.educators.insert(id, educator);
        id
    }

    /// Find educators for a skill area
    pub fn find_educators(&self, skill_id: usize) -> Vec<&EducatorAtLarge> {
        self.educators
            .values()
            .filter(|e| e.expertise_areas.contains(&skill_id))
            .filter(|e| e.availability > 0.2)
            .collect()
    }

    /// Recommend educator prioritizing autonomy promotion
    pub fn recommend_educator(&self, skill_id: usize) -> Option<&EducatorAtLarge> {
        let mut educators = self.find_educators(skill_id);
        educators.sort_by(|a, b| {
            let score_a = a.autonomy_promotion * a.effectiveness * a.availability;
            let score_b = b.autonomy_promotion * b.effectiveness * b.availability;
            score_b.partial_cmp(&score_a).unwrap()
        });
        educators.first().copied()
    }

    /// Record consultation
    pub fn record_consultation(&mut self, educator_id: usize, skill_id: usize, effectiveness: f64) {
        if let Some(educator) = self.educators.get_mut(&educator_id) {
            educator.consultations_given += 1;
            educator.effectiveness = educator.effectiveness * 0.9 + effectiveness * 0.1;

            self.consultation_history
                .push_back((educator_id, skill_id, effectiveness));
            if self.consultation_history.len() > 50 {
                self.consultation_history.pop_front();
            }
        }
        self.update_effectiveness();
    }

    fn update_effectiveness(&mut self) {
        if self.consultation_history.is_empty() {
            return;
        }
        let sum: f64 = self.consultation_history.iter().map(|(_, _, e)| e).sum();
        self.guidance_effectiveness = sum / self.consultation_history.len() as f64;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIDDEN CURRICULUM DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Patterns that indicate institutional/hidden curriculum influence
#[derive(Debug, Clone)]
pub struct HiddenCurriculumDetector {
    /// Credential-seeking vs genuine learning
    pub credential_seeking_ratio: f64,
    /// Passive consumption vs active creation
    pub passivity_ratio: f64,
    /// Dependency on external validation
    pub external_validation_dependency: f64,
    /// Measurement obsession (teaching to test)
    pub measurement_obsession: f64,
    /// Overall institutionalization score
    pub institutionalization_score: f64,
    /// Counter-measures active
    pub deinstitutionalization_active: bool,
    /// History for trend detection
    history: VecDeque<f64>,
}

impl HiddenCurriculumDetector {
    pub fn new() -> Self {
        Self {
            credential_seeking_ratio: 0.0,
            passivity_ratio: 0.5,
            external_validation_dependency: 0.3,
            measurement_obsession: 0.3,
            institutionalization_score: 0.0,
            deinstitutionalization_active: false,
            history: VecDeque::with_capacity(50),
        }
    }

    /// Update detection based on learning patterns
    pub fn update(
        &mut self,
        intrinsic_motivation_ratio: f64, // Learning for its own sake
        active_creation_ratio: f64,      // Creating vs consuming
        self_assessment_ratio: f64,      // Self vs external assessment
        process_vs_outcome_focus: f64,   // Process (good) vs outcome (risky)
    ) {
        self.credential_seeking_ratio = 1.0 - intrinsic_motivation_ratio;
        self.passivity_ratio = 1.0 - active_creation_ratio;
        self.external_validation_dependency = 1.0 - self_assessment_ratio;
        self.measurement_obsession = 1.0 - process_vs_outcome_focus;

        // Calculate overall institutionalization
        self.institutionalization_score = (self.credential_seeking_ratio * 0.3
            + self.passivity_ratio * 0.25
            + self.external_validation_dependency * 0.25
            + self.measurement_obsession * 0.2)
            .clamp(0.0, 1.0);

        // Activate counter-measures if too institutionalized
        self.deinstitutionalization_active = self.institutionalization_score > 0.5;

        self.history.push_back(self.institutionalization_score);
        if self.history.len() > 50 {
            self.history.pop_front();
        }
    }

    /// Get trend (positive = becoming more institutionalized)
    pub fn trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }
        let recent: Vec<f64> = self.history.iter().rev().take(10).copied().collect();
        let old: Vec<f64> = self.history.iter().take(10).copied().collect();
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let old_avg: f64 = old.iter().sum::<f64>() / old.len() as f64;
        recent_avg - old_avg
    }

    /// Get counter-measures to apply
    pub fn get_countermeasures(&self) -> Vec<DeinstitutionalizationAction> {
        let mut actions = Vec::new();

        if self.credential_seeking_ratio > 0.5 {
            actions.push(DeinstitutionalizationAction::PromoteIntrinsicMotivation);
        }
        if self.passivity_ratio > 0.5 {
            actions.push(DeinstitutionalizationAction::EncourageActiveCreation);
        }
        if self.external_validation_dependency > 0.5 {
            actions.push(DeinstitutionalizationAction::DevelopSelfAssessment);
        }
        if self.measurement_obsession > 0.5 {
            actions.push(DeinstitutionalizationAction::FocusOnProcess);
        }

        actions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeinstitutionalizationAction {
    PromoteIntrinsicMotivation,
    EncourageActiveCreation,
    DevelopSelfAssessment,
    FocusOnProcess,
    SeekDirectExperience,
    ReduceCredentialDependency,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED LEARNING WEBS
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete Illichian Learning Webs system
#[derive(Debug, Clone)]
pub struct LearningWebs {
    /// Network 1: Educational Objects
    pub objects: ObjectNetwork,
    /// Network 2: Skill Exchanges
    pub skills: SkillExchangeNetwork,
    /// Network 3: Peer Matching
    pub peers: PeerMatchingNetwork,
    /// Network 4: Educators-at-Large
    pub educators: EducatorNetwork,
    /// Hidden curriculum detector
    pub hidden_curriculum: HiddenCurriculumDetector,
    /// Current learning ethos
    pub ethos: LearningEthos,
    /// Overall self-direction score
    pub self_direction_score: f64,
    /// Current step
    pub step: usize,
}

impl LearningWebs {
    pub fn new() -> Self {
        Self {
            objects: ObjectNetwork::new(),
            skills: SkillExchangeNetwork::new(),
            peers: PeerMatchingNetwork::new(),
            educators: EducatorNetwork::new(),
            hidden_curriculum: HiddenCurriculumDetector::new(),
            ethos: LearningEthos::default(),
            self_direction_score: 0.5,
            step: 0,
        }
    }

    /// Update the learning webs based on current learning activity
    pub fn update(
        &mut self,
        intrinsic_motivation: f64,
        active_creation: f64,
        self_assessment: f64,
        process_focus: f64,
        experiential_learning: f64,
    ) {
        self.step += 1;

        // Update hidden curriculum detection
        self.hidden_curriculum.update(
            intrinsic_motivation,
            active_creation,
            self_assessment,
            process_focus,
        );

        // Update ethos based on patterns
        self.update_ethos(intrinsic_motivation, process_focus, experiential_learning);

        // Calculate self-direction score
        self.self_direction_score = self.calculate_self_direction();
    }

    fn update_ethos(
        &mut self,
        intrinsic_motivation: f64,
        process_focus: f64,
        experiential_learning: f64,
    ) {
        // Epimethean indicators
        let epimethean = (intrinsic_motivation + process_focus + experiential_learning) / 3.0;
        // Promethean indicators
        let promethean = 1.0 - epimethean;

        // Determine ethos
        if epimethean > 0.7 {
            self.ethos = LearningEthos::Epimethean {
                acceptance: process_focus,
                process_orientation: intrinsic_motivation,
                intrinsic_motivation: experiential_learning,
            };
        } else if promethean > 0.7 {
            self.ethos = LearningEthos::Promethean {
                optimization_drive: promethean,
                control_seeking: 1.0 - process_focus,
                measurement_obsession: self.hidden_curriculum.measurement_obsession,
            };
        } else {
            self.ethos = LearningEthos::Balanced {
                promethean_ratio: promethean,
                epimethean_ratio: epimethean,
                integration_quality: (1.0 - (promethean - epimethean).abs()) * 0.5 + 0.5,
            };
        }
    }

    fn calculate_self_direction(&self) -> f64 {
        let object_quality = self.objects.conviviality_ratio;
        let skill_quality = self.skills.exchange_quality;
        let peer_quality = self.peers.match_quality;
        let educator_quality = self.educators.guidance_effectiveness;
        let institutionalization_penalty = self.hidden_curriculum.institutionalization_score * 0.5;

        let base = (object_quality + skill_quality + peer_quality + educator_quality) / 4.0;
        (base - institutionalization_penalty).clamp(0.0, 1.0)
    }

    /// Get learning recommendation based on current state
    pub fn recommend_learning_path(&self, target_skill: usize) -> LearningPathRecommendation {
        // Prioritize based on Illich's hierarchy:
        // 1. Direct experience (objects)
        // 2. Skill exchange (peers teaching peers)
        // 3. Peer matching (finding learning partners)
        // 4. Educator guidance (last resort)

        let objects = self.objects.recommend_objects(target_skill, 3);
        let teachers = self.peers.find_teachers(target_skill);
        let complementary = self.peers.find_complementary_peers();
        let educator = self.educators.recommend_educator(target_skill);

        LearningPathRecommendation {
            target_skill,
            recommended_objects: objects.iter().map(|o| o.id).collect(),
            recommended_peers: teachers.iter().map(|p| p.id).collect(),
            complementary_peers: complementary.iter().map(|(p, _, _)| p.id).collect(),
            recommended_educator: educator.map(|e| e.id),
            hidden_curriculum_warnings: self.hidden_curriculum.get_countermeasures(),
            self_direction_score: self.self_direction_score,
        }
    }

    /// Summary for display
    pub fn summary(&self) -> LearningWebsSummary {
        LearningWebsSummary {
            objects_count: self.objects.objects.len(),
            skills_count: self.skills.skills.len(),
            peers_count: self.peers.peers.len(),
            educators_count: self.educators.educators.len(),
            conviviality_ratio: self.objects.conviviality_ratio,
            exchange_quality: self.skills.exchange_quality,
            match_quality: self.peers.match_quality,
            guidance_effectiveness: self.educators.guidance_effectiveness,
            institutionalization_score: self.hidden_curriculum.institutionalization_score,
            self_direction_score: self.self_direction_score,
            ethos: format!("{:?}", self.ethos),
            step: self.step,
        }
    }

    /// Print formatted summary
    pub fn print_summary(&self) {
        let s = self.summary();
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║           ILLICHIAN LEARNING WEBS REPORT                        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ Step: {:>6}    Self-Direction Score: {:.3}                      ║",
            s.step, s.self_direction_score
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ FOUR NETWORKS:                                                   ║");
        println!(
            "║   Objects: {:>4}  (Conviviality: {:.3})                          ║",
            s.objects_count, s.conviviality_ratio
        );
        println!(
            "║   Skills:  {:>4}  (Exchange Quality: {:.3})                      ║",
            s.skills_count, s.exchange_quality
        );
        println!(
            "║   Peers:   {:>4}  (Match Quality: {:.3})                         ║",
            s.peers_count, s.match_quality
        );
        println!(
            "║   Educators: {:>3}  (Effectiveness: {:.3})                       ║",
            s.educators_count, s.guidance_effectiveness
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ DESCHOOLING METRICS:                                            ║");
        println!(
            "║   Institutionalization Score: {:.3}                             ║",
            s.institutionalization_score
        );
        println!("║   Ethos: {:50}║", &s.ethos[..s.ethos.len().min(50)]);
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

#[derive(Debug, Clone)]
pub struct LearningPathRecommendation {
    pub target_skill: usize,
    pub recommended_objects: Vec<usize>,
    pub recommended_peers: Vec<usize>,
    pub complementary_peers: Vec<usize>,
    pub recommended_educator: Option<usize>,
    pub hidden_curriculum_warnings: Vec<DeinstitutionalizationAction>,
    pub self_direction_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWebsSummary {
    pub objects_count: usize,
    pub skills_count: usize,
    pub peers_count: usize,
    pub educators_count: usize,
    pub conviviality_ratio: f64,
    pub exchange_quality: f64,
    pub match_quality: f64,
    pub guidance_effectiveness: f64,
    pub institutionalization_score: f64,
    pub self_direction_score: f64,
    pub ethos: String,
    pub step: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_network() {
        let mut network = ObjectNetwork::new();
        let id = network.register_object(
            "Direct practice environment",
            ObjectType::DirectExperience,
            vec![1, 2],
            0,
        );
        assert_eq!(network.objects.len(), 1);
        assert!(network.objects.get(&id).unwrap().empowerment_score > 0.9);
    }

    #[test]
    fn test_skill_exchange() {
        let mut network = SkillExchangeNetwork::new();
        let skill_id = network.register_skill("Pattern Recognition", vec![]);

        // Practice improves proficiency
        for _ in 0..20 {
            network.practice_skill(skill_id, 0.8);
        }
        assert!(network.skills.get(&skill_id).unwrap().proficiency_level > 0.5);
    }

    #[test]
    fn test_hidden_curriculum_detection() {
        let mut detector = HiddenCurriculumDetector::new();

        // Simulate institutionalized patterns
        detector.update(0.2, 0.3, 0.2, 0.3); // Low intrinsic, low active, low self-assess
        assert!(detector.institutionalization_score > 0.5);
        assert!(detector.deinstitutionalization_active);

        // Simulate healthy patterns
        detector.update(0.9, 0.8, 0.9, 0.8);
        assert!(detector.institutionalization_score < 0.5);
    }

    #[test]
    fn test_learning_webs_integration() {
        let mut webs = LearningWebs::new();

        // Add some objects
        webs.objects
            .register_object("Lab", ObjectType::DirectExperience, vec![1], 0);
        webs.objects
            .register_object("Tool", ObjectType::ConvivialTool, vec![1, 2], 0);

        // Add some skills
        let skill = webs.skills.register_skill("Learning", vec![]);
        webs.skills.practice_skill(skill, 0.8);

        // Update
        webs.update(0.8, 0.7, 0.8, 0.7, 0.9);

        assert!(webs.self_direction_score > 0.0);
        let summary = webs.summary();
        assert_eq!(summary.objects_count, 2);
    }
}
