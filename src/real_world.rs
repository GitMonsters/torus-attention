//! # Real-World Interface for AGI
//!
//! This module provides traits and implementations for connecting AGI systems
//! to real-world sensors and actuators, enabling embodied intelligence.
//!
//! ## Key Features
//!
//! 1. **Sensor Traits**: Abstract interfaces for various sensor types
//! 2. **Actuator Traits**: Abstract interfaces for various actuator types
//! 3. **Sensor Fusion**: Combine multiple sensor modalities
//! 4. **Time Synchronization**: Handle asynchronous sensor data
//! 5. **Calibration**: Sensor calibration and normalization
//!
//! ## Supported Sensor Types
//!
//! - Vision (camera, depth camera)
//! - Audio (microphone, speech recognition)
//! - Proprioception (joint angles, velocities)
//! - Touch (pressure, temperature)
//! - Localization (GPS, IMU, odometry)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                   Real-World Interface                              │
//! │                                                                     │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                   Sensor Manager                             │   │
//! │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐│   │
//! │  │  │ Vision │  │ Audio  │  │ Touch  │  │  IMU   │  │  GPS   ││   │
//! │  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘│   │
//! │  │      └───────────┴───────────┴───────────┴───────────┘     │   │
//! │  │                          │                                  │   │
//! │  │                   ┌──────┴──────┐                           │   │
//! │  │                   │   Sensor    │                           │   │
//! │  │                   │   Fusion    │                           │   │
//! │  │                   └──────┬──────┘                           │   │
//! │  └──────────────────────────┼──────────────────────────────────┘   │
//! │                             ▼                                       │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                  Unified Perception                          │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the real-world interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealWorldConfig {
    /// Maximum sensor buffer size
    pub max_buffer_size: usize,
    /// Sensor fusion window (milliseconds)
    pub fusion_window_ms: u64,
    /// Enable automatic calibration
    pub auto_calibration: bool,
    /// Sensor timeout (milliseconds)
    pub sensor_timeout_ms: u64,
    /// Enable data normalization
    pub normalize_data: bool,
    /// Enable noise filtering
    pub filter_noise: bool,
    /// Noise filter strength (0-1)
    pub filter_strength: f64,
}

impl Default for RealWorldConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 100,
            fusion_window_ms: 50,
            auto_calibration: true,
            sensor_timeout_ms: 1000,
            normalize_data: true,
            filter_noise: true,
            filter_strength: 0.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENSOR TRAITS
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for a sensor
pub type SensorId = u64;

/// Type of sensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorType {
    /// Camera (RGB)
    Camera,
    /// Depth camera
    DepthCamera,
    /// Stereo camera
    StereoCamera,
    /// Microphone
    Microphone,
    /// Pressure sensor
    PressureSensor,
    /// Temperature sensor
    TemperatureSensor,
    /// IMU (accelerometer + gyroscope)
    IMU,
    /// GPS
    GPS,
    /// Lidar
    Lidar,
    /// Radar
    Radar,
    /// Joint encoder
    JointEncoder,
    /// Force/torque sensor
    ForceTorque,
    /// Custom sensor
    Custom(u64),
}

/// State of a sensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorState {
    /// Sensor is not initialized
    Uninitialized,
    /// Sensor is ready
    Ready,
    /// Sensor is actively reading
    Active,
    /// Sensor error
    Error,
    /// Sensor is calibrating
    Calibrating,
    /// Sensor is offline
    Offline,
}

/// Timestamped sensor reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Sensor ID
    pub sensor_id: SensorId,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Sensor type
    pub sensor_type: SensorType,
    /// Reading data
    pub data: SensorData,
    /// Reading quality (0-1)
    pub quality: f64,
}

/// Different types of sensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorData {
    /// Single scalar value
    Scalar(f64),
    /// Vector of values
    Vector(Vec<f64>),
    /// 2D image data (width, height, channels, data)
    Image {
        width: usize,
        height: usize,
        channels: usize,
        data: Vec<f64>,
    },
    /// Point cloud (list of 3D points)
    PointCloud {
        points: Vec<(f64, f64, f64)>,
        colors: Option<Vec<(f64, f64, f64)>>,
    },
    /// Pose (position + orientation quaternion)
    Pose {
        position: (f64, f64, f64),
        orientation: (f64, f64, f64, f64), // quaternion (w, x, y, z)
    },
    /// IMU data
    IMUData {
        linear_acceleration: (f64, f64, f64),
        angular_velocity: (f64, f64, f64),
    },
    /// Audio samples
    Audio {
        sample_rate: u32,
        channels: usize,
        samples: Vec<f64>,
    },
}

/// Trait for sensor implementations
pub trait Sensor: Send + Sync + std::fmt::Debug {
    /// Get sensor ID
    fn id(&self) -> SensorId;

    /// Get sensor type
    fn sensor_type(&self) -> SensorType;

    /// Get current state
    fn state(&self) -> SensorState;

    /// Initialize the sensor
    fn initialize(&mut self) -> Result<(), SensorError>;

    /// Read current sensor value
    fn read(&mut self) -> Result<SensorReading, SensorError>;

    /// Calibrate the sensor
    fn calibrate(&mut self) -> Result<(), SensorError>;

    /// Shutdown the sensor
    fn shutdown(&mut self) -> Result<(), SensorError>;
}

/// Sensor error type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorError {
    /// Sensor not initialized
    NotInitialized,
    /// Sensor timeout
    Timeout,
    /// Communication error
    CommunicationError(String),
    /// Calibration error
    CalibrationError(String),
    /// Hardware error
    HardwareError(String),
    /// Data error
    DataError(String),
    /// Unknown error
    Unknown(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTUATOR TRAITS
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an actuator
pub type ActuatorId = u64;

/// Type of actuator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActuatorType {
    /// Servo motor
    Servo,
    /// DC motor
    DCMotor,
    /// Stepper motor
    Stepper,
    /// Linear actuator
    Linear,
    /// Pneumatic actuator
    Pneumatic,
    /// Speaker
    Speaker,
    /// Display
    Display,
    /// LED
    LED,
    /// Gripper
    Gripper,
    /// Custom actuator
    Custom(u64),
}

/// State of an actuator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuatorState {
    /// Actuator is not initialized
    Uninitialized,
    /// Actuator is ready
    Ready,
    /// Actuator is moving
    Moving,
    /// Actuator reached target
    AtTarget,
    /// Actuator error
    Error,
    /// Actuator is disabled (e-stop)
    Disabled,
}

/// Command for an actuator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorCommand {
    /// Actuator ID
    pub actuator_id: ActuatorId,
    /// Command type
    pub command_type: CommandType,
    /// Target value
    pub target: ActuatorTarget,
    /// Execution deadline (optional, nanoseconds since epoch)
    pub deadline_ns: Option<u64>,
}

/// Type of actuator command
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandType {
    /// Set position
    SetPosition,
    /// Set velocity
    SetVelocity,
    /// Set force/torque
    SetForce,
    /// Stop
    Stop,
    /// Home
    Home,
    /// Emergency stop
    EStop,
}

/// Target value for actuator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActuatorTarget {
    /// Single value
    Scalar(f64),
    /// Vector of values
    Vector(Vec<f64>),
    /// Position in space
    Position(f64, f64, f64),
    /// Boolean state
    State(bool),
}

/// Result of actuator command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorResult {
    /// Was command successful
    pub success: bool,
    /// Actual achieved value (if applicable)
    pub achieved: Option<ActuatorTarget>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
}

/// Trait for actuator implementations
pub trait Actuator: Send + Sync + std::fmt::Debug {
    /// Get actuator ID
    fn id(&self) -> ActuatorId;

    /// Get actuator type
    fn actuator_type(&self) -> ActuatorType;

    /// Get current state
    fn state(&self) -> ActuatorState;

    /// Initialize the actuator
    fn initialize(&mut self) -> Result<(), ActuatorError>;

    /// Execute a command
    fn execute(&mut self, command: ActuatorCommand) -> Result<ActuatorResult, ActuatorError>;

    /// Get current position/value
    fn get_position(&self) -> Option<ActuatorTarget>;

    /// Stop immediately
    fn stop(&mut self) -> Result<(), ActuatorError>;

    /// Shutdown the actuator
    fn shutdown(&mut self) -> Result<(), ActuatorError>;
}

/// Actuator error type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActuatorError {
    /// Actuator not initialized
    NotInitialized,
    /// Command timeout
    Timeout,
    /// Position limit exceeded
    LimitExceeded,
    /// Force limit exceeded
    ForceLimitExceeded,
    /// Communication error
    CommunicationError(String),
    /// Hardware error
    HardwareError(String),
    /// Unknown error
    Unknown(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK IMPLEMENTATIONS FOR TESTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Mock camera sensor for testing
#[derive(Debug, Clone)]
pub struct MockCamera {
    id: SensorId,
    state: SensorState,
    width: usize,
    height: usize,
    frame_count: u64,
}

impl MockCamera {
    pub fn new(id: SensorId, width: usize, height: usize) -> Self {
        Self {
            id,
            state: SensorState::Uninitialized,
            width,
            height,
            frame_count: 0,
        }
    }
}

impl Sensor for MockCamera {
    fn id(&self) -> SensorId {
        self.id
    }

    fn sensor_type(&self) -> SensorType {
        SensorType::Camera
    }

    fn state(&self) -> SensorState {
        self.state
    }

    fn initialize(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Ready;
        Ok(())
    }

    fn read(&mut self) -> Result<SensorReading, SensorError> {
        if self.state != SensorState::Ready && self.state != SensorState::Active {
            return Err(SensorError::NotInitialized);
        }

        self.state = SensorState::Active;
        self.frame_count += 1;

        // Generate mock image data
        let data: Vec<f64> = (0..self.width * self.height * 3)
            .map(|i| ((i as f64 + self.frame_count as f64) % 256.0) / 255.0)
            .collect();

        Ok(SensorReading {
            sensor_id: self.id,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sensor_type: SensorType::Camera,
            data: SensorData::Image {
                width: self.width,
                height: self.height,
                channels: 3,
                data,
            },
            quality: 1.0,
        })
    }

    fn calibrate(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Calibrating;
        // Simulate calibration
        self.state = SensorState::Ready;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Offline;
        Ok(())
    }
}

/// Mock IMU sensor for testing
#[derive(Debug, Clone)]
pub struct MockIMU {
    id: SensorId,
    state: SensorState,
    sample_count: u64,
}

impl MockIMU {
    pub fn new(id: SensorId) -> Self {
        Self {
            id,
            state: SensorState::Uninitialized,
            sample_count: 0,
        }
    }
}

impl Sensor for MockIMU {
    fn id(&self) -> SensorId {
        self.id
    }

    fn sensor_type(&self) -> SensorType {
        SensorType::IMU
    }

    fn state(&self) -> SensorState {
        self.state
    }

    fn initialize(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Ready;
        Ok(())
    }

    fn read(&mut self) -> Result<SensorReading, SensorError> {
        if self.state != SensorState::Ready && self.state != SensorState::Active {
            return Err(SensorError::NotInitialized);
        }

        self.state = SensorState::Active;
        self.sample_count += 1;

        // Generate mock IMU data with some noise
        let t = self.sample_count as f64 * 0.01;

        Ok(SensorReading {
            sensor_id: self.id,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sensor_type: SensorType::IMU,
            data: SensorData::IMUData {
                linear_acceleration: (
                    0.0 + (t * 0.1).sin() * 0.1,
                    0.0 + (t * 0.2).cos() * 0.1,
                    9.81 + (t * 0.05).sin() * 0.05,
                ),
                angular_velocity: (
                    (t * 0.3).sin() * 0.01,
                    (t * 0.4).cos() * 0.01,
                    (t * 0.2).sin() * 0.01,
                ),
            },
            quality: 0.95,
        })
    }

    fn calibrate(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Calibrating;
        self.state = SensorState::Ready;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), SensorError> {
        self.state = SensorState::Offline;
        Ok(())
    }
}

/// Mock servo actuator for testing
#[derive(Debug, Clone)]
pub struct MockServo {
    id: ActuatorId,
    state: ActuatorState,
    position: f64,
    min_position: f64,
    max_position: f64,
}

impl MockServo {
    pub fn new(id: ActuatorId, min: f64, max: f64) -> Self {
        Self {
            id,
            state: ActuatorState::Uninitialized,
            position: (min + max) / 2.0,
            min_position: min,
            max_position: max,
        }
    }
}

impl Actuator for MockServo {
    fn id(&self) -> ActuatorId {
        self.id
    }

    fn actuator_type(&self) -> ActuatorType {
        ActuatorType::Servo
    }

    fn state(&self) -> ActuatorState {
        self.state
    }

    fn initialize(&mut self) -> Result<(), ActuatorError> {
        self.state = ActuatorState::Ready;
        Ok(())
    }

    fn execute(&mut self, command: ActuatorCommand) -> Result<ActuatorResult, ActuatorError> {
        if self.state == ActuatorState::Uninitialized {
            return Err(ActuatorError::NotInitialized);
        }

        match command.command_type {
            CommandType::SetPosition => {
                if let ActuatorTarget::Scalar(target) = command.target {
                    if target < self.min_position || target > self.max_position {
                        return Err(ActuatorError::LimitExceeded);
                    }
                    self.state = ActuatorState::Moving;
                    self.position = target;
                    self.state = ActuatorState::AtTarget;

                    Ok(ActuatorResult {
                        success: true,
                        achieved: Some(ActuatorTarget::Scalar(self.position)),
                        error: None,
                        execution_time_ns: 1_000_000, // 1ms simulated
                    })
                } else {
                    Err(ActuatorError::Unknown("Invalid target type".to_string()))
                }
            }
            CommandType::Stop | CommandType::EStop => {
                self.state = ActuatorState::Ready;
                Ok(ActuatorResult {
                    success: true,
                    achieved: Some(ActuatorTarget::Scalar(self.position)),
                    error: None,
                    execution_time_ns: 100_000,
                })
            }
            CommandType::Home => {
                self.position = (self.min_position + self.max_position) / 2.0;
                self.state = ActuatorState::AtTarget;
                Ok(ActuatorResult {
                    success: true,
                    achieved: Some(ActuatorTarget::Scalar(self.position)),
                    error: None,
                    execution_time_ns: 5_000_000,
                })
            }
            _ => Err(ActuatorError::Unknown("Unsupported command".to_string())),
        }
    }

    fn get_position(&self) -> Option<ActuatorTarget> {
        Some(ActuatorTarget::Scalar(self.position))
    }

    fn stop(&mut self) -> Result<(), ActuatorError> {
        self.state = ActuatorState::Ready;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), ActuatorError> {
        self.state = ActuatorState::Uninitialized;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENSOR FUSION
// ═══════════════════════════════════════════════════════════════════════════════

/// Fused perception from multiple sensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedPerception {
    /// Timestamp of fusion
    pub timestamp_ns: u64,
    /// Estimated position
    pub position: Option<(f64, f64, f64)>,
    /// Estimated orientation (quaternion)
    pub orientation: Option<(f64, f64, f64, f64)>,
    /// Estimated velocity
    pub velocity: Option<(f64, f64, f64)>,
    /// Visual features (if camera available)
    pub visual_features: Vec<f64>,
    /// Audio features (if microphone available)
    pub audio_features: Vec<f64>,
    /// Contact points (if touch sensors available)
    pub contact_points: Vec<(f64, f64, f64)>,
    /// Overall perception confidence
    pub confidence: f64,
    /// Contributing sensor IDs
    pub sensor_ids: Vec<SensorId>,
}

/// Sensor fusion system
#[derive(Debug)]
pub struct SensorFusion {
    /// Sensor readings buffer
    readings: HashMap<SensorId, VecDeque<SensorReading>>,
    /// Sensor weights for fusion
    weights: HashMap<SensorId, f64>,
    /// Configuration
    config: RealWorldConfig,
    /// Last fusion result
    last_fusion: Option<FusedPerception>,
}

impl SensorFusion {
    pub fn new(config: RealWorldConfig) -> Self {
        Self {
            readings: HashMap::new(),
            weights: HashMap::new(),
            config,
            last_fusion: None,
        }
    }

    /// Add a sensor reading
    pub fn add_reading(&mut self, reading: SensorReading) {
        let buffer = self
            .readings
            .entry(reading.sensor_id)
            .or_insert_with(|| VecDeque::with_capacity(self.config.max_buffer_size));

        buffer.push_back(reading);

        // Prune old readings
        while buffer.len() > self.config.max_buffer_size {
            buffer.pop_front();
        }
    }

    /// Set sensor weight for fusion
    pub fn set_weight(&mut self, sensor_id: SensorId, weight: f64) {
        self.weights.insert(sensor_id, weight.max(0.0).min(1.0));
    }

    /// Perform sensor fusion
    pub fn fuse(&mut self) -> FusedPerception {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let window_ns = self.config.fusion_window_ms * 1_000_000;

        // Collect recent readings
        let mut position_estimates: Vec<((f64, f64, f64), f64)> = Vec::new();
        let mut orientation_estimates: Vec<((f64, f64, f64, f64), f64)> = Vec::new();
        let mut visual_features: Vec<f64> = Vec::new();
        let mut sensor_ids: Vec<SensorId> = Vec::new();

        for (&sensor_id, buffer) in &self.readings {
            let weight = *self.weights.get(&sensor_id).unwrap_or(&1.0);

            for reading in buffer.iter().rev() {
                if now_ns.saturating_sub(reading.timestamp_ns) > window_ns {
                    break;
                }

                if !sensor_ids.contains(&sensor_id) {
                    sensor_ids.push(sensor_id);
                }

                match &reading.data {
                    SensorData::Pose {
                        position,
                        orientation,
                    } => {
                        position_estimates.push((*position, weight * reading.quality));
                        orientation_estimates.push((*orientation, weight * reading.quality));
                    }
                    SensorData::IMUData {
                        linear_acceleration,
                        angular_velocity: _,
                    } => {
                        // Simple dead-reckoning (very simplified)
                        if let Some(last) = &self.last_fusion {
                            if let Some(pos) = last.position {
                                let dt = 0.01; // Assume 100Hz
                                let new_pos = (
                                    pos.0 + linear_acceleration.0 * dt * dt / 2.0,
                                    pos.1 + linear_acceleration.1 * dt * dt / 2.0,
                                    pos.2 + (linear_acceleration.2 - 9.81) * dt * dt / 2.0,
                                );
                                position_estimates.push((new_pos, weight * reading.quality * 0.5));
                            }
                        }
                    }
                    SensorData::Image { data, .. } => {
                        // Extract simple visual features (mean of regions)
                        if data.len() > 0 {
                            let chunk_size = data.len() / 10;
                            for chunk in data.chunks(chunk_size.max(1)).take(10) {
                                let mean: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
                                visual_features.push(mean);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Weighted average for position
        let position = if position_estimates.is_empty() {
            None
        } else {
            let total_weight: f64 = position_estimates.iter().map(|(_, w)| w).sum();
            if total_weight > 0.0 {
                let x = position_estimates.iter().map(|(p, w)| p.0 * w).sum::<f64>() / total_weight;
                let y = position_estimates.iter().map(|(p, w)| p.1 * w).sum::<f64>() / total_weight;
                let z = position_estimates.iter().map(|(p, w)| p.2 * w).sum::<f64>() / total_weight;
                Some((x, y, z))
            } else {
                None
            }
        };

        // Weighted average for orientation (simplified - proper quaternion averaging is more complex)
        let orientation = if orientation_estimates.is_empty() {
            None
        } else {
            let total_weight: f64 = orientation_estimates.iter().map(|(_, w)| w).sum();
            if total_weight > 0.0 {
                let w = orientation_estimates
                    .iter()
                    .map(|(o, wt)| o.0 * wt)
                    .sum::<f64>()
                    / total_weight;
                let x = orientation_estimates
                    .iter()
                    .map(|(o, wt)| o.1 * wt)
                    .sum::<f64>()
                    / total_weight;
                let y = orientation_estimates
                    .iter()
                    .map(|(o, wt)| o.2 * wt)
                    .sum::<f64>()
                    / total_weight;
                let z = orientation_estimates
                    .iter()
                    .map(|(o, wt)| o.3 * wt)
                    .sum::<f64>()
                    / total_weight;
                // Normalize quaternion
                let norm = (w * w + x * x + y * y + z * z).sqrt();
                if norm > 0.0 {
                    Some((w / norm, x / norm, y / norm, z / norm))
                } else {
                    Some((1.0, 0.0, 0.0, 0.0))
                }
            } else {
                None
            }
        };

        // Calculate overall confidence
        let confidence = if sensor_ids.is_empty() {
            0.0
        } else {
            let avg_weight: f64 = sensor_ids
                .iter()
                .map(|id| self.weights.get(id).unwrap_or(&1.0))
                .sum::<f64>()
                / sensor_ids.len() as f64;
            avg_weight * 0.5 + 0.5 * (sensor_ids.len() as f64 / 5.0).min(1.0)
        };

        let result = FusedPerception {
            timestamp_ns: now_ns,
            position,
            orientation,
            velocity: None, // Would require more sophisticated state estimation
            visual_features,
            audio_features: vec![],
            contact_points: vec![],
            confidence,
            sensor_ids,
        };

        self.last_fusion = Some(result.clone());
        result
    }

    /// Get last fusion result
    pub fn last_perception(&self) -> Option<&FusedPerception> {
        self.last_fusion.as_ref()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED REAL-WORLD INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified real-world interface
#[derive(Debug)]
pub struct RealWorldInterface {
    /// Registered sensors (using Box for trait objects)
    sensors: HashMap<SensorId, Box<dyn Sensor>>,
    /// Registered actuators (using Box for trait objects)
    actuators: HashMap<ActuatorId, Box<dyn Actuator>>,
    /// Sensor fusion
    pub fusion: SensorFusion,
    /// Configuration
    config: RealWorldConfig,
    /// Next sensor ID
    next_sensor_id: SensorId,
    /// Next actuator ID
    next_actuator_id: ActuatorId,
}

impl RealWorldInterface {
    pub fn new(config: RealWorldConfig) -> Self {
        Self {
            sensors: HashMap::new(),
            actuators: HashMap::new(),
            fusion: SensorFusion::new(config.clone()),
            config,
            next_sensor_id: 0,
            next_actuator_id: 0,
        }
    }

    /// Register a sensor
    pub fn register_sensor(
        &mut self,
        mut sensor: Box<dyn Sensor>,
    ) -> Result<SensorId, SensorError> {
        sensor.initialize()?;
        let id = sensor.id();
        self.sensors.insert(id, sensor);
        Ok(id)
    }

    /// Register an actuator
    pub fn register_actuator(
        &mut self,
        mut actuator: Box<dyn Actuator>,
    ) -> Result<ActuatorId, ActuatorError> {
        actuator.initialize()?;
        let id = actuator.id();
        self.actuators.insert(id, actuator);
        Ok(id)
    }

    /// Read all sensors
    pub fn read_sensors(&mut self) -> Vec<SensorReading> {
        let mut readings = Vec::new();

        for sensor in self.sensors.values_mut() {
            if let Ok(reading) = sensor.read() {
                self.fusion.add_reading(reading.clone());
                readings.push(reading);
            }
        }

        readings
    }

    /// Read a specific sensor
    pub fn read_sensor(&mut self, sensor_id: SensorId) -> Option<SensorReading> {
        if let Some(sensor) = self.sensors.get_mut(&sensor_id) {
            if let Ok(reading) = sensor.read() {
                self.fusion.add_reading(reading.clone());
                return Some(reading);
            }
        }
        None
    }

    /// Execute an actuator command
    pub fn execute(&mut self, command: ActuatorCommand) -> Result<ActuatorResult, ActuatorError> {
        if let Some(actuator) = self.actuators.get_mut(&command.actuator_id) {
            actuator.execute(command)
        } else {
            Err(ActuatorError::Unknown("Actuator not found".to_string()))
        }
    }

    /// Get fused perception
    pub fn get_perception(&mut self) -> FusedPerception {
        self.fusion.fuse()
    }

    /// Shutdown all devices
    pub fn shutdown(&mut self) {
        for sensor in self.sensors.values_mut() {
            let _ = sensor.shutdown();
        }
        for actuator in self.actuators.values_mut() {
            let _ = actuator.shutdown();
        }
    }

    /// Get summary
    pub fn summary(&self) -> RealWorldSummary {
        RealWorldSummary {
            sensor_count: self.sensors.len(),
            actuator_count: self.actuators.len(),
            active_sensors: self
                .sensors
                .values()
                .filter(|s| s.state() == SensorState::Active || s.state() == SensorState::Ready)
                .count(),
            active_actuators: self
                .actuators
                .values()
                .filter(|a| {
                    a.state() != ActuatorState::Uninitialized && a.state() != ActuatorState::Error
                })
                .count(),
        }
    }
}

/// Summary of real-world interface state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealWorldSummary {
    pub sensor_count: usize,
    pub actuator_count: usize,
    pub active_sensors: usize,
    pub active_actuators: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_camera() {
        let mut camera = MockCamera::new(0, 64, 64);

        assert_eq!(camera.state(), SensorState::Uninitialized);

        camera.initialize().unwrap();
        assert_eq!(camera.state(), SensorState::Ready);

        let reading = camera.read().unwrap();
        assert_eq!(reading.sensor_type, SensorType::Camera);

        if let SensorData::Image {
            width,
            height,
            channels,
            ..
        } = reading.data
        {
            assert_eq!(width, 64);
            assert_eq!(height, 64);
            assert_eq!(channels, 3);
        } else {
            panic!("Expected image data");
        }

        camera.shutdown().unwrap();
        assert_eq!(camera.state(), SensorState::Offline);
    }

    #[test]
    fn test_mock_imu() {
        let mut imu = MockIMU::new(1);

        imu.initialize().unwrap();

        let reading = imu.read().unwrap();
        assert_eq!(reading.sensor_type, SensorType::IMU);

        if let SensorData::IMUData {
            linear_acceleration,
            angular_velocity,
        } = reading.data
        {
            // Check gravity approximately correct
            assert!((linear_acceleration.2 - 9.81).abs() < 0.1);
            // Check angular velocity small
            assert!(angular_velocity.0.abs() < 0.1);
        } else {
            panic!("Expected IMU data");
        }
    }

    #[test]
    fn test_mock_servo() {
        let mut servo = MockServo::new(0, -90.0, 90.0);

        servo.initialize().unwrap();

        let result = servo
            .execute(ActuatorCommand {
                actuator_id: 0,
                command_type: CommandType::SetPosition,
                target: ActuatorTarget::Scalar(45.0),
                deadline_ns: None,
            })
            .unwrap();

        assert!(result.success);

        if let Some(ActuatorTarget::Scalar(pos)) = servo.get_position() {
            assert!((pos - 45.0).abs() < 0.01);
        } else {
            panic!("Expected scalar position");
        }

        // Test limit exceeded
        let result = servo.execute(ActuatorCommand {
            actuator_id: 0,
            command_type: CommandType::SetPosition,
            target: ActuatorTarget::Scalar(180.0),
            deadline_ns: None,
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_sensor_fusion() {
        let config = RealWorldConfig::default();
        let mut fusion = SensorFusion::new(config);

        // Add a pose reading
        fusion.add_reading(SensorReading {
            sensor_id: 0,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sensor_type: SensorType::GPS,
            data: SensorData::Pose {
                position: (1.0, 2.0, 3.0),
                orientation: (1.0, 0.0, 0.0, 0.0),
            },
            quality: 0.9,
        });

        let fused = fusion.fuse();

        assert!(fused.position.is_some());
        let pos = fused.position.unwrap();
        assert!((pos.0 - 1.0).abs() < 0.01);
        assert!((pos.1 - 2.0).abs() < 0.01);
        assert!((pos.2 - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_real_world_interface() {
        let config = RealWorldConfig::default();
        let mut interface = RealWorldInterface::new(config);

        // Register mock sensors
        let camera = MockCamera::new(0, 32, 32);
        let imu = MockIMU::new(1);

        interface.register_sensor(Box::new(camera)).unwrap();
        interface.register_sensor(Box::new(imu)).unwrap();

        // Register mock actuator
        let servo = MockServo::new(0, -90.0, 90.0);
        interface.register_actuator(Box::new(servo)).unwrap();

        let summary = interface.summary();
        assert_eq!(summary.sensor_count, 2);
        assert_eq!(summary.actuator_count, 1);

        // Read sensors
        let readings = interface.read_sensors();
        assert_eq!(readings.len(), 2);

        // Execute actuator command
        let result = interface
            .execute(ActuatorCommand {
                actuator_id: 0,
                command_type: CommandType::SetPosition,
                target: ActuatorTarget::Scalar(30.0),
                deadline_ns: None,
            })
            .unwrap();

        assert!(result.success);

        // Shutdown
        interface.shutdown();
    }

    #[test]
    fn test_sensor_weights() {
        let config = RealWorldConfig::default();
        let mut fusion = SensorFusion::new(config);

        fusion.set_weight(0, 0.9);
        fusion.set_weight(1, 0.5);

        // Add readings with different weights
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        fusion.add_reading(SensorReading {
            sensor_id: 0,
            timestamp_ns: now,
            sensor_type: SensorType::GPS,
            data: SensorData::Pose {
                position: (10.0, 0.0, 0.0),
                orientation: (1.0, 0.0, 0.0, 0.0),
            },
            quality: 1.0,
        });

        fusion.add_reading(SensorReading {
            sensor_id: 1,
            timestamp_ns: now,
            sensor_type: SensorType::GPS,
            data: SensorData::Pose {
                position: (0.0, 0.0, 0.0),
                orientation: (1.0, 0.0, 0.0, 0.0),
            },
            quality: 1.0,
        });

        let fused = fusion.fuse();
        let pos = fused.position.unwrap();

        // Should be weighted towards sensor 0 (0.9 vs 0.5)
        assert!(pos.0 > 5.0); // Closer to 10 than to 0
    }
}
