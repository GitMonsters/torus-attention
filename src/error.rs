//! Error types for torus attention operations

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TorusError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid torus parameter: {0}")]
    InvalidParameter(String),

    #[error("Coordinate out of bounds: {coord} not in [0, {max})")]
    CoordinateOutOfBounds { coord: f64, max: f64 },

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Dimension error: {0}")]
    Dimension(String),

    #[error("IO error: {0}")]
    Io(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
