//! # Training Metrics and Logging
//!
//! Provides Tensorboard-compatible logging for training metrics.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::metrics::MetricsLogger;
//!
//! let mut logger = MetricsLogger::new("./logs")?;
//! logger.log_scalar("train/loss", 1.5, 100)?;
//! logger.log_scalar("train/lr", 0.001, 100)?;
//! logger.flush()?;
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::TorusResult;

/// Metrics logger with Tensorboard support
pub struct MetricsLogger {
    /// Tensorboard summary writer (None if disabled)
    writer: Option<SummaryWriter>,
    /// Log directory
    log_dir: PathBuf,
    /// Whether logging is enabled
    enabled: bool,
}

impl MetricsLogger {
    /// Create a new metrics logger
    ///
    /// Creates a timestamped subdirectory for this training run.
    pub fn new(log_dir: impl AsRef<Path>) -> TorusResult<Self> {
        let log_dir = log_dir.as_ref();
        
        // Create timestamped run directory
        let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let run_dir = log_dir.join(format!("run_{}", timestamp));
        fs::create_dir_all(&run_dir)?;
        
        let writer = SummaryWriter::new(&run_dir);
        
        log::info!("Tensorboard logs: {:?}", run_dir);
        log::info!("View with: tensorboard --logdir {:?}", log_dir);
        
        Ok(Self {
            writer: Some(writer),
            log_dir: run_dir,
            enabled: true,
        })
    }
    
    /// Create a disabled logger (for when logging is not wanted)
    pub fn disabled() -> Self {
        Self {
            writer: None,
            log_dir: PathBuf::new(),
            enabled: false,
        }
    }
    
    /// Log a scalar value
    pub fn log_scalar(&mut self, tag: &str, value: f64, step: usize) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        if let Some(ref mut writer) = self.writer {
            writer.add_scalar(tag, value as f32, step);
        }
        Ok(())
    }
    
    /// Log multiple scalars at once
    pub fn log_scalars(&mut self, values: &[(&str, f64)], step: usize) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        if let Some(ref mut writer) = self.writer {
            for (tag, value) in values {
                writer.add_scalar(tag, *value as f32, step);
            }
        }
        Ok(())
    }
    
    /// Log a histogram of values
    /// 
    /// Note: This is a no-op in the current version of tensorboard-rs
    /// as histogram functionality is not fully implemented.
    #[allow(unused_variables)]
    pub fn log_histogram(&mut self, tag: &str, values: &[f64], step: usize) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        // Histogram not implemented in tensorboard-rs 0.5.x
        // Just log the mean as a scalar instead
        if !values.is_empty() {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            self.log_scalar(&format!("{}_mean", tag), mean, step)?;
        }
        Ok(())
    }
    
    /// Log training metrics from DynamicTrainingStats
    pub fn log_training_stats(
        &mut self,
        stats: &crate::dynamic_trainer::DynamicTrainingStats,
    ) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let step = stats.step;
        
        // Loss metrics
        self.log_scalar("train/loss", stats.loss, step)?;
        self.log_scalar("train/avg_loss", stats.avg_loss, step)?;
        
        // Learning rate
        self.log_scalar("train/learning_rate", stats.learning_rate, step)?;
        
        // Batch size
        self.log_scalar("train/batch_size", stats.batch_size as f64, step)?;
        
        // Coherence
        self.log_scalar("coherence/soc_score", stats.coherence, step)?;
        
        // Growth progress
        self.log_scalar("growth/progress", stats.growth_progress, step)?;
        
        // Gradient norm
        self.log_scalar("train/grad_norm", stats.grad_norm, step)?;
        
        // Per-layer learning rates
        for (i, &lr) in stats.layer_lrs.iter().enumerate() {
            self.log_scalar(&format!("layer_lr/layer_{}", i), lr, step)?;
        }
        
        // Per-layer EMA alphas
        for (i, &alpha) in stats.ema_alphas.iter().enumerate() {
            self.log_scalar(&format!("ema_alpha/layer_{}", i), alpha, step)?;
        }
        
        // Tokens per second (if available)
        if stats.tokens_per_sec > 0.0 {
            self.log_scalar("perf/tokens_per_sec", stats.tokens_per_sec, step)?;
        }
        
        Ok(())
    }
    
    /// Log coherence components
    pub fn log_coherence(
        &mut self,
        comprehensibility: f64,
        manageability: f64,
        meaningfulness: f64,
        step: usize,
    ) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        self.log_scalar("coherence/comprehensibility", comprehensibility, step)?;
        self.log_scalar("coherence/manageability", manageability, step)?;
        self.log_scalar("coherence/meaningfulness", meaningfulness, step)?;
        
        Ok(())
    }
    
    /// Flush the writer
    pub fn flush(&mut self) -> TorusResult<()> {
        if !self.enabled {
            return Ok(());
        }
        if let Some(ref mut writer) = self.writer {
            writer.flush();
        }
        Ok(())
    }
    
    /// Get the log directory
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }
    
    /// Check if logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Drop for MetricsLogger {
    fn drop(&mut self) {
        if self.enabled {
            let _ = self.flush();
        }
    }
}

/// Training metrics collector for periodic logging
#[derive(Debug, Default)]
pub struct MetricsCollector {
    /// Loss values in current window
    losses: Vec<f64>,
    /// Gradient norms in current window
    grad_norms: Vec<f64>,
    /// Throughput (tokens/sec) samples
    throughput: Vec<f64>,
    /// Window size for averaging
    window_size: usize,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(window_size: usize) -> Self {
        Self {
            losses: Vec::with_capacity(window_size),
            grad_norms: Vec::with_capacity(window_size),
            throughput: Vec::with_capacity(window_size),
            window_size,
        }
    }
    
    /// Record a training step
    pub fn record(&mut self, loss: f64, grad_norm: f64, tokens_per_sec: Option<f64>) {
        self.losses.push(loss);
        self.grad_norms.push(grad_norm);
        if let Some(tps) = tokens_per_sec {
            self.throughput.push(tps);
        }
        
        // Keep window size
        if self.losses.len() > self.window_size {
            self.losses.remove(0);
        }
        if self.grad_norms.len() > self.window_size {
            self.grad_norms.remove(0);
        }
        if self.throughput.len() > self.window_size {
            self.throughput.remove(0);
        }
    }
    
    /// Get average loss in window
    pub fn avg_loss(&self) -> f64 {
        if self.losses.is_empty() {
            0.0
        } else {
            self.losses.iter().sum::<f64>() / self.losses.len() as f64
        }
    }
    
    /// Get average gradient norm in window
    pub fn avg_grad_norm(&self) -> f64 {
        if self.grad_norms.is_empty() {
            0.0
        } else {
            self.grad_norms.iter().sum::<f64>() / self.grad_norms.len() as f64
        }
    }
    
    /// Get average throughput in window
    pub fn avg_throughput(&self) -> f64 {
        if self.throughput.is_empty() {
            0.0
        } else {
            self.throughput.iter().sum::<f64>() / self.throughput.len() as f64
        }
    }
    
    /// Get loss variance (for stability monitoring)
    pub fn loss_variance(&self) -> f64 {
        if self.losses.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_loss();
        self.losses.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.losses.len() as f64
    }
    
    /// Clear all metrics
    pub fn clear(&mut self) {
        self.losses.clear();
        self.grad_norms.clear();
        self.throughput.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(10);
        
        for i in 0..20 {
            collector.record(1.0 + i as f64 * 0.1, 0.5, Some(1000.0));
        }
        
        // Should only keep last 10
        assert!(collector.losses.len() <= 10);
        assert!(collector.avg_loss() > 0.0);
    }

    #[test]
    fn test_metrics_logger_disabled() {
        let mut logger = MetricsLogger::disabled();
        assert!(!logger.is_enabled());
        
        // Should not fail even when disabled
        logger.log_scalar("test", 1.0, 0).unwrap();
        logger.flush().unwrap();
    }
}
