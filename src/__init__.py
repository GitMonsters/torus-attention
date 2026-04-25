"""
Torus Attention Module
======================

Attention mechanism on torus topology with vortex dynamics and residual streams.

Main Components:
- TorusMultiHeadAttention: Multi-head attention with torus + vortex dynamics
- TorusTransformerBlock: Pre-norm block with separate attention/FFN residual scales
- AttentionResidualStream: Cross-block highway for gradient flow
- TorusAttentionConfig: Configuration dataclass
- apply_torus_attention: Convenience function

Kimi Attention Residuals (paper-faithful, 2025):
- LayerAttentionResidual: cross-layer Q/K/V attention, zero-init alpha
- BlockAttentionResiduals: intra-block full attention + O(1) block summary
- build_kimi_model: convenience stacked-block builder

Parallel Compound Attention (compiled integration):
- ParallelCompoundAttention: N streams × BlockAttentionResiduals +
  CrossStreamCohesion + CompoundSummaryAccumulator
- build_parallel_compound: flat-hyperparameter constructor

Usage:
    from src import build_parallel_compound

    pca = build_parallel_compound(d_model=512, n_streams=4, n_blocks=3, block_size=4)
    out, metrics = pca.forward(x)   # x: [B, L, 512]
"""

from .torus_attention_mechanism import (
    TorusAttentionConfig,
    TorusPositionalEncoding,
    VortexAttentionHead,
    TorusMultiHeadAttention,
    AttentionResidualStream,
    TorusTransformerBlock,
    apply_torus_attention,
    # Kimi Attention Residuals (paper-faithful implementation)
    LayerAttentionResidual,
    BlockAttentionResiduals,
    build_kimi_model,
)
from .kimi_parallel_compound import (
    CrossStreamCohesion,
    CompoundSummaryAccumulator,
    ParallelCompoundAttention,
    build_parallel_compound,
)
from .advanced_torus_topology import AdvancedTorusConfig, TorusCoordinateSystem
from .tinygrad_compatibility import Sequential, MultiheadAttention

__all__ = [
    # Core torus attention
    'TorusAttentionConfig',
    'TorusPositionalEncoding',
    'VortexAttentionHead',
    'TorusMultiHeadAttention',
    'AttentionResidualStream',
    'TorusTransformerBlock',
    'apply_torus_attention',
    'AdvancedTorusConfig',
    'TorusCoordinateSystem',
    # Kimi Attention Residuals
    'LayerAttentionResidual',
    'BlockAttentionResiduals',
    'build_kimi_model',
    # Parallel Compound Attention (compiled)
    'CrossStreamCohesion',
    'CompoundSummaryAccumulator',
    'ParallelCompoundAttention',
    'build_parallel_compound',
]

__version__ = "1.3.0"