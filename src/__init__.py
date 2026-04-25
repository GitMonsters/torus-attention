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

Usage:
    from src import TorusTransformerBlock, AttentionResidualStream, TorusAttentionConfig

    config = TorusAttentionConfig(d_model=512, n_heads=8)
    stream = AttentionResidualStream(config.d_model, n_blocks=12)
    stream.reset()

    blocks = [TorusTransformerBlock(config, stream, i) for i in range(12)]
"""

from .torus_attention_mechanism import (
    TorusAttentionConfig,
    TorusPositionalEncoding,
    VortexAttentionHead,
    TorusMultiHeadAttention,
    AttentionResidualStream,
    TorusTransformerBlock,
    apply_torus_attention,
)
from .advanced_torus_topology import AdvancedTorusConfig, TorusCoordinateSystem
from .tinygrad_compatibility import Sequential, MultiheadAttention

__all__ = [
    'TorusAttentionConfig',
    'TorusPositionalEncoding',
    'VortexAttentionHead',
    'TorusMultiHeadAttention',
    'AttentionResidualStream',
    'TorusTransformerBlock',
    'apply_torus_attention',
    'AdvancedTorusConfig',
    'TorusCoordinateSystem',
]

__version__ = "1.1.0"