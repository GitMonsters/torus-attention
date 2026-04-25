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

Block AttnRes (Moonshot AI, arXiv:2603.15031) — paper-faithful:
- AttnResLayer:           single pseudo-query softmax over block summaries
- BlockAttentionResiduals: full stack of BlockAttnResLayer transformer layers
- build_kimi_model:       convenience builder

Parallel Streams:
- StreamBlockAttnRes:      apply inter-stream Block AttnRes over N streams
- ParallelCompoundAttention: intra + inter stream Block AttnRes
- build_parallel_compound: flat-hyperparameter constructor

Usage::

    from src import build_parallel_compound, build_kimi_model

    # Single-stream Block AttnRes:
    model = build_kimi_model(config, n_layers=24, block_size=8)
    out, metrics = model.forward(x)

    # Multi-stream:
    pca = build_parallel_compound(d_model=512, n_streams=4)
    out, metrics = pca.forward(x)
"""

from .torus_attention_mechanism import (
    TorusAttentionConfig,
    TorusPositionalEncoding,
    VortexAttentionHead,
    TorusMultiHeadAttention,
    AttentionResidualStream,
    TorusTransformerBlock,
    apply_torus_attention,
    # Block AttnRes (Moonshot AI, arXiv:2603.15031)
    _attn_res_block,
    AttnResLayer,
    BlockAttnResLayer,
    BlockAttentionResiduals,
    build_kimi_model,
    # Legacy alias
    LayerAttentionResidual,
)
from .kimi_parallel_compound import (
    StreamBlockAttnRes,
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
    # Block AttnRes — paper-faithful (Moonshot AI, arXiv:2603.15031)
    '_attn_res_block',
    'AttnResLayer',
    'BlockAttnResLayer',
    'BlockAttentionResiduals',
    'build_kimi_model',
    'LayerAttentionResidual',   # legacy alias for AttnResLayer
    # Parallel Streams
    'StreamBlockAttnRes',
    'ParallelCompoundAttention',
    'build_parallel_compound',
]

__version__ = "1.4.0"