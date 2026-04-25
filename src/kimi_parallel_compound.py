#!/usr/bin/env python3
"""
Kimi Block AttnRes — Applied to Parallel Cognitive Streams
===========================================================

Paper: Attention Residuals (Moonshot AI, arXiv:2603.15031)
Repo:  https://github.com/MoonshotAI/Attention-Residuals

Core mechanism (Block AttnRes, paper §3.2):

    V = stack(blocks + [partial])          # [N+1, B, T, D]
    K = norm(V)
    logits = einsum('d, nbTd -> nbT', w, K)  # w = learned pseudo-query
    h = einsum('nbT, nbTd -> bTd', softmax(logits, dim=0), V)

Applied TWICE per transformer layer: before attention and before MLP.

Parallel streams variant:
    - Each cognitive stream is treated as one "completed block"
    - Stream i attends over all completed streams 0..i-1 + its own partial
    - Equivalent to N rounds of inter-block attention, one per stream
    - Output: N updated stream tensors with selective cross-stream information

Architecture::

    Input: N streams, each [B, L, D]
                      │
    Stream 0  ────────┤  (first stream has no history → h = stream_0)
    Stream 1  ─ block_attn_res([stream_0],          stream_1)
    Stream 2  ─ block_attn_res([stream_0, stream_1], stream_2)
    ...
    Stream N  ─ block_attn_res([s0..s_{N-1}],       stream_N)
                      │
    Output: N updated streams [B, L, D]

Complexity: O(N × L) — single dot product per stream (no L² attention).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from tinygrad.tensor import Tensor
from tinygrad import nn

from .torus_attention_mechanism import (
    TorusAttentionConfig,
    _attn_res_block,
    AttnResLayer,
    BlockAttentionResiduals,
    build_kimi_model,
)


class StreamBlockAttnRes(object):
    """
    Apply Block AttnRes over N parallel cognitive streams.

    Treats each completed stream output as a block summary.  Stream i
    attends over streams 0..i-1 (completed blocks) + its own value (partial)
    using a single learned pseudo-query per stream.

    One AttnResLayer per stream (two proj/norm pairs: before-attn + before-mlp).
    The 'before-attn' path is used to update stream input; the 'before-mlp'
    path is exposed for callers who want to apply it at a second point.
    """

    def __init__(self, d_model: int, n_streams: int):
        self.n_streams = n_streams
        # One AttnResLayer per stream
        self.layers = [AttnResLayer(d_model) for _ in range(n_streams)]

    def forward(self, streams: List[Tensor]) -> Tuple[List[Tensor], Dict]:
        """
        Args:
            streams: List[Tensor[B, T, D]], one per cognitive stream.

        Returns:
            updated:  List[Tensor[B, T, D]] — each stream after block AttnRes.
            metrics:  Dict with n_streams.
        """
        completed: List[Tensor] = []
        updated: List[Tensor] = []

        for i, s in enumerate(streams):
            if completed:
                h = self.layers[i].before_attn(completed, s)
            else:
                h = s  # first stream — no history
            updated.append(h)
            completed.append(h)

        return updated, {'n_streams': self.n_streams, 'method': 'block_attn_res'}


class ParallelCompoundAttention(object):
    """
    Block AttnRes applied to N parallel cognitive streams (TinyGrad).

    Applies the Moonshot AI Block AttnRes mechanism across N streams,
    treating each stream as a completed block in the inter-block attention.

    Also exposes a full BlockAttentionResiduals stack for single-stream use.

    Parameters
    ----------
    config    : TorusAttentionConfig
    n_streams : Number of parallel streams.
    n_layers  : Transformer layers per stream (for intra-stream BlockAttentionResiduals).
    block_size: Sub-layers per block (attn+MLP counts as 2).

    Usage
    -----
    ::

        config = TorusAttentionConfig(d_model=512)
        pca = ParallelCompoundAttention(config, n_streams=4)

        streams = [x] * 4   # or different projections of x
        updated, metrics = pca.forward(streams)
    """

    def __init__(
        self,
        config: TorusAttentionConfig,
        n_streams: int = 4,
        n_layers: int = 12,
        block_size: int = 8,
    ):
        self.config = config
        self.n_streams = n_streams
        d_model = config.d_model

        # Intra-stream Block AttnRes stacks (one full stack per stream)
        self.intra_stream = [
            BlockAttentionResiduals(config, n_layers=n_layers, block_size=block_size)
            for _ in range(n_streams)
        ]

        # Inter-stream Block AttnRes (cross-stream fusion)
        self.inter_stream = StreamBlockAttnRes(d_model, n_streams)

        # Stream input projections (diversify)
        self.stream_in = [
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_streams)
        ]
        self.stream_in_norms = [nn.LayerNorm(d_model) for _ in range(n_streams)]

        # Output merge
        self.output_proj = nn.Linear(d_model * n_streams, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """
        Args:
            x: Input [B, L, D].

        Returns:
            output:  Unified tensor [B, L, D].
            metrics: Diagnostic dict.
        """
        metrics: Dict = {}

        # Diversify input into N streams
        stream_ins = [
            norm(proj(x))
            for proj, norm in zip(self.stream_in, self.stream_in_norms)
        ]

        # Intra-stream: each stream runs through its own Block AttnRes stack
        intra_outs = []
        for i, (si, stack) in enumerate(zip(stream_ins, self.intra_stream)):
            out, m = stack.forward(si)
            intra_outs.append(out)
            metrics[f'stream_{i}'] = m

        # Inter-stream: apply Block AttnRes across streams
        updated, inter_m = self.inter_stream.forward(intra_outs)
        metrics['inter_stream'] = inter_m

        # Merge streams
        merged = Tensor.cat(updated, dim=-1)          # [B, L, D * n_streams]
        output = self.output_norm(self.output_proj(merged))

        metrics['n_streams'] = self.n_streams
        return output, metrics


def build_parallel_compound(
    d_model: int = 512,
    n_heads: int = 8,
    n_streams: int = 4,
    n_layers: int = 12,
    block_size: int = 8,
    **torus_kwargs,
) -> ParallelCompoundAttention:
    """
    Build a ParallelCompoundAttention from flat hyperparameters.

    Example::

        pca = build_parallel_compound(d_model=512, n_streams=4, n_layers=24)
        out, metrics = pca.forward(tokens)  # tokens: [B, L, 512]
    """
    config = TorusAttentionConfig(d_model=d_model, n_heads=n_heads, **torus_kwargs)
    return ParallelCompoundAttention(config, n_streams, n_layers, block_size)
