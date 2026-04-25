#!/usr/bin/env python3
"""
Parallel Compound Attention
===========================

Compiles LayerAttentionResidual, BlockAttentionResiduals, and build_kimi_model
into a single unified module that operates under three principles:

    PRINCIPLE 1 — Think in parallel
        N independent BlockAttentionResiduals streams decompose complexity
        simultaneously.  Each stream processes the same input through its own
        chain of cross-layer-attention blocks.  No stream sees another's
        internals during its own computation (true parallelism).

    PRINCIPLE 2 — Integrate with cohesion
        After each block level, stream outputs are unified via a cross-stream
        LayerAttentionResidual.  Each stream attends over ALL other streams'
        block summaries, then updates its own representation.  This is the
        binding step — streams learn what to borrow from each other.

    PRINCIPLE 3 — Compound insight
        Block summaries from level k become additional context (history) for
        level k+1 across ALL streams.  Insight is not discarded at block
        boundaries; it is carried forward and available to every subsequent
        block.  The compounding is multiplicative: stream × block summaries
        attend together.

Architecture diagram::

    Input [B, L, D]
        │
        ├─ Stream 0 ─ Block_0_0 ─┬─ Block_1_0 ─┬─ Block_2_0 ─┐
        ├─ Stream 1 ─ Block_0_1 ─┤ cross-stream ├─ Block_2_1 ─┤ cross-stream → output
        ├─ Stream 2 ─ Block_0_2 ─┤ integration  ├─ Block_2_2 ─┤
        └─ Stream N ─ Block_0_N ─┘              └─ Block_2_N ─┘
                           ↑                          ↑
                       summaries                  summaries compound
                       broadcast                  from level 0+1
                       to level 1

Complexity:
    Time:  O(n_streams * n_blocks * block_size * L²)
    Memory: O(n_streams * n_blocks * block_size * L * D)
    Inter-block comms: O(1) per block boundary (inherited from BlockAttentionResiduals)

Reported results on base LLMs (Kimi paper, 2025):
    • 1.25× compute reduction for same performance
    • +7.5 pts GPQA Diamond
    • Bounded signal magnitude, even gradient distribution
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from tinygrad.tensor import Tensor
from tinygrad import nn

from .torus_attention_mechanism import (
    TorusAttentionConfig,
    LayerAttentionResidual,
    BlockAttentionResiduals,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Stream Cohesion Layer
# ─────────────────────────────────────────────────────────────────────────────

class CrossStreamCohesion(object):
    """
    Binds N parallel streams into a coherent shared representation.

    Each stream queries all other streams' block summaries via
    LayerAttentionResidual, updating its own hidden state.  This is the
    "integrate with cohesion" step — streams share relevant insight without
    collapsing into a single stream.

    Runs once per block level (after all streams finish their block).
    """

    def __init__(self, d_model: int, n_streams: int, n_heads: int = 8):
        self.n_streams = n_streams
        # Each stream has its own cross-stream attention module
        self.cross_attn = [
            LayerAttentionResidual(d_model, n_heads=n_heads)
            for _ in range(n_streams)
        ]
        # Post-cohesion gate: controls how much cross-stream info to absorb
        self.cohesion_gate = nn.Linear(d_model, d_model, bias=False)
        self.cohesion_norm = nn.LayerNorm(d_model)
        self.cohesion_scale = nn.Parameter(Tensor.ones(1) * 0.5)

    def forward(
        self,
        stream_hiddens: List[Tensor],
        stream_summaries: List[Tensor],
    ) -> List[Tensor]:
        """
        Args:
            stream_hiddens:   List[Tensor[B, L, D]] — current hidden state per stream.
            stream_summaries: List[Tensor[B, L, D]] — block summary per stream.

        Returns:
            List[Tensor[B, L, D]] — updated hidden states after cross-stream binding.
        """
        updated = []
        for i, h in enumerate(stream_hiddens):
            # Other streams' summaries as history for this stream
            other_summaries = [s for j, s in enumerate(stream_summaries) if j != i]
            if not other_summaries:
                updated.append(h)
                continue
            # Cross-stream attention: this stream attends over all others
            h_cross = self.cross_attn[i].forward(h, other_summaries)
            # Cohesion gate: learned blend
            gate = self.cohesion_gate(h_cross).sigmoid()
            h_bound = self.cohesion_norm(h + self.cohesion_scale * gate * h_cross)
            updated.append(h_bound)
        return updated


# ─────────────────────────────────────────────────────────────────────────────
# Compound Summary Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class CompoundSummaryAccumulator(object):
    """
    Compounds block summaries across levels and streams.

    At each block level, the summaries from ALL previous levels × ALL streams
    are available as additional context.  A learned attention mechanism selects
    what to carry forward, preventing unbounded accumulation while preserving
    useful long-range insight.

    This embodies "compound insight at every step."
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        self.attn = LayerAttentionResidual(d_model, n_heads=n_heads)
        self.compress = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        current_summary: Tensor,
        history: List[Tensor],
    ) -> Tensor:
        """
        Args:
            current_summary: Summary from the just-completed block level [B, L, D].
            history:         All previous summaries (any level/stream) [B, L, D] each.

        Returns:
            Enriched summary [B, L, D] that compounds all prior insight.
        """
        if not history:
            return current_summary
        enriched = self.attn.forward(current_summary, history)
        return self.norm(self.compress(enriched) + current_summary)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Compound Attention  (the compiled module)
# ─────────────────────────────────────────────────────────────────────────────

class ParallelCompoundAttention(object):
    """
    Compiled Kimi Attention Residuals with parallel streams and compound binding.

    Brings together:
        LayerAttentionResidual  → within each block (cross-layer)
        BlockAttentionResiduals → within each stream (block boundary compression)
        CrossStreamCohesion     → between streams (cohesion after each block level)
        CompoundSummaryAccumulator → across block levels (compounding insight)

    Parameters
    ----------
    config      : TorusAttentionConfig  — shared attention config for all blocks.
    n_streams   : int  — number of parallel processing streams.
    n_blocks    : int  — number of block levels per stream.
    block_size  : int  — layers per block (cross-layer attention within block).

    Usage
    -----
    ::

        config = TorusAttentionConfig(d_model=512, n_heads=8)
        pca = ParallelCompoundAttention(config, n_streams=4, n_blocks=3, block_size=4)

        x = embed(tokens)            # [B, L, D]
        out, metrics = pca.forward(x)
        # out: [B, L, D]  unified representation
    """

    def __init__(
        self,
        config: TorusAttentionConfig,
        n_streams: int = 4,
        n_blocks: int = 3,
        block_size: int = 4,
    ):
        self.config = config
        self.n_streams = n_streams
        self.n_blocks = n_blocks
        self.block_size = block_size
        d_model = config.d_model
        n_heads = config.n_heads

        # PRINCIPLE 1: n_streams × n_blocks independent BlockAttentionResiduals
        # streams[stream_idx][block_idx]
        self.streams: List[List[BlockAttentionResiduals]] = [
            [BlockAttentionResiduals(config, block_size=block_size)
             for _ in range(n_blocks)]
            for _ in range(n_streams)
        ]

        # PRINCIPLE 2: cross-stream cohesion after each block level
        self.cohesion_layers = [
            CrossStreamCohesion(d_model, n_streams, n_heads=n_heads)
            for _ in range(n_blocks)
        ]

        # PRINCIPLE 3: compound summary accumulator (shared across all streams/levels)
        self.accumulator = CompoundSummaryAccumulator(d_model, n_heads=n_heads)

        # Stream input projections (diversify streams from same input)
        self.stream_in_proj = [
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_streams)
        ]
        self.stream_in_norms = [nn.LayerNorm(d_model) for _ in range(n_streams)]

        # Output: merge all stream final outputs
        self.output_proj = nn.Linear(d_model * n_streams, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Full parallel compound forward pass.

        Step-by-step:
            For each block level b in 0..n_blocks:
                For each stream s in 0..n_streams (parallel):
                    h_s, summary_s = streams[s][b](h_s)           # block forward
                    summary_s = accumulator(summary_s, history)   # compound
                Cross-stream cohesion: each stream attends others  # integrate

        Args:
            x    : Input tensor [B, L, D].
            mask : Optional attention mask.

        Returns:
            output  : Unified tensor [B, L, D].
            metrics : Diagnostic dict.
        """
        metrics: Dict = {}

        # ── Initialise per-stream hidden states (diversified projections) ───
        stream_hiddens: List[Tensor] = [
            norm(proj(x))
            for proj, norm in zip(self.stream_in_proj, self.stream_in_norms)
        ]

        # Global summary history (all streams × all block levels)
        global_summary_history: List[Tensor] = []

        # ── Main loop: block levels ──────────────────────────────────────────
        for b in range(self.n_blocks):
            block_summaries: List[Tensor] = []
            block_hiddens: List[Tensor] = []
            level_metrics: Dict = {}

            # PRINCIPLE 1 — Think in parallel
            for s in range(self.n_streams):
                h, summary, blk_m = self.streams[s][b].forward(
                    stream_hiddens[s], mask
                )
                # PRINCIPLE 3 — Compound insight: enrich summary with history
                summary = self.accumulator.forward(summary, global_summary_history)

                block_hiddens.append(h)
                block_summaries.append(summary)
                global_summary_history.append(summary)
                level_metrics[f's{s}_alpha'] = blk_m.get('block_alpha_mean', 0.0)

            # PRINCIPLE 2 — Integrate with cohesion
            stream_hiddens = self.cohesion_layers[b].forward(
                block_hiddens, block_summaries
            )

            metrics[f'block_{b}'] = level_metrics

        # ── Output: merge streams ────────────────────────────────────────────
        merged = Tensor.cat(stream_hiddens, dim=-1)   # [B, L, D * n_streams]
        output = self.output_norm(self.output_proj(merged))

        metrics['n_streams'] = self.n_streams
        metrics['n_blocks'] = self.n_blocks
        metrics['block_size'] = self.block_size
        metrics['total_layers'] = self.n_streams * self.n_blocks * self.block_size
        metrics['summary_history_len'] = len(global_summary_history)

        return output, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructor
# ─────────────────────────────────────────────────────────────────────────────

def build_parallel_compound(
    d_model: int = 512,
    n_heads: int = 8,
    n_streams: int = 4,
    n_blocks: int = 3,
    block_size: int = 4,
    **torus_kwargs,
) -> ParallelCompoundAttention:
    """
    Build a ParallelCompoundAttention from flat hyperparameters.

    Equivalent effective depth = n_streams × n_blocks × block_size layers.
    With n_streams=4, n_blocks=3, block_size=4 → 48 effective layers.

    Example::

        pca = build_parallel_compound(d_model=512, n_streams=4)
        out, metrics = pca.forward(tokens)  # tokens: [B, L, 512]
    """
    config = TorusAttentionConfig(
        d_model=d_model,
        n_heads=n_heads,
        **torus_kwargs,
    )
    return ParallelCompoundAttention(config, n_streams, n_blocks, block_size)
