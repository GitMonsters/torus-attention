# Torus Attention

**Attention mechanism on torus topology with vortex dynamics.**

Part of the [Enhanced Multi-PINNACLE Torus System](https://github.com/GitMonsters/enhanced-multi-pinnacle-torus-complete).

## Why Torus Topology?

| Property | Sphere | Flat | **Torus** |
|---|---|---|---|
| Singularities | ❌ Poles | ❌ Edges | ✅ None |
| Cyclical data | ❌ | ❌ | ✅ Natural |
| Gradient flow | ✗ Vanishes | ✗ Limited | ✅ Continuous |
| Long-range memory | ✗ | ✗ | ✅ Circulation loops |

## Features

- **Vortex dynamics** — dual-scale processing (local vortices + global circulation)
- **No singularities** — continuous manifold, smooth gradient flow
- **TinyGrad-native** — runs on any hardware (CPU/CUDA/Metal/ROCm)
- **Drop-in attention** — replaces standard softmax attention

## Quick Start

```python
from src.torus_attention_mechanism import TorusAttentionMechanism, TorusAttentionConfig

config = TorusAttentionConfig(
    d_model=512,
    n_heads=8,
    major_radius=1.0,
    minor_radius=0.3,
)
attn = TorusAttentionMechanism(config)
```

## Architecture

```
Input → Torus Coordinate Projection
           ↓
    Vortex Dynamics Layer
    (local circulation + global flow)
           ↓
    Dual-Scale Attention
    (major circle × minor circle)
           ↓
    Output Projection
```

## Related

- [torus-collider](https://github.com/GitMonsters/torus-collider) — full torus topology system
- [enhanced-multi-pinnacle-torus-complete](https://github.com/GitMonsters/enhanced-multi-pinnacle-torus-complete) — complete implementation

## License

MIT
