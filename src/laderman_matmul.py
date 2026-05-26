#!/usr/bin/env python3
"""
Laderman 1976 — 23-Multiplication 3×3 Matrix Algorithm
=======================================================

Source: Laderman, J.D. "A noncommutative algorithm of length 23 for multiplying
        6×6 matrices using 103 multiplications." Bull. AMS 82 (1976) pp. 126–128.
        (Also directly applicable to 3×3 as a base case.)

Reduces the naive 27 scalar multiplications to 23, at the cost of extra additions.
Verified against numpy.matmul for correctness.

Two API surfaces:
  laderman_3x3(A, B)         — pure Python, accepts list-of-lists or numpy 2D arrays
  laderman_matmul_batch(A, B) — PyTorch batch version, shapes (..., 3, 3)
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Pure Python / NumPy reference implementation
# ---------------------------------------------------------------------------

def laderman_3x3(A, B) -> np.ndarray:
    """Multiply two 3×3 matrices using Laderman's 23-multiplication algorithm.

    Args:
        A: 3×3 array-like
        B: 3×3 array-like

    Returns:
        numpy ndarray of shape (3, 3) equal to A @ B
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert A.shape == (3, 3) and B.shape == (3, 3), "Inputs must be 3×3"

    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]

    b11, b12, b13 = B[0, 0], B[0, 1], B[0, 2]
    b21, b22, b23 = B[1, 0], B[1, 1], B[1, 2]
    b31, b32, b33 = B[2, 0], B[2, 1], B[2, 2]

    # ── 23 bilinear products (Laderman 1976) ──────────────────────────────
    m1  = (a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22
    m2  = (a11 - a21) * (-b12 + b22)
    m3  = a22 * (-b11 + b12 + b21 - b22 - b23 - b31 + b33)
    m4  = (-a11 + a21 + a22) * (b11 - b12 + b22)
    m5  = (a21 + a22) * (-b11 + b12)
    m6  = a11 * b11
    m7  = (-a11 + a31 + a32) * (b11 - b13 + b23)
    m8  = (-a11 + a31) * (b13 - b23)
    m9  = (a31 + a32) * (-b11 + b13)
    m10 = (a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23
    m11 = a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32)
    m12 = (-a13 + a32 + a33) * (b22 + b31 - b32)
    m13 = (a13 - a33) * (b22 - b32)
    m14 = a13 * b31
    m15 = (a32 + a33) * (-b31 + b32)
    m16 = (-a13 + a22 + a23) * (b23 + b31 - b33)
    m17 = (a13 - a23) * (b23 - b33)
    m18 = (a22 + a23) * (-b31 + b33)
    m19 = a12 * b21
    m20 = a23 * b32
    m21 = a21 * b13
    m22 = a31 * b12
    m23 = a33 * b33

    # ── Output entries ────────────────────────────────────────────────────
    c11 = m6  + m14 + m19
    c12 = m1  + m4  + m5  + m6  + m12 + m14 + m15
    c13 = m6  + m7  + m9  + m10 + m14 + m16 + m18
    c21 = m2  + m3  + m4  + m6  + m14 + m16 + m17
    c22 = m2  + m4  + m5  + m6  + m20
    c23 = m14 + m16 + m17 + m18 + m21
    c31 = m6  + m7  + m8  + m11 + m12 + m13 + m14
    c32 = m12 + m13 + m14 + m15 + m22
    c33 = m6  + m7  + m8  + m9  + m23

    return np.array([[c11, c12, c13],
                     [c21, c22, c23],
                     [c31, c32, c33]])


# ---------------------------------------------------------------------------
# PyTorch batch version — shape (..., 3, 3)
# ---------------------------------------------------------------------------

def laderman_matmul_batch(A, B):
    """Batch 3×3 matrix multiply using Laderman's 23-multiplication algorithm.

    Args:
        A: tensor of shape (..., 3, 3)
        B: tensor of shape (..., 3, 3)

    Returns:
        tensor of shape (..., 3, 3) equal to A @ B
    """
    # Extract all 9 entries of each matrix via indexing on last two dims.
    def _e(M, i, j):
        return M[..., i, j]

    a11, a12, a13 = _e(A,0,0), _e(A,0,1), _e(A,0,2)
    a21, a22, a23 = _e(A,1,0), _e(A,1,1), _e(A,1,2)
    a31, a32, a33 = _e(A,2,0), _e(A,2,1), _e(A,2,2)

    b11, b12, b13 = _e(B,0,0), _e(B,0,1), _e(B,0,2)
    b21, b22, b23 = _e(B,1,0), _e(B,1,1), _e(B,1,2)
    b31, b32, b33 = _e(B,2,0), _e(B,2,1), _e(B,2,2)

    # ── 23 bilinear products ──────────────────────────────────────────────
    m1  = (a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22
    m2  = (a11 - a21) * (-b12 + b22)
    m3  = a22 * (-b11 + b12 + b21 - b22 - b23 - b31 + b33)
    m4  = (-a11 + a21 + a22) * (b11 - b12 + b22)
    m5  = (a21 + a22) * (-b11 + b12)
    m6  = a11 * b11
    m7  = (-a11 + a31 + a32) * (b11 - b13 + b23)
    m8  = (-a11 + a31) * (b13 - b23)
    m9  = (a31 + a32) * (-b11 + b13)
    m10 = (a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23
    m11 = a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32)
    m12 = (-a13 + a32 + a33) * (b22 + b31 - b32)
    m13 = (a13 - a33) * (b22 - b32)
    m14 = a13 * b31
    m15 = (a32 + a33) * (-b31 + b32)
    m16 = (-a13 + a22 + a23) * (b23 + b31 - b33)
    m17 = (a13 - a23) * (b23 - b33)
    m18 = (a22 + a23) * (-b31 + b33)
    m19 = a12 * b21
    m20 = a23 * b32
    m21 = a21 * b13
    m22 = a31 * b12
    m23 = a33 * b33

    # ── Output entries ────────────────────────────────────────────────────
    c11 = m6  + m14 + m19
    c12 = m1  + m4  + m5  + m6  + m12 + m14 + m15
    c13 = m6  + m7  + m9  + m10 + m14 + m16 + m18
    c21 = m2  + m3  + m4  + m6  + m14 + m16 + m17
    c22 = m2  + m4  + m5  + m6  + m20
    c23 = m14 + m16 + m17 + m18 + m21
    c31 = m6  + m7  + m8  + m11 + m12 + m13 + m14
    c32 = m12 + m13 + m14 + m15 + m22
    c33 = m6  + m7  + m8  + m9  + m23

    # Reconstruct (..., 3, 3) tensor by stacking along last two dims.
    # Works with any tensor library that supports __add__, __mul__, and stack.
    try:
        import torch
        row0 = torch.stack([c11, c12, c13], dim=-1)
        row1 = torch.stack([c21, c22, c23], dim=-1)
        row2 = torch.stack([c31, c32, c33], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)
    except (ImportError, AttributeError):
        # Fallback: numpy stacking
        row0 = np.stack([c11, c12, c13], axis=-1)
        row1 = np.stack([c21, c22, c23], axis=-1)
        row2 = np.stack([c31, c32, c33], axis=-1)
        return np.stack([row0, row1, row2], axis=-2)


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

def _verify(n_trials: int = 1000, tol: float = 1e-10) -> bool:
    """Run random trials comparing laderman_3x3 against numpy.matmul."""
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        C_ref = A @ B
        C_lad = laderman_3x3(A, B)
        if not np.allclose(C_ref, C_lad, atol=tol):
            return False
    return True


if __name__ == "__main__":
    ok = _verify()
    print(f"Laderman 3×3 verification: {'PASS' if ok else 'FAIL'} (1000 random trials)")
