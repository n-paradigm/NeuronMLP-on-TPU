"""Numpy/JAX reference implementations of the SVD-Flash MLP.

These are the golden implementations that the Pallas kernels in this package
must match. They mirror the math of NeuronMM's NKI kernels:

  fused_gate_up : gate = X @ Ug @ Vg ; up = X @ Uu @ Vu ; out = silu(gate) * up
  fused_xuv     : out  = X @ Ud @ Vd

Shapes follow the SVD_LlamaMLP layout in
NeuronMM/models/llama/modeling_llama.py:1250, where a dense weight W (out, in)
is factored into u (out, r) and v (r, in) with W ~= u @ v.

Nothing here is performance-sensitive; correctness and readability win.
"""

from __future__ import annotations

import numpy as np


def low_rank_dim(hidden_size: int, intermediate_size: int, compress_ratio: float) -> int:
    """Rank used by SVD-Flash, aligned to a multiple of 128.

    This is the exact formula from NeuronMM (llama_inference.py:693 and
    modeling_llama.py:2464). The 128 alignment matters because both Trainium's
    PE array and the TPU MXU are 128-wide; an unaligned rank wastes a fraction
    of every systolic pass.

    For ratio=0.8 this reproduces the hardcoded values in
    NeuronMM/test_speedup.py: 1280 (1B), 1792 (3B), 2560 (8B).
    """
    numer = intermediate_size * hidden_size * compress_ratio
    denom = (intermediate_size + hidden_size) * 128
    return round(numer / denom) * 128


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish, computed in float32 to avoid bf16 error in the exp.

    Uses the branch-stable form of the sigmoid. The naive 1/(1+exp(-x))
    overflows exp for x << 0; the result happens to still be correct (x/inf
    rounds to 0) but it raises warnings and relies on inf arithmetic. Splitting
    on the sign keeps every exp argument <= 0.
    """
    x32 = x.astype(np.float32)
    pos = x32 >= 0
    # exp arg is <= 0 in both branches, so no overflow is possible.
    z_neg = np.exp(np.where(pos, -np.inf, x32))        # exp(x) for x < 0
    z_pos = np.exp(np.where(pos, -x32, -np.inf))       # exp(-x) for x >= 0
    sigmoid = np.where(pos, 1.0 / (1.0 + z_pos), z_neg / (1.0 + z_neg))
    return x32 * sigmoid


def fused_gate_up_ref(
    x: np.ndarray,      # (S, H)
    u_gate: np.ndarray,  # (H, r)
    v_gate: np.ndarray,  # (r, I)
    u_up: np.ndarray,    # (H, r)
    v_up: np.ndarray,    # (r, I)
) -> np.ndarray:
    """Returns (S, I) = silu(X @ Ug @ Vg) * (X @ Uu @ Vu).

    Note the output layout differs from NKI's fused_mlp_up_T, which returns
    (I, S) because Trainium wanted the transposed layout to feed the next
    kernel. On TPU we keep the natural (S, I) layout and let the down kernel
    consume it directly -- there is no partition-dim constraint forcing a
    transpose.
    """
    acc = np.float32
    gate = (x.astype(acc) @ u_gate.astype(acc)) @ v_gate.astype(acc)
    up = (x.astype(acc) @ u_up.astype(acc)) @ v_up.astype(acc)
    return (silu(gate) * up).astype(x.dtype)


def fused_xuv_ref(
    x: np.ndarray,   # (S, I)
    u: np.ndarray,   # (I, r)
    v: np.ndarray,   # (r, H)
) -> np.ndarray:
    """Returns (S, H) = X @ U @ V."""
    acc = np.float32
    return ((x.astype(acc) @ u.astype(acc)) @ v.astype(acc)).astype(x.dtype)


def svd_mlp_ref(
    x: np.ndarray,
    u_gate: np.ndarray, v_gate: np.ndarray,
    u_up: np.ndarray, v_up: np.ndarray,
    u_down: np.ndarray, v_down: np.ndarray,
) -> np.ndarray:
    """Full SVD SwiGLU block: the reference for the two-kernel pipeline."""
    h = fused_gate_up_ref(x, u_gate, v_gate, u_up, v_up)
    return fused_xuv_ref(h, u_down, v_down)


def dense_mlp_ref(
    x: np.ndarray,
    w_gate: np.ndarray,  # (H, I)
    w_up: np.ndarray,    # (H, I)
    w_down: np.ndarray,  # (I, H)
) -> np.ndarray:
    """Uncompressed SwiGLU, for the dense arm of the three-way benchmark."""
    acc = np.float32
    gate = x.astype(acc) @ w_gate.astype(acc)
    up = x.astype(acc) @ w_up.astype(acc)
    return ((silu(gate) * up) @ w_down.astype(acc)).astype(x.dtype)
