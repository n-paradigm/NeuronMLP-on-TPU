"""Whole-model transformer E2E benchmark: fused SVD MLP vs unfused SVD MLP.

Both paths use identical SVD-factored weights (U_gate, V_gate, U_up, V_up,
U_down, V_down).  The only difference is *how* those three matmuls are
executed:

  unfused  — standard JAX matmuls run sequentially by XLA
  fused    — Pallas kernels that fuse Gate/Up/SwiGLU + Down into fewer
              memory passes

This matches the paper's claim: the Pallas kernel is faster than an
equivalent unfused SVD MLP for the same compressed weights.  Attention is
identical in both paths and is not the subject of comparison.

The benchmark runs a full N-layer prefill forward pass (no KV cache needed;
prefill processes the entire sequence in one shot), matching NeuronMLP's
context-encoding phase.

Usage (on TPU):
    python svdflash_tpu/bench/e2e_block.py \\
        --model llama3-3b --seq-lens 128 512 1024 --warmup 3 --repeats 10

    # Sanity check with 2 layers instead of 28:
    python svdflash_tpu/bench/e2e_block.py --seq-lens 128 --layers 2

Output: one JSON record per sequence length with whole-model latency for each
path and speedup_fused_vs_unfused = unfused_avg / fused_avg.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.reference import low_rank_dim
from svdflash_tpu.models.hf_svd import load_mlp_factors, checkpoint_metadata
from svdflash_tpu.test_speedup import (
    _positive,
    get_fused_mlp_up_T_params,
    get_fused_three_mm_XTUV_params,
    nki_matmul_fully_optimized_,
    svd_mlp_with_fused_kernel,
)

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

_SHAPES = {
    "llama3-3b": dict(hidden=3072, intermediate=8192, heads=24, kv_heads=8, layers=28),
    "llama3-8b": dict(hidden=4096, intermediate=14336, heads=32, kv_heads=8, layers=32),
}

_DEFAULT_COMPRESS_RATIO = 0.8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rand(key: jax.Array, shape: tuple, scale: float = 1.0) -> jax.Array:
    return (jax.random.normal(key, shape, dtype=jnp.bfloat16) * scale).astype(
        jnp.bfloat16
    )


def _rmsnorm(x: jax.Array, w: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + 1e-6)
    return (x32 / rms * w.astype(jnp.float32)).astype(x.dtype)


# ---------------------------------------------------------------------------
# Weight initialisation — SVD factors only, no dense MLP weights
# ---------------------------------------------------------------------------

# Key order must be stable; it is used to reconstruct per-layer dicts inside JIT.
_WEIGHT_KEYS = (
    "norm1_w", "norm2_w",
    "wq", "wk", "wv", "wo",
    "ug", "vg", "uu", "vu", "ud", "vd",
)
_N_WEIGHTS = len(_WEIGHT_KEYS)


def _make_layer_weights(
    key: jax.Array, H: int, I: int, R: int, num_heads: int, num_kv_heads: int
) -> dict:
    """Random weights for one transformer layer (SVD-factored MLP only)."""
    head_dim = H // num_heads
    kv_dim   = num_kv_heads * head_dim
    keys = jax.random.split(key, _N_WEIGHTS)
    s = lambda d: 1.0 / math.sqrt(d)  # noqa: E731
    values = (
        jnp.ones(H, dtype=jnp.bfloat16),        # norm1_w
        jnp.ones(H, dtype=jnp.bfloat16),        # norm2_w
        _rand(keys[2],  (H, H),      s(H)),      # wq
        _rand(keys[3],  (H, kv_dim), s(H)),      # wk
        _rand(keys[4],  (H, kv_dim), s(H)),      # wv
        _rand(keys[5],  (H, H),      s(H)),      # wo
        _rand(keys[6],  (H, R),      s(H)),      # ug
        _rand(keys[7],  (R, I),      s(R)),      # vg
        _rand(keys[8],  (H, R),      s(H)),      # uu
        _rand(keys[9],  (R, I),      s(R)),      # vu
        _rand(keys[10], (I, R),      s(I)),      # ud
        _rand(keys[11], (R, H),      s(R)),      # vd
    )
    return dict(zip(_WEIGHT_KEYS, values))


def _load_checkpoint_weights(
    model_path: str,
    layer_indices: list[int],
    dtype: jnp.dtype = jnp.bfloat16,
) -> list[dict]:
    """Load real weights for selected layers from an SVD-decomposed HF checkpoint.

    MLP factors (ug/vg/uu/vu/ud/vd) come from load_mlp_factors.
    Attention projections and RMSNorm weights are read directly from safetensors.

    PyTorch Linear stores weights as (output, input); we transpose to (input, output)
    so that x @ w works in the standard JAX matmul convention.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "loading real weights requires safetensors: "
            "python -m pip install safetensors"
        ) from exc

    from pathlib import Path
    import json as _json

    model_path = Path(model_path)
    config_path = model_path / "config.json"
    index_path  = model_path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")

    weight_map = _json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shard_names = sorted({weight_map[k] for k in weight_map
                          if any(f"layers.{l}." in k for l in layer_indices)})

    # Load attention + norm keys for all requested layers at once.
    attn_keys = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
    raw: dict[str, jax.Array] = {}
    with __import__("contextlib").ExitStack() as stack:
        shards = {
            name: stack.enter_context(
                safe_open(str(model_path / name), framework="flax")
            )
            for name in shard_names
        }
        for layer in layer_indices:
            for suffix in attn_keys:
                key = f"model.layers.{layer}.{suffix}"
                if key not in weight_map:
                    raise KeyError(f"checkpoint missing key: {key}")
                t = shards[weight_map[key]].get_tensor(key)
                # Transpose 2-D projection weights; leave 1-D norm weights as-is.
                raw[key] = jax.device_put(t.T if t.ndim == 2 else t)

    jax.block_until_ready(list(raw.values()))

    # Load SVD MLP factors via the existing loader (returns ug,vg,uu,vu,ud,vd).
    mlp_by_layer = {
        idx: factors
        for idx, factors in zip(layer_indices, load_mlp_factors(model_path, layer_indices))
    }

    weights = []
    for layer in layer_indices:
        pfx = f"model.layers.{layer}."
        ug, vg, uu, vu, ud, vd = (f.astype(dtype) for f in mlp_by_layer[layer])
        w = dict(
            norm1_w=raw[pfx + "input_layernorm.weight"].astype(dtype),
            norm2_w=raw[pfx + "post_attention_layernorm.weight"].astype(dtype),
            wq=raw[pfx + "self_attn.q_proj.weight"].astype(dtype),
            wk=raw[pfx + "self_attn.k_proj.weight"].astype(dtype),
            wv=raw[pfx + "self_attn.v_proj.weight"].astype(dtype),
            wo=raw[pfx + "self_attn.o_proj.weight"].astype(dtype),
            ug=ug, vg=vg, uu=uu, vu=vu, ud=ud, vd=vd,
        )
        jax.block_until_ready(list(w.values()))
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Attention (GQA, prefill)
# ---------------------------------------------------------------------------

def _attention(
    x: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    num_heads: int,
    num_kv_heads: int,
) -> jax.Array:
    S, H = x.shape
    head_dim = H // num_heads
    groups   = num_heads // num_kv_heads

    q = (x @ wq).reshape(S, num_heads,    head_dim).transpose(1, 0, 2)
    k = (x @ wk).reshape(S, num_kv_heads, head_dim).transpose(1, 0, 2)
    v = (x @ wv).reshape(S, num_kv_heads, head_dim).transpose(1, 0, 2)

    k = jnp.repeat(k, groups, axis=0)
    v = jnp.repeat(v, groups, axis=0)

    out = jax.nn.dot_product_attention(q, k, v)
    return out.transpose(1, 0, 2).reshape(S, H) @ wo


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

def _unfused_svd_block(
    x: jax.Array, w: dict, num_heads: int, num_kv_heads: int
) -> jax.Array:
    """One layer: attention unchanged; MLP runs each SVD matmul via nki_matmul_fully_optimized_.

    Using the same Pallas matmul kernel as the fused path means the speedup
    delta isolates fusion (fewer memory passes) rather than Pallas vs XLA.
    nki_matmul_fully_optimized_(lhsT, rhs) computes lhsT.T @ rhs where
    lhsT=(K,M) and rhs=(K,N), producing (M,N).
    """
    h = _rmsnorm(x, w["norm1_w"])
    h = _attention(h, w["wq"], w["wk"], w["wv"], w["wo"], num_heads, num_kv_heads)
    x = x + h
    h = _rmsnorm(x, w["norm2_w"])
    # Gate: h @ ug @ vg
    xu_gate = nki_matmul_fully_optimized_(h.T,        w["ug"])   # (H,S),(H,R) -> (S,R)
    gate    = nki_matmul_fully_optimized_(xu_gate.T,  w["vg"])   # (R,S),(R,I) -> (S,I)
    # Up: h @ uu @ vu
    xu_up   = nki_matmul_fully_optimized_(h.T,        w["uu"])   # (H,S),(H,R) -> (S,R)
    up      = nki_matmul_fully_optimized_(xu_up.T,    w["vu"])   # (R,S),(R,I) -> (S,I)
    # SwiGLU
    hidden  = jax.nn.silu(gate.astype(jnp.float32)).astype(h.dtype) * up
    # Down: hidden @ ud @ vd
    xu_down = nki_matmul_fully_optimized_(hidden.T,   w["ud"])   # (I,S),(I,R) -> (S,R)
    out     = nki_matmul_fully_optimized_(xu_down.T,  w["vd"])   # (R,S),(R,H) -> (S,H)
    return x + out


def _fused_svd_block(
    x: jax.Array, w: dict, num_heads: int, num_kv_heads: int, S: int
) -> jax.Array:
    """One layer: attention unchanged; MLP uses fused Pallas kernels."""
    h = _rmsnorm(x, w["norm1_w"])
    h = _attention(h, w["wq"], w["wk"], w["wv"], w["wo"], num_heads, num_kv_heads)
    x = x + h
    h = _rmsnorm(x, w["norm2_w"])
    h = svd_mlp_with_fused_kernel(
        h,
        w["uu"], w["vu"], w["ug"], w["vg"], w["ud"], w["vd"],
        up_T_params=get_fused_mlp_up_T_params(S),
        XTUV_params=get_fused_three_mm_XTUV_params(S),
    )
    return x + h


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_fn(fn, args, *, warmup: int, repeats: int) -> dict:
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    return dict(
        latency_ms_avg=statistics.fmean(samples),
        latency_ms_p50=statistics.median(samples),
        latency_ms_p90=samples[math.ceil(0.90 * len(samples)) - 1],
        latency_ms_min=samples[0],
        latency_ms_max=samples[-1],
        iterations=repeats,
    )


# ---------------------------------------------------------------------------
# Full-model benchmark
# ---------------------------------------------------------------------------

def _bench_model(S: int, cfg: dict, args: argparse.Namespace) -> dict:
    _positive("S", S)
    H        = cfg["hidden"]
    I        = cfg["intermediate"]
    n_heads  = cfg["heads"]
    kv_heads = cfg["kv_heads"]
    n_layers = args.layers if args.layers is not None else cfg["layers"]

    if args.model_path:
        # Real SVD-decomposed checkpoint weights.
        layer_indices = list(range(n_layers))
        print(f"S={S}: loading real checkpoint weights for {n_layers} layers...", flush=True)
        weights = _load_checkpoint_weights(args.model_path, layer_indices)
        # Infer R from the first layer's ug shape (H, R).
        R = weights[0]["ug"].shape[1]
        compress_ratio = R / low_rank_dim(H, I, 1.0) if H and I else args.compress_ratio
    else:
        # Synthetic random weights — useful without a checkpoint.
        R = low_rank_dim(H, I, args.compress_ratio)
        compress_ratio = args.compress_ratio
        base_key   = jax.random.key(args.seed)
        layer_keys = jax.random.split(base_key, n_layers)
        weights = [
            _make_layer_weights(layer_keys[i], H, I, R, n_heads, kv_heads)
            for i in range(n_layers)
        ]

    x_key = jax.random.key(args.seed ^ 0xDEAD)
    x = (jax.random.normal(x_key, (S, H), dtype=jnp.bfloat16) * 0.02).astype(jnp.bfloat16)
    jax.block_until_ready(x)

    # Stack weights along a leading layer axis so jax.lax.scan can iterate over
    # them without unrolling the loop in HLO.  Unrolling 28 layers of Pallas
    # matmuls produces an enormous graph that takes 10–30 min to compile.
    stacked_ws = {k: jnp.stack([w[k] for w in weights], axis=0) for k in _WEIGHT_KEYS}

    def unfused_model(x, stacked):
        def body(carry, w_slice):
            return _unfused_svd_block(carry, w_slice, n_heads, kv_heads), None
        x, _ = jax.lax.scan(body, x, stacked)
        return x

    def fused_model(x, stacked):
        def body(carry, w_slice):
            return _fused_svd_block(carry, w_slice, n_heads, kv_heads, S), None
        x, _ = jax.lax.scan(body, x, stacked)
        return x

    unfused_stats = _time_fn(unfused_model, (x, stacked_ws), warmup=args.warmup, repeats=args.repeats)
    fused_stats   = _time_fn(fused_model,   (x, stacked_ws), warmup=args.warmup, repeats=args.repeats)

    speedup = unfused_stats["latency_ms_avg"] / fused_stats["latency_ms_avg"]
    return dict(
        seq_len=S,
        hidden=H,
        intermediate=I,
        rank=R,
        compress_ratio=compress_ratio,
        num_layers=n_layers,
        num_heads=n_heads,
        num_kv_heads=kv_heads,
        weights="checkpoint" if args.model_path else "synthetic",
        unfused_block=unfused_stats,
        fused_block=fused_stats,
        speedup_fused_vs_unfused=speedup,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whole-model N-layer transformer E2E: fused vs unfused SVD MLP."
    )
    p.add_argument(
        "--model", choices=list(_SHAPES), default="llama3-3b",
    )
    p.add_argument(
        "--seq-lens", type=int, nargs="+", default=[128, 512, 1024], metavar="S",
    )
    p.add_argument(
        "--layers", type=int, default=None,
        help="Override layer count (default: model depth)",
    )
    p.add_argument(
        "--model-path", type=str, default=None, metavar="PATH",
        help="Path to an SVD-decomposed HF checkpoint directory. "
             "If omitted, synthetic random weights are used.",
    )
    p.add_argument("--compress-ratio", type=float, default=_DEFAULT_COMPRESS_RATIO)
    p.add_argument("--warmup",  type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--output",  type=str, default=None,
                   help="Write full JSON report to this path")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"E2E block benchmark requires a TPU backend, got {jax.default_backend()!r}."
        )

    cfg = _SHAPES[args.model]
    results = []
    for S in args.seq_lens:
        rec = _bench_model(S, cfg, args)
        results.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    report = dict(
        model=args.model,
        backend=jax.default_backend(),
        jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        compress_ratio=args.compress_ratio,
        cases=results,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
