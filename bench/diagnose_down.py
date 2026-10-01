"""Diagnostic tests to identify why Down kernel is slower than JAX.

Measures:
1. VMEM usage across different configurations
2. Impact of tile sizes on latency
3. Comparison between XUV and dense matmul
4. Breakdown of DMA vs compute time
"""

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.config import (
    MODELS,
    vmem_bytes_xuv,
    fits_vmem,
    XUV_INPUT_TILE,
    XUV_OUTPUT_TILE,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
)
from svdflash_tpu.kernels.pallas_ops import xuv_matmul, dense_matmul


def measure_latency(fn, args, warmup=5, iterations=30):
    """Measure synchronized latency."""
    compiled = jax.jit(fn)

    # Warmup
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))

    # Measure
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        jax.block_until_ready(compiled(*args))
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        samples.append(elapsed_ms)

    samples.sort()
    return {
        "avg_ms": sum(samples) / len(samples),
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "p90_ms": samples[int(0.9 * len(samples))],
    }


def random_inputs(key, seq_len, intermediate, rank, hidden):
    """Generate test inputs."""
    keys = jax.random.split(key, 3)
    x = jax.random.normal(keys[0], (seq_len, intermediate), dtype=jnp.bfloat16)
    u = jax.random.normal(keys[1], (intermediate, rank), dtype=jnp.bfloat16) / math.sqrt(intermediate)
    v = jax.random.normal(keys[2], (rank, hidden), dtype=jnp.bfloat16) / math.sqrt(rank)
    jax.block_until_ready((x, u, v))
    return x, u, v


def test_vmem_usage(model_name, seq_len):
    """Check VMEM requirements for Down kernel."""
    model = MODELS[model_name]

    # Down projection: (seq_len, intermediate) @ (intermediate, rank) @ (rank, hidden)
    rank_080 = model.rank_at_080

    vmem_bytes = vmem_bytes_xuv(
        input_size=model.intermediate_size,
        rank=rank_080,
        output_size=model.hidden_size,
    )

    vmem_mb = vmem_bytes / (1024 * 1024)
    fits = fits_vmem(vmem_bytes)
    utilization = vmem_bytes / (16 * 1024 * 1024) * 100

    return {
        "model": model_name,
        "seq_len": seq_len,
        "intermediate_size": model.intermediate_size,
        "rank": rank_080,
        "hidden_size": model.hidden_size,
        "vmem_bytes": vmem_bytes,
        "vmem_mb": vmem_mb,
        "fits_in_budget": fits,
        "utilization_percent": utilization,
        "tiles": {
            "row": XUV_ROW_TILE,
            "input": XUV_INPUT_TILE,
            "rank": XUV_RANK_TILE,
            "output": XUV_OUTPUT_TILE,
        },
        "dma_copies_per_row_strip": {
            "u_tiles": math.ceil(model.intermediate_size / XUV_INPUT_TILE) * math.ceil(rank_080 / XUV_RANK_TILE),
            "v_tiles": math.ceil(rank_080 / XUV_RANK_TILE) * math.ceil(model.hidden_size / XUV_OUTPUT_TILE),
        }
    }


def test_xuv_vs_dense(model_name, seq_len, warmup, iterations):
    """Compare XUV matmul with dense matmul for Down projection."""
    model = MODELS[model_name]
    rank = model.rank_at_080

    key = jax.random.key(42)
    x, u, v = random_inputs(key, seq_len, model.intermediate_size, rank, model.hidden_size)

    # XUV matmul (fused)
    xuv_fn = lambda x_, u_, v_: xuv_matmul(x_, u_, v_, mixed_precision=True)
    xuv_timing = measure_latency(xuv_fn, (x, u, v), warmup, iterations)

    # Dense matmul (unfused, two separate matmuls)
    def dense_two_stage(x_, u_, v_):
        xu = dense_matmul(x_, u_, mixed_precision=True)
        return dense_matmul(xu, v_, mixed_precision=True)

    dense_timing = measure_latency(dense_two_stage, (x, u, v), warmup, iterations)

    # JAX baseline (highest precision)
    def jax_baseline(x_, u_, v_):
        with jax.default_matmul_precision("highest"):
            return (x_ @ u_) @ v_

    jax_timing = measure_latency(jax_baseline, (x, u, v), warmup, iterations)

    # JAX default precision
    jax_default_fn = lambda x_, u_, v_: (x_ @ u_) @ v_
    jax_default_timing = measure_latency(jax_default_fn, (x, u, v), warmup, iterations)

    return {
        "model": model_name,
        "seq_len": seq_len,
        "shape": f"({seq_len}, {model.intermediate_size}) @ ({model.intermediate_size}, {rank}) @ ({rank}, {model.hidden_size})",
        "xuv_fused": xuv_timing,
        "dense_two_stage": dense_timing,
        "jax_highest_precision": jax_timing,
        "jax_default_precision": jax_default_timing,
        "speedup_xuv_vs_jax_highest": jax_timing["median_ms"] / xuv_timing["median_ms"],
        "speedup_xuv_vs_jax_default": jax_default_timing["median_ms"] / xuv_timing["median_ms"],
        "speedup_xuv_vs_dense": dense_timing["median_ms"] / xuv_timing["median_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS.keys()), default="llama-3.1-8b")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Diagnostics require TPU, got {jax.default_backend()}")

    print(f"Running diagnostics for {args.model} with seq_len={args.seq_len}")

    # Test 1: VMEM usage analysis
    print("\n=== Test 1: VMEM Usage ===")
    vmem_report = test_vmem_usage(args.model, args.seq_len)
    print(f"VMEM: {vmem_report['vmem_mb']:.2f} MB ({vmem_report['utilization_percent']:.1f}% of 16 MB budget)")
    print(f"Fits in budget: {vmem_report['fits_in_budget']}")
    print(f"U tiles per row strip: {vmem_report['dma_copies_per_row_strip']['u_tiles']}")
    print(f"V tiles per row strip: {vmem_report['dma_copies_per_row_strip']['v_tiles']}")

    # Test 2: XUV vs Dense vs JAX
    print("\n=== Test 2: Performance Comparison ===")
    perf_report = test_xuv_vs_dense(args.model, args.seq_len, args.warmup, args.iterations)
    print(f"XUV fused:              {perf_report['xuv_fused']['median_ms']:.4f} ms")
    print(f"Dense two-stage:        {perf_report['dense_two_stage']['median_ms']:.4f} ms")
    print(f"JAX highest precision:  {perf_report['jax_highest_precision']['median_ms']:.4f} ms")
    print(f"JAX default precision:  {perf_report['jax_default_precision']['median_ms']:.4f} ms")
    print(f"\nSpeedup XUV vs JAX (highest): {perf_report['speedup_xuv_vs_jax_highest']:.3f}x")
    print(f"Speedup XUV vs JAX (default): {perf_report['speedup_xuv_vs_jax_default']:.3f}x")
    print(f"Speedup XUV vs Dense:         {perf_report['speedup_xuv_vs_dense']:.3f}x")

    # Save full report
    report = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "vmem_analysis": vmem_report,
        "performance_comparison": perf_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
