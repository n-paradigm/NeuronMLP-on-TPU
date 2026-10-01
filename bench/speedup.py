"""NKI test_speedup.py parity benchmark for Pallas on TPU.

The benchmark compares a three-projection dense SwiGLU MLP with the two-kernel
low-rank Pallas MLP. Compilation, random input creation, and warmup are excluded
from steady-state latency. Every timed call is synchronized with the TPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.config import (
    MODELS,
    GateUpBlocks,
    XUV_INPUT_TILE,
    XUV_OUTPUT_TILE,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
    rank_for,
    vmem_bytes_gate_up,
    vmem_bytes_xuv,
)
from svdflash_tpu.kernels.pallas_ops import dense_mlp, svd_mlp


def _parse_sequence_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return lengths


def _random_array(key, shape, scale):
    return (jax.random.normal(key, shape, dtype=jnp.bfloat16) * scale).astype(
        jnp.bfloat16
    )


def _make_inputs(seed: int, rows: int, hidden: int, intermediate: int, rank: int):
    keys = jax.random.split(jax.random.key(seed), 10)
    x = _random_array(keys[0], (rows, hidden), 1.0)
    dense = (
        _random_array(keys[1], (hidden, intermediate), 1.0 / math.sqrt(hidden)),
        _random_array(keys[2], (hidden, intermediate), 1.0 / math.sqrt(hidden)),
        _random_array(
            keys[3], (intermediate, hidden), 1.0 / math.sqrt(intermediate)
        ),
    )
    svd = (
        _random_array(keys[4], (hidden, rank), 1.0 / math.sqrt(hidden)),
        _random_array(keys[5], (rank, intermediate), 1.0 / math.sqrt(rank)),
        _random_array(keys[6], (hidden, rank), 1.0 / math.sqrt(hidden)),
        _random_array(keys[7], (rank, intermediate), 1.0 / math.sqrt(rank)),
        _random_array(keys[8], (intermediate, rank), 1.0 / math.sqrt(intermediate)),
        _random_array(keys[9], (rank, hidden), 1.0 / math.sqrt(rank)),
    )
    jax.block_until_ready((x, dense, svd))
    return x, dense, svd


def _vendor_dense_mlp(x, w_gate, w_up, w_down):
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def _vendor_svd_mlp(x, u_gate, v_gate, u_up, v_up, u_down, v_down):
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    return ((jax.nn.silu(gate) * up) @ u_down) @ v_down


def _compile_and_measure(
    fn: Callable,
    args: Sequence[jax.Array],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    started = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_seconds = time.perf_counter() - started

    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))

    samples = []
    for _ in range(iterations):
        started_ns = time.perf_counter_ns()
        jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter_ns() - started_ns) / 1e6)
    samples.sort()
    measured_seconds = sum(samples) / 1000.0
    return {
        "compile_seconds": compile_seconds,
        "latency_ms_avg": statistics.fmean(samples),
        "latency_ms_p50": statistics.median(samples),
        "latency_ms_p90": samples[math.ceil(0.90 * len(samples)) - 1],
        "latency_ms_p99": samples[math.ceil(0.99 * len(samples)) - 1],
        "latency_ms_min": samples[0],
        "latency_ms_max": samples[-1],
        "iterations": iterations,
        "measured_seconds": measured_seconds,
    }


def _relative_l2(got, expected) -> float:
    got = got.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    error = jnp.linalg.norm(got - expected) / jnp.maximum(
        jnp.linalg.norm(expected), 1e-12
    )
    return float(jax.device_get(error))


def _bench_case(args, model_name: str, sequence_length: int) -> dict:
    model = MODELS[model_name]
    rows = args.batch_size * sequence_length
    rank = rank_for(model, args.compress_ratio)
    if rows != 128:
        raise ValueError(
            "the NKI S=128 speedup benchmark configuration requires "
            f"batch_size * sequence_length == 128, got {rows}"
        )
    blocks = GateUpBlocks(bs=128, bh=1024, br=512, bn=512)
    gate_up_padded_rank = math.ceil(rank / blocks.br) * blocks.br
    down_padded_rank = math.ceil(rank / XUV_RANK_TILE) * XUV_RANK_TILE
    x, dense_weights, svd_factors = _make_inputs(
        args.seed + rows,
        rows,
        model.hidden_size,
        model.intermediate_size,
        rank,
    )

    pallas_dense_fn = lambda *xs: dense_mlp(*xs)  # noqa: E731
    pallas_svd_fn = lambda *xs: svd_mlp(  # noqa: E731
        *xs,
        gate_up_blocks=blocks,
    )
    functions = {
        "pallas_dense_mlp": (pallas_dense_fn, (x, *dense_weights)),
        "pallas_svd_mlp": (pallas_svd_fn, (x, *svd_factors)),
        "vendor_dense_mlp": (_vendor_dense_mlp, (x, *dense_weights)),
        "vendor_svd_mlp": (_vendor_svd_mlp, (x, *svd_factors)),
    }

    results = {
        name: _compile_and_measure(
            fn,
            fn_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for name, (fn, fn_args) in functions.items()
    }

    verification = None
    if not args.no_verify:
        pallas_dense = jax.jit(pallas_dense_fn)(x, *dense_weights)
        vendor_dense = jax.jit(_vendor_dense_mlp)(x, *dense_weights)
        pallas_svd = jax.jit(pallas_svd_fn)(x, *svd_factors)
        vendor_svd = jax.jit(_vendor_svd_mlp)(x, *svd_factors)
        verification = {
            "pallas_dense_vs_vendor_relative_l2": _relative_l2(
                pallas_dense, vendor_dense
            ),
            "pallas_svd_vs_vendor_relative_l2": _relative_l2(
                pallas_svd, vendor_svd
            ),
        }
    for timing in results.values():
        timing["throughput_tokens_per_second"] = (
            rows * timing["iterations"] / timing["measured_seconds"]
        )

    pallas_svd_latency = results["pallas_svd_mlp"]["latency_ms_avg"]
    return {
        "model": model_name,
        "batch_size": args.batch_size,
        "sequence_length": sequence_length,
        "rows": rows,
        "hidden_size": model.hidden_size,
        "intermediate_size": model.intermediate_size,
        "rank": rank,
        "logical_rank": rank,
        "gate_up_padded_rank": gate_up_padded_rank,
        "down_padded_rank": down_padded_rank,
        "padded_hidden_size_for_down": (
            math.ceil(model.hidden_size / XUV_OUTPUT_TILE) * XUV_OUTPUT_TILE
        ),
        "compress_ratio": args.compress_ratio,
        "input_dtype": str(x.dtype),
        "scratch_dtype": str(x.dtype),
        "accumulator_dtype": "float32",
        "dma_strategy_by_implementation": {
            "pallas_dense_mlp": "double_buffered_async",
            "pallas_svd_mlp": "single_buffer_synchronous",
            "vendor_dense_mlp": "compiler_managed",
            "vendor_svd_mlp": "compiler_managed",
        },
        "tensor_cores_per_device": int(
            getattr(jax.devices()[0], "num_cores", 1)
        ),
        "gate_up_row_tiles": math.ceil(rows / blocks.bs),
        "down_row_tiles": math.ceil(rows / XUV_ROW_TILE),
        "active_tensor_cores_gate_up": min(
            int(getattr(jax.devices()[0], "num_cores", 1)),
            math.ceil(rows / blocks.bs),
        ),
        "active_tensor_cores_down": min(
            int(getattr(jax.devices()[0], "num_cores", 1)),
            math.ceil(rows / XUV_ROW_TILE),
        ),
        "gate_up_blocks": blocks._asdict(),
        "down_xuv_blocks": {
            "rows": XUV_ROW_TILE,
            "input": XUV_INPUT_TILE,
            "rank": XUV_RANK_TILE,
            "output": XUV_OUTPUT_TILE,
        },
        "estimated_pallas_svd_vmem_bytes": {
            "gate_up": vmem_bytes_gate_up(
                blocks, gate_up_padded_rank, model.hidden_size
            ),
            "down_xuv": vmem_bytes_xuv(
                model.intermediate_size,
                down_padded_rank,
                model.hidden_size,
            ),
        },
        "verification": verification,
        "results": results,
        # Direct counterpart of the NKI script's dur_baseline / dur_mm.
        "speedup_pallas_svd_vs_pallas_dense": (
            results["pallas_dense_mlp"]["latency_ms_avg"] / pallas_svd_latency
        ),
        "speedup_pallas_svd_vs_vendor_dense": (
            results["vendor_dense_mlp"]["latency_ms_avg"] / pallas_svd_latency
        ),
        "speedup_pallas_svd_vs_vendor_svd": (
            results["vendor_svd_mlp"]["latency_ms_avg"] / pallas_svd_latency
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=("llama-3.2-3b", "llama-3.1-8b"),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_sequence_lengths,
        default=(128,),
        help="comma-separated token counts",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratio", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"NKI-parity speedup benchmark requires TPU, got {jax.default_backend()!r}"
        )
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch-size and iterations must be positive; warmup may be zero")
    if not 0.0 < args.compress_ratio <= 1.0:
        raise ValueError("compress-ratio must be in (0, 1]")

    cases = [
        _bench_case(args, model_name, sequence_length)
        for model_name in args.models
        for sequence_length in args.sequence_lengths
    ]
    report = {
        "benchmark_scope": "single_synthetic_mlp_nki_test_speedup_parity",
        "timing_is_device_synchronized": True,
        "compilation_and_warmup_excluded": True,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "cases": cases,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
