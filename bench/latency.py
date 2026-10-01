"""Synthetic-weight MLP latency benchmark for SVD-Flash on TPU v5e.

The generated arrays have the exact low-rank factor shapes used by the target
models. They are not SVDs of pretrained weights: random factors are sufficient
for compilation, memory-traffic, and latency measurements, but not accuracy or
perplexity evaluation. This benchmark compares vendor-XLA baselines with the
original two-Pallas-kernel path and a path that contracts Down-U before the
HBM kernel boundary.
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
    fits_vmem,
    gate_up_block_params,
    rank_for,
    vmem_bytes_gate_up,
    vmem_bytes_gate_up_down_u,
    vmem_bytes_xuv,
)
from svdflash_tpu.kernels.pallas_ops import (
    contracted_svd_mlp,
    fused_gate_up,
    svd_mlp,
    xuv_matmul,
)


def _parse_sequence_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive integers")
    return lengths


def _random_array(key: jax.Array, shape: tuple[int, int], scale: float) -> jax.Array:
    return (jax.random.normal(key, shape, dtype=jnp.bfloat16) * scale).astype(
        jnp.bfloat16
    )


def _make_factors(seed: int, hidden: int, intermediate: int, rank: int):
    keys = jax.random.split(jax.random.key(seed), 6)
    hidden_scale = 1.0 / math.sqrt(hidden)
    intermediate_scale = 1.0 / math.sqrt(intermediate)
    rank_scale = 1.0 / math.sqrt(rank)
    factors = (
        _random_array(keys[0], (hidden, rank), hidden_scale),
        _random_array(keys[1], (rank, intermediate), rank_scale),
        _random_array(keys[2], (hidden, rank), hidden_scale),
        _random_array(keys[3], (rank, intermediate), rank_scale),
        _random_array(keys[4], (intermediate, rank), intermediate_scale),
        _random_array(keys[5], (rank, hidden), rank_scale),
    )
    jax.block_until_ready(factors)
    return factors


def _make_dense_weights(seed: int, hidden: int, intermediate: int):
    keys = jax.random.split(jax.random.key(seed), 3)
    hidden_scale = 1.0 / math.sqrt(hidden)
    intermediate_scale = 1.0 / math.sqrt(intermediate)
    weights = (
        _random_array(keys[0], (hidden, intermediate), hidden_scale),
        _random_array(keys[1], (hidden, intermediate), hidden_scale),
        _random_array(keys[2], (intermediate, hidden), intermediate_scale),
    )
    jax.block_until_ready(weights)
    return weights


def _nearest_rank_percentile(samples: Sequence[float], percentile: float) -> float:
    index = min(
        len(samples) - 1,
        max(0, math.ceil(percentile / 100.0 * len(samples)) - 1),
    )
    return samples[index]


def _measure(
    fn: Callable[..., jax.Array],
    args: Sequence[jax.Array],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    # Explicit compilation and warmup keep both out of the reported latency.
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        jax.block_until_ready(compiled(*args))
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    samples_ms.sort()
    total_seconds = sum(samples_ms) / 1000.0
    mean_ms = statistics.fmean(samples_ms)
    return {
        "latency_ms_avg": mean_ms,
        "latency_ms_p50": statistics.median(samples_ms),
        "latency_ms_p90": _nearest_rank_percentile(samples_ms, 90),
        "latency_ms_p95": _nearest_rank_percentile(samples_ms, 95),
        "latency_ms_p99": _nearest_rank_percentile(samples_ms, 99),
        "latency_ms_p100": samples_ms[-1],
        "latency_ms_min": samples_ms[0],
        "iterations": len(samples_ms),
        "measured_seconds": total_seconds,
    }


def _dense_mlp(x, w_gate, w_up, w_down):
    gate = x @ w_gate
    up = x @ w_up
    return (jax.nn.silu(gate) * up) @ w_down


def _unfused_svd_mlp(x, u_gate, v_gate, u_up, v_up, u_down, v_down):
    """Six-matmul JAX baseline; each low-rank projection is a separate dot."""
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    hidden = jax.nn.silu(gate) * up
    return (hidden @ u_down) @ v_down


def _vendor_gate_up(x, u_gate, v_gate, u_up, v_up):
    """Equivalent Gate/Up path lowered to vendor XLA dots."""
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    return jax.nn.silu(gate) * up


def _vendor_down_xuv(x, u_down, v_down):
    """Equivalent Down path lowered to two vendor XLA dots."""
    return (x @ u_down) @ v_down


def _load_tuned_blocks(path: str | None) -> dict[tuple[str, int, str], tuple[float, object]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    tuned = {}
    for record in report.get("best", []):
        kernel = record["kernel"]
        if kernel == "gate-up":
            blocks = GateUpBlocks(**record["blocks"])
        else:
            continue
        key = (record["model"], int(record["rows"]), kernel)
        tuned[key] = (float(record["compress_ratio"]), blocks)
    return tuned


def _select_blocks(
    args: argparse.Namespace,
    model_name: str,
    rows: int,
    kernel: str,
    default,
):
    entry = args.tuned_blocks.get((model_name, rows, kernel))
    if entry is None:
        return default
    tuned_ratio, blocks = entry
    if not math.isclose(tuned_ratio, args.compress_ratio, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"tuned {kernel} blocks for {model_name}, rows={rows} use "
            f"compress_ratio={tuned_ratio}, requested {args.compress_ratio}"
        )
    return blocks


def _bench_model(args: argparse.Namespace, model_name: str) -> list[dict]:
    model = MODELS[model_name]
    rank = rank_for(model, args.compress_ratio)
    factors = None
    dense_weights = None
    if args.benchmark in (
        "xuv",
        "gate-up",
        "unfused",
        "mlp",
        "contracted-mlp",
        "nki-parity",
        "all",
    ):
        factors = _make_factors(
            args.seed, model.hidden_size, model.intermediate_size, rank
        )
    if args.benchmark in ("dense", "nki-parity", "all"):
        dense_weights = _make_dense_weights(
            args.seed + 10_000, model.hidden_size, model.intermediate_size
        )
    results = []

    for sequence_length in args.sequence_lengths:
        rows = args.batch_size * sequence_length
        x = _random_array(
            jax.random.key(args.seed + rows),
            (rows, model.hidden_size),
            1.0,
        )
        down_input = _random_array(
            jax.random.key(args.seed + rows + 1),
            (rows, model.intermediate_size),
            1.0,
        )
        jax.block_until_ready((x, down_input))
        gate_blocks = _select_blocks(
            args,
            model_name,
            rows,
            "gate-up",
            gate_up_block_params(rows, model, args.compress_ratio),
        )
        # The extra Down-U buffers need a narrower output tile to stay below
        # Mosaic's scoped VMEM limit with the default Llama 3 8B blocks.
        contracted_blocks = gate_blocks._replace(bn=min(gate_blocks.bn, 512))
        gate_padded_rank = math.ceil(rank / gate_blocks.br) * gate_blocks.br
        contracted_padded_rank = (
            math.ceil(rank / contracted_blocks.br) * contracted_blocks.br
        )
        down_padded_rank = math.ceil(rank / XUV_RANK_TILE) * XUV_RANK_TILE
        gate_vmem = vmem_bytes_gate_up(
            gate_blocks, gate_padded_rank, model.hidden_size
        )
        down_vmem = vmem_bytes_xuv(
            model.intermediate_size,
            down_padded_rank,
            model.hidden_size,
        )
        contracted_vmem = vmem_bytes_gate_up_down_u(
            contracted_blocks,
            contracted_padded_rank,
            contracted_padded_rank,
            model.hidden_size,
        )

        if not fits_vmem(gate_vmem) or not fits_vmem(down_vmem):
            raise ValueError(
                f"estimated VMEM exceeds the safety budget: gate/up={gate_vmem}, "
                f"down={down_vmem}"
            )
        if args.benchmark in ("contracted-mlp", "all") and not fits_vmem(
            contracted_vmem
        ):
            raise ValueError(
                "estimated contracted MLP VMEM exceeds the safety budget: "
                f"{contracted_vmem}"
            )

        common = {
            "model": model_name,
            "batch_size": args.batch_size,
            "sequence_length": sequence_length,
            "rows": rows,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "rank": rank,
            "logical_rank": rank,
            "compress_ratio": args.compress_ratio,
            "device": str(jax.devices()[0]),
            "gate_up_blocks": gate_blocks._asdict(),
            "contracted_blocks": contracted_blocks._asdict(),
            "down_tiles": {
                "rows": XUV_ROW_TILE,
                "input": XUV_INPUT_TILE,
                "rank": XUV_RANK_TILE,
                "output": XUV_OUTPUT_TILE,
            },
            "gate_up_padded_rank": gate_padded_rank,
            "down_padded_rank": down_padded_rank,
            "input_dtype": str(x.dtype),
            "scratch_dtype": str(x.dtype),
            "accumulator_dtype": "float32",
            "pallas_svd_dma_strategy": "single_buffer_synchronous",
            "tensor_cores_per_device": int(
                getattr(jax.devices()[0], "num_cores", 1)
            ),
            "gate_up_row_tiles": math.ceil(rows / gate_blocks.bs),
            "down_row_tiles": math.ceil(rows / XUV_ROW_TILE),
            "active_tensor_cores_gate_up": min(
                int(getattr(jax.devices()[0], "num_cores", 1)),
                math.ceil(rows / gate_blocks.bs),
            ),
            "active_tensor_cores_down": min(
                int(getattr(jax.devices()[0], "num_cores", 1)),
                math.ceil(rows / XUV_ROW_TILE),
            ),
            "estimated_gate_up_vmem_bytes": gate_vmem,
            "estimated_down_vmem_bytes": down_vmem,
            "estimated_contracted_vmem_bytes": contracted_vmem,
            "comparison_scope": (
                "dense_vs_fused_low_rank_mlp"
                if args.benchmark == "nki-parity"
                else args.benchmark
            ),
        }

        if args.benchmark in ("dense", "nki-parity", "all"):
            timing = _measure(
                _dense_mlp,
                (x, *dense_weights),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append({**common, "benchmark": "dense_mlp", **timing})

        if args.benchmark in ("xuv", "all"):
            u_gate, v_gate, u_up, v_up, u_down, v_down = factors
            if args.benchmark == "all":
                timing = _measure(
                    _vendor_down_xuv,
                    (down_input, u_down, v_down),
                    args.warmup,
                    args.iterations,
                )
                timing["throughput_tokens_per_second"] = (
                    rows * timing["iterations"] / timing["measured_seconds"]
                )
                results.append(
                    {**common, "benchmark": "vendor_down_xuv", **timing}
                )
            xuv_fn = lambda x_, u_, v_: xuv_matmul(x_, u_, v_)  # noqa: E731
            timing = _measure(
                xuv_fn,
                (down_input, u_down, v_down),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append({**common, "benchmark": "down_xuv", **timing})

        if args.benchmark in ("gate-up", "all"):
            u_gate, v_gate, u_up, v_up, u_down, v_down = factors
            if args.benchmark == "all":
                timing = _measure(
                    _vendor_gate_up,
                    (x, u_gate, v_gate, u_up, v_up),
                    args.warmup,
                    args.iterations,
                )
                timing["throughput_tokens_per_second"] = (
                    rows * timing["iterations"] / timing["measured_seconds"]
                )
                results.append(
                    {**common, "benchmark": "vendor_gate_up", **timing}
                )
            gate_fn = lambda *xs: fused_gate_up(  # noqa: E731
                *xs, blocks=gate_blocks
            )
            timing = _measure(
                gate_fn,
                (x, u_gate, v_gate, u_up, v_up),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append({**common, "benchmark": "fused_gate_up", **timing})

        if args.benchmark in ("unfused", "all"):
            u_gate, v_gate, u_up, v_up, u_down, v_down = factors
            timing = _measure(
                _unfused_svd_mlp,
                (x, u_gate, v_gate, u_up, v_up, u_down, v_down),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append({**common, "benchmark": "unfused_svd_mlp", **timing})

        if args.benchmark in ("mlp", "nki-parity", "all"):
            u_gate, v_gate, u_up, v_up, u_down, v_down = factors
            mlp_fn = lambda *xs: svd_mlp(  # noqa: E731
                *xs, gate_up_blocks=gate_blocks
            )
            timing = _measure(
                mlp_fn,
                (x, u_gate, v_gate, u_up, v_up, u_down, v_down),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append({**common, "benchmark": "two_kernel_svd_mlp", **timing})

        if args.benchmark in ("contracted-mlp", "all"):
            u_gate, v_gate, u_up, v_up, u_down, v_down = factors
            contracted_fn = lambda *xs: contracted_svd_mlp(  # noqa: E731
                *xs, gate_up_blocks=contracted_blocks
            )
            timing = _measure(
                contracted_fn,
                (x, u_gate, v_gate, u_up, v_up, u_down, v_down),
                args.warmup,
                args.iterations,
            )
            timing["throughput_tokens_per_second"] = (
                rows * timing["iterations"] / timing["measured_seconds"]
            )
            results.append(
                {**common, "benchmark": "contracted_hidden_svd_mlp", **timing}
            )

    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=("qwen3-8b", "llama-3.1-8b"),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_sequence_lengths,
        default=(1, 128, 512, 1024),
        help="comma-separated token counts",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratio", type=float, default=0.8)
    parser.add_argument(
        "--benchmark",
        choices=(
            "dense",
            "xuv",
            "gate-up",
            "unfused",
            "mlp",
            "contracted-mlp",
            "nki-parity",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tuning-config",
        type=str,
        help="JSON generated by svdflash_tpu.bench.tune",
    )
    parser.add_argument("--output", type=str)
    return parser


def _speedup_summary(results: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], dict[str, dict]] = {}
    for result in results:
        key = (
            result["model"],
            result["batch_size"],
            result["sequence_length"],
        )
        groups.setdefault(key, {})[result["benchmark"]] = result

    summaries = []
    for (model, batch_size, sequence_length), timings in groups.items():
        fused = timings.get("two_kernel_svd_mlp")
        contracted = timings.get("contracted_hidden_svd_mlp")
        if fused is None and contracted is None:
            continue
        summary = {
            "model": model,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        }
        dense = timings.get("dense_mlp")
        unfused = timings.get("unfused_svd_mlp")
        if fused is not None:
            summary["two_kernel_latency_ms"] = fused["latency_ms_avg"]
        if dense is not None and fused is not None:
            summary["speedup_vs_dense"] = (
                dense["latency_ms_avg"] / fused["latency_ms_avg"]
            )
        if unfused is not None and fused is not None:
            summary["speedup_vs_unfused_svd"] = (
                unfused["latency_ms_avg"] / fused["latency_ms_avg"]
            )
        if contracted is not None:
            summary["contracted_hidden_latency_ms"] = contracted["latency_ms_avg"]
        if fused is not None and contracted is not None:
            summary["speedup_contracted_vs_two_kernel"] = (
                fused["latency_ms_avg"] / contracted["latency_ms_avg"]
            )
        if unfused is not None and contracted is not None:
            summary["speedup_contracted_vs_unfused_svd"] = (
                unfused["latency_ms_avg"] / contracted["latency_ms_avg"]
            )
        summaries.append(summary)
    return summaries


def main() -> None:
    args = _parser().parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"latency benchmark requires TPU, got backend {jax.default_backend()!r}"
        )
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch-size and iterations must be positive; warmup may be zero")
    if not 0.0 < args.compress_ratio <= 1.0:
        raise ValueError("compress-ratio must be in the interval (0, 1]")
    args.tuned_blocks = _load_tuned_blocks(args.tuning_config)

    results = []
    for model_name in args.models:
        results.extend(_bench_model(args, model_name))

    report = {
        "synthetic_factors": True,
        "accuracy_claim": False,
        "benchmark_scope": "single_transformer_mlp_layer",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "tuning_config": args.tuning_config,
        "tuned_workloads": len(args.tuned_blocks),
        "results": results,
        "speedups": _speedup_summary(results),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
