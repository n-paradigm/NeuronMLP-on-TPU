"""Measure every real SVD MLP layer in a Hugging Face checkpoint on TPU.

This is an MLP-only sweep: it executes the selected layers' Gate/Up/SwiGLU
and Down projections, but excludes attention, RMSNorm, residual connections,
KV cache, tokenization, and sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.config import (
    GateUpBlocks,
    MODELS,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
    gate_up_block_params,
    vmem_bytes_gate_up,
    vmem_bytes_gate_up_down_u,
    vmem_bytes_xuv,
)
from svdflash_tpu.kernels.pallas_ops import contracted_svd_mlp, svd_mlp
from svdflash_tpu.models.hf_svd import (
    checkpoint_metadata,
    load_mlp_factors,
    validate_factor_shapes,
)


def _parse_layers(value: str, num_layers: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(num_layers))
    selected = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            bounds = item.split(":")
            if len(bounds) != 2:
                raise ValueError(f"invalid layer range {item!r}")
            start = int(bounds[0] or 0)
            stop = int(bounds[1] or num_layers)
            selected.extend(range(start, stop))
        else:
            selected.append(int(item))
    layers = tuple(dict.fromkeys(selected))
    if not layers or min(layers) < 0 or max(layers) >= num_layers:
        raise ValueError(f"layers must be in [0, {num_layers}), got {layers}")
    return layers


def _load_tuned_blocks(path: str | None, *, model: str, rows: int, ratio: float):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    for record in report.get("best", []):
        if (
            record.get("model") == model
            and int(record.get("rows", -1)) == rows
            and record.get("kernel") == "gate-up"
        ):
            tuned_ratio = float(record["compress_ratio"])
            if not math.isclose(tuned_ratio, ratio, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"tuning config ratio={tuned_ratio} does not match checkpoint "
                    f"ratio={ratio}"
                )
            return GateUpBlocks(**record["blocks"])
    return None


def _unfused_svd_mlp(x, u_gate, v_gate, u_up, v_up, u_down, v_down):
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    hidden = jax.nn.silu(gate) * up
    return (hidden @ u_down) @ v_down


def _compile_layer(kind: str, blocks, example_x, example_factors):
    if kind == "pallas":
        fn = lambda x, *f: svd_mlp(x, *f, gate_up_blocks=blocks)  # noqa: E731
    elif kind == "contracted":
        fn = lambda x, *f: contracted_svd_mlp(  # noqa: E731
            x, *f, gate_up_blocks=blocks
        )
    elif kind == "unfused":
        fn = _unfused_svd_mlp
    else:
        raise ValueError(f"unsupported benchmark {kind!r}")
    return jax.jit(fn).lower(example_x, *example_factors).compile()


def _compile_sweep(kind: str, blocks, example_x, example_layers):
    """Compile the selected layer chain as one executable.

    Calling one compiled layer from a Python loop adds a host dispatch for
    every layer.  That penalizes the two-kernel Pallas path disproportionately
    (64 device launches for 32 layers), so timing uses one JIT-compiled sweep.
    """
    if kind == "pallas":
        layer_fn = lambda x, *f: svd_mlp(x, *f, gate_up_blocks=blocks)  # noqa: E731
    elif kind == "contracted":
        layer_fn = lambda x, *f: contracted_svd_mlp(  # noqa: E731
            x, *f, gate_up_blocks=blocks
        )
    elif kind == "unfused":
        layer_fn = _unfused_svd_mlp
    else:
        raise ValueError(f"unsupported benchmark {kind!r}")

    layer_count = len(example_layers)

    def sweep(x, *flat_factors):
        value = x
        for index in range(layer_count):
            start = index * 6
            value = layer_fn(value, *flat_factors[start : start + 6])
        return value

    flat_factors = tuple(factor for layer in example_layers for factor in layer)
    return jax.jit(sweep).lower(example_x, *flat_factors).compile()


def _percentile(samples: Sequence[float], percentile: float) -> float:
    index = min(
        len(samples) - 1,
        max(0, math.ceil(percentile / 100.0 * len(samples)) - 1),
    )
    return sorted(samples)[index]


def _measure(
    compiled, x, layers, *, warmup: int, iterations: int, compile_sweep: bool
):
    flat_factors = tuple(factor for layer in layers for factor in layer)

    def run():
        if compile_sweep:
            return compiled(x, *flat_factors)
        value = x
        for factors in layers:
            value = compiled(value, *factors)
        return value

    for _ in range(warmup):
        jax.block_until_ready(run())

    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = run()
        jax.block_until_ready(output)
        samples.append((time.perf_counter_ns() - start) / 1e6)

    total = sum(samples) / 1000.0
    mean = statistics.fmean(samples)
    return {
        "latency_ms_avg": mean,
        "latency_ms_per_layer_avg": mean / len(layers),
        "latency_ms_p50": statistics.median(samples),
        "latency_ms_p90": _percentile(samples, 90),
        "latency_ms_p99": _percentile(samples, 99),
        "latency_ms_min": min(samples),
        "latency_ms_max": max(samples),
        "iterations": iterations,
        "measured_seconds": total,
    }


def _verify_layer(
    pallas: Callable[..., jax.Array],
    unfused: Callable[..., jax.Array],
    x: jax.Array,
    factors: Sequence[jax.Array],
) -> dict[str, float | int]:
    got = pallas(x, *factors).astype(jnp.float32)
    expected = unfused(x, *factors).astype(jnp.float32)
    diff = got - expected
    max_abs = jnp.max(jnp.abs(diff))
    relative_l2 = jnp.linalg.norm(diff) / jnp.maximum(
        jnp.linalg.norm(expected), 1e-12
    )
    actual_max_abs = jnp.max(jnp.abs(got))
    expected_max_abs = jnp.max(jnp.abs(expected))
    actual_rms = jnp.sqrt(jnp.mean(jnp.square(got)))
    expected_rms = jnp.sqrt(jnp.mean(jnp.square(expected)))
    actual_nonfinite = jnp.sum(~jnp.isfinite(got))
    expected_nonfinite = jnp.sum(~jnp.isfinite(expected))
    values = jax.device_get(
        (
            max_abs,
            relative_l2,
            actual_max_abs,
            expected_max_abs,
            actual_rms,
            expected_rms,
            actual_nonfinite,
            expected_nonfinite,
        )
    )
    return {
        "max_abs_error": float(values[0]),
        "relative_l2_error": float(values[1]),
        "actual_max_abs": float(values[2]),
        "reference_max_abs": float(values[3]),
        "actual_rms": float(values[4]),
        "reference_rms": float(values[5]),
        "actual_nonfinite_count": int(values[6]),
        "reference_nonfinite_count": int(values[7]),
    }


def _array_stats(value: jax.Array) -> dict[str, float | int]:
    value32 = value.astype(jnp.float32)
    max_abs, rms, nonfinite = jax.device_get(
        (
            jnp.max(jnp.abs(value32)),
            jnp.sqrt(jnp.mean(jnp.square(value32))),
            jnp.sum(~jnp.isfinite(value32)),
        )
    )
    return {
        "max_abs": float(max_abs),
        "rms": float(rms),
        "nonfinite_count": int(nonfinite),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layers", default="all", help="all, 0:32, or 0,4,8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--benchmark",
        choices=("pallas", "contracted", "unfused", "both", "all"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--tuning-config",
        help="JSON from bench.tune; selects a measured Gate/Up block for this rows",
    )
    parser.add_argument(
        "--compile-sweep",
        action="store_true",
        help="compile the selected layer chain as one executable",
    )
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"all-layer latency requires TPU, got {jax.default_backend()!r}"
        )
    if (
        args.batch_size <= 0
        or args.sequence_length <= 0
        or args.warmup < 0
        or args.iterations <= 0
    ):
        raise ValueError("batch, sequence, and iterations must be positive")

    config, _ = checkpoint_metadata(args.model_path)
    num_layers = int(config["num_hidden_layers"])
    selected_layers = _parse_layers(args.layers, num_layers)
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    model = MODELS["llama-3.1-8b"]
    if (hidden_size, intermediate_size, num_layers) != (
        model.hidden_size,
        model.intermediate_size,
        model.num_layers,
    ):
        raise ValueError(
            "this benchmark currently supports the Llama 3 8B shape only; "
            f"checkpoint reports H={hidden_size}, I={intermediate_size}, "
            f"layers={num_layers}"
        )

    print(f"Loading {len(selected_layers)} MLP layers into TPU HBM...", flush=True)
    factors = load_mlp_factors(args.model_path, selected_layers)
    rank = validate_factor_shapes(
        factors[0],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    retained_parameter_ratio = (
        rank * (hidden_size + intermediate_size) / (hidden_size * intermediate_size)
    )
    rows = args.batch_size * args.sequence_length
    blocks = gate_up_block_params(rows, model, retained_parameter_ratio)
    tuned_blocks = _load_tuned_blocks(
        args.tuning_config,
        model="llama-3.1-8b",
        rows=rows,
        ratio=retained_parameter_ratio,
    )
    if tuned_blocks is not None:
        blocks = tuned_blocks
    contracted_blocks = blocks._replace(bn=min(blocks.bn, 512))
    gate_up_padded_rank = math.ceil(rank / blocks.br) * blocks.br
    contracted_padded_rank = (
        math.ceil(rank / contracted_blocks.br) * contracted_blocks.br
    )
    down_padded_rank = math.ceil(rank / XUV_RANK_TILE) * XUV_RANK_TILE
    gate_up_vmem_bytes = vmem_bytes_gate_up(
        blocks, gate_up_padded_rank, hidden_size
    )
    contracted_vmem_bytes = vmem_bytes_gate_up_down_u(
        contracted_blocks,
        contracted_padded_rank,
        contracted_padded_rank,
        hidden_size,
    )
    down_vmem_bytes = vmem_bytes_xuv(
        intermediate_size, down_padded_rank, hidden_size
    )
    x = jax.random.normal(
        jax.random.key(args.seed),
        (rows, hidden_size),
        dtype=jnp.bfloat16,
    )
    jax.block_until_ready(x)

    compiled = {}
    compile_seconds = {}
    if args.benchmark == "both":
        kinds = ("pallas", "unfused")
    elif args.benchmark == "all":
        kinds = ("pallas", "contracted", "unfused")
    else:
        kinds = (args.benchmark,)
    blocks_by_kind = {
        "pallas": blocks,
        "contracted": contracted_blocks,
        "unfused": blocks,
    }
    for kind in kinds:
        start = time.perf_counter()
        compiled[kind] = (
            _compile_sweep(kind, blocks_by_kind[kind], x, factors)
            if args.compile_sweep
            else _compile_layer(kind, blocks_by_kind[kind], x, factors[0])
        )
        compile_seconds[kind] = time.perf_counter() - start

    verification = None
    if not args.no_verify and any(kind != "unfused" for kind in kinds):
        unfused = _compile_layer("unfused", blocks, x, factors[0])
        verification = {}
        for kind in kinds:
            if kind == "unfused":
                continue
            candidate = _compile_layer(kind, blocks_by_kind[kind], x, factors[0])
            per_layer = [
                {
                    "layer": layer_index,
                    **_verify_layer(candidate, unfused, x, layer_factors),
                }
                for layer_index, layer_factors in zip(selected_layers, factors)
            ]
            verification[kind] = {
                "layers_checked": len(per_layer),
                "max_abs_error": max(item["max_abs_error"] for item in per_layer),
                "max_relative_l2_error": max(
                    item["relative_l2_error"] for item in per_layer
                ),
                "any_actual_nonfinite": any(
                    item["actual_nonfinite_count"] for item in per_layer
                ),
                "any_reference_nonfinite": any(
                    item["reference_nonfinite_count"] for item in per_layer
                ),
                "per_layer": per_layer,
            }

    results = {}
    for kind in kinds:
        results[kind] = {
            "compile_seconds": compile_seconds[kind],
            **_measure(
                compiled[kind],
                x,
                factors,
                warmup=args.warmup,
                iterations=args.iterations,
                compile_sweep=args.compile_sweep,
            ),
        }
        results[kind]["throughput_tokens_per_second"] = (
            rows * args.iterations / results[kind]["measured_seconds"]
        )

    speedup = None
    if "pallas" in results and "unfused" in results:
        speedup = (
            results["unfused"]["latency_ms_avg"]
            / results["pallas"]["latency_ms_avg"]
        )

    report = {
        "synthetic_factors": False,
        "checkpoint": str(Path(args.model_path).resolve()),
        "benchmark_scope": "all_selected_mlp_layers_only",
        "end_to_end_model_claim": False,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rank": rank,
        "logical_rank": rank,
        "gate_up_padded_rank": gate_up_padded_rank,
        "contracted_padded_rank": contracted_padded_rank,
        "down_padded_rank": down_padded_rank,
        "retained_parameter_ratio_actual": retained_parameter_ratio,
        "gate_up_effective_parameter_ratio_after_padding": (
            gate_up_padded_rank
            * (hidden_size + intermediate_size)
            / (hidden_size * intermediate_size)
        ),
        "down_effective_parameter_ratio_after_padding": (
            down_padded_rank
            * (hidden_size + intermediate_size)
            / (hidden_size * intermediate_size)
        ),
        "checkpoint_layer_count": num_layers,
        "layers_measured": list(selected_layers),
        "measured_kinds": list(kinds),
        "host_dispatches_per_sweep": (
            1 if args.compile_sweep else len(selected_layers)
        ),
        "explicit_pallas_kernel_calls_per_sweep": {
            kind: (
                2 * len(selected_layers)
                if kind == "pallas"
                else len(selected_layers) if kind == "contracted" else 0
            )
            for kind in kinds
        },
        "compiled_sweep": args.compile_sweep,
        "gate_up_blocks_source": "tuning_config" if tuned_blocks else "default",
        "weights_resident_in_tpu_hbm": True,
        "input_source": "jax.random.normal_bfloat16",
        "input_stats": _array_stats(x),
        "input_dtype": str(x.dtype),
        "factor_dtype": str(factors[0][0].dtype),
        "scratch_dtype": str(x.dtype),
        "accumulator_dtype": "float32",
        "pallas_svd_dma_strategy": "single_buffer_synchronous",
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
        "contracted_path_measured": "contracted" in results,
        "contracted_blocks": (
            contracted_blocks._asdict() if "contracted" in results else None
        ),
        "contracted_intermediate": (
            "down_xu" if "contracted" in results else None
        ),
        "contracted_intermediate_shape": (
            [rows, rank] if "contracted" in results else None
        ),
        "contracted_kernel_padded_output_shape": (
            [
                math.ceil(rows / contracted_blocks.bs) * contracted_blocks.bs,
                contracted_padded_rank,
            ]
            if "contracted" in results
            else None
        ),
        "unfused_uses_logical_rank": "unfused" in results,
        "pallas_uses_zero_padded_rank": any(
            kind in results for kind in ("pallas", "contracted")
        ),
        "estimated_vmem_bytes": {
            "gate_up": gate_up_vmem_bytes,
            "down_xuv": down_vmem_bytes,
            "contracted": (
                contracted_vmem_bytes if "contracted" in results else None
            ),
        },
        "verification": verification,
        "results": results,
        "speedup_pallas_vs_unfused_svd": speedup,
        "speedup_contracted_vs_pallas": (
            results["pallas"]["latency_ms_avg"]
            / results["contracted"]["latency_ms_avg"]
            if "pallas" in results and "contracted" in results
            else None
        ),
        "speedup_contracted_vs_unfused_svd": (
            results["unfused"]["latency_ms_avg"]
            / results["contracted"]["latency_ms_avg"]
            if "unfused" in results and "contracted" in results
            else None
        ),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
