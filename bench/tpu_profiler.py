"""Capture isolated XProf traces for SVD-Flash workloads on TPU.

This is the TPU/JAX counterpart of NeuronMM's ``nki_profiler.py``: inputs are
created once, each workload is compiled and warmed up before tracing, and only
steady-state executions are captured. Open the resulting run directory with
TensorBoard's profile plugin, or pass ``--perfetto`` to also emit a Perfetto
trace.

Example::

    python -m svdflash_tpu.bench.tpu_profiler \
      --model llama-3.1-8b --sequence-length 128 \
      --workloads pallas-svd-mlp pallas-contracted-mlp \
      --warmup 5 --iterations 10 --perfetto

    tensorboard --logdir /tmp/svdflash_tpu_profiles

The manifest contains reproducibility metadata and static VMEM estimates. HBM
traffic, actual VMEM use, and MXU utilization must be read from the captured
XProf trace; the estimates in the manifest are not hardware measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.config import (
    MODELS,
    XUV_INPUT_TILE,
    XUV_OUTPUT_TILE,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
    GateUpBlocks,
    fits_vmem,
    gate_up_block_params,
    rank_for,
    vmem_bytes_gate_up,
    vmem_bytes_gate_up_down_u,
    vmem_bytes_xuv,
)
from svdflash_tpu.kernels.pallas_ops import (
    contracted_svd_mlp,
    dense_mlp,
    fused_gate_up,
    svd_mlp,
    xuv_matmul,
)


WORKLOADS = (
    "vendor-dense-mlp",
    "pallas-dense-mlp",
    "vendor-svd-mlp",
    "pallas-gate-up",
    "vendor-down-xuv",
    "pallas-down-xuv",
    "pallas-svd-mlp",
    "pallas-contracted-mlp",
)
DEFAULT_WORKLOADS = ("pallas-svd-mlp",)


@dataclass(frozen=True)
class Workload:
    fn: Callable[..., jax.Array]
    args: tuple[jax.Array, ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_blocks(value: str) -> GateUpBlocks:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "blocks must be four comma-separated integers: bs,bh,br,bn"
        ) from error
    if len(values) != 4 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "blocks must be four positive integers: bs,bh,br,bn"
        )
    return GateUpBlocks(*values)


def _random_array(key, shape: tuple[int, int], scale: float) -> jax.Array:
    return (jax.random.normal(key, shape, dtype=jnp.bfloat16) * scale).astype(
        jnp.bfloat16
    )


def _make_svd_inputs(seed: int, rows: int, hidden: int, intermediate: int, rank: int):
    keys = jax.random.split(jax.random.key(seed), 8)
    x = _random_array(keys[0], (rows, hidden), 1.0)
    down_input = _random_array(keys[1], (rows, intermediate), 1.0)
    factors = (
        _random_array(keys[2], (hidden, rank), 1.0 / math.sqrt(hidden)),
        _random_array(keys[3], (rank, intermediate), 1.0 / math.sqrt(rank)),
        _random_array(keys[4], (hidden, rank), 1.0 / math.sqrt(hidden)),
        _random_array(keys[5], (rank, intermediate), 1.0 / math.sqrt(rank)),
        _random_array(keys[6], (intermediate, rank), 1.0 / math.sqrt(intermediate)),
        _random_array(keys[7], (rank, hidden), 1.0 / math.sqrt(rank)),
    )
    jax.block_until_ready((x, down_input, factors))
    return x, down_input, factors


def _make_dense_inputs(seed: int, rows: int, hidden: int, intermediate: int):
    keys = jax.random.split(jax.random.key(seed), 4)
    x = _random_array(keys[0], (rows, hidden), 1.0)
    weights = (
        _random_array(keys[1], (hidden, intermediate), 1.0 / math.sqrt(hidden)),
        _random_array(keys[2], (hidden, intermediate), 1.0 / math.sqrt(hidden)),
        _random_array(
            keys[3], (intermediate, hidden), 1.0 / math.sqrt(intermediate)
        ),
    )
    jax.block_until_ready((x, weights))
    return x, weights


def _vendor_dense_mlp(x, w_gate, w_up, w_down):
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def _vendor_svd_mlp(x, u_gate, v_gate, u_up, v_up, u_down, v_down):
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    return ((jax.nn.silu(gate) * up) @ u_down) @ v_down


def _vendor_down_xuv(x, u, v):
    return (x @ u) @ v


def _expand_workloads(requested: Sequence[str]) -> tuple[str, ...]:
    if "all" in requested:
        return WORKLOADS
    return tuple(dict.fromkeys(requested))


def _build_workloads(
    selected: Sequence[str],
    *,
    seed: int,
    rows: int,
    hidden: int,
    intermediate: int,
    rank: int,
    gate_blocks: GateUpBlocks,
    contracted_blocks: GateUpBlocks,
) -> dict[str, Workload]:
    selected_set = set(selected)
    workloads: dict[str, Workload] = {}

    dense_names = {"vendor-dense-mlp", "pallas-dense-mlp"}
    if selected_set & dense_names:
        dense_x, dense_weights = _make_dense_inputs(
            seed + 10_000, rows, hidden, intermediate
        )
        if "vendor-dense-mlp" in selected_set:
            workloads["vendor-dense-mlp"] = Workload(
                _vendor_dense_mlp, (dense_x, *dense_weights)
            )
        if "pallas-dense-mlp" in selected_set:
            workloads["pallas-dense-mlp"] = Workload(
                lambda *xs: dense_mlp(*xs), (dense_x, *dense_weights)
            )

    if selected_set - dense_names:
        x, down_input, factors = _make_svd_inputs(
            seed, rows, hidden, intermediate, rank
        )
        u_gate, v_gate, u_up, v_up, u_down, v_down = factors
        if "vendor-svd-mlp" in selected_set:
            workloads["vendor-svd-mlp"] = Workload(
                _vendor_svd_mlp, (x, *factors)
            )
        if "pallas-gate-up" in selected_set:
            workloads["pallas-gate-up"] = Workload(
                lambda *xs: fused_gate_up(*xs, blocks=gate_blocks),
                (x, u_gate, v_gate, u_up, v_up),
            )
        if "vendor-down-xuv" in selected_set:
            workloads["vendor-down-xuv"] = Workload(
                _vendor_down_xuv, (down_input, u_down, v_down)
            )
        if "pallas-down-xuv" in selected_set:
            workloads["pallas-down-xuv"] = Workload(
                lambda *xs: xuv_matmul(*xs), (down_input, u_down, v_down)
            )
        if "pallas-svd-mlp" in selected_set:
            workloads["pallas-svd-mlp"] = Workload(
                lambda *xs: svd_mlp(*xs, gate_up_blocks=gate_blocks),
                (x, *factors),
            )
        if "pallas-contracted-mlp" in selected_set:
            workloads["pallas-contracted-mlp"] = Workload(
                lambda *xs: contracted_svd_mlp(
                    *xs, gate_up_blocks=contracted_blocks
                ),
                (x, *factors),
            )

    return {name: workloads[name] for name in selected}


def _percentile(sorted_samples: Sequence[float], percentile: float) -> float:
    index = min(
        len(sorted_samples) - 1,
        max(0, math.ceil(percentile / 100.0 * len(sorted_samples)) - 1),
    )
    return sorted_samples[index]


def _compile_and_warm_up(workload: Workload, warmup: int):
    started = time.perf_counter()
    compiled = jax.jit(workload.fn).lower(*workload.args).compile()
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(compiled(*workload.args))
    return compiled, compile_seconds


def _capture(
    name: str,
    workload: Workload,
    compiled,
    trace_dir: Path,
    iterations: int,
    create_perfetto_trace: bool,
) -> dict:
    trace_dir.mkdir(parents=True, exist_ok=True)
    samples_ms = []
    jax.profiler.start_trace(
        str(trace_dir), create_perfetto_trace=create_perfetto_trace
    )
    try:
        for step in range(iterations):
            started_ns = time.perf_counter_ns()
            with jax.profiler.StepTraceAnnotation(name, step_num=step):
                output = compiled(*workload.args)
                jax.block_until_ready(output)
            samples_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
    finally:
        jax.profiler.stop_trace()

    samples_ms.sort()
    return {
        "trace_dir": str(trace_dir),
        "iterations": iterations,
        "capture_seconds": sum(samples_ms) / 1000.0,
        "latency_ms_avg": statistics.fmean(samples_ms),
        "latency_ms_p50": statistics.median(samples_ms),
        "latency_ms_p90": _percentile(samples_ms, 90),
        "latency_ms_p99": _percentile(samples_ms, 99),
        "latency_ms_min": samples_ms[0],
        "latency_ms_max": samples_ms[-1],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="llama-3.1-8b")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--sequence-length", type=_positive_int, default=128)
    parser.add_argument("--compress-ratio", type=float, default=0.8)
    parser.add_argument("--warmup", type=_nonnegative_int, default=5)
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gate-up-blocks",
        type=_parse_blocks,
        help="override Gate/Up blocks as bs,bh,br,bn",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=(*WORKLOADS, "all"),
        default=DEFAULT_WORKLOADS,
    )
    parser.add_argument("--trace-root", default="/tmp/svdflash_tpu_profiles")
    parser.add_argument(
        "--run-name",
        help="trace subdirectory name; defaults to a UTC timestamp and shape",
    )
    parser.add_argument(
        "--output",
        help="manifest path; defaults to <run-directory>/manifest.json",
    )
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="also create perfetto_trace.json.gz in each trace directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"TPU profiling requires the TPU backend, got {jax.default_backend()!r}"
        )
    if not 0.0 < args.compress_ratio <= 1.0:
        raise ValueError("compress-ratio must be in the interval (0, 1]")

    selected = _expand_workloads(args.workloads)
    model = MODELS[args.model]
    rows = args.batch_size * args.sequence_length
    rank = rank_for(model, args.compress_ratio)
    gate_blocks = args.gate_up_blocks or gate_up_block_params(
        rows, model, args.compress_ratio
    )
    contracted_blocks = gate_blocks._replace(bn=min(gate_blocks.bn, 512))

    gate_padded_rank = math.ceil(rank / gate_blocks.br) * gate_blocks.br
    contracted_padded_rank = (
        math.ceil(rank / contracted_blocks.br) * contracted_blocks.br
    )
    down_padded_rank = math.ceil(rank / XUV_RANK_TILE) * XUV_RANK_TILE
    gate_vmem = vmem_bytes_gate_up(gate_blocks, gate_padded_rank, model.hidden_size)
    contracted_vmem = vmem_bytes_gate_up_down_u(
        contracted_blocks,
        contracted_padded_rank,
        contracted_padded_rank,
        model.hidden_size,
    )
    down_vmem = vmem_bytes_xuv(
        model.intermediate_size, down_padded_rank, model.hidden_size
    )

    if any(name in selected for name in ("pallas-gate-up", "pallas-svd-mlp")):
        if not fits_vmem(gate_vmem):
            raise ValueError(f"estimated Gate/Up VMEM exceeds safety budget: {gate_vmem}")
    if "pallas-contracted-mlp" in selected and not fits_vmem(contracted_vmem):
        raise ValueError(
            f"estimated contracted MLP VMEM exceeds safety budget: {contracted_vmem}"
        )
    if any(name in selected for name in ("pallas-down-xuv", "pallas-svd-mlp")):
        if not fits_vmem(down_vmem):
            raise ValueError(f"estimated Down XUV VMEM exceeds safety budget: {down_vmem}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{stamp}_{args.model}_rows{rows}_rank{rank}"
    run_dir = Path(args.trace_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    workloads = _build_workloads(
        selected,
        seed=args.seed,
        rows=rows,
        hidden=model.hidden_size,
        intermediate=model.intermediate_size,
        rank=rank,
        gate_blocks=gate_blocks,
        contracted_blocks=contracted_blocks,
    )

    report = {
        "profile_scope": "single_synthetic_transformer_mlp_layer",
        "synthetic_inputs": True,
        "hardware_metrics_source": "XProf trace (not parsed into this manifest)",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "model": args.model,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "rows": rows,
        "hidden_size": model.hidden_size,
        "intermediate_size": model.intermediate_size,
        "compress_ratio": args.compress_ratio,
        "rank": rank,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gate_up_blocks": gate_blocks._asdict(),
        "contracted_blocks": contracted_blocks._asdict(),
        "xuv_tiles": {
            "rows": XUV_ROW_TILE,
            "input": XUV_INPUT_TILE,
            "rank": XUV_RANK_TILE,
            "output": XUV_OUTPUT_TILE,
        },
        "static_vmem_estimates_bytes": {
            "gate_up": gate_vmem,
            "down_xuv": down_vmem,
            "contracted_mlp": contracted_vmem,
        },
        "perfetto_trace_requested": args.perfetto,
        "run_dir": str(run_dir),
        "profiles": {},
    }

    for name, workload in workloads.items():
        print(f"compiling and warming up {name} ...", flush=True)
        compiled, compile_seconds = _compile_and_warm_up(workload, args.warmup)
        print(f"capturing {name} ...", flush=True)
        profile = _capture(
            name,
            workload,
            compiled,
            run_dir / name,
            args.iterations,
            args.perfetto,
        )
        profile["compile_seconds"] = compile_seconds
        report["profiles"][name] = profile
        print(f"saved {name} trace to {profile['trace_dir']}", flush=True)

    output_path = Path(args.output) if args.output else run_dir / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved manifest to {output_path}")


if __name__ == "__main__":
    main()
