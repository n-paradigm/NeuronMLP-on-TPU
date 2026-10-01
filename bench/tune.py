"""Autotune the configurable Gate/Up Pallas kernel on TPU v5e."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from svdflash_tpu.bench.latency import (
    _make_factors,
    _measure,
    _parse_sequence_lengths,
    _random_array,
)
from svdflash_tpu.kernels.config import (
    MODELS,
    GateUpBlocks,
    gate_up_tuning_candidates,
    rank_for,
    vmem_bytes_gate_up,
)
from svdflash_tpu.kernels.pallas_ops import fused_gate_up


def _limit_candidates(candidates: Sequence, maximum: int) -> Sequence:
    return candidates if maximum <= 0 else candidates[:maximum]


def _run_candidate(
    *,
    kernel: str,
    blocks: GateUpBlocks,
    fn,
    fn_args,
    rows: int,
    rank: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    expected,
    max_relative_l2_error: float,
) -> dict:
    padded_rank = math.ceil(rank / blocks.br) * blocks.br
    vmem_bytes = vmem_bytes_gate_up(blocks, padded_rank, hidden_size)

    record = {
        "kernel": kernel,
        "blocks": blocks._asdict(),
        "padded_rank": padded_rank,
        "estimated_vmem_bytes": vmem_bytes,
    }
    try:
        timing = _measure(fn, fn_args, warmup, iterations)
        timing["throughput_tokens_per_second"] = (
            rows * timing["iterations"] / timing["measured_seconds"]
        )
        record.update(timing)
        if expected is not None:
            actual = fn(*fn_args)
            actual32 = actual.astype(jnp.float32)
            expected32 = expected.astype(jnp.float32)
            difference = actual32 - expected32
            max_abs_error, relative_l2_error = jax.device_get(
                (
                    jnp.max(jnp.abs(difference)),
                    jnp.linalg.norm(difference)
                    / jnp.maximum(jnp.linalg.norm(expected32), 1e-12),
                )
            )
            record["max_abs_error"] = float(max_abs_error)
            record["relative_l2_error"] = float(relative_l2_error)
            if not math.isfinite(float(relative_l2_error)) or float(
                relative_l2_error
            ) > max_relative_l2_error:
                record.update(
                    {
                        "status": "incorrect",
                        "error": (
                            f"relative L2 error {float(relative_l2_error):.6g} exceeds "
                            f"{max_relative_l2_error:.6g}"
                        ),
                    }
                )
            else:
                record["status"] = "ok"
        else:
            record["status"] = "ok"
    except Exception as error:  # TPU compiler/runtime failures are candidate results.
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        )
    return record


def _tune_model(args: argparse.Namespace, model_name: str) -> tuple[list[dict], list[dict]]:
    model = MODELS[model_name]
    rank = rank_for(model, args.compress_ratio)
    factors = _make_factors(
        args.seed, model.hidden_size, model.intermediate_size, rank
    )
    u_gate, v_gate, u_up, v_up, _, _ = factors
    records = []
    best = []

    for sequence_length in args.sequence_lengths:
        rows = args.batch_size * sequence_length
        x = _random_array(
            jax.random.key(args.seed + rows),
            (rows, model.hidden_size),
            1.0,
        )
        jax.block_until_ready(x)

        workloads = []
        if args.kernel == "gate-up":
            gate_expected = None
            if args.verify:
                gate_reference = jax.jit(
                    lambda x_, ug_, vg_, uu_, vu_: jax.nn.silu(
                        (x_ @ ug_) @ vg_
                    )
                    * ((x_ @ uu_) @ vu_)
                )
                gate_expected = gate_reference(x, u_gate, v_gate, u_up, v_up)
                jax.block_until_ready(gate_expected)
            candidates = gate_up_tuning_candidates(
                rows, model, args.compress_ratio, args.profile
            )
            workloads.append(
                (
                    "gate-up",
                    _limit_candidates(candidates, args.max_candidates),
                    lambda blocks: (
                        lambda *xs: fused_gate_up(*xs, blocks=blocks),
                        (x, u_gate, v_gate, u_up, v_up),
                    ),
                    gate_expected,
                )
            )
        for kernel, candidates, make_call, expected in workloads:
            print(
                f"Tuning {model_name} rows={rows} kernel={kernel} "
                f"candidates={len(candidates)}",
                file=sys.stderr,
            )
            kernel_records = []
            for index, blocks in enumerate(candidates, start=1):
                fn, fn_args = make_call(blocks)
                record = _run_candidate(
                    kernel=kernel,
                    blocks=blocks,
                    fn=fn,
                    fn_args=fn_args,
                    rows=rows,
                    rank=rank,
                    hidden_size=model.hidden_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    expected=expected,
                    max_relative_l2_error=args.max_relative_l2_error,
                )
                record.update(
                    {
                        "model": model_name,
                        "batch_size": args.batch_size,
                        "sequence_length": sequence_length,
                        "rows": rows,
                        "compress_ratio": args.compress_ratio,
                        "rank": rank,
                        "candidate_index": index,
                    }
                )
                records.append(record)
                kernel_records.append(record)
                status = (
                    f"{record['latency_ms_avg']:.4f} ms"
                    if record["status"] == "ok"
                    else record.get("error_type", record["status"])
                )
                print(
                    f"  [{index}/{len(candidates)}] {blocks}: {status}",
                    file=sys.stderr,
                )
                if not args.keep_compiler_cache:
                    jax.clear_caches()

            successful = [
                record for record in kernel_records if record["status"] == "ok"
            ]
            if successful:
                winner = min(successful, key=lambda record: record["latency_ms_avg"])
                best.append(winner)

    return records, best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=("qwen3-8b",),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_sequence_lengths,
        default=(1, 128, 512, 1024),
        help="comma-separated token counts",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratio", type=float, default=0.8)
    parser.add_argument("--kernel", choices=("gate-up",), default="gate-up")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="limit candidates per kernel/workload; zero means all",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare every candidate with an unfused JAX reference",
    )
    parser.add_argument("--max-relative-l2-error", type=float, default=0.05)
    parser.add_argument("--keep-compiler-cache", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"Pallas tuning requires TPU, got backend {jax.default_backend()!r}"
        )
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch-size and iterations must be positive; warmup may be zero")
    if args.max_candidates < 0:
        raise ValueError("max-candidates may not be negative")
    if args.max_relative_l2_error <= 0:
        raise ValueError("max-relative-l2-error must be positive")
    if not 0.0 < args.compress_ratio <= 1.0:
        raise ValueError("compress-ratio must be in the interval (0, 1]")

    records = []
    best = []
    for model_name in args.models:
        model_records, model_best = _tune_model(args, model_name)
        records.extend(model_records)
        best.extend(model_best)

    report = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "profile": args.profile,
        "records": records,
        "best": best,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
