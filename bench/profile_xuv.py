"""Capture TPU traces for vendor, Pallas, and unfused XUV matmul paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.pallas_ops import xuv_matmul


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _vendor_xuv(x, u, v):
    return (x @ u) @ v


def _unfused_xuv(x, u, v):
    xu = x @ u
    return xu @ v


def _pallas_xuv(x, u, v):
    return xuv_matmul(
        x,
        u,
        v,
        mixed_precision=True,
        interpret=False,
    )


def _compile(fn, args):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(5):
        jax.block_until_ready(compiled(*args))
    return compiled


def _capture(name, compiled, args, trace_root: Path, iterations: int):
    trace_dir = trace_root / name
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    try:
        for _ in range(iterations):
            jax.block_until_ready(compiled(*args))
    finally:
        jax.profiler.stop_trace()
    return str(trace_dir)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=_positive_int, default=128)
    parser.add_argument("--input-size", type=_positive_int, default=14336)
    parser.add_argument("--rank", type=_positive_int, default=2816)
    parser.add_argument("--output-size", type=_positive_int, default=4096)
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--trace-root", default="/tmp/xuv_profiles")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("vendor", "pallas", "unfused"),
        default=("vendor", "pallas", "unfused"),
    )
    parser.add_argument("--output", default="/tmp/xuv_profiles/manifest.json")
    return parser.parse_args()


def main():
    args = _parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"TPU profiling requires the TPU backend, got {jax.default_backend()!r}"
        )

    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(
        keys[0], (args.rows, args.input_size), dtype=jnp.bfloat16
    )
    u = jax.random.normal(
        keys[1], (args.input_size, args.rank), dtype=jnp.bfloat16
    )
    v = jax.random.normal(
        keys[2], (args.rank, args.output_size), dtype=jnp.bfloat16
    )
    args_tuple = (x, u, v)
    jax.block_until_ready(args_tuple)

    functions = {
        "vendor": _vendor_xuv,
        "pallas": _pallas_xuv,
        "unfused": _unfused_xuv,
    }
    trace_root = Path(args.trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "shape": {
            "x": [args.rows, args.input_size],
            "u": [args.input_size, args.rank],
            "v": [args.rank, args.output_size],
        },
        "iterations": args.iterations,
        "traces": {},
    }

    for mode in args.modes:
        compiled = _compile(functions[mode], args_tuple)
        manifest["traces"][mode] = _capture(
            mode,
            compiled,
            args_tuple,
            trace_root,
            args.iterations,
        )
        print(f"saved {mode} trace to {manifest['traces'][mode]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest to {output_path}")


if __name__ == "__main__":
    main()
