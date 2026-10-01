"""Compile and execute both NeuronMLP Pallas kernels on a real TPU."""

from __future__ import annotations

import json
import math
import time

import jax
import jax.numpy as jnp

from svdflash_tpu.kernels.config import GateUpBlocks
from svdflash_tpu.kernels.pallas_ops import fused_gate_up, xuv_matmul


GATE_UP_BLOCKS = GateUpBlocks(bs=128, bh=128, br=128, bn=128)


def _random(key, shape):
    return (
        jax.random.normal(key, shape, dtype=jnp.bfloat16) * jnp.bfloat16(0.05)
    ).astype(jnp.bfloat16)


def _compile_and_run(fn, args):
    started = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    output = compiled(*args)
    jax.block_until_ready(output)
    first_run_ms = (time.perf_counter() - started) * 1000.0
    return output, compile_seconds, first_run_ms


def _errors(actual, expected):
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    difference = actual - expected
    max_abs, relative_l2 = jax.device_get(
        (
            jnp.max(jnp.abs(difference)),
            jnp.linalg.norm(difference)
            / jnp.maximum(jnp.linalg.norm(expected), 1e-12),
        )
    )
    return float(max_abs), float(relative_l2)


def main() -> None:
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"TPU smoke test requires the TPU backend, got {jax.default_backend()!r}"
        )

    keys = jax.random.split(jax.random.key(0), 8)

    xuv_args = (
        _random(keys[0], (128, 1024)),
        _random(keys[1], (1024, 512)),
        _random(keys[2], (512, 1024)),
    )
    xuv_reference = jax.jit(lambda x, u, v: (x @ u) @ v)(*xuv_args)
    xuv_output, xuv_compile, xuv_first_run = _compile_and_run(
        xuv_matmul, xuv_args
    )
    xuv_max_abs, xuv_relative_l2 = _errors(xuv_output, xuv_reference)

    gate_args = (
        _random(keys[3], (128, 128)),
        _random(keys[4], (128, 128)),
        _random(keys[5], (128, 128)),
        _random(keys[6], (128, 128)),
        _random(keys[7], (128, 128)),
    )
    gate_reference = jax.jit(
        lambda x, ug, vg, uu, vu: jax.nn.silu((x @ ug) @ vg)
        * ((x @ uu) @ vu)
    )(*gate_args)
    gate_fn = lambda *args: fused_gate_up(  # noqa: E731
        *args, blocks=GATE_UP_BLOCKS
    )
    gate_output, gate_compile, gate_first_run = _compile_and_run(
        gate_fn, gate_args
    )
    gate_max_abs, gate_relative_l2 = _errors(gate_output, gate_reference)

    max_relative_l2 = 0.05
    passed = all(
        math.isfinite(error) and error <= max_relative_l2
        for error in (xuv_relative_l2, gate_relative_l2)
    )
    report = {
        "passed": passed,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "xuv": {
            "compile_seconds": xuv_compile,
            "first_run_ms": xuv_first_run,
            "max_abs_error": xuv_max_abs,
            "relative_l2_error": xuv_relative_l2,
        },
        "gate_up": {
            "compile_seconds": gate_compile,
            "first_run_ms": gate_first_run,
            "max_abs_error": gate_max_abs,
            "relative_l2_error": gate_relative_l2,
        },
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
