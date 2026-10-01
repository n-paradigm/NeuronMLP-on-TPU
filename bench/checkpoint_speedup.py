"""Independent real-checkpoint MLP measurements for test_speedup.py.

Layers share a reproducible synthetic input, never each other's MLP output.
Only one layer's factors reside on the device at a time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from svdflash_tpu.models.hf_svd import inspect_mlp_checkpoint, load_mlp_factors


def _errors(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"output shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("MLP output or reference contains NaN/Inf")
    difference = actual - expected
    return {
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "reference_rms": float(np.sqrt(np.mean(expected**2))),
    }


def summarize(records, selected_layers, seq_lens):
    """Ratio of sums of layer medians, not the mean of layer speedups."""
    summaries = []
    for S in seq_lens:
        rows = [row for row in records if row["S"] == S]
        complete = (len(rows) == len(selected_layers)
                    and {row["layer"] for row in rows} == set(selected_layers))
        passed = complete and all(row["passed"] for row in rows)
        result = {"S": S, "layers_completed": len(rows), "layers_expected": len(selected_layers),
                  "complete": complete, "passed": passed,
                  "failed_layers": [row["layer"] for row in rows if not row["passed"]]}
        if rows:
            result["max_fused_relative_l2"] = max(row["fused_vs_fp32"]["relative_l2"] for row in rows)
        if passed and all("fused_ms" in row and "unfused_low_rank_ms" in row for row in rows):
            fused_ms = sum(row["fused_ms"] for row in rows)
            unfused_ms = sum(row["unfused_low_rank_ms"] for row in rows)
            result.update(fused_ms_sum=fused_ms, unfused_low_rank_ms_sum=unfused_ms,
                          speedup_vs_unfused=unfused_ms/fused_ms)
        summaries.append(result)
    return summaries


def _save_report(report, path):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    columns = ("layer", "S", "rank", "dtype", "down_backend", "passed", "fused_ms", "unfused_low_rank_ms",
               "speedup_vs_unfused", "fused_relative_l2", "unfused_relative_l2", "max_abs_error")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in report["records"]:
            row = {key: record[key] for key in columns if key in record}
            row.update(fused_relative_l2=record["fused_vs_fp32"]["relative_l2"],
                       unfused_relative_l2=record["unfused_vs_fp32"]["relative_l2"],
                       max_abs_error=record["fused_vs_fp32"]["max_abs_error"])
            writer.writerow(row)


def run(args, kernels):
    """Run the --model-path branch of the original test_speedup CLI."""
    metadata = inspect_mlp_checkpoint(args.model_path, args.layers)
    if len(set(args.seq_lens)) != len(args.seq_lens):
        raise ValueError("sequence lengths must not contain duplicates")
    if args.input_scale <= 0 or not np.isfinite(args.input_scale):
        raise ValueError("input-scale must be finite and positive")
    if args.max_relative_l2 is not None and (
        args.max_relative_l2 <= 0 or not np.isfinite(args.max_relative_l2)
    ):
        raise ValueError("max-relative-l2 must be finite and positive")
    mode = ("checkpoint_inspection" if args.inspect_only else
            "interpret_correctness_only" if args.interpret else
            "tpu_correctness_only" if args.check_only else "tpu_benchmark")
    report = {
        "mode": mode, "complete": False, "jax": jax.__version__,
        "async_dma": not args.sync_dma, "down_backend": args.down_backend,
        "checkpoint": metadata, "seq_lens": args.seq_lens,
        "input": {"kind": "synthetic_standard_normal", "scale": args.input_scale,
                  "seed": args.seed, "shared_across_layers": True, "chain_layers": False},
        "requested_dtype": args.dtype or "checkpoint",
        "timing": {"statistic": "median_ms", "warmup": args.warmup,
                   "repeats": args.repeats, "load_and_compile_included": False,
                   "aggregation": "sum_unfused_layer_medians / sum_fused_layer_medians"},
        "records": [], "summaries": [],
    }
    H, I = metadata["hidden_size"], metadata["intermediate_size"]
    print(json.dumps({"checkpoint": metadata["model_path"], "shards": len(metadata["shards"]),
                      "selected_layers": metadata["selected_layers"], "H": H, "I": I,
                      "ranks": sorted({row["rank"] for row in metadata["layers"]}),
                      "mode": mode}), flush=True)
    if args.inspect_only:
        report["complete"] = True
        _save_report(report, args.output)
        return report
    if not args.interpret and jax.default_backend() != "tpu":
        raise RuntimeError("real-checkpoint latency requires TPU; use --inspect-only or --interpret")
    report["devices"] = [str(device) for device in jax.devices()]
    _save_report(report, args.output)

    profiling = getattr(args, "profile_dir", None) is not None
    if profiling:
        from pathlib import Path
        profile_dir = Path(args.profile_dir).expanduser().resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        jax.block_until_ready(())
        print(f"Starting JAX profiler: {profile_dir}", flush=True)
        jax.profiler.start_trace(str(profile_dir))

    # Reuse the same jitted callable across layers. Weights are dynamic arguments,
    # so equal shapes/dtypes reuse compilation instead of embedding checkpoint constants.
    functions = {}
    inputs = {}
    rng = np.random.default_rng(args.seed)
    for S in args.seq_lens:
        inputs[S] = rng.normal(size=(S, H)).astype(np.float32) * args.input_scale

    try:
        for info in metadata["layers"]:
            layer = info["layer"]
            print(f"Layer {layer}: loading six real SVD factors (rank={info['rank']})...", flush=True)
            loaded, = load_mlp_factors(args.model_path, [layer])
            # hf_svd order is gate/up/down; NeuronMM test_speedup order is up/gate/down.
            ug, vg, uu, vu, ud, vd = loaded
            dtype = loaded[0].dtype if args.dtype is None else (
                jnp.bfloat16 if args.dtype == "bf16" else jnp.float32)
            factors = tuple(a.astype(dtype) for a in (uu, vu, ug, vg, ud, vd))
            jax.block_until_ready(factors)
            del loaded, ug, vg, uu, vu, ud, vd
            try:
                for S in args.seq_lens:
                    mixed = jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16)
                    tolerance = args.max_relative_l2 or (0.12 if mixed else 0.001)
                    up_params = kernels.get_fused_mlp_up_T_params(S)
                    down_params = kernels.get_fused_three_mm_XTUV_params(S)
                    up_shape = kernels._fusion_shape(
                        H, info["rank"], S, I, dtype, up_params["M_tiles_in_block"],
                        up_params["r_tiles_in_block"], up_params["K_tiles_in_block"],
                        up_params["N_tiles_in_block"], True, async_dma=not args.sync_dma)
                    down_shape = kernels._fusion_shape(
                        I, info["rank"], S, H, dtype, down_params["M_tiles_in_block"],
                        down_params["r_tiles_in_block"], down_params["K_tiles_in_block"],
                        down_params["N_tiles_in_block"], False, async_dma=not args.sync_dma) if args.down_backend == "pallas" else None
                    key = (S, str(jnp.dtype(dtype)))
                    if key not in functions:
                        # Bind params as defaults: later loop iterations must not change
                        # a cached callable that retraces for a different layer rank.
                        def fused(x, *f, up=up_params, down=down_params, mp=mixed):
                            return kernels.svd_mlp_with_fused_kernel(
                                x, *f, up, down, mixed_precision=mp, interpret=args.interpret,
                                async_dma=not args.sync_dma, down_backend=args.down_backend)
                        def unfused(x, *f):
                            with jax.default_matmul_precision("highest"):
                                return kernels._unfused_low_rank(x, *f)
                        def reference(x, *f):
                            with jax.default_matmul_precision("highest"):
                                return kernels._unfused_reference(x, *f)
                        functions[key] = (jax.jit(fused), jax.jit(unfused), jax.jit(reference))
                fused, unfused, reference = functions[key]
                x = jnp.asarray(inputs[S], dtype=dtype)
                call_args = (x, *factors)
                print(f"Layer {layer}, S={S}: compile/reuse executables and verify...", flush=True)
                started = time.perf_counter()
                expected = np.asarray(reference(*call_args), dtype=np.float32)
                actual_unfused = np.asarray(unfused(*call_args), dtype=np.float32)
                actual_fused = np.asarray(fused(*call_args), dtype=np.float32)
                fused_error = _errors(actual_fused, expected)
                unfused_error = _errors(actual_unfused, expected)
                comparison = _errors(actual_fused, actual_unfused)
                passed = (fused_error["relative_l2"] <= tolerance
                          and unfused_error["relative_l2"] <= tolerance)
                record = {
                    "layer": layer, "S": S, "rank": info["rank"],
                    "dtype": str(jnp.dtype(dtype)), "up_params": up_params,
                    "down_backend": args.down_backend,
                    "down_params": down_params if down_shape is not None else None,
                    "estimated_vmem_bytes": {"gate_up": up_shape.vmem_bytes, "down": down_shape.vmem_bytes if down_shape is not None else None},
                    "fused_vs_fp32": fused_error, "unfused_vs_fp32": unfused_error,
                    "fused_vs_unfused": comparison, "max_relative_l2": tolerance, "passed": passed,
                    "compile_and_verify_seconds": time.perf_counter() - started,
                }
                if passed and not args.check_only and not args.interpret:
                    # Alternate measurement order between layers to reduce systematic bias.
                    jobs = [("fused_ms", fused), ("unfused_low_rank_ms", unfused)]
                    if layer % 2:
                        jobs.reverse()
                    for name, fn in jobs:
                        record[name] = kernels._time_ms(fn, call_args, args.warmup, args.repeats)
                    record["speedup_vs_unfused"] = record["unfused_low_rank_ms"] / record["fused_ms"]
                report["records"].append(record)
                report["summaries"] = summarize(report["records"], metadata["selected_layers"], args.seq_lens)
                _save_report(report, args.output)
                print(json.dumps(record, allow_nan=False), flush=True)
                del x, call_args, actual_fused, actual_unfused, expected
            finally:
                del factors
    finally:
        if profiling:
            jax.profiler.stop_trace()
            print(f"JAX profiler trace saved to: {profile_dir}", flush=True)

    report["complete"] = True
    _save_report(report, args.output)
    for summary in report["summaries"]:
        print(json.dumps({"summary": summary}), flush=True)
    if not all(row["passed"] for row in report["records"]):
        raise RuntimeError("some layers failed numerical verification; see the saved per-layer report")
    return report
