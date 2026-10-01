"""Compare current Pallas and JAX with identical inputs and FP32 references.
Each version uses its own default tiling unless explicitly overridden.

Example: python -m svdflash_tpu.bench.compare_dma
  --model-path /home/zouji/models/llama3-8b_0.9_svd --output comparison.json
Optionally add --baseline saved_kernel.py for saved/current comparisons.
--include-hybrid adds a full Pallas Gate/Up + JAX Down measurement.
Weights, compilation, verification and tracing are excluded from timing.
JAX Gate/Up and Down match the transposed Pallas stage interfaces. Standalone
stage timings cannot be added to infer the full-MLP gap. --interpret validates
small CPU examples and never emits performance measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
from importlib.machinery import SourceFileLoader
import json
import math
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from svdflash_tpu import test_speedup as installed
from svdflash_tpu import profiler as p


def load_module(path, name="_dma_baseline"):
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def errors(actual, expected):
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Non-finite output/reference")
    return dict(relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)),
                max_abs_error=float(np.max(np.abs(a-b))))



def jax_gate_up_t(x, ug, vg, uu, vu):
    """Match Pallas Gate/Up's BF16 boundaries and (intermediate, sequence) output."""
    with jax.default_matmul_precision("highest"):
        gate = (x @ ug) @ vg
        up = (x @ uu) @ vu
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(x.dtype) * up
        return hidden.T


def jax_down(hidden_t, ud, vd):
    """Same transposed hidden input and normal output as the Pallas Down stage."""
    with jax.default_matmul_precision("highest"):
        return (hidden_t.T @ ud) @ vd


def measure_interleaved(compiled, workloads, warmup, repeats):
    # Each round measures each workload once, in a seeded shuffled order.
    # Compilation and input generation happen before this function.
    for name, (_, inputs) in workloads.items():
        for _ in range(warmup):
            jax.block_until_ready(compiled[name](*inputs))
    samples = {name: [] for name in workloads}
    order_rng = np.random.default_rng(17)
    for _ in range(repeats):
        for name in order_rng.permutation(list(workloads)):
            inputs = workloads[name][1]
            start = time.perf_counter_ns()
            jax.block_until_ready(compiled[name](*inputs))
            samples[name].append((time.perf_counter_ns()-start)/1e6)
    return {name: dict(median_ms=statistics.median(ds),
                       p10_ms=float(np.percentile(ds, 10)),
                       p90_ms=float(np.percentile(ds, 90)), samples_ms=ds)
            for name, ds in samples.items()}


def summarize_stages(latency):
    comparisons = {}
    for stage, jax_name in (("gate-up", "jax/gate-up"), ("down", "jax/down"),
                            ("fused", "optimized/unfused")):
        current = latency["optimized/"+stage]
        vendor = latency[jax_name]
        # Samples at the same index were taken in the same shuffled round.
        paired = 1000*(np.asarray(current["samples_ms"]) - np.asarray(vendor["samples_ms"]))
        comparisons[stage] = dict(
            pallas_ms=current["median_ms"],
            jax_ms=vendor["median_ms"],
            speedup_vs_jax=vendor["median_ms"]/current["median_ms"],
            excess_us=(current["median_ms"]-vendor["median_ms"])*1000,
            paired_excess_us_median=float(np.median(paired)),
            paired_excess_us_p10=float(np.percentile(paired, 10)),
            paired_excess_us_p90=float(np.percentile(paired, 90)),
        )
        if "baseline/"+stage in latency:
            comparisons[stage]["baseline_ms"] = latency["baseline/"+stage]["median_ms"]
    return comparisons


def print_stage_comparison(comparisons):
    has_baseline = all("baseline_ms" in row for row in comparisons.values())
    saved_header = "   saved_ms" if has_baseline else ""
    print(f"stage     {saved_header}   pallas_ms      jax_ms  jax/pallas  excess_us", flush=True)
    for stage, row in comparisons.items():
        saved_value = f" {row['baseline_ms']:10.6f}" if has_baseline else ""
        print(f"{stage:10s}{saved_value} {row['pallas_ms']:10.6f} "
              f"{row['jax_ms']:11.6f} {row['speedup_vs_jax']:11.3f} {row['excess_us']:10.2f}", flush=True)
    print("Positive excess_us means Pallas is slower. Standalone stage gaps are not additive.", flush=True)


def summarize_full_mlp(latency):
    """Compare complete executables; never add standalone stage timings."""
    jax_ms = latency["optimized/unfused"]["median_ms"]
    pallas_ms = latency["optimized/fused"]["median_ms"]
    names = [("pallas", "optimized/fused"), ("jax", "optimized/unfused")]
    if "optimized/hybrid" in latency:
        names.insert(1, ("hybrid", "optimized/hybrid"))
    return {name: dict(median_ms=latency[key]["median_ms"],
                       p10_ms=latency[key]["p10_ms"], p90_ms=latency[key]["p90_ms"],
                       speedup_vs_jax=jax_ms/latency[key]["median_ms"],
                       speedup_vs_pallas=pallas_ms/latency[key]["median_ms"])
            for name, key in names}


def print_full_mlp_comparison(comparisons):
    print("full MLP        median_ms      p10_ms      p90_ms   jax/path  pallas/path", flush=True)
    for name, row in comparisons.items():
        print(f"{name:12s} {row['median_ms']:12.6f} {row['p10_ms']:11.6f} {row['p90_ms']:11.6f} "
              f"{row['speedup_vs_jax']:10.3f} {row['speedup_vs_pallas']:12.3f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="optional saved kernel source for old/current comparison")
    parser.add_argument("--interpret", action="store_true", help="CPU correctness only; use small explicit dimensions")
    parser.add_argument("--case", choices=("synthetic", "checkpoint", "both"),
                        help="default: both with --model-path, otherwise synthetic")
    parser.add_argument("--include-hybrid", action="store_true",
                        help="also benchmark complete Pallas Gate/Up + JAX Down using the candidate")
    parser.add_argument("--candidate", help="optional candidate source; otherwise use installed kernel")
    parser.add_argument("--model-path")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=p.positive, default=128)
    parser.add_argument("--up-tiles", type=p.tile_params)
    parser.add_argument("--baseline-up-tiles", type=p.tile_params)
    parser.add_argument("--down-tiles", type=p.tile_params)
    parser.add_argument("--baseline-down-tiles", type=p.tile_params)
    parser.add_argument("--hidden", type=p.positive, default=3072)
    parser.add_argument("--intermediate", type=p.positive, default=8192)
    parser.add_argument("--rank", type=p.positive, default=1792)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--warmup", type=p.positive, default=10)
    parser.add_argument("--repeats", type=p.positive, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    baseline_path = Path(args.baseline).resolve() if args.baseline else None
    if baseline_path is not None and not baseline_path.is_file():
        parser.error(f"--baseline file not found: {baseline_path}. Omit --baseline to compare current Pallas with JAX.")
    if baseline_path is None and (args.require_bitwise or args.baseline_up_tiles or args.baseline_down_tiles):
        parser.error("--require-bitwise and --baseline-*-tiles require --baseline; omit these flags for Pallas/JAX comparison")
    if not args.interpret and jax.default_backend() != "tpu":
        parser.error("Performance comparison requires a TPU")
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    selected_case = args.case or ("both" if args.model_path else "synthetic")
    if selected_case in ("checkpoint", "both") and not args.model_path:
        parser.error("--case checkpoint/both requires --model-path")
    if args.interpret and not args.model_path and any(
            value > 2048 for value in (args.hidden, args.intermediate, args.rank, args.seq_len)):
        parser.error("--interpret is for small checks; provide --hidden/--intermediate/--rank/--seq-len <= 2048")
    baseline = load_module(baseline_path) if baseline_path is not None else None
    candidate = load_module(Path(args.candidate).resolve(), "_dma_candidate") if args.candidate else installed
    if args.include_hybrid and "down_backend" not in inspect.signature(candidate.svd_mlp_with_fused_kernel).parameters:
        parser.error("--include-hybrid requires an updated candidate test_speedup.py with down_backend support")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 <= 0:
        parser.error("--max-relative-l2 must be positive and finite")
    versions = [("optimized", candidate)]
    base_up = base_down = None
    if baseline is not None:
        versions.insert(0, ("baseline", baseline))
        base_up = args.baseline_up_tiles or args.up_tiles or baseline.get_fused_mlp_up_T_params(args.seq_len)
        base_down = args.baseline_down_tiles or args.down_tiles or baseline.get_fused_three_mm_XTUV_params(args.seq_len)
    up = args.up_tiles or candidate.get_fused_mlp_up_T_params(args.seq_len)
    down = args.down_tiles or candidate.get_fused_three_mm_XTUV_params(args.seq_len)
    hidden_source, hidden_module = versions[0]
    hidden_up = base_up if baseline is not None else up
    H, I, R = args.hidden, args.intermediate, args.rank
    if args.model_path:
        from svdflash_tpu.bench.checkpoint_speedup import inspect_mlp_checkpoint, load_mlp_factors
        metadata = inspect_mlp_checkpoint(args.model_path, [args.layer])
        H, I, R = metadata["hidden_size"], metadata["intermediate_size"], metadata["layers"][0]["rank"]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((args.seq_len, H), dtype=np.float32), jnp.bfloat16)
    shapes = ((H,R), (R,I), (H,R), (R,I), (I,R), (R,H))
    factors = tuple(jnp.asarray(rng.standard_normal(s, dtype=np.float32)/math.sqrt(s[0]),
                               jnp.bfloat16) for s in shapes)
    report = dict(complete=False, mode="interpret_correctness_only" if args.interpret else "tpu_benchmark", jax=jax.__version__, devices=[str(d) for d in jax.devices()],
                  shape=dict(H=H, I=I, rank=R, S=args.seq_len), dtype="bfloat16",
                  up_tiles=up, down_tiles=down, include_hybrid=args.include_hybrid,
                  hybrid_implementation="Pallas Gate/Up + JAX Down; one enclosing jit" if args.include_hybrid else None,
                  require_bitwise=args.require_bitwise, max_relative_l2=args.max_relative_l2,
                  warmup=args.warmup, repeats=args.repeats,
                  timing="disabled" if args.interpret else "randomized interleaved host wall time; block_until_ready; no trace",
                  stage_comparison_scope="identical array shapes, dtype, values, and transposed stage interfaces; XLA chooses internal layouts",
                  standalone_stage_times_additive=False,
                  stage_down_input_source=f"{hidden_source} Gate/Up output, shared by all Down implementations",
                  matmul_precision="highest for JAX baselines and FP32 references",
                  candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  cases={})
    if baseline is not None:
        report.update(baseline_up_tiles=base_up, baseline_down_tiles=base_down,
                      baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    compiled = {}
    reference = jax.jit(p._reference)
    try:
        for case in (["synthetic", "checkpoint"] if selected_case == "both" else [selected_case]):
            if case == "checkpoint":
                ug, vg, uu, vu, ud, vd = next(iter(load_mlp_factors(args.model_path, [args.layer])))
                factors = tuple(a.astype(jnp.bfloat16) for a in (uu,vu,ug,vg,ud,vd))
            jax.block_until_ready((x, factors))
            # Isolated Down uses identical hidden values for every implementation.
            uu, vu, ug, vg, ud, vd = factors
            gate_args = (x, ug, vg, uu, vu)
            hidden_key = hidden_source+"/gate-up"
            if hidden_key not in compiled:
                print("Compiling "+hidden_key, flush=True)
                fn = lambda *a: hidden_module.fused_mlp_up_T(*a, **hidden_up, interpret=args.interpret)
                compiled[hidden_key], _ = p._compile(fn, gate_args, args.warmup)
            hidden = jax.block_until_ready(compiled[hidden_key](*gate_args))
            workloads = {}
            for prefix, module in versions:
                entries = p.build_workloads(module, x, factors, hidden,
                                            base_up if prefix == "baseline" else up,
                                            base_down if prefix == "baseline" else down, interpret=args.interpret)
                for name, entry in entries.items():
                    if name != "unfused" or prefix == "optimized":
                        workloads[prefix+"/"+name] = entry
            if args.include_hybrid:
                def hybrid_fn(x, *f):
                    return candidate.svd_mlp_with_fused_kernel(
                        x, *f, up_T_params=up, mixed_precision=x.dtype == jnp.bfloat16,
                        interpret=args.interpret, down_backend="jax")
                workloads["optimized/hybrid"] = (hybrid_fn, (x, *factors))
            workloads["jax/gate-up"] = (jax_gate_up_t, gate_args)
            workloads["jax/down"] = (jax_down, (hidden, ud, vd))
            results = {}
            for name, (fn, inputs) in workloads.items():
                if name not in compiled:
                    print("Compiling "+name, flush=True)
                    compiled[name], _ = p._compile(fn, inputs, args.warmup)
                results[name] = np.asarray(jax.block_until_ready(compiled[name](*inputs)), np.float32)
            expected = np.asarray(reference(x, *factors), np.float32)
            verification = {}
            if baseline is not None:
                for stage in ("gate-up", "down", "fused"):
                    a, b = results["optimized/"+stage], results["baseline/"+stage]
                    if args.require_bitwise:
                        np.testing.assert_array_equal(a, b, err_msg=stage+" changed output")
                    verification[stage] = dict(bitwise_equal=bool(np.array_equal(a,b)), **errors(a,b))
            full_versions = [prefix+"/fused" for prefix, _ in versions] + ["optimized/unfused"]
            if args.include_hybrid:
                full_versions.append("optimized/hybrid")
                verification["hybrid_vs_pallas"] = errors(results["optimized/hybrid"], results["optimized/fused"])
                verification["hybrid_vs_jax"] = errors(results["optimized/hybrid"], results["optimized/unfused"])
            for version in full_versions:
                verification[version+"_vs_fp32"] = errors(results[version], expected)
                if verification[version+"_vs_fp32"]["relative_l2"] > args.max_relative_l2:
                    raise AssertionError(version+" failed FP32 tolerance")
            with jax.default_matmul_precision("highest"):
                gate_ref = jax.jit(lambda x,ug,vg,uu,vu:
                    (jax.nn.silu((x.astype(jnp.float32) @ ug.astype(jnp.float32)) @ vg.astype(jnp.float32)) *
                     ((x.astype(jnp.float32) @ uu.astype(jnp.float32)) @ vu.astype(jnp.float32))).T)
                down_ref = jax.jit(lambda h,u,v:
                    (h.astype(jnp.float32).T @ u.astype(jnp.float32)) @ v.astype(jnp.float32))
                references = {"gate-up": np.asarray(gate_ref(*gate_args),np.float32),
                              "down": np.asarray(down_ref(hidden, ud, vd),np.float32)}
            for stage, expected_stage in references.items():
                for version in [prefix for prefix, _ in versions] + ["jax"]:
                    name = version+"/"+stage
                    verification[name+"_vs_fp32"] = errors(results[name], expected_stage)
                    if verification[name+"_vs_fp32"]["relative_l2"] > args.max_relative_l2:
                        raise AssertionError(name+" failed FP32 tolerance")
            for stage in ("gate-up", "down"):
                verification["optimized/"+stage+"_vs_jax"] = errors(
                    results["optimized/"+stage], results["jax/"+stage])
            print(case+" verification passed", flush=True)
            case_report = dict(
                verification=verification,
                checkpoint=args.model_path if case == "checkpoint" else None,
                layer=args.layer if case == "checkpoint" else None,
                input="synthetic standard normal; seed=0",
            )
            if not args.interpret:
                latency = measure_interleaved(compiled, workloads, args.warmup, args.repeats)
                comparisons = summarize_stages(latency)
                case_report.update(latency=latency, stage_comparison=comparisons)
                if baseline is not None:
                    case_report["speedup_vs_saved"] = {
                        stage: latency["baseline/"+stage]["median_ms"] / latency["optimized/"+stage]["median_ms"]
                        for stage in ("gate-up", "down", "fused")}
                print_stage_comparison(comparisons)
                if args.include_hybrid:
                    case_report["full_mlp_comparison"] = summarize_full_mlp(latency)
                    print_full_mlp_comparison(case_report["full_mlp_comparison"])
            report["cases"][case] = case_report
            save()
        report["complete"] = True
        save()
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise
    return report


if __name__ == "__main__":
    main()
