"""Why Gate/Up beats JAX but Down loses: a roofline/fusion comparison.

Both Pallas stages use the same synchronous-DMA strategy, so DMA count alone
cannot explain the split result. This script computes, per stage:

  * MXU FLOPs and weight bytes streamed -> arithmetic intensity
  * the serial (non-overlapped) DMA cost implied by pltpu.sync_copy
  * the HBM traffic a JAX/XLA implementation must pay for intermediates,
    which is exactly what Pallas fusion removes

The fusion saving is the only term that differs qualitatively between the two
stages, and it is what pays for the synchronous-DMA penalty.

Bandwidth/FLOPs defaults are TPU v5e figures; override to match your part.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from svdflash_tpu.kernels.config import (
    MODELS,
    XUV_INPUT_TILE,
    XUV_OUTPUT_TILE,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
)

BF16 = 2


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def gate_up_profile(rows, hidden, rank, intermediate, blocks):
    """Pallas fused_gate_up: X@U then XU@V, for gate and up together."""
    bs, bh, br, bn = blocks

    hidden_blocks = _ceil_div(hidden, bh)
    rank_blocks = _ceil_div(rank, br)
    out_blocks = _ceil_div(intermediate, bn)
    row_strips = _ceil_div(rows, bs)

    # phase 1: for each (rank, hidden) pair load u_gate and u_up tiles
    u_copies = rank_blocks * hidden_blocks * 2
    u_bytes = u_copies * bh * br * BF16
    # phase 2: for each (out, rank) pair load v_gate and v_up tiles
    v_copies = out_blocks * rank_blocks * 2
    v_bytes = v_copies * br * bn * BF16

    x_bytes = rows * hidden * BF16  # one sync_copy per row strip

    flops = 2 * (2 * rows * hidden * rank) + 2 * (2 * rows * rank * intermediate)

    # JAX must land gate and up in HBM, then read both back for silu*mul.
    jax_intermediate_bytes = 2 * rows * intermediate * BF16 * 2

    return {
        "stage": "gate-up",
        "blocks": {"bs": bs, "bh": bh, "br": br, "bn": bn},
        "row_strips": row_strips,
        "sync_copies_per_row_strip": u_copies + v_copies,
        "weight_bytes": row_strips * (u_bytes + v_bytes),
        "x_bytes": x_bytes,
        "flops": flops,
        "jax_intermediate_bytes": jax_intermediate_bytes,
    }


def down_profile(rows, intermediate, rank, hidden):
    """Pallas xuv_matmul on the Down shape: (S,I) @ (I,r) @ (r,H)."""
    input_blocks = _ceil_div(intermediate, XUV_INPUT_TILE)
    rank_blocks = _ceil_div(rank, XUV_RANK_TILE)
    out_blocks = _ceil_div(hidden, XUV_OUTPUT_TILE)
    row_strips = _ceil_div(rows, XUV_ROW_TILE)

    u_copies = rank_blocks * input_blocks
    u_bytes = u_copies * XUV_INPUT_TILE * XUV_RANK_TILE * BF16
    v_copies = out_blocks * rank_blocks
    v_bytes = v_copies * XUV_RANK_TILE * XUV_OUTPUT_TILE * BF16

    x_bytes = rows * intermediate * BF16

    flops = 2 * rows * intermediate * rank + 2 * rows * rank * hidden

    # JAX only has to land XU, which is (S, r) -- small, and XLA may even
    # fuse the pair of matmuls so this is an upper bound on what fusion saves.
    jax_intermediate_bytes = rows * rank * BF16 * 2

    return {
        "stage": "down",
        "blocks": {
            "bs": XUV_ROW_TILE,
            "input": XUV_INPUT_TILE,
            "rank": XUV_RANK_TILE,
            "output": XUV_OUTPUT_TILE,
        },
        "row_strips": row_strips,
        "sync_copies_per_row_strip": u_copies + v_copies,
        "weight_bytes": row_strips * (u_bytes + v_bytes),
        "x_bytes": x_bytes,
        "flops": flops,
        "jax_intermediate_bytes": jax_intermediate_bytes,
    }


def add_roofline(profile, bandwidth_gbps, tflops):
    """Attach timing estimates: serial DMA vs MXU, and the fusion credit."""
    bytes_per_us = bandwidth_gbps * 1e9 / 1e6
    flops_per_us = tflops * 1e12 / 1e6

    weight_us = profile["weight_bytes"] / bytes_per_us
    x_us = profile["x_bytes"] / bytes_per_us
    mxu_us = profile["flops"] / flops_per_us
    fusion_us = profile["jax_intermediate_bytes"] / bytes_per_us

    # sync_copy forbids overlap, so DMA and MXU add instead of maxing.
    serial_us = weight_us + x_us + mxu_us
    # XLA double-buffers, so its weight streaming hides behind compute, but it
    # still pays for the intermediates it has to materialize.
    overlapped_us = max(weight_us + x_us, mxu_us) + fusion_us

    profile["roofline_us"] = {
        "weight_dma": weight_us,
        "x_dma": x_us,
        "mxu": mxu_us,
        "fusion_credit": fusion_us,
        "pallas_serial_total": serial_us,
        "overlapped_reference_total": overlapped_us,
    }
    profile["arithmetic_intensity_flop_per_byte"] = profile["flops"] / (
        profile["weight_bytes"] + profile["x_bytes"]
    )
    profile["dma_stall_fraction"] = (weight_us + x_us) / serial_us
    # >1 means the synchronous-DMA penalty is covered by what fusion saves.
    profile["fusion_pays_for_sync_dma"] = fusion_us / max(
        serial_us - overlapped_us + fusion_us, 1e-9
    )
    profile["predicted_ratio_vs_overlapped"] = serial_us / overlapped_us
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="llama-3.1-8b")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--compress-ratio", type=float, default=0.8)
    parser.add_argument("--bandwidth-gbps", type=float, default=819.0)
    parser.add_argument("--tflops", type=float, default=197.0)
    parser.add_argument("--bs", type=int, default=128,
                        help="Gate/Up row block; speedup.py hardcodes 128")
    parser.add_argument("--output")
    args = parser.parse_args()

    model = MODELS[args.model]
    rank = model.rank_at_080
    rows = args.seq_len
    # Only Gate/Up takes a row block; the Down path pins XUV_ROW_TILE globally.
    blocks = (args.bs, 1024, 512, 512)

    gate_up = add_roofline(
        gate_up_profile(
            rows, model.hidden_size, rank, model.intermediate_size, blocks
        ),
        args.bandwidth_gbps,
        args.tflops,
    )
    down = add_roofline(
        down_profile(rows, model.intermediate_size, rank, model.hidden_size),
        args.bandwidth_gbps,
        args.tflops,
    )

    print(f"{args.model}  S={rows}  H={model.hidden_size}  "
          f"I={model.intermediate_size}  r={rank}\n")

    header = f"{'':28s}{'gate-up':>14s}{'down':>14s}"
    print(header)
    print("-" * len(header))

    def row(label, key, fmt="{:>14,.1f}", source=lambda p: p):
        left = fmt.format(source(gate_up)[key])
        right = fmt.format(source(down)[key])
        print(f"{label:28s}{left}{right}")

    roof = lambda p: p["roofline_us"]  # noqa: E731
    row("row strips", "row_strips", "{:>14,d}")
    row("sync copies / row strip", "sync_copies_per_row_strip", "{:>14,d}")
    row("weight bytes (MB)", "weight_bytes", "{:>14,.2f}",
        lambda p: {"weight_bytes": p["weight_bytes"] / 1024**2})
    row("X bytes (MB)", "x_bytes", "{:>14,.2f}",
        lambda p: {"x_bytes": p["x_bytes"] / 1024**2})
    row("FLOPs (G)", "flops", "{:>14,.2f}",
        lambda p: {"flops": p["flops"] / 1e9})
    row("intensity (FLOP/byte)", "arithmetic_intensity_flop_per_byte")
    print()
    row("MXU time (us)", "mxu", "{:>14,.2f}", roof)
    row("weight DMA (us)", "weight_dma", "{:>14,.2f}", roof)
    row("X DMA (us)", "x_dma", "{:>14,.2f}", roof)
    row("Pallas serial total (us)", "pallas_serial_total", "{:>14,.2f}", roof)
    print()
    row("fusion credit (us)", "fusion_credit", "{:>14,.2f}", roof)
    row("overlapped ref (us)", "overlapped_reference_total", "{:>14,.2f}", roof)
    row("DMA stall fraction", "dma_stall_fraction", "{:>14,.3f}")
    row("predicted Pallas/ref", "predicted_ratio_vs_overlapped", "{:>14,.3f}")

    print("\nA predicted ratio below 1.0 means Pallas should win; above 1.0 "
          "means the synchronous-DMA penalty outweighs the fusion credit.")

    if args.output:
        report = {
            "model": args.model,
            "seq_len": rows,
            "rank": rank,
            "assumed_bandwidth_gbps": args.bandwidth_gbps,
            "assumed_tflops": args.tflops,
            "gate_up": gate_up,
            "down": down,
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
