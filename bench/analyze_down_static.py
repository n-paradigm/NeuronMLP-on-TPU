"""Static analysis of Down kernel configuration without requiring TPU.

Analyzes VMEM usage, DMA patterns, and tile efficiency to identify
why the Down projection might be slower than JAX baseline.
"""

import argparse
import json
import math
from pathlib import Path

from svdflash_tpu.kernels.config import (
    MODELS,
    vmem_bytes_xuv,
    fits_vmem,
    XUV_INPUT_TILE,
    XUV_OUTPUT_TILE,
    XUV_RANK_TILE,
    XUV_ROW_TILE,
    VMEM_BUDGET_BYTES,
)


def analyze_down_projection(model_name, seq_len):
    """Analyze Down projection configuration and identify issues."""
    model = MODELS[model_name]
    rank = model.rank_at_080

    # Down: (seq_len, intermediate) @ (intermediate, rank) @ (rank, hidden)
    rows = seq_len
    input_dim = model.intermediate_size
    output_dim = model.hidden_size

    # VMEM usage breakdown
    padded_input = math.ceil(input_dim / XUV_INPUT_TILE) * XUV_INPUT_TILE
    padded_rank = math.ceil(rank / XUV_RANK_TILE) * XUV_RANK_TILE
    padded_output = math.ceil(output_dim / XUV_OUTPUT_TILE) * XUV_OUTPUT_TILE
    padded_rows = math.ceil(rows / XUV_ROW_TILE) * XUV_ROW_TILE

    dtype_size = 2  # bf16

    # VMEM components (from config.py:vmem_bytes_xuv)
    x_block = XUV_ROW_TILE * padded_input * dtype_size
    u_tile = XUV_INPUT_TILE * XUV_RANK_TILE * dtype_size
    v_tile = XUV_RANK_TILE * XUV_OUTPUT_TILE * dtype_size
    xu_accumulator = XUV_ROW_TILE * padded_rank * dtype_size  # Full rank!
    out_tile = XUV_ROW_TILE * XUV_OUTPUT_TILE * dtype_size
    dot_acc = XUV_ROW_TILE * XUV_OUTPUT_TILE * 4  # fp32

    total_vmem = x_block + u_tile + v_tile + xu_accumulator + out_tile + dot_acc

    # DMA analysis per row strip
    input_tiles = math.ceil(input_dim / XUV_INPUT_TILE)
    rank_tiles = math.ceil(rank / XUV_RANK_TILE)
    output_tiles = math.ceil(output_dim / XUV_OUTPUT_TILE)

    # For X @ U phase
    u_dma_per_row = input_tiles * rank_tiles

    # For XU @ V phase
    v_dma_per_row = rank_tiles * output_tiles

    total_dma_per_row = u_dma_per_row + v_dma_per_row

    row_strips = math.ceil(rows / XUV_ROW_TILE)
    total_dma_ops = total_dma_per_row * row_strips

    # Issues identification
    issues = []

    # Issue 1: VMEM pressure
    vmem_utilization = total_vmem / VMEM_BUDGET_BYTES
    if vmem_utilization > 0.75:
        issues.append({
            "severity": "HIGH" if vmem_utilization > 0.9 else "MEDIUM",
            "issue": "VMEM pressure near limit",
            "detail": f"Using {vmem_utilization*100:.1f}% of 16MB budget. XU accumulator alone is {xu_accumulator/(1024*1024):.2f}MB",
            "impact": "Compiler may spill to HBM, adding latency"
        })

    # Issue 2: Large input dimension
    if input_tiles > 10:
        issues.append({
            "severity": "HIGH",
            "issue": "Too many input tiles",
            "detail": f"{input_tiles} tiles of size {XUV_INPUT_TILE} for dimension {input_dim}",
            "impact": f"Each row strip does {u_dma_per_row} U-tile DMAs synchronously"
        })

    # Issue 3: Small tiles for large dimensions
    if XUV_INPUT_TILE < input_dim / 10:
        issues.append({
            "severity": "MEDIUM",
            "issue": "Input tile undersized",
            "detail": f"Tile {XUV_INPUT_TILE} covers only {XUV_INPUT_TILE/input_dim*100:.1f}% of dimension",
            "impact": "Poor DMA amortization"
        })

    # Issue 4: Synchronous DMA
    issues.append({
        "severity": "HIGH",
        "issue": "Synchronous DMA strategy",
        "detail": "pltpu.sync_copy waits for each tile before computation",
        "impact": "MXU stalls during DMA; no double-buffering"
    })

    # Issue 5: Full-rank resident buffer
    if xu_accumulator > 1024 * 1024:  # > 1MB
        issues.append({
            "severity": "MEDIUM",
            "issue": "Large XU resident buffer",
            "detail": f"XU accumulator is {xu_accumulator/(1024*1024):.2f}MB for rank {rank}",
            "impact": "Limits VMEM available for double-buffering"
        })

    return {
        "model": model_name,
        "seq_len": seq_len,
        "dimensions": {
            "rows": rows,
            "input": input_dim,
            "rank": rank,
            "output": output_dim,
        },
        "padded_dimensions": {
            "rows": padded_rows,
            "input": padded_input,
            "rank": padded_rank,
            "output": padded_output,
        },
        "tiles": {
            "row": XUV_ROW_TILE,
            "input": XUV_INPUT_TILE,
            "rank": XUV_RANK_TILE,
            "output": XUV_OUTPUT_TILE,
        },
        "vmem_breakdown_bytes": {
            "x_block": x_block,
            "u_tile": u_tile,
            "v_tile": v_tile,
            "xu_accumulator": xu_accumulator,
            "out_tile": out_tile,
            "dot_accumulator_fp32": dot_acc,
            "total": total_vmem,
        },
        "vmem_breakdown_mb": {k: v/(1024*1024) for k, v in {
            "x_block": x_block,
            "u_tile": u_tile,
            "v_tile": v_tile,
            "xu_accumulator": xu_accumulator,
            "out_tile": out_tile,
            "dot_accumulator_fp32": dot_acc,
            "total": total_vmem,
        }.items()},
        "vmem_utilization_percent": vmem_utilization * 100,
        "fits_in_budget": fits_vmem(total_vmem),
        "dma_analysis": {
            "input_tiles_per_row": input_tiles,
            "rank_tiles": rank_tiles,
            "output_tiles_per_row": output_tiles,
            "u_dma_ops_per_row_strip": u_dma_per_row,
            "v_dma_ops_per_row_strip": v_dma_per_row,
            "total_dma_ops_per_row_strip": total_dma_per_row,
            "row_strips_total": row_strips,
            "total_dma_ops": total_dma_ops,
        },
        "issues": issues,
    }


def suggest_improvements(analysis):
    """Suggest concrete improvements based on analysis."""
    suggestions = []

    model_name = analysis["model"]
    dims = analysis["dimensions"]
    tiles = analysis["tiles"]

    # Suggestion 1: Increase input tile
    new_input_tile = 2048
    if dims["input"] > tiles["input"] * 2:
        new_u_dma = math.ceil(dims["input"] / new_input_tile) * math.ceil(dims["rank"] / tiles["rank"])
        old_u_dma = analysis["dma_analysis"]["u_dma_ops_per_row_strip"]
        suggestions.append({
            "change": f"Increase XUV_INPUT_TILE from {tiles['input']} to {new_input_tile}",
            "benefit": f"Reduces U-tile DMAs from {old_u_dma} to {new_u_dma} per row strip",
            "tradeoff": "Increases U-tile VMEM by 2x; verify VMEM budget"
        })

    # Suggestion 2: Implement async DMA
    suggestions.append({
        "change": "Implement double-buffered async DMA like pallas_dense.py",
        "benefit": "Overlaps DMA with MXU computation; hides memory latency",
        "tradeoff": "Adds 2x tile buffer VMEM; more complex kernel code",
        "reference": "pallas/svdflash_tpu/kernels/pallas_dense.py:60-95"
    })

    # Suggestion 3: Reduce rank tile if VMEM tight
    if analysis["vmem_utilization_percent"] > 80:
        new_rank_tile = 256
        old_xu_size = analysis["vmem_breakdown_mb"]["xu_accumulator"]
        new_xu_size = XUV_ROW_TILE * math.ceil(dims["rank"] / new_rank_tile) * new_rank_tile * 2 / (1024*1024)
        suggestions.append({
            "change": f"Reduce XUV_RANK_TILE from {tiles['rank']} to {new_rank_tile}",
            "benefit": f"Reduces XU accumulator from {old_xu_size:.2f}MB to {new_xu_size:.2f}MB",
            "tradeoff": "More rank iterations; balance against VMEM relief"
        })

    # Suggestion 4: Alternative tiling strategy
    suggestions.append({
        "change": "Switch from row-major to output-major tiling",
        "benefit": "Process all rows for one output tile, may improve V-tile reuse",
        "tradeoff": "Requires kernel restructure; experimental"
    })

    return suggestions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS.keys()), default="llama-3.1-8b")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Analyzing Down kernel for {args.model} with seq_len={args.seq_len}\n")

    analysis = analyze_down_projection(args.model, args.seq_len)
    suggestions = suggest_improvements(analysis)

    # Print summary
    print("=== VMEM Usage ===")
    print(f"Total: {analysis['vmem_breakdown_mb']['total']:.2f} MB ({analysis['vmem_utilization_percent']:.1f}% of budget)")
    print(f"  X block:        {analysis['vmem_breakdown_mb']['x_block']:.2f} MB")
    print(f"  U tile:         {analysis['vmem_breakdown_mb']['u_tile']:.2f} MB")
    print(f"  V tile:         {analysis['vmem_breakdown_mb']['v_tile']:.2f} MB")
    print(f"  XU accumulator: {analysis['vmem_breakdown_mb']['xu_accumulator']:.2f} MB  <-- LARGE")
    print(f"  Output tile:    {analysis['vmem_breakdown_mb']['out_tile']:.2f} MB")
    print(f"  Dot acc (fp32): {analysis['vmem_breakdown_mb']['dot_accumulator_fp32']:.2f} MB")

    print(f"\n=== DMA Operations ===")
    print(f"Per row strip: {analysis['dma_analysis']['total_dma_ops_per_row_strip']} sync copies")
    print(f"  U-tiles: {analysis['dma_analysis']['u_dma_ops_per_row_strip']} (input={analysis['dma_analysis']['input_tiles_per_row']} × rank={analysis['dma_analysis']['rank_tiles']})")
    print(f"  V-tiles: {analysis['dma_analysis']['v_dma_ops_per_row_strip']} (rank={analysis['dma_analysis']['rank_tiles']} × output={analysis['dma_analysis']['output_tiles_per_row']})")
    print(f"Total across all rows: {analysis['dma_analysis']['total_dma_ops']} operations")

    print(f"\n=== Issues Found ({len(analysis['issues'])}) ===")
    for i, issue in enumerate(analysis['issues'], 1):
        print(f"{i}. [{issue['severity']}] {issue['issue']}")
        print(f"   {issue['detail']}")
        print(f"   Impact: {issue['impact']}\n")

    print(f"=== Suggested Improvements ({len(suggestions)}) ===")
    for i, sug in enumerate(suggestions, 1):
        print(f"{i}. {sug['change']}")
        print(f"   Benefit: {sug['benefit']}")
        print(f"   Tradeoff: {sug['tradeoff']}")
        if 'reference' in sug:
            print(f"   Reference: {sug['reference']}")
        print()

    # Save report
    report = {
        "analysis": analysis,
        "suggestions": suggestions,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()
