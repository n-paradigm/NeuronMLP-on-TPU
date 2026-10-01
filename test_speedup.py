"""Low-rank TPU kernels with streaming DMA and FP32 contractions.

Keep XU in VMEM; round once at each GEMM boundary. Weight prefetch spans
outer tiles and rank tails execute only valid R_TILEs. Gate/Up/SwiGLU
and Down remain separate kernels. Python loops specialize at trace time.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# TPU v5e MXU 是 128 x 128；这些是软件 tile，不是 NKI 的硬件上限。
M_TILE = 256
R_TILE = 256
K_TILE = 128
N_TILE = 512
VMEM_LIMIT = 16 * 1024 * 1024
VMEM_HEADROOM = 0.8


def _round_up(n, tile):
    return ((n + tile - 1) // tile) * tile


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _validate_arrays(*arrays):
    for a in arrays:
        if a.ndim != 2 or min(a.shape) <= 0:
            raise ValueError(f"expected a nonempty 2D array, got {a.shape}")
        if a.dtype != arrays[0].dtype:
            raise TypeError("all inputs must have the same dtype")
    if arrays[0].dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise TypeError("supported dtypes are bfloat16 and float32")


def _pad(a, rows, cols):
    return jnp.pad(a, ((0, rows - a.shape[0]), (0, cols - a.shape[1])))


def _mesh(interpret):
    if interpret:
        return (pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
                pltpu.InterpretParams(num_cores_or_threads=1))
    return pltpu.TensorCoreMesh(axis_name="core"), False


def _dot(stationary, moving, mixed_precision):
    # 对应 nisa.nc_matmul(stationary, moving) = stationary.T @ moving。
    dtype = jnp.bfloat16 if mixed_precision else jnp.float32
    return jnp.dot(stationary.astype(dtype).T, moving.astype(dtype),
                   preferred_element_type=jnp.float32,
                   precision=None if mixed_precision else lax.Precision.HIGHEST)


def _vmem_type(mesh, interpret, shape, dtype):
    # JAX 0.11.1 CPU interpreter recognizes VMEM, but not mesh-bound VMEM@TC.
    memory = pltpu.VMEM if interpret else pltpu.VMEM @ mesh
    return memory(shape, dtype)


def _tile(ref, row, col, rows, cols):
    return ref.at[pl.dslice(row, rows), pl.dslice(col, cols)]


def _check_load(input_ref, vmem_ref, ofs, load_shape, transpose=False):
    if len(ofs) != 2 or len(load_shape) != 3:
        raise ValueError("ofs must be (row, col); load_shape must be (block, par, free)")
    block_size, par_size, free_size = load_shape
    for name, value in zip(("block_size", "par_size", "free_size"), load_shape):
        _positive(name, value)
    if tuple(vmem_ref.shape) != (block_size * par_size, free_size):
        raise ValueError("vmem_ref must have shape (block_size * par_size, free_size)")
    if len(input_ref.shape) != 2:
        raise ValueError("input_ref must be 2D")
    # DMA 不使用 NKI 的 mask；launcher 先补零。静态偏移可在 tracing 时检查。
    extent = (free_size, block_size * par_size) if transpose else (
        block_size * par_size, free_size)
    for offset, size, bound in zip(ofs, extent, input_ref.shape):
        if isinstance(offset, (int, np.integer)) and (offset < 0 or offset + size > bound):
            raise ValueError("load exceeds input bounds; pad the input before launching")


def load_tensor_block(input_ref, vmem_ref, ofs: tuple[int, int],
                      load_shape: tuple[int, int, int]):
    """HBM -> VMEM；原 (par, block, free) 缓存改为 (block * par, free)。"""
    _check_load(input_ref, vmem_ref, ofs, load_shape)
    block_size, par_size, free_size = load_shape
    row_offset, column_offset = ofs
    # The grouped tile is one rectangle in both HBM and VMEM.
    pltpu.sync_copy(
        _tile(input_ref, row_offset, column_offset, block_size * par_size, free_size),
        vmem_ref)
    return pltpu.load(vmem_ref)


def _load_weight_block(hbm_src_ref, vmem_dst_ref, ofs, load_shape, *, sem=None):
    """Load an aligned weight block, zero-filling a partial final rank block.

    Launchers align HBM rank to R_TILE, rather than the larger r_block. Only
    this helper accepts a partial block; load_tensor_block remains strict.
    Offsets and extents are static because the contraction loops use range.
    """
    block_size, par_size, free_size = load_shape
    rows, cols = block_size * par_size, free_size
    if tuple(vmem_dst_ref.shape) != (rows, cols):
        raise ValueError("weight cache shape does not match load_shape")
    nr = min(rows, hbm_src_ref.shape[0] - ofs[0])
    nc = min(cols, hbm_src_ref.shape[1] - ofs[1])
    if min(ofs) < 0 or nr <= 0 or nc <= 0:
        raise ValueError("weight block starts outside input bounds")
    if nr != rows or nc != cols:
        # Only clear the regions not covered by DMA to avoid unnecessary writes.
        # Clear right padding (beyond loaded columns) for all loaded rows.
        if nc < cols:
            pltpu.store(_tile(vmem_dst_ref, 0, nc, nr, cols - nc),
                       jnp.zeros((nr, cols - nc), vmem_dst_ref.dtype))
        # Clear bottom padding (beyond loaded rows) for all columns.
        if nr < rows:
            pltpu.store(_tile(vmem_dst_ref, nr, 0, rows - nr, cols),
                       jnp.zeros((rows - nr, cols), vmem_dst_ref.dtype))
    source = _tile(hbm_src_ref, *ofs, nr, nc)
    destination = _tile(vmem_dst_ref, 0, 0, nr, nc)
    if sem is None:
        pltpu.sync_copy(source, destination)
        return None
    return pltpu.async_copy(source, destination, sem)


def _wait_weight_block(copy):
    # The matching start may precede computation of the previous rank block.
    copy.wait()



def load_tensor_block_T(input_ref, vmem_ref, ofs: tuple[int, int],
                        load_shape: tuple[int, int, int], *, staging_ref):
    """HBM -> staging -> 转置 VMEM；staging_ref 形状是 (free_size, par_size)。"""
    _check_load(input_ref, vmem_ref, ofs, load_shape, transpose=True)
    block_size, par_size, free_size = load_shape
    if tuple(staging_ref.shape) != (free_size, par_size):
        raise ValueError("staging_ref must have shape (free_size, par_size)")
    row_offset, column_offset = ofs
    for block_id in range(block_size):
        pltpu.sync_copy(
            _tile(input_ref, row_offset, column_offset + block_id * par_size,
                  free_size, par_size), staging_ref)
        pltpu.store(_tile(vmem_ref, block_id * par_size, 0, par_size, free_size),
                    pltpu.load(staging_ref).T)
    return pltpu.load(vmem_ref)



def get_fused_mlp_up_T_params(S):
    """Use a full 512-token strip when it does not add sequence padding."""
    _positive("S", S)
    if S % 512 == 0:
        # Smaller M/N staging leaves space for FP32 accumulators and prefetch.
        return dict(M_tiles_in_block=2, r_tiles_in_block=2,
                    K_tiles_in_block=4, N_tiles_in_block=1)
    return dict(M_tiles_in_block=4, r_tiles_in_block=2,
                K_tiles_in_block=1 if S < 512 else 2, N_tiles_in_block=2)


def get_fused_three_mm_XTUV_params(S):
    _positive("S", S)
    if S % 512 == 0:
        # Down kernel's K dimension equals S; K-blocks = ceil(S / kb).
        # K_tiles=4 (kb=512) gives only 1 block at S=512 and 2 at S=1024,
        # leaving cores idle. K_tiles=2 (kb=256) yields 4 blocks at S=1024,
        # saturating all cores; only step up to 4 at S>=2048 where K-blocks>=4.
        k_tiles = 2 if S < 2048 else 4
        return dict(M_tiles_in_block=4, r_tiles_in_block=2,
                    K_tiles_in_block=k_tiles, N_tiles_in_block=2)
    return dict(M_tiles_in_block=4, r_tiles_in_block=2,
                K_tiles_in_block=1 if S < 512 else 2, N_tiles_in_block=2)



class _FusionShape(NamedTuple):
    M_block: int
    r_block: int
    K_block: int
    N_block: int
    M_padded: int
    r_padded: int
    K_padded: int
    N_padded: int
    vmem_bytes: int



def _projection_memory_plan(mb, rb, kb, nb, mp, rp, npad, dtype, gate, async_dma):
    """Estimate peak live VMEM; only XU survives the U/V stage boundary.

    Keep the existing temporary allowance in each stage. Scoped allocations
    permit reuse, but Mosaic's actual allocation still requires TPU checking.

    Returns detailed breakdown for diagnostics.
    """
    itemsize = jnp.dtype(dtype).itemsize
    branches = 2 if gate else 1
    rn = (rp+rb-1)//rb
    u_slots = 2 if async_dma and (mp//mb)*rn > 1 else 1
    v_slots = 2 if async_dma and (npad//nb)*rn > 1 else 1
    persistent = branches*rp*kb*itemsize
    transpose = kb*M_TILE*itemsize if gate else 0
    output = (N_TILE*kb if gate else K_TILE*nb)*itemsize
    u_work = branches*(u_slots*mb*rb*itemsize + min(rb, rp)*kb*4)
    v_work = branches*(v_slots*rb*nb*itemsize + nb*kb*4)
    temporaries = branches*max(R_TILE*K_TILE, N_TILE*K_TILE)*8
    v_peak = persistent + output + v_work + temporaries

    def u_peak(rows):
        return persistent + rows*kb*itemsize + transpose + u_work + temporaries

    # Full X needs to fit only during XU, not alongside the XUV working set.
    cache_rows = mp if max(u_peak(mp), v_peak) < VMEM_LIMIT*VMEM_HEADROOM else mb

    return dict(
        persistent_bytes=persistent,
        xu_stage_peak_bytes=u_peak(cache_rows),
        xuv_stage_peak_bytes=v_peak,
        estimated_vmem_bytes=max(u_peak(cache_rows), v_peak),
        input_cache_rows=cache_rows,
        # Additional diagnostics
        breakdown=dict(
            persistent_xu_buffers_bytes=persistent,
            input_cache_bytes=cache_rows*kb*itemsize,
            transpose_staging_bytes=transpose,
            u_weight_cache_bytes=u_slots*mb*rb*itemsize*branches,
            v_weight_cache_bytes=v_slots*rb*nb*itemsize*branches,
            output_tile_bytes=output,
            temporaries_bytes=temporaries,
        ),
        config=dict(
            branches=branches,
            u_slots=u_slots,
            v_slots=v_slots,
            cache_rows=cache_rows,
            cache_full_input=(cache_rows == mp),
        ),
    )


def _projection_memory(mb, rb, kb, nb, mp, rp, npad, dtype, gate, async_dma):
    plan = _projection_memory_plan(mb, rb, kb, nb, mp, rp, npad, dtype, gate, async_dma)
    return plan["estimated_vmem_bytes"], plan["input_cache_rows"]



def _fusion_shape(M, r, K, N, dtype, M_count, r_count, K_count, N_count, gate,
                  async_dma=True):
    for name, value in zip(("M_tiles_in_block", "r_tiles_in_block",
                            "K_tiles_in_block", "N_tiles_in_block"),
                           (M_count, r_count, K_count, N_count)):
        _positive(name, value)
    mb, rb, kb, nb = M_TILE*M_count, R_TILE*r_count, K_TILE*K_count, N_TILE*N_count
    mp, rp, kp, npad = (_round_up(M, mb), _round_up(r, R_TILE),
                        _round_up(K, kb), _round_up(N, nb))

    # Get detailed memory plan for diagnostics
    plan = _projection_memory_plan(mb, rb, kb, nb, mp, rp, npad, dtype, gate, async_dma)
    estimate = plan["estimated_vmem_bytes"]

    # Check VMEM budget with enhanced error reporting
    try:
        _check_vmem(estimate)
    except ValueError as e:
        # Add configuration details to help debugging
        config_info = (
            f"\nCurrent configuration:\n"
            f"  M_tiles_in_block={M_count}, r_tiles_in_block={r_count}\n"
            f"  K_tiles_in_block={K_count}, N_tiles_in_block={N_count}\n"
            f"  Block sizes: mb={mb}, rb={rb}, kb={kb}, nb={nb}\n"
            f"  Stage: {'gate_up' if gate else 'down'}\n"
            f"  Async DMA: {async_dma}\n"
            f"\nMemory breakdown:\n"
        )
        for key, value in plan["breakdown"].items():
            config_info += f"  {key}: {value / (1024*1024):.2f} MiB\n"
        raise ValueError(str(e) + config_info)

    return _FusionShape(mb, rb, kb, nb, mp, rp, kp, npad, estimate)



def _check_vmem(estimate):
    """Validate VMEM estimate against budget with detailed diagnostics."""
    budget_bytes = VMEM_LIMIT * VMEM_HEADROOM
    if estimate >= budget_bytes:
        # Provide actionable guidance on which dimensions to reduce
        overage_mb = (estimate - budget_bytes) / (1024 * 1024)
        raise ValueError(
            f"Estimated VMEM {estimate/2**20:.2f} MiB exceeds "
            f"{budget_bytes/2**20:.2f} MiB budget (overage: {overage_mb:.2f} MiB).\n"
            f"Reduce tile counts to fit within budget:\n"
            f"  - Decrease M_tiles_in_block (affects input cache size)\n"
            f"  - Decrease r_tiles_in_block (affects XU buffer size)\n"
            f"  - Decrease K_tiles_in_block or N_tiles_in_block (affects output staging)\n"
            f"Try halving the largest tile count first."
        )




def _weight_stream(sources, buffers, semaphores, offsets, load_shape, *,
                   async_dma, stage, source_names):
    """Static double-buffered stream; prefetch continues across outer blocks."""
    slots = buffers.shape[1]

    def load(index):
        copies = []
        for branch, source in enumerate(sources):
            with jax.named_scope(stage + "/HBM_to_VMEM/" + source_names[branch]):
                copies.append(_load_weight_block(
                    source, buffers.at[branch, index % slots], offsets[index], load_shape,
                    sem=semaphores.at[branch, index % slots] if async_dma else None))
        return copies

    if async_dma:
        pending = load(0)
    for index in range(len(offsets)):
        if async_dma:
            for copy in pending:
                _wait_weight_block(copy)
            if index + 1 < len(offsets):
                pending = load(index + 1)
        else:
            load(index)
        yield index, tuple(buffers.at[b, index % slots] for b in range(len(sources)))


def _stream_projection(x, us, vs, shape, *, gate, mixed_precision, interpret, async_dma):
    """Keep each contraction in FP32, casting only at its mathematical boundary."""
    mb, rb, kb, nb, mp, rp, kp, npad, _ = shape
    mt, rt, kt, nt = mb // M_TILE, rb // R_TILE, kb // K_TILE, nb // N_TILE
    mn, rn, nn = mp // mb, (rp + rb - 1) // rb, npad // nb
    branches = len(us)
    u_slots = 2 if async_dma and mn * rn > 1 else 1
    v_slots = 2 if async_dma and nn * rn > 1 else 1
    mesh, interpretation = _mesh(interpret)
    dtype = x.dtype
    stage = "gate_up" if gate else "down"

    def vmem(shape):
        return _vmem_type(mesh, interpret, shape, dtype)

    # Only valid R_TILEs are resident. A partial rb block is never dotted
    # beyond rp, even though its DMA staging buffer has the full rb extent.
    output_tile_shape = (N_TILE, kb) if gate else (K_TILE, nb)
    _, cache_rows = _projection_memory(mb, rb, kb, nb, mp, rp, npad, dtype, gate, async_dma)
    cache_full_input = cache_rows == mp
    # Only XU is live across both phases. Scope input/transpose storage to
    # XU and output staging to XUV, allowing their VMEM allocations to overlap.
    scratch = (vmem((rp, kb)),) * branches
    u_scratch = dict(
        buffers=vmem((branches, u_slots, mb, rb)),
        semaphores=pltpu.SemaphoreType.DMA((branches, u_slots)),
        x_cache=vmem((cache_rows, kb)),
    )
    if gate:
        u_scratch["staging"] = vmem((kb, M_TILE))
    out_shape = (npad, kp) if gate else (kp, npad)

    @pl.kernel(mesh=mesh, interpret=interpretation,
               out_type=jax.ShapeDtypeStruct(out_shape, dtype), scratch_types=scratch,
               name="fused_mlp_up_T_v5e" if gate else "fused_three_mm_XTUV_v5e")
    def kernel(x_ref, *refs):
        u_refs, v_refs = refs[:branches], refs[branches:2*branches]
        out_ref = refs[2*branches]
        xu_refs = refs[2*branches+1:3*branches+1]

        for sequence_block in range(kp // kb):
            @pl.when(lax.axis_index("core") == sequence_block % mesh.num_cores)
            def process_sequence():
                # Rank-major order allows a small FP32 accumulator to finish
                # all M blocks without repeatedly loading/storing XU in BF16.
                @functools.partial(pl.run_scoped, **u_scratch)
                def compute_u_stage(buffers, semaphores, x_cache, staging=None):
                    offsets = tuple((m*mb, r*rb) for r in range(rn) for m in range(mn))
                    stream = _weight_stream(
                        u_refs, buffers, semaphores, offsets, (mt, M_TILE, rb),
                        async_dma=async_dma, stage=stage,
                        source_names=("U_ref", "U_ref_1")[:branches])
                    for index, caches in stream:
                        rank_block, input_block = divmod(index, mn)
                        valid_rt = min(rb, rp-rank_block*rb) // R_TILE
                        if input_block == 0:
                            accum = [[[jnp.zeros((R_TILE, K_TILE), jnp.float32)
                                       for _ in range(kt)] for _ in range(valid_rt)]
                                     for _ in range(branches)]
                        if not cache_full_input or index == 0:
                            input_offset = 0 if cache_full_input else input_block*mb
                            with jax.named_scope(stage + "/HBM_to_VMEM/X_ref"):
                                if gate:
                                    load_tensor_block_T(
                                        x_ref, x_cache, (sequence_block*kb, input_offset),
                                        (cache_rows//M_TILE, M_TILE, kb), staging_ref=staging)
                                else:
                                    load_tensor_block(
                                        x_ref, x_cache, (input_offset, sequence_block*kb),
                                        (cache_rows//M_TILE, M_TILE, kb))
                        cache_offset = input_block*mb if cache_full_input else 0
                        with jax.named_scope(stage + "/MXU/XU"):
                            for k in range(kt):
                                for r in range(valid_rt):
                                    for m in range(mt):
                                        xt = pltpu.load(_tile(
                                            x_cache, cache_offset+m*M_TILE, k*K_TILE, M_TILE, K_TILE))
                                        for b in range(branches):
                                            ut = pltpu.load(_tile(
                                                caches[b], m*M_TILE, r*R_TILE, M_TILE, R_TILE))
                                            accum[b][r][k] += _dot(ut, xt, mixed_precision)
                        if input_block == mn-1:
                            for b in range(branches):
                                for r in range(valid_rt):
                                    for k in range(kt):
                                        pltpu.store(_tile(
                                            xu_refs[b], rank_block*rb+r*R_TILE,
                                            k*K_TILE, R_TILE, K_TILE),
                                            accum[b][r][k].astype(dtype))

                @functools.partial(
                    pl.run_scoped,
                    buffers=vmem((branches, v_slots, rb, nb)),
                    semaphores=pltpu.SemaphoreType.DMA((branches, v_slots)),
                    output_tile=vmem(output_tile_shape))
                def compute_v_stage(buffers, semaphores, output_tile):
                    offsets = tuple((r*rb, n*nb) for n in range(nn) for r in range(rn))
                    stream = _weight_stream(
                        v_refs, buffers, semaphores, offsets, (rt, R_TILE, nb),
                        async_dma=async_dma, stage=stage,
                        source_names=("V_ref", "V_ref_1")[:branches])
                    tile_shape = (N_TILE, K_TILE) if gate else (K_TILE, N_TILE)
                    for index, caches in stream:
                        output_block, rank_block = divmod(index, rn)
                        valid_rt = min(rb, rp-rank_block*rb) // R_TILE
                        if rank_block == 0:
                            accum = [[[jnp.zeros(tile_shape, jnp.float32)
                                       for _ in range(nt)] for _ in range(kt)]
                                     for _ in range(branches)]
                        with jax.named_scope(stage + "/MXU/XUV"):
                            for k in range(kt):
                                for n in range(nt):
                                    for r in range(valid_rt):
                                        for b in range(branches):
                                            xu = pltpu.load(_tile(
                                                xu_refs[b], rank_block*rb+r*R_TILE,
                                                k*K_TILE, R_TILE, K_TILE))
                                            vt = pltpu.load(_tile(
                                                caches[b], r*R_TILE, n*N_TILE, R_TILE, N_TILE))
                                            accum[b][k][n] += (
                                                _dot(vt, xu, mixed_precision) if gate
                                                else _dot(xu, vt, mixed_precision))
                        if rank_block == rn-1:
                            if gate:
                                for n in range(nt):
                                    for k in range(kt):
                                        # Match BF16 GEMM and activation boundaries.
                                        g = accum[0][k][n].astype(dtype)
                                        u = accum[1][k][n].astype(dtype)
                                        activated = jax.nn.silu(g.astype(jnp.float32)).astype(dtype)
                                        pltpu.store(_tile(
                                            output_tile, 0, k*K_TILE, N_TILE, K_TILE),
                                            (activated*u).astype(dtype))
                                    with jax.named_scope(stage + "/output_DMA"):
                                        pltpu.sync_copy(output_tile, _tile(
                                            out_ref, output_block*nb+n*N_TILE,
                                            sequence_block*kb, N_TILE, kb))
                            else:
                                for k in range(kt):
                                    for n in range(nt):
                                        pltpu.store(_tile(
                                            output_tile, 0, n*N_TILE, K_TILE, N_TILE),
                                            accum[0][k][n].astype(dtype))
                                    with jax.named_scope(stage + "/output_DMA"):
                                        pltpu.sync_copy(output_tile, _tile(
                                            out_ref, sequence_block*kb+k*K_TILE,
                                            output_block*nb, K_TILE, nb))

    return kernel(x, *us, *vs)


def fused_three_mm_XTUV(
    X_ref, U_ref, V_ref,
    mixed_precision=True, r_tiles_in_block=2, K_tiles_in_block=1,
    N_tiles_in_block=2, M_tiles_in_block=4, *, interpret=False, async_dma=True,
):
    """Return X_ref.T @ U_ref @ V_ref, with FP32 contraction accumulators."""
    _validate_arrays(X_ref, U_ref, V_ref)
    M, K = X_ref.shape
    if U_ref.shape[0] != M or V_ref.shape[0] != U_ref.shape[1]:
        raise ValueError("X, U, V contraction dimensions do not match")
    r, N = V_ref.shape
    shape = _fusion_shape(M, r, K, N, X_ref.dtype, M_tiles_in_block,
                          r_tiles_in_block, K_tiles_in_block, N_tiles_in_block,
                          gate=False, async_dma=async_dma)
    mp, rp, kp, npad = shape[4:8]
    output = _stream_projection(
        _pad(X_ref, mp, kp), (_pad(U_ref, mp, rp),), (_pad(V_ref, rp, npad),),
        shape, gate=False, mixed_precision=mixed_precision, interpret=interpret,
        async_dma=async_dma)
    return output[:K, :N]


def fused_mlp_up_T(
    X_ref, U_ref, V_ref, U_ref_1, V_ref_1,
    mixed_precision=True, r_tiles_in_block=2, K_tiles_in_block=1,
    N_tiles_in_block=2, M_tiles_in_block=4, *, interpret=False, async_dma=True,
):
    """Return transposed SiLU((X U) V) * ((X U_1) V_1)."""
    _validate_arrays(X_ref, U_ref, V_ref, U_ref_1, V_ref_1)
    K, M = X_ref.shape
    if U_ref.shape[0] != M or V_ref.shape[0] != U_ref.shape[1]:
        raise ValueError("X, U, V contraction dimensions do not match")
    if U_ref_1.shape != U_ref.shape or V_ref_1.shape != V_ref.shape:
        raise ValueError("gate/up factors must have matching shapes")
    r, N = V_ref.shape
    shape = _fusion_shape(M, r, K, N, X_ref.dtype, M_tiles_in_block,
                          r_tiles_in_block, K_tiles_in_block, N_tiles_in_block,
                          gate=True, async_dma=async_dma)
    mp, rp, kp, npad = shape[4:8]
    output = _stream_projection(
        _pad(X_ref, kp, mp), (_pad(U_ref, mp, rp), _pad(U_ref_1, mp, rp)),
        (_pad(V_ref, rp, npad), _pad(V_ref_1, rp, npad)), shape,
        gate=True, mixed_precision=mixed_precision, interpret=interpret,
        async_dma=async_dma)
    return output[:N, :K]



def nki_matmul_fully_optimized_(
    lhsT, rhs, TILES_IN_BLOCK_M=2, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4,
    *, mixed_precision=True, interpret=False,
):
    """保留 NKI 名称/布局：lhsT=(K,M), rhs=(K,N)，返回 (M,N)。"""
    _validate_arrays(lhsT, rhs)
    K, M = lhsT.shape
    if rhs.shape[0] != K:
        raise ValueError("lhsT and rhs must have the same contraction dimension")
    N = rhs.shape[1]
    for name, value in zip(("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K"),
                           (TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K)):
        _positive(name, value)
    TILE_M, TILE_K, TILE_N = K_TILE, M_TILE, N_TILE
    BLOCK_M, BLOCK_K, BLOCK_N = (TILE_M*TILES_IN_BLOCK_M,
                                TILE_K*TILES_IN_BLOCK_K, TILE_N*TILES_IN_BLOCK_N)
    Mp, Kp, Np = _round_up(M, BLOCK_M), _round_up(K, BLOCK_K), _round_up(N, BLOCK_N)
    NUM_BLOCK_M, NUM_BLOCK_K, NUM_BLOCK_N = Mp//BLOCK_M, Kp//BLOCK_K, Np//BLOCK_N
    estimate = ((Mp*BLOCK_N + BLOCK_K*BLOCK_N + BLOCK_K*BLOCK_M + TILE_M*BLOCK_N)
                * lhsT.dtype.itemsize + TILE_M*TILE_N*8)
    _check_vmem(estimate)
    mesh, interpretation = _mesh(interpret)
    def vmem(rows, cols):
        return _vmem_type(mesh, interpret, (rows, cols), lhsT.dtype)

    @pl.kernel(
        mesh=mesh, interpret=interpretation,
        out_type=jax.ShapeDtypeStruct((Mp, Np), lhsT.dtype),
        scratch_types=(vmem(Mp, BLOCK_N), vmem(BLOCK_K, BLOCK_N),
                       vmem(BLOCK_K, BLOCK_M), vmem(TILE_M, BLOCK_N)),
        name="nki_matmul_fully_optimized_v5e",
    )
    def kernel(lhsT, rhs, result, result_tiles, rhs_tiles, lhsT_tiles, result_packed):
        # 与原代码一致：n -> k -> m -> bn -> bm -> bk。
        for n in range(NUM_BLOCK_N):
            @pl.when(lax.axis_index("core") == n % mesh.num_cores)
            def process_N_block():
                pltpu.store(result_tiles, jnp.zeros(result_tiles.shape, result_tiles.dtype))
                for k in range(NUM_BLOCK_K):
                    for bk_r in range(TILES_IN_BLOCK_K):
                        k_index = (k*TILES_IN_BLOCK_K + bk_r)*TILE_K
                        pltpu.sync_copy(_tile(rhs, k_index, BLOCK_N*n, TILE_K, BLOCK_N),
                                        _tile(rhs_tiles, bk_r*TILE_K, 0, TILE_K, BLOCK_N))
                    for m in range(NUM_BLOCK_M):
                        for bk_l in range(TILES_IN_BLOCK_K):
                            k_index = (k*TILES_IN_BLOCK_K + bk_l)*TILE_K
                            pltpu.sync_copy(_tile(lhsT, k_index, BLOCK_M*m, TILE_K, BLOCK_M),
                                            _tile(lhsT_tiles, bk_l*TILE_K, 0, TILE_K, BLOCK_M))
                        for bn in range(TILES_IN_BLOCK_N):
                            for bm in range(TILES_IN_BLOCK_M):
                                res_tile = jnp.zeros((TILE_M, TILE_N), jnp.float32)
                                for bk in range(TILES_IN_BLOCK_K):
                                    a = pltpu.load(_tile(lhsT_tiles, bk*TILE_K, bm*TILE_M, TILE_K, TILE_M))
                                    b = pltpu.load(_tile(rhs_tiles, bk*TILE_K, bn*TILE_N, TILE_K, TILE_N))
                                    res_tile += _dot(a, b, mixed_precision)
                                result_tile_ref = _tile(result_tiles, m*BLOCK_M + bm*TILE_M,
                                                        bn*TILE_N, TILE_M, TILE_N)
                                pltpu.store(result_tile_ref,
                                    (pltpu.load(result_tile_ref).astype(jnp.float32) + res_tile).astype(result_tiles.dtype))
                for m in range(NUM_BLOCK_M):
                    for bm in range(TILES_IN_BLOCK_M):
                        for bn in range(TILES_IN_BLOCK_N):
                            pltpu.store(_tile(result_packed, 0, bn*TILE_N, TILE_M, TILE_N),
                                pltpu.load(_tile(result_tiles, m*BLOCK_M + bm*TILE_M,
                                                 bn*TILE_N, TILE_M, TILE_N)))
                        pltpu.sync_copy(result_packed, _tile(
                            result, m*BLOCK_M + bm*TILE_M, BLOCK_N*n, TILE_M, BLOCK_N))
    return kernel(_pad(lhsT, Kp, Mp), _pad(rhs, Kp, Np))[:M, :N]


def jax_low_rank_down(hidden_t, u_down, v_down):
    """JAX Down from (I,S), retaining the input dtype at GEMM boundaries.

    JIT the enclosing MLP to include layout conversion and both GEMMs in one
    executable. No Pallas Down tiling or VMEM estimate applies to this path.
    """
    _validate_arrays(hidden_t, u_down, v_down)
    if hidden_t.shape[0] != u_down.shape[0] or u_down.shape[1] != v_down.shape[0]:
        raise ValueError("hidden, U, V contraction dimensions do not match")
    with jax.default_matmul_precision("highest"):
        return (hidden_t.T @ u_down) @ v_down


def svd_mlp_with_fused_kernel(x, u_up, v_up, u_gate, v_gate, u_down, v_down,
                             up_T_params=None, XTUV_params=None, *,
                             mixed_precision=True, interpret=False, async_dma=True,
                             down_backend="pallas"):
    """Pallas Gate/Up followed by Pallas (default) or JAX Down.

    For JAX Down, input dtypes determine GEMM boundaries and precision is
    highest; mixed_precision/async_dma configure the Pallas stages only.
    XTUV_params are unused with down_backend="jax".
    """
    if down_backend not in ("pallas", "jax"):
        raise ValueError("down_backend must be 'pallas' or 'jax'")
    S = x.shape[0]
    if up_T_params is None:
        up_T_params = get_fused_mlp_up_T_params(S)
    if down_backend == "pallas" and XTUV_params is None:
        XTUV_params = get_fused_three_mm_XTUV_params(S)
    activated_gate_t = fused_mlp_up_T(
        x, u_gate, v_gate, u_up, v_up, **up_T_params,
        mixed_precision=mixed_precision, interpret=interpret, async_dma=async_dma)
    if down_backend == "jax":
        return jax_low_rank_down(activated_gate_t, u_down, v_down)
    return fused_three_mm_XTUV(
        activated_gate_t, u_down, v_down, **XTUV_params,
        mixed_precision=mixed_precision, interpret=interpret, async_dma=async_dma)


def nki_mm(x, up_v_proj, up_u_proj, gate_v_proj, gate_u_proj,
           down_v_proj, down_u_proj, *, mixed_precision=True, interpret=False,
           up_T_params=None, XTUV_params=None, async_dma=True, down_backend="pallas"):
    # 与 torch.nn.Linear 的权重布局一致；内部使用 JAX 数组的 .T。
    return svd_mlp_with_fused_kernel(
        x, up_v_proj.T, up_u_proj.T, gate_v_proj.T, gate_u_proj.T,
        down_v_proj.T, down_u_proj.T, up_T_params, XTUV_params,
        mixed_precision=mixed_precision, interpret=interpret, async_dma=async_dma,
        down_backend=down_backend)


def nki_baseline(x, up_proj, gate_proj, down_proj, *,
                 mixed_precision=True, interpret=False, matmul_params=None):
    params = {} if matmul_params is None else matmul_params
    up = nki_matmul_fully_optimized_(x.T, up_proj.T, **params,
                                   mixed_precision=mixed_precision, interpret=interpret)
    gate = nki_matmul_fully_optimized_(x.T, gate_proj.T, **params,
                                     mixed_precision=mixed_precision, interpret=interpret)
    act = jax.nn.silu(gate.astype(jnp.float32)).astype(x.dtype) * up
    return nki_matmul_fully_optimized_(act.T, down_proj.T, **params,
                                      mixed_precision=mixed_precision, interpret=interpret)


def _unfused_reference(x, u_up, v_up, u_gate, v_gate, u_down, v_down):
    # FP32 golden reference；和 kernel 使用相同的 SVD 因子。
    x, u_up, v_up, u_gate, v_gate, u_down, v_down = [
        a.astype(jnp.float32) for a in (x, u_up, v_up, u_gate, v_gate, u_down, v_down)]
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    return ((jax.nn.silu(gate) * up) @ u_down) @ v_down


def _unfused_low_rank(x, u_up, v_up, u_gate, v_gate, u_down, v_down):
    # BF16 benchmark 对照：与融合实现同精度、同因子、同布局。
    # 使用JAX默认matmul（XLA编译器）
    gate = (x @ u_gate) @ v_gate
    up = (x @ u_up) @ v_up
    hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(x.dtype) * up
    return (hidden @ u_down) @ v_down


def _unfused_low_rank_pallas(x, u_up, v_up, u_gate, v_gate, u_down, v_down, *,
                              mixed_precision=True, interpret=False):
    """Unfused baseline using Pallas matmul - fair comparison to fused kernel.

    This uses the same nki_matmul_fully_optimized_ as the dense baseline,
    allowing us to isolate the fusion benefit from the Pallas matmul benefit.

    nki_matmul_fully_optimized_(lhsT, rhs) expects lhsT=(K,M), rhs=(K,N) and computes
    lhsT.T @ rhs = (M,K) @ (K,N) = (M,N).
    """
    # Gate projection: x @ u_gate @ v_gate
    # x=(S,H), u_gate=(H,R) -> want (S,R)
    # Call with lhsT=x.T=(H,S), rhs=u_gate=(H,R) -> output=(S,R)
    xu_gate = nki_matmul_fully_optimized_(
        x.T, u_gate, mixed_precision=mixed_precision, interpret=interpret)
    # xu_gate=(S,R), v_gate=(R,I) -> want (S,I)
    # Call with lhsT=xu_gate.T=(R,S), rhs=v_gate=(R,I) -> output=(S,I)
    gate = nki_matmul_fully_optimized_(
        xu_gate.T, v_gate, mixed_precision=mixed_precision, interpret=interpret)

    # Up projection: x @ u_up @ v_up
    xu_up = nki_matmul_fully_optimized_(
        x.T, u_up, mixed_precision=mixed_precision, interpret=interpret)
    up = nki_matmul_fully_optimized_(
        xu_up.T, v_up, mixed_precision=mixed_precision, interpret=interpret)

    # SwiGLU activation
    hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(x.dtype) * up

    # Down projection: hidden @ u_down @ v_down
    # hidden=(S,I), u_down=(I,R) -> want (S,R)
    hidden_xu = nki_matmul_fully_optimized_(
        hidden.T, u_down, mixed_precision=mixed_precision, interpret=interpret)
    # hidden_xu=(S,R), v_down=(R,H) -> want (S,H)
    result = nki_matmul_fully_optimized_(
        hidden_xu.T, v_down, mixed_precision=mixed_precision, interpret=interpret)

    return result


def _time_ms(fn, args, warmup, repeats):
    # 编译和预热不计入；每次都等待 TPU 执行完成。
    jax.block_until_ready(fn(*args))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="offline SVD checkpoint directory; test real layers")
    parser.add_argument("--layers", type=int, nargs="+", help="layer indices; defaults to all checkpoint layers")
    parser.add_argument("--output", help="real-layer JSON report path; also writes a sibling CSV")
    parser.add_argument("--inspect-only", action="store_true", help="validate real checkpoint headers without loading tensors")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--model", choices=("llama3-3b", "llama3-8b"), default="llama3-3b")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1, 128, 512])
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), help="defaults to checkpoint dtype, or bf16 for synthetic tests")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interpret", action="store_true", help="CPU 数值验证，不报告 TPU speedup")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--down-backend", choices=("pallas", "jax"), default="pallas",
                        help="jax: Pallas Gate/Up + JAX Down; pallas: original full Pallas path")
    parser.add_argument("--sync-dma", action="store_true", help="disable weight prefetch for an otherwise identical baseline")
    parser.add_argument("--profile-dir", default=None, help="JAX profiler trace directory, e.g. ./profiles/test_speedup")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be >= 0 and repeats must be > 0")
    for s in args.seq_lens:
        _positive("sequence length", s)
    if args.model_path:
        if any(v is not None for v in (args.hidden, args.intermediate, args.rank)):
            parser.error("real-checkpoint dimensions come from the tensors; omit --hidden/--intermediate/--rank")
        import sys
        from pathlib import Path
        if not __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from svdflash_tpu.bench.checkpoint_speedup import run
        return run(args, sys.modules[__name__])
    if args.layers is not None or args.inspect_only or args.output or args.max_relative_l2 is not None:
        parser.error("--layers/--inspect-only/--output/--max-relative-l2 require --model-path")
    args.dtype = args.dtype or "bf16"
    base = (3072, 8192, 1792) if args.model == "llama3-3b" else (4096, 14336, 2560)
    if args.interpret:
        base = (129, 137, 131)
    H, I, R = tuple(value if value is not None else default
                    for value, default in zip((args.hidden, args.intermediate, args.rank), base))
    for name, value in zip(("hidden", "intermediate", "rank"), (H, I, R)):
        _positive(name, value)
    if not args.interpret and jax.default_backend() != "tpu":
        parser.error("TPU backend required for benchmarking; use --interpret for CPU checks")
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    mixed = args.dtype == "bf16"
    rng = np.random.default_rng(args.seed)
    def random_matrix(rows, cols):
        return jnp.asarray(rng.normal(size=(rows, cols)).astype(np.float32) / math.sqrt(rows), dtype=dtype)
    factors = (random_matrix(H, R), random_matrix(R, I),
               random_matrix(H, R), random_matrix(R, I),
               random_matrix(I, R), random_matrix(R, H))
    # Dense 权重由相同因子重建；重建成本在计时之外。
    uu, vu, ug, vg, ud, vd = factors
    weights = tuple((u.astype(jnp.float32) @ v.astype(jnp.float32)).astype(dtype).T
                    for u, v in ((uu, vu), (ug, vg), (ud, vd)))
    print(json.dumps({"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
                      "mode": "interpret_correctness_only" if args.interpret else "tpu",
                      "shape": {"H": H, "I": I, "R": R}, "dtype": args.dtype,
                      "down_backend": args.down_backend}))
    profiling = args.profile_dir is not None
    if profiling:
        from pathlib import Path

        profile_dir = Path(args.profile_dir).expanduser().resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)

        # 确保前面的输入准备和权重重建完成，再开始采集。
        jax.block_until_ready((factors, weights))

        print(f"Starting JAX profiler: {profile_dir}", flush=True)
        jax.profiler.start_trace(str(profile_dir))

    try:
        for S in args.seq_lens:
            # 把原来 for S 循环中的全部代码放在这里。
            # 从 x = jnp.asarray(...) 开始，
            # 到 print(json.dumps(record), flush=True) 结束。
            x = jnp.asarray(rng.normal(size=(S, H)).astype(np.float32), dtype=dtype)
            up_params = get_fused_mlp_up_T_params(S)
            down_params = get_fused_three_mm_XTUV_params(S)
            if args.interpret:
                up_params = down_params = dict(M_tiles_in_block=1, r_tiles_in_block=1,
                                            K_tiles_in_block=1, N_tiles_in_block=1)
            fused = jax.jit(lambda x, *f: svd_mlp_with_fused_kernel(
                 x, *f, up_params, down_params, mixed_precision=mixed, interpret=args.interpret,
                 async_dma=not args.sync_dma, down_backend=args.down_backend))
            dense = jax.jit(lambda x, *w: nki_baseline(
                 x, *w, mixed_precision=mixed, interpret=args.interpret))
            unfused = jax.jit(lambda x, *f: _unfused_low_rank_pallas(
                x, *f, mixed_precision=mixed, interpret=args.interpret))
            call_args = (x, *factors)
            with jax.default_matmul_precision("highest"):
                expected = np.asarray(jax.jit(_unfused_reference)(*call_args))
            got = np.asarray(fused(*call_args)).astype(np.float32)
            if not np.isfinite(got).all():
                raise AssertionError("fused output contains NaN/Inf")
            rel_l2 = float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-12))
            tolerance = 0.12 if mixed else 0.001
            if rel_l2 > tolerance:
                raise AssertionError(f"S={S}: relative L2 error {rel_l2:.6g} > {tolerance}")
            record = {"S": S, "async_dma": not args.sync_dma, "down_backend": args.down_backend,
                      "up_params": up_params, "down_params": down_params if args.down_backend == "pallas" else None,
                      "relative_l2": rel_l2, "max_abs_error": float(np.max(np.abs(got-expected)))}
            if not args.interpret and not args.check_only:
                print(f"S={S}: correctness passed; compiling/warming dense and unfused baselines.", flush=True)
                fused_ms = _time_ms(fused, call_args, args.warmup, args.repeats)
                dense_ms = _time_ms(dense, (x, *weights), args.warmup, args.repeats)
                unfused_ms = _time_ms(unfused, call_args, args.warmup, args.repeats)
                record.update(fused_ms=fused_ms, dense_ms=dense_ms, unfused_low_rank_ms=unfused_ms,
                              speedup_vs_dense=dense_ms/fused_ms, speedup_vs_unfused=unfused_ms/fused_ms)
            print(json.dumps(record), flush=True)
            
    finally:
        if profiling:
            jax.profiler.stop_trace()
            print(f"JAX profiler trace saved to: {profile_dir}", flush=True)
if __name__ == "__main__":
    main()


     


