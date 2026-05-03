from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_pool_ptr,
    v_pool_ptr,
    page_table_ptr,
    seq_lens_ptr,
    current_lengths_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_ptb,
    stride_ptp,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    scale,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_pages: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    d_offsets = tl.arange(0, head_dim)
    q_ptrs = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + d_offsets * stride_qd
    q = tl.load(q_ptrs)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    cur_len = tl.load(current_lengths_ptr + batch_idx)
    query_abs_pos = seq_len - cur_len

    running_max = -float("inf")
    running_denom = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for page_idx in tl.static_range(0, max_pages):
        block_id = tl.load(page_table_ptr + batch_idx * stride_ptb + page_idx * stride_ptp)
        if block_id >= 0:
            for block_offset in tl.static_range(0, block_size):
                token_idx = page_idx * block_size + block_offset
                if token_idx < seq_len and token_idx <= query_abs_pos:
                    k_ptrs = (
                        k_pool_ptr
                        + block_id * stride_kb
                        + head_idx * stride_kh
                        + block_offset * stride_kt
                        + d_offsets * stride_kd
                    )
                    v_ptrs = (
                        v_pool_ptr
                        + block_id * stride_kb
                        + head_idx * stride_kh
                        + block_offset * stride_kt
                        + d_offsets * stride_kd
                    )
                    k = tl.load(k_ptrs)
                    v = tl.load(v_ptrs)
                    score = tl.sum(q * k, axis=0) * scale
                    new_max = tl.maximum(running_max, score)
                    alpha = tl.exp(running_max - new_max)
                    beta = tl.exp(score - new_max)
                    acc = acc * alpha + beta * v
                    running_denom = running_denom * alpha + beta
                    running_max = new_max

    out = acc / tl.maximum(running_denom, 1e-9)
    out_ptrs = out_ptr + batch_idx * stride_ob + head_idx * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(q.dtype))


@triton.jit
def _paged_attention_prefill_kernel(
    q_ptr,
    k_pool_ptr,
    v_pool_ptr,
    page_table_ptr,
    seq_lens_ptr,
    current_lengths_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_ptb,
    stride_ptp,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    scale,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_pages: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_idx = tl.program_id(2)

    d_offsets = tl.arange(0, head_dim)
    q_ptrs = (
        q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + query_idx * stride_qt
        + d_offsets * stride_qd
    )
    q = tl.load(q_ptrs)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    cur_len = tl.load(current_lengths_ptr + batch_idx)
    query_abs_pos = seq_len - cur_len + query_idx

    running_max = -float("inf")
    running_denom = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for page_idx in tl.static_range(0, max_pages):
        block_id = tl.load(page_table_ptr + batch_idx * stride_ptb + page_idx * stride_ptp)
        if block_id >= 0:
            for block_offset in tl.static_range(0, block_size):
                token_idx = page_idx * block_size + block_offset
                if token_idx < seq_len and token_idx <= query_abs_pos:
                    k_ptrs = (
                        k_pool_ptr
                        + block_id * stride_kb
                        + head_idx * stride_kh
                        + block_offset * stride_kt
                        + d_offsets * stride_kd
                    )
                    v_ptrs = (
                        v_pool_ptr
                        + block_id * stride_kb
                        + head_idx * stride_kh
                        + block_offset * stride_kt
                        + d_offsets * stride_kd
                    )
                    k = tl.load(k_ptrs)
                    v = tl.load(v_ptrs)
                    score = tl.sum(q * k, axis=0) * scale
                    new_max = tl.maximum(running_max, score)
                    alpha = tl.exp(running_max - new_max)
                    beta = tl.exp(score - new_max)
                    acc = acc * alpha + beta * v
                    running_denom = running_denom * alpha + beta
                    running_max = new_max

    out = acc / tl.maximum(running_denom, 1e-9)
    out_ptrs = (
        out_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_idx * stride_ot
        + d_offsets * stride_od
    )
    tl.store(out_ptrs, out.to(q.dtype))


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    current_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (batch_size, num_heads)
    _paged_attention_decode_kernel[grid](
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        current_lengths,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        k_pool.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        1.0 / math.sqrt(head_dim),
        head_dim=head_dim,
        block_size=k_pool.shape[2],
        max_pages=max(page_table.shape[1], 1),
    )
    return out


def paged_attention_prefill_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    current_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, query_len, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (batch_size, num_heads, query_len)
    _paged_attention_prefill_kernel[grid](
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        current_lengths,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        k_pool.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        1.0 / math.sqrt(head_dim),
        head_dim=head_dim,
        block_size=k_pool.shape[2],
        max_pages=max(page_table.shape[1], 1),
    )
    return out
