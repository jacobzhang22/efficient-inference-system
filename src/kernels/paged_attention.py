from __future__ import annotations

import torch

from src.cache.paged_kv import BatchedPagedKVCache
from src.kernels.reference_paged_attention import paged_attention_reference
from src.kernels.triton_paged_attention import (
    paged_attention_decode_triton,
    paged_attention_prefill_triton,
)


def paged_attention(
    backend: str,
    q: torch.Tensor,
    kv_cache: BatchedPagedKVCache,
    current_lengths: int | torch.Tensor | None,
) -> torch.Tensor:
    if backend not in {"paged_reference", "triton_paged"}:
        raise ValueError(f"Unsupported paged attention backend: {backend}")

    if backend == "paged_reference" or not q.is_cuda:
        return paged_attention_reference(q=q, kv_cache=kv_cache, current_lengths=current_lengths)

    page_table = kv_cache.page_table_tensor(device=q.device)
    seq_lens = kv_cache.seq_lens_tensor(device=q.device)
    if current_lengths is None:
        current_lengths_tensor = torch.full((q.shape[0],), q.shape[2], device=q.device, dtype=torch.int32)
    elif isinstance(current_lengths, int):
        current_lengths_tensor = torch.full((q.shape[0],), current_lengths, device=q.device, dtype=torch.int32)
    else:
        current_lengths_tensor = current_lengths.to(device=q.device, dtype=torch.int32)

    if q.shape[2] == 1:
        return paged_attention_decode_triton(
            q=q,
            k_pool=kv_cache.pool.k_blocks,
            v_pool=kv_cache.pool.v_blocks,
            page_table=page_table,
            seq_lens=seq_lens,
            current_lengths=current_lengths_tensor,
        )

    return paged_attention_prefill_triton(
        q=q,
        k_pool=kv_cache.pool.k_blocks,
        v_pool=kv_cache.pool.v_blocks,
        page_table=page_table,
        seq_lens=seq_lens,
        current_lengths=current_lengths_tensor,
    )
