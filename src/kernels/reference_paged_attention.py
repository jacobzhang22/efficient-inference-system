from __future__ import annotations

import math

import torch

from src.cache.paged_kv import BatchedPagedKVCache


def paged_attention_reference(
    q: torch.Tensor,
    kv_cache: BatchedPagedKVCache,
    current_lengths: int | torch.Tensor | None,
) -> torch.Tensor:
    batch_size, num_heads, query_len, head_dim = q.shape
    out = torch.zeros_like(q)

    if current_lengths is None:
        current_lengths_tensor = torch.full(
            (batch_size,),
            query_len,
            device=q.device,
            dtype=torch.long,
        )
    elif isinstance(current_lengths, int):
        current_lengths_tensor = torch.full((batch_size,), current_lengths, device=q.device, dtype=torch.long)
    else:
        current_lengths_tensor = current_lengths.to(device=q.device, dtype=torch.long)

    scale = 1.0 / math.sqrt(head_dim)
    for batch_idx, state in enumerate(kv_cache.states):
        seq_len = state.seq_len
        past_len = seq_len - int(current_lengths_tensor[batch_idx].item())
        for head_idx in range(num_heads):
            for query_idx in range(query_len):
                query_abs_pos = past_len + query_idx
                valid_scores = []
                valid_values = []

                for token_idx in range(seq_len):
                    if token_idx > query_abs_pos:
                        break

                    block_id = state.block_ids[token_idx // state.pool.block_size]
                    block_offset = token_idx % state.pool.block_size
                    k_vec = state.pool.k_blocks[block_id, head_idx, block_offset, :]
                    v_vec = state.pool.v_blocks[block_id, head_idx, block_offset, :]
                    valid_scores.append(torch.dot(q[batch_idx, head_idx, query_idx], k_vec) * scale)
                    valid_values.append(v_vec)

                if not valid_scores:
                    continue

                scores = torch.stack(valid_scores)
                weights = torch.softmax(scores, dim=0)
                values = torch.stack(valid_values, dim=0)
                out[batch_idx, head_idx, query_idx] = torch.sum(weights.unsqueeze(-1) * values, dim=0)

    return out
