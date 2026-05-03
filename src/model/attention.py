import math
import torch
import torch.nn as nn

from src.cache.paged_kv import BatchedPagedKVCache, PagedKVCacheState
from src.kernels.paged_attention import paged_attention


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, attention_backend: str = "triton_paged"):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCacheState | BatchedPagedKVCache | None = None,
        use_cache: bool = False,
        current_lengths: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PagedKVCacheState | BatchedPagedKVCache | None]:
        _, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k_new = self._split_heads(self.k_proj(x))
        v_new = self._split_heads(self.v_proj(x))

        if use_cache:
            if isinstance(kv_cache, PagedKVCacheState):
                kv_cache = BatchedPagedKVCache(states=[kv_cache], pool=kv_cache.pool)
            elif kv_cache is None:
                raise ValueError("Paged cached attention requires initialized paged cache state.")

            kv_cache.append_batch(k_new, v_new, current_lengths=current_lengths)
            out = paged_attention(
                backend=self.attention_backend,
                q=q,
                kv_cache=kv_cache,
                current_lengths=current_lengths,
            )
            out = self._merge_heads(out)
            out = self.out_proj(out)
            if len(kv_cache.states) == 1:
                return out, kv_cache.states[0]
            return out, kv_cache

        scores = torch.matmul(q, k_new.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_new)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, None
