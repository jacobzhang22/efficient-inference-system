import math
import torch
import torch.nn as nn

from src.cache.kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

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

    def _apply_causal_mask_with_cache(
        self,
        scores: torch.Tensor,
        current_seq_len: int,
        total_kv_len: int,
        past_kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        scores: [B, H, T, total_kv_len]
        Query positions correspond to absolute positions:
            past_kv_len, ..., past_kv_len + current_seq_len - 1
        Key positions correspond to:
            0, ..., total_kv_len - 1

        A query at absolute position q can attend to keys <= q.
        """
        query_positions = torch.arange(
            past_kv_len,
            past_kv_len + current_seq_len,
            device=device,
        ).unsqueeze(-1)  # [T, 1]

        key_positions = torch.arange(total_kv_len, device=device).unsqueeze(0)  # [1, K]
        causal_mask = key_positions <= query_positions  # [T, K]

        return scores.masked_fill(~causal_mask.view(1, 1, current_seq_len, total_kv_len), float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        _, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k_new = self._split_heads(self.k_proj(x))
        v_new = self._split_heads(self.v_proj(x))

        if use_cache:
            past_kv_len = 0 if kv_cache is None else kv_cache.seq_len

            if kv_cache is None:
                kv_cache = KVCache()

            kv_cache.append(k_new, v_new)
            k_all = kv_cache.k
            v_all = kv_cache.v
            total_kv_len = kv_cache.seq_len

            scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = self._apply_causal_mask_with_cache(
                scores=scores,
                current_seq_len=seq_len,
                total_kv_len=total_kv_len,
                past_kv_len=past_kv_len,
                device=x.device,
            )

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v_all)
            out = self._merge_heads(out)
            out = self.out_proj(out)
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