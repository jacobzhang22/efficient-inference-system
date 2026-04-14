import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.kv_cache import BatchedKVCache, KVCache


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
        past_kv_len: int | torch.Tensor,
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
        key_positions = torch.arange(total_kv_len, device=device)

        if isinstance(past_kv_len, int):
            query_positions = torch.arange(
                past_kv_len,
                past_kv_len + current_seq_len,
                device=device,
            ).unsqueeze(-1)
            causal_mask = key_positions.unsqueeze(0) <= query_positions
            return scores.masked_fill(
                ~causal_mask.view(1, 1, current_seq_len, total_kv_len),
                float("-inf"),
            )

        query_offsets = torch.arange(current_seq_len, device=device).view(1, current_seq_len, 1)
        query_positions = past_kv_len.view(-1, 1, 1) + query_offsets
        valid_key_mask = key_positions.view(1, 1, total_kv_len) < (
            past_kv_len.view(-1, 1, 1) + current_seq_len
        )
        causal_mask = key_positions.view(1, 1, total_kv_len) <= query_positions
        combined_mask = causal_mask & valid_key_mask
        return scores.masked_fill(~combined_mask.unsqueeze(1), float("-inf"))

    def _build_attention_mask(
        self,
        total_kv_lens: torch.Tensor,
        current_seq_len: int,
        total_kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        key_positions = torch.arange(total_kv_len, device=device).view(1, 1, total_kv_len)
        past_kv_lens = total_kv_lens - current_seq_len
        query_positions = past_kv_lens.view(-1, 1, 1) + torch.arange(current_seq_len, device=device).view(1, current_seq_len, 1)
        valid_key_mask = key_positions < total_kv_lens.view(-1, 1, 1)
        causal_mask = key_positions <= query_positions
        return valid_key_mask & causal_mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | BatchedKVCache | None = None,
        use_cache: bool = False,
        current_lengths: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | BatchedKVCache | None]:
        _, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k_new = self._split_heads(self.k_proj(x))
        v_new = self._split_heads(self.v_proj(x))

        if use_cache:
            if isinstance(kv_cache, BatchedKVCache):
                total_kv_lens = kv_cache.append_batch(k_new, v_new, current_lengths=current_lengths)
                k_all, v_all, total_kv_lens = kv_cache.to_padded_tensors()
                total_kv_len = k_all.shape[2]

                attn_mask = self._build_attention_mask(
                    total_kv_lens=total_kv_lens,
                    current_seq_len=seq_len,
                    total_kv_len=total_kv_len,
                    device=x.device,
                )

                q_flat = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
                k_flat = k_all.reshape(k_all.shape[0] * k_all.shape[1], k_all.shape[2], k_all.shape[3])
                v_flat = v_all.reshape(v_all.shape[0] * v_all.shape[1], v_all.shape[2], v_all.shape[3])
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                attn_mask = attn_mask.reshape(q.shape[0] * q.shape[1], seq_len, total_kv_len)

                out = F.scaled_dot_product_attention(
                    q_flat,
                    k_flat,
                    v_flat,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                )
                out = out.view(q.shape[0], self.num_heads, seq_len, self.head_dim)
                out = self._merge_heads(out)
                out = self.out_proj(out)
                return out, kv_cache

            if kv_cache is None:
                past_kv_len = 0
            else:
                past_kv_len = kv_cache.seq_len

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
