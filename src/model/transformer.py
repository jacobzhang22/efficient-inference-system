import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.cache.kv_cache import KVCache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv_cache


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache | None] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache | None] | None]:
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(position_offset, position_offset + T, device=device)
        positions = positions.unsqueeze(0).expand(B, T)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if use_cache and kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if use_cache else None
            x, cache_i = block(x, kv_cache=cache_i, use_cache=use_cache)
            if use_cache:
                new_caches.append(cache_i)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches