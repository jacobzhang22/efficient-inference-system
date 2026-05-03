import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.cache.paged_kv import BatchedPagedKVCache, LayerBlockPool, PagedKVCacheState


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0, attention_backend: str = "triton_paged"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout, attention_backend=attention_backend)
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
        kv_cache: PagedKVCacheState | BatchedPagedKVCache | None = None,
        use_cache: bool = False,
        current_lengths: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PagedKVCacheState | BatchedPagedKVCache | None]:
        attn_out, kv_cache = self.attn(
            self.ln1(x),
            kv_cache=kv_cache,
            use_cache=use_cache,
            current_lengths=current_lengths,
        )
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
        attention_backend: str = "triton_paged",
        kv_block_size: int = 16,
        kv_pool_initial_blocks: int = 64,
        kv_pool_growth_factor: float = 2.0,
        enable_attention_correctness_checks: bool = False,
    ):
        super().__init__()
        if attention_backend not in {"paged_reference", "triton_paged"}:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.attention_backend = attention_backend
        self.kv_block_size = kv_block_size
        self.kv_pool_initial_blocks = kv_pool_initial_blocks
        self.kv_pool_growth_factor = kv_pool_growth_factor
        self.enable_attention_correctness_checks = enable_attention_correctness_checks
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, attention_backend=attention_backend) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.layer_block_pools: list[LayerBlockPool] | None = None

    def _ensure_layer_block_pools(self) -> list[LayerBlockPool]:
        if self.layer_block_pools is None:
            self.layer_block_pools = [
                LayerBlockPool(
                    block_size=self.kv_block_size,
                    initial_blocks=self.kv_pool_initial_blocks,
                    growth_factor=self.kv_pool_growth_factor,
                )
                for _ in self.blocks
            ]
        return self.layer_block_pools

    def reset_paged_cache_pools(self) -> None:
        self.layer_block_pools = None

    def create_request_kv_caches(self, batch_size: int) -> list[list[PagedKVCacheState]]:
        pools = self._ensure_layer_block_pools()
        return [
            [PagedKVCacheState(pool=pools[layer_idx]) for layer_idx in range(len(self.blocks))]
            for _ in range(batch_size)
        ]

    def release_request_caches(self, request_caches: list[PagedKVCacheState | None] | None) -> None:
        if request_caches is None:
            return

        for cache in request_caches:
            if isinstance(cache, PagedKVCacheState):
                cache.release()

    def release_kv_caches(self, kv_caches: list[PagedKVCacheState | BatchedPagedKVCache | None] | None) -> None:
        if kv_caches is None:
            return

        for cache in kv_caches:
            if isinstance(cache, BatchedPagedKVCache):
                for state in cache.states:
                    state.release()
            elif isinstance(cache, PagedKVCacheState):
                cache.release()

    def set_attention_backend(self, backend_name: str) -> None:
        if backend_name not in {"paged_reference", "triton_paged"}:
            raise ValueError(f"Unsupported attention backend: {backend_name}")
        self.attention_backend = backend_name
        for block in self.blocks:
            block.attn.attention_backend = backend_name

    def paged_memory_stats(self) -> dict[str, int]:
        pools = self.layer_block_pools or []
        reserved = sum(pool.reserved_bytes() for pool in pools)
        allocated = reserved - sum(pool.bytes_per_block() * len(pool.free_block_ids) for pool in pools)
        return {
            "allocated_kv_block_bytes": allocated,
            "reserved_kv_bytes": reserved,
            "free_kv_pool_bytes": max(reserved - allocated, 0),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[PagedKVCacheState | BatchedPagedKVCache | None] | None = None,
        use_cache: bool = False,
        position_offset: int | torch.Tensor = 0,
        current_lengths: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[PagedKVCacheState | BatchedPagedKVCache | None] | None]:
        B, T = input_ids.shape
        device = input_ids.device

        if isinstance(position_offset, int):
            positions = torch.arange(position_offset, position_offset + T, device=device)
            positions = positions.unsqueeze(0).expand(B, T)
        else:
            positions = position_offset.view(B, 1) + torch.arange(T, device=device).view(1, T)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if use_cache and kv_caches is None:
            per_request = self.create_request_kv_caches(B)
            if B > 1:
                pools = self._ensure_layer_block_pools()
                kv_caches = [
                    BatchedPagedKVCache(
                        states=[per_request[request_idx][layer_idx] for request_idx in range(B)],
                        pool=pools[layer_idx],
                    )
                    for layer_idx in range(len(self.blocks))
                ]
            else:
                kv_caches = [per_request[0][layer_idx] for layer_idx in range(len(self.blocks))]

        new_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if use_cache else None
            x, cache_i = block(
                x,
                kv_cache=cache_i,
                use_cache=use_cache,
                current_lengths=current_lengths,
            )
            if use_cache:
                new_caches.append(cache_i)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches
