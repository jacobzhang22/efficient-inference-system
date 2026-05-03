import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, ModelConfig
from src.model.transformer import TinyTransformerLM
from src.cache.paged_kv import BatchedPagedKVCache, PagedKVCacheState
from src.utils.metrics import bytes_to_mb
from src.utils.seed import set_seed


def _cache_metric_totals(kv_caches, model) -> tuple[int, int, int, int]:
    live = 0
    allocated_blocks = 0
    for cache in kv_caches:
        if isinstance(cache, BatchedPagedKVCache):
            live += cache.live_bytes()
            allocated_blocks += cache.reserved_bytes()
        elif isinstance(cache, PagedKVCacheState):
            live += cache.live_bytes()
            allocated_blocks += cache.reserved_bytes()

    pool_stats = model.paged_memory_stats() if hasattr(model, "paged_memory_stats") else {}
    reserved_pool = pool_stats.get("reserved_kv_bytes", allocated_blocks)
    fragmentation = max(reserved_pool - live, 0)
    return live, allocated_blocks, reserved_pool, fragmentation


@torch.no_grad()
def run():
    ecfg = ExperimentConfig()
    mcfg = ModelConfig()
    set_seed(ecfg.seed)

    os.makedirs(f"{ecfg.output_dir}/raw", exist_ok=True)

    device = ecfg.device
    model = TinyTransformerLM(
        vocab_size=mcfg.vocab_size,
        d_model=mcfg.d_model,
        num_heads=mcfg.num_heads,
        num_layers=mcfg.num_layers,
        d_ff=mcfg.d_ff,
        max_seq_len=mcfg.max_seq_len,
        dropout=mcfg.dropout,
        attention_backend=mcfg.attention_backend,
        kv_block_size=mcfg.kv_block_size,
        kv_pool_initial_blocks=mcfg.kv_pool_initial_blocks,
        kv_pool_growth_factor=mcfg.kv_pool_growth_factor,
        enable_attention_correctness_checks=mcfg.enable_attention_correctness_checks,
    ).to(device)
    model.eval()

    prompt_len = 16
    max_new_tokens = 64
    prompt = torch.randint(0, mcfg.vocab_size, (1, prompt_len), device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []

    # Prefill
    logits, kv_caches = model(prompt, use_cache=True, position_offset=0)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    live_bytes, allocated_block_bytes, reserved_pool_bytes, fragmentation_bytes = _cache_metric_totals(kv_caches, model)

    current_alloc_mb = bytes_to_mb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
    peak_alloc_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

    rows.append(
        {
            "stage": "prefill",
            "backend_name": model.attention_backend,
            "decoded_tokens": 0,
            "cache_memory_mb": bytes_to_mb(live_bytes),
            "live_cache_memory_mb": bytes_to_mb(live_bytes),
            "allocated_block_memory_mb": bytes_to_mb(allocated_block_bytes),
            "reserved_cache_memory_mb": bytes_to_mb(reserved_pool_bytes),
            "fragmentation_memory_mb": bytes_to_mb(fragmentation_bytes),
            "gpu_allocated_mb": current_alloc_mb,
            "gpu_peak_allocated_mb": peak_alloc_mb,
        }
    )

    last_token = next_token

    for step in range(max_new_tokens - 1):
        current_pos = prompt_len + step

        logits, kv_caches = model(
            last_token,
            kv_caches=kv_caches,
            use_cache=True,
            position_offset=current_pos,
        )
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        live_bytes, allocated_block_bytes, reserved_pool_bytes, fragmentation_bytes = _cache_metric_totals(kv_caches, model)
        current_alloc_mb = bytes_to_mb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
        peak_alloc_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

        rows.append(
            {
                "stage": "decode",
                "backend_name": model.attention_backend,
                "decoded_tokens": step + 1,
                "cache_memory_mb": bytes_to_mb(live_bytes),
                "live_cache_memory_mb": bytes_to_mb(live_bytes),
                "allocated_block_memory_mb": bytes_to_mb(allocated_block_bytes),
                "reserved_cache_memory_mb": bytes_to_mb(reserved_pool_bytes),
                "fragmentation_memory_mb": bytes_to_mb(fragmentation_bytes),
                "gpu_allocated_mb": current_alloc_mb,
                "gpu_peak_allocated_mb": peak_alloc_mb,
            }
        )

    df = pd.DataFrame(rows)
    output_path = f"{ecfg.output_dir}/raw/memory_growth.csv"
    df.to_csv(output_path, index=False)
    print(df.head())
    print(f"Saved memory growth results to {output_path}")
    model.release_kv_caches(kv_caches)


if __name__ == "__main__":
    run()
