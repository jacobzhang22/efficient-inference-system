import os

import pandas as pd
import torch

from src.config import ExperimentConfig, ModelConfig
from src.model.transformer import TinyTransformerLM
from src.utils.metrics import bytes_to_mb
from src.utils.seed import set_seed


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

    total_cache_bytes = sum(cache.bytes_used() for cache in kv_caches if cache is not None)

    current_alloc_mb = bytes_to_mb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
    peak_alloc_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

    rows.append(
        {
            "stage": "prefill",
            "decoded_tokens": 0,
            "cache_memory_mb": bytes_to_mb(total_cache_bytes),
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

        total_cache_bytes = sum(cache.bytes_used() for cache in kv_caches if cache is not None)
        current_alloc_mb = bytes_to_mb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
        peak_alloc_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

        rows.append(
            {
                "stage": "decode",
                "decoded_tokens": step + 1,
                "cache_memory_mb": bytes_to_mb(total_cache_bytes),
                "gpu_allocated_mb": current_alloc_mb,
                "gpu_peak_allocated_mb": peak_alloc_mb,
            }
        )

    df = pd.DataFrame(rows)
    output_path = f"{ecfg.output_dir}/raw/memory_growth.csv"
    df.to_csv(output_path, index=False)
    print(df.head())
    print(f"Saved memory growth results to {output_path}")


if __name__ == "__main__":
    run()
