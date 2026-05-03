import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, ModelConfig
from src.inference.generate_with_cache import generate_with_cache
from src.model.transformer import TinyTransformerLM
from src.utils.metrics import bytes_to_mb, mean
from src.utils.seed import set_seed


def build_model(device: str):
    mcfg = ModelConfig()
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
    return model, mcfg


def run():
    ecfg = ExperimentConfig()
    set_seed(ecfg.seed)

    os.makedirs(f"{ecfg.output_dir}/raw", exist_ok=True)

    model, mcfg = build_model(ecfg.device)
    rows = []

    for prompt_len in ecfg.prompt_lengths:
        model.reset_paged_cache_pools()
        if ecfg.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"Running prompt_len={prompt_len}")

        cache_totals = []
        cache_prefills = []
        cache_avg_generated_tokens = []
        cache_avg_decode_only_tokens = []
        live_cache_mem_mb = []
        reserved_cache_mem_mb = []
        fragmentation_mem_mb = []

        for repeat_idx in range(ecfg.repeats + ecfg.warmup_runs):
            prompt = torch.randint(0, mcfg.vocab_size, (1, prompt_len), device=ecfg.device)

            cache_result = generate_with_cache(model, prompt, ecfg.max_new_tokens)

            if repeat_idx >= ecfg.warmup_runs:
                cache_totals.append(cache_result["total_time_ms"])
                cache_prefills.append(cache_result["prefill_time_ms"])
                cache_avg_generated_tokens.append(cache_result["avg_generated_token_time_ms"])
                cache_avg_decode_only_tokens.append(cache_result["avg_decode_only_token_time_ms"])
                live_cache_mem_mb.append(bytes_to_mb(cache_result["live_kv_bytes"]))
                reserved_cache_mem_mb.append(bytes_to_mb(cache_result["reserved_kv_bytes"]))
                fragmentation_mem_mb.append(bytes_to_mb(cache_result["fragmentation_bytes"]))
            model.release_kv_caches(cache_result.get("kv_caches"))

        cache_total = mean(cache_totals)
        cache_avg_gen = mean(cache_avg_generated_tokens)
        cache_avg_decode = mean(cache_avg_decode_only_tokens)

        rows.append(
            {
                "backend_name": model.attention_backend,
                "prompt_len": prompt_len,
                "cache_total_ms": cache_total,
                "cache_prefill_ms": mean(cache_prefills),
                "cache_avg_generated_token_ms": cache_avg_gen,
                "cache_avg_decode_only_token_ms": cache_avg_decode,
                "live_cache_memory_mb": mean(live_cache_mem_mb),
                "reserved_cache_memory_mb": mean(reserved_cache_mem_mb),
                "fragmentation_memory_mb": mean(fragmentation_mem_mb),
            }
        )

    df = pd.DataFrame(rows)
    output_path = f"{ecfg.output_dir}/raw/benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(df)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    run()
