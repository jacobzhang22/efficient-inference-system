import os

import pandas as pd
import torch

from src.config import ExperimentConfig, ModelConfig
from src.inference.generate_no_cache import generate_no_cache
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
        print(f"Running prompt_len={prompt_len}")

        no_cache_totals = []
        no_cache_avg_generated_tokens = []

        cache_totals = []
        cache_prefills = []
        cache_avg_generated_tokens = []
        cache_avg_decode_only_tokens = []
        cache_mem_mb = []

        for repeat_idx in range(ecfg.repeats + ecfg.warmup_runs):
            prompt = torch.randint(0, mcfg.vocab_size, (1, prompt_len), device=ecfg.device)

            no_cache_result = generate_no_cache(model, prompt, ecfg.max_new_tokens)
            cache_result = generate_with_cache(model, prompt, ecfg.max_new_tokens)

            if repeat_idx >= ecfg.warmup_runs:
                no_cache_totals.append(no_cache_result["total_time_ms"])
                no_cache_avg_generated_tokens.append(no_cache_result["avg_generated_token_time_ms"])

                cache_totals.append(cache_result["total_time_ms"])
                cache_prefills.append(cache_result["prefill_time_ms"])
                cache_avg_generated_tokens.append(cache_result["avg_generated_token_time_ms"])
                cache_avg_decode_only_tokens.append(cache_result["avg_decode_only_token_time_ms"])
                cache_mem_mb.append(bytes_to_mb(cache_result["cache_bytes"]))

        no_cache_total = mean(no_cache_totals)
        cache_total = mean(cache_totals)
        no_cache_avg_gen = mean(no_cache_avg_generated_tokens)
        cache_avg_gen = mean(cache_avg_generated_tokens)
        cache_avg_decode = mean(cache_avg_decode_only_tokens)

        rows.append(
            {
                "prompt_len": prompt_len,
                "no_cache_total_ms": no_cache_total,
                "no_cache_avg_generated_token_ms": no_cache_avg_gen,
                "cache_total_ms": cache_total,
                "cache_prefill_ms": mean(cache_prefills),
                "cache_avg_generated_token_ms": cache_avg_gen,
                "cache_avg_decode_only_token_ms": cache_avg_decode,
                "cache_memory_mb": mean(cache_mem_mb),
                "speedup_total": (no_cache_total / cache_total) if cache_total > 0 else 0.0,
                "speedup_avg_generated_token": (
                    no_cache_avg_gen / cache_avg_gen if cache_avg_gen > 0 else 0.0
                ),
            }
        )

    df = pd.DataFrame(rows)
    output_path = f"{ecfg.output_dir}/raw/benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(df)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    run()
