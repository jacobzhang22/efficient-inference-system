import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig


def run():
    ecfg = ExperimentConfig()
    raw_dir = f"{ecfg.output_dir}/raw"
    plot_dir = f"{ecfg.output_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    benchmark_path = f"{raw_dir}/benchmark_results.csv"
    memory_path = f"{raw_dir}/memory_growth.csv"

    if os.path.exists(benchmark_path):
        df = pd.read_csv(benchmark_path)
        if not df.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["cache_avg_generated_token_ms"], marker="o")
            plt.xlabel("Prompt length")
            plt.ylabel("Average generated-token time (ms)")
            plt.title("Paged engine generated-token latency vs prompt length")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/avg_generated_token_latency.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["cache_avg_decode_only_token_ms"], marker="o")
            plt.xlabel("Prompt length")
            plt.ylabel("Decode-only token time (ms)")
            plt.title("Paged engine decode-only latency vs prompt length")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/cache_decode_only_latency.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["cache_total_ms"], marker="o")
            plt.xlabel("Prompt length")
            plt.ylabel("Total generation time (ms)")
            plt.title("Paged engine total generation time vs prompt length")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/total_generation_time.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["cache_prefill_ms"], marker="o")
            plt.xlabel("Prompt length")
            plt.ylabel("Prefill time (ms)")
            plt.title("Paged engine prefill time vs prompt length")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/prefill_time.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["live_cache_memory_mb"], marker="o", label="Live KV")
            plt.plot(df["prompt_len"], df["reserved_cache_memory_mb"], marker="o", label="Reserved KV")
            plt.xlabel("Prompt length")
            plt.ylabel("KV memory (MB)")
            plt.title("Paged engine KV memory vs prompt length")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/cache_memory_vs_prompt.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(df["prompt_len"], df["fragmentation_memory_mb"], marker="o")
            plt.xlabel("Prompt length")
            plt.ylabel("Fragmentation (MB)")
            plt.title("Paged engine fragmentation vs prompt length")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/fragmentation_vs_prompt.png")
            plt.close()

    if os.path.exists(memory_path):
        mem_df = pd.read_csv(memory_path)

        decode_df = mem_df[mem_df["stage"] == "decode"].copy()

        plt.figure(figsize=(8, 5))
        live_col = "live_cache_memory_mb" if "live_cache_memory_mb" in decode_df.columns else "cache_memory_mb"
        plt.plot(decode_df["decoded_tokens"], decode_df[live_col], marker="o", label="Live KV")
        if "reserved_cache_memory_mb" in decode_df.columns:
            plt.plot(
                decode_df["decoded_tokens"],
                decode_df["reserved_cache_memory_mb"],
                marker="o",
                label="Reserved pool",
            )
        plt.xlabel("Decoded tokens")
        plt.ylabel("KV cache memory (MB)")
        plt.title("KV cache growth during decoding")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/memory_growth_over_decode.png")
        plt.close()

        if "gpu_peak_allocated_mb" in mem_df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(mem_df.index, mem_df["gpu_peak_allocated_mb"], marker="o")
            plt.xlabel("Measurement step")
            plt.ylabel("GPU peak allocated (MB)")
            plt.title("GPU peak allocated memory over run")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/gpu_peak_memory_over_run.png")
            plt.close()

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    run()
