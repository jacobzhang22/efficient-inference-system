import os

import matplotlib.pyplot as plt
import pandas as pd

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

        plt.figure(figsize=(8, 5))
        plt.plot(df["prompt_len"], df["no_cache_avg_generated_token_ms"], marker="o", label="No cache")
        plt.plot(df["prompt_len"], df["cache_avg_generated_token_ms"], marker="o", label="With KV cache")
        plt.xlabel("Prompt length")
        plt.ylabel("Average end-to-end time per generated token (ms)")
        plt.title("Avg generated-token latency vs prompt length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/avg_generated_token_latency.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df["prompt_len"], df["cache_avg_decode_only_token_ms"], marker="o")
        plt.xlabel("Prompt length")
        plt.ylabel("Average decode-only token time (ms)")
        plt.title("Cached decode-only latency vs prompt length")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/cache_decode_only_latency.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df["prompt_len"], df["no_cache_total_ms"], marker="o", label="No cache")
        plt.plot(df["prompt_len"], df["cache_total_ms"], marker="o", label="With KV cache")
        plt.xlabel("Prompt length")
        plt.ylabel("Total generation time (ms)")
        plt.title("Total generation time vs prompt length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/total_generation_time.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df["prompt_len"], df["cache_prefill_ms"], marker="o")
        plt.xlabel("Prompt length")
        plt.ylabel("Prefill time (ms)")
        plt.title("Prefill time vs prompt length")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/prefill_time.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df["prompt_len"], df["cache_memory_mb"], marker="o")
        plt.xlabel("Prompt length")
        plt.ylabel("KV cache memory (MB)")
        plt.title("KV cache memory vs prompt length")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/cache_memory_vs_prompt.png")
        plt.close()

    if os.path.exists(memory_path):
        mem_df = pd.read_csv(memory_path)

        decode_df = mem_df[mem_df["stage"] == "decode"].copy()

        plt.figure(figsize=(8, 5))
        plt.plot(decode_df["decoded_tokens"], decode_df["cache_memory_mb"], marker="o")
        plt.xlabel("Decoded tokens")
        plt.ylabel("KV cache memory (MB)")
        plt.title("KV cache growth during decoding")
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