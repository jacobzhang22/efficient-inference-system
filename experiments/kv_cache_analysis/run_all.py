from experiments.kv_cache_analysis.benchmark_kv_cache import run as run_benchmark
from experiments.kv_cache_analysis.memory_growth import run as run_memory
from experiments.kv_cache_analysis.plot_results import run as run_plots


if __name__ == "__main__":
    print("Running KV cache benchmark...")
    run_benchmark()

    print("Running memory growth analysis...")
    run_memory()

    print("Generating plots...")
    run_plots()

    print("Done.")