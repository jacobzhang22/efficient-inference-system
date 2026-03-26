from experiments.dynamic_batching.benchmark_scheduler import run as run_benchmark
from experiments.dynamic_batching.plot_scheduler_results import run as run_plots


if __name__ == "__main__":
    print("Running dynamic batching benchmark...")
    run_benchmark()

    print("Generating scheduler plots...")
    run_plots()

    print("Done.")