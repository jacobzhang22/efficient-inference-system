import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.batching.benchmark_scheduler import run as run_benchmark
from experiments.batching.plot_scheduler_results import run as run_plots


if __name__ == "__main__":
    print("Running batching benchmark...")
    run_benchmark()

    print("Generating scheduler plots...")
    run_plots()

    print("Done.")
