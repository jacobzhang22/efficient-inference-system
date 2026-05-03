import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.batching.benchmark_scheduler import run_with_config
from experiments.batching.plot_scheduler_results import run as run_plots
from src.config import SchedulingExperimentConfig


def run():
    cfg = SchedulingExperimentConfig(
        arrival_rates=[4.0],
        max_batch_sizes=[2],
        batch_timeouts_ms=[0.0],
        prefill_chunk_sizes=[32],
        max_tokens_per_iteration=256,
        num_requests=8,
        repeats=1,
        output_dir="results/batching_smoke",
    )

    print("Running batching smoke test...")
    run_with_config(cfg)

    print("Generating smoke-test plots...")
    run_plots(cfg)

    print("Smoke test done.")


if __name__ == "__main__":
    run()
