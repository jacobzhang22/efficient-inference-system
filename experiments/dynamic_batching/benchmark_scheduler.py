import os
import pandas as pd
import torch

from src.config import ModelConfig, SchedulingExperimentConfig
from src.model.transformer import TinyTransformerLM
from src.serving.batched_generate import run_batch_generate
from src.serving.loadgen import generate_requests
from src.serving.metrics import batches_to_rows, requests_to_rows, summarize_run
from src.serving.scheduler import DynamicBatchingScheduler
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
    cfg = SchedulingExperimentConfig()
    set_seed(cfg.seed)

    raw_dir = f"{cfg.output_dir}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    model, mcfg = build_model(cfg.device)

    summary_rows = []
    request_rows = []
    batch_rows = []

    for arrival_rate_rps in cfg.arrival_rates:
        for max_batch_size in cfg.max_batch_sizes:
            for batch_timeout_ms in cfg.batch_timeouts_ms:
                for run_idx in range(cfg.repeats):
                    print(
                        f"arrival_rate={arrival_rate_rps}, "
                        f"max_batch_size={max_batch_size}, "
                        f"batch_timeout_ms={batch_timeout_ms}, "
                        f"run={run_idx}"
                    )

                    requests = generate_requests(
                        num_requests=cfg.num_requests,
                        arrival_rate_rps=arrival_rate_rps,
                        prompt_len=cfg.prompt_len,
                        max_new_tokens=cfg.max_new_tokens,
                        vocab_size=mcfg.vocab_size,
                        seed=cfg.seed + run_idx,
                    )

                    scheduler = DynamicBatchingScheduler(
                        max_batch_size=max_batch_size,
                        batch_timeout_ms=batch_timeout_ms,
                    )

                    def batch_executor(batch):
                        return run_batch_generate(
                            model=model,
                            requests=batch,
                            device=cfg.device,
                        )

                    with torch.no_grad():
                        completed, batch_records = scheduler.run(
                            requests=requests,
                            batch_executor=batch_executor,
                        )

                    summary_rows.append(
                        summarize_run(
                            completed_requests=completed,
                            batch_records=batch_records,
                            arrival_rate_rps=arrival_rate_rps,
                            max_batch_size=max_batch_size,
                            batch_timeout_ms=batch_timeout_ms,
                            run_idx=run_idx,
                        )
                    )

                    request_rows.extend(
                        requests_to_rows(
                            completed_requests=completed,
                            arrival_rate_rps=arrival_rate_rps,
                            max_batch_size=max_batch_size,
                            batch_timeout_ms=batch_timeout_ms,
                            run_idx=run_idx,
                        )
                    )

                    batch_rows.extend(
                        batches_to_rows(
                            batch_records=batch_records,
                            arrival_rate_rps=arrival_rate_rps,
                            max_batch_size=max_batch_size,
                            batch_timeout_ms=batch_timeout_ms,
                            run_idx=run_idx,
                        )
                    )

    summary_df = pd.DataFrame(summary_rows)
    request_df = pd.DataFrame(request_rows)
    batch_df = pd.DataFrame(batch_rows)

    summary_path = f"{raw_dir}/summary.csv"
    request_path = f"{raw_dir}/requests.csv"
    batch_path = f"{raw_dir}/batches.csv"

    summary_df.to_csv(summary_path, index=False)
    request_df.to_csv(request_path, index=False)
    batch_df.to_csv(batch_path, index=False)

    print(summary_df.head())
    print(f"Saved summary to {summary_path}")
    print(f"Saved request-level results to {request_path}")
    print(f"Saved batch-level results to {batch_path}")


if __name__ == "__main__":
    run()