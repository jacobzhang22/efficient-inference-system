import os
import pandas as pd
import torch

from src.config import ModelConfig, SchedulingExperimentConfig
from src.model.transformer import TinyTransformerLM
from src.serving.batched_generate import run_batch_generate
from src.serving.continuous_generate import run_decode_step, run_prefill_chunk
from src.serving.loadgen import generate_requests
from src.serving.metrics import batches_to_rows, requests_to_rows, summarize_run
from src.serving.scheduler import ContinuousBatchingScheduler, DynamicBatchingScheduler, StaticBatchingScheduler
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


def run_with_config(cfg: SchedulingExperimentConfig):
    set_seed(cfg.seed)

    raw_dir = f"{cfg.output_dir}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    model, mcfg = build_model(cfg.device)

    summary_rows = []
    request_rows = []
    batch_rows = []

    for scheduler_mode in cfg.scheduler_modes:
        for arrival_rate_rps in cfg.arrival_rates:
            for max_batch_size in cfg.max_batch_sizes:
                if scheduler_mode == "dynamic":
                    policy_values = cfg.batch_timeouts_ms
                elif scheduler_mode == "static":
                    policy_values = [0.0]
                else:
                    policy_values = cfg.prefill_chunk_sizes
                for policy_value in policy_values:
                    for run_idx in range(cfg.repeats):
                        print(
                            f"scheduler_mode={scheduler_mode}, "
                            f"arrival_rate={arrival_rate_rps}, "
                            f"max_batch_size={max_batch_size}, "
                            f"policy={policy_value}, "
                            f"run={run_idx}"
                        )

                        requests = generate_requests(
                            num_requests=cfg.num_requests,
                            arrival_rate_rps=arrival_rate_rps,
                            vocab_size=mcfg.vocab_size,
                            workload_profiles=cfg.resolved_request_workload_profiles(),
                            seed=cfg.seed + run_idx,
                        )

                        if scheduler_mode == "dynamic":
                            scheduler = DynamicBatchingScheduler(
                                max_batch_size=max_batch_size,
                                batch_timeout_ms=policy_value,
                            )

                            def batch_executor(batch):
                                for req in batch:
                                    req.scheduler_mode = "dynamic"
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
                            batch_timeout_ms = policy_value
                            scheduling_policy_value = policy_value
                        elif scheduler_mode == "static":
                            scheduler = StaticBatchingScheduler(max_batch_size=max_batch_size)

                            def batch_executor(batch):
                                for req in batch:
                                    req.scheduler_mode = "static"
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
                            batch_timeout_ms = -1.0
                            scheduling_policy_value = 0.0
                        else:
                            scheduler = ContinuousBatchingScheduler(
                                max_batch_size=max_batch_size,
                                prefill_chunk_size=policy_value,
                                max_tokens_per_iteration=cfg.max_tokens_per_iteration,
                            )

                            def prefill_executor(batch, chunk_size, current_time_ms, event_id):
                                return run_prefill_chunk(
                                    model=model,
                                    requests=batch,
                                    chunk_size=chunk_size,
                                    device=cfg.device,
                                    start_time_ms=current_time_ms,
                                    event_id=event_id,
                                )

                            def decode_executor(batch, current_time_ms, event_id):
                                return run_decode_step(
                                    model=model,
                                    requests=batch,
                                    device=cfg.device,
                                    start_time_ms=current_time_ms,
                                    event_id=event_id,
                                )

                            with torch.no_grad():
                                completed, batch_records = scheduler.run(
                                    requests=requests,
                                    prefill_executor=prefill_executor,
                                    decode_executor=decode_executor,
                                )
                            batch_timeout_ms = -1.0
                            scheduling_policy_value = policy_value

                        summary_rows.append(
                            summarize_run(
                                completed_requests=completed,
                                batch_records=batch_records,
                                arrival_rate_rps=arrival_rate_rps,
                                max_batch_size=max_batch_size,
                                batch_timeout_ms=batch_timeout_ms,
                                scheduler_mode=scheduler_mode,
                                scheduling_policy_value=scheduling_policy_value,
                                run_idx=run_idx,
                            )
                        )

                        request_rows.extend(
                            requests_to_rows(
                                completed_requests=completed,
                                arrival_rate_rps=arrival_rate_rps,
                                max_batch_size=max_batch_size,
                                batch_timeout_ms=batch_timeout_ms,
                                scheduler_mode=scheduler_mode,
                                scheduling_policy_value=scheduling_policy_value,
                                run_idx=run_idx,
                            )
                        )

                        batch_rows.extend(
                            batches_to_rows(
                                batch_records=batch_records,
                                arrival_rate_rps=arrival_rate_rps,
                                max_batch_size=max_batch_size,
                                batch_timeout_ms=batch_timeout_ms,
                                scheduler_mode=scheduler_mode,
                                scheduling_policy_value=scheduling_policy_value,
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


def run():
    cfg = SchedulingExperimentConfig()
    if cfg.heterogeneous_requests:
        print("Request mix:")
        for profile in cfg.resolved_request_workload_profiles():
            print(
                f"  {profile.name}: prompt_len=[{profile.prompt_len_min}, {profile.prompt_len_max}], "
                f"max_new_tokens=[{profile.max_new_tokens_min}, {profile.max_new_tokens_max}], "
                f"weight={profile.weight}"
            )
    run_with_config(cfg)


if __name__ == "__main__":
    run()
