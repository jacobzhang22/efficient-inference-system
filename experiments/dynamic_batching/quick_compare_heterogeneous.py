import pandas as pd
import torch

from experiments.dynamic_batching.benchmark_scheduler import build_model
from src.config import SchedulingExperimentConfig, default_serving_request_mix
from src.serving.batched_generate import run_batch_generate
from src.serving.continuous_generate import run_decode_step, run_prefill_chunk
from src.serving.loadgen import generate_requests
from src.serving.metrics import summarize_run
from src.serving.scheduler import ContinuousBatchingScheduler, DynamicBatchingScheduler
from src.utils.seed import set_seed


def _make_requests(model, cfg: SchedulingExperimentConfig, arrival_rate: float):
    return generate_requests(
        num_requests=cfg.num_requests,
        arrival_rate_rps=arrival_rate,
        vocab_size=model.vocab_size,
        workload_profiles=cfg.resolved_request_workload_profiles(),
        seed=cfg.seed,
    )


def _run_dynamic(model, cfg: SchedulingExperimentConfig, arrival_rate: float) -> dict:
    requests = _make_requests(model, cfg, arrival_rate)
    scheduler = DynamicBatchingScheduler(
        max_batch_size=cfg.max_batch_sizes[0],
        batch_timeout_ms=cfg.batch_timeouts_ms[0],
    )

    def batch_executor(batch):
        return run_batch_generate(model=model, requests=batch, device=cfg.device)

    with torch.no_grad():
        completed, batch_records = scheduler.run(requests, batch_executor)

    return summarize_run(
        completed_requests=completed,
        batch_records=batch_records,
        arrival_rate_rps=arrival_rate,
        max_batch_size=cfg.max_batch_sizes[0],
        batch_timeout_ms=cfg.batch_timeouts_ms[0],
        scheduler_mode="dynamic",
        scheduling_policy_value=cfg.batch_timeouts_ms[0],
        run_idx=0,
    )


def _run_continuous(
    model,
    cfg: SchedulingExperimentConfig,
    arrival_rate: float,
    prefill_chunk_size: int,
    max_tokens_per_iteration: int,
) -> dict:
    requests = _make_requests(model, cfg, arrival_rate)
    scheduler = ContinuousBatchingScheduler(
        max_batch_size=cfg.max_batch_sizes[0],
        prefill_chunk_size=prefill_chunk_size,
        max_tokens_per_iteration=max_tokens_per_iteration,
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

    summary = summarize_run(
        completed_requests=completed,
        batch_records=batch_records,
        arrival_rate_rps=arrival_rate,
        max_batch_size=cfg.max_batch_sizes[0],
        batch_timeout_ms=-1.0,
        scheduler_mode="continuous",
        scheduling_policy_value=prefill_chunk_size,
        run_idx=0,
    )
    summary["max_tokens_per_iteration"] = max_tokens_per_iteration
    return summary


def run():
    cfg = SchedulingExperimentConfig(
        scheduler_modes=["dynamic", "continuous"],
        arrival_rates=[36.0, 44.0],
        max_batch_sizes=[8],
        batch_timeouts_ms=[0.0],
        prefill_chunk_sizes=[256],
        max_tokens_per_iteration=1024,
        num_requests=100,
        heterogeneous_requests=True,
        request_workload_profiles=default_serving_request_mix(),
        repeats=1,
    )
    set_seed(cfg.seed)
    model, _ = build_model(cfg.device)

    print("Request mix:")
    for profile in cfg.resolved_request_workload_profiles():
        print(
            f"  {profile.name}: prompt_len=[{profile.prompt_len_min}, {profile.prompt_len_max}], "
            f"max_new_tokens=[{profile.max_new_tokens_min}, {profile.max_new_tokens_max}], "
            f"weight={profile.weight}"
        )

    rows = []
    for arrival_rate in cfg.arrival_rates:
        rows.append(_run_dynamic(model, cfg, arrival_rate))
        rows.append(
            _run_continuous(
                model,
                cfg,
                arrival_rate,
                prefill_chunk_size=cfg.prefill_chunk_sizes[0],
                max_tokens_per_iteration=cfg.max_tokens_per_iteration,
            )
        )

    df = pd.DataFrame(rows)
    cols = [
        "scheduler_mode",
        "arrival_rate_rps",
        "throughput_rps",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_first_token_latency_ms",
        "mean_prompt_len",
        "mean_max_new_tokens",
        "mean_active_requests",
        "mean_tokens_scheduled",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    run()
