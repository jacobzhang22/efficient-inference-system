import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dynamic_batching.benchmark_scheduler import build_model
from src.config import SchedulingExperimentConfig, default_serving_request_mix
from src.serving.batched_generate import run_batch_generate
from src.serving.continuous_generate import run_decode_step, run_prefill_chunk
from src.serving.loadgen import generate_requests
from src.serving.metrics import summarize_run
from src.serving.scheduler import (
    ContinuousBatchingScheduler,
    DynamicBatchingScheduler,
    StaticBatchingScheduler,
)
from src.utils.seed import set_seed


def _make_requests(model, cfg: SchedulingExperimentConfig, arrival_rate: float):
    return generate_requests(
        num_requests=cfg.num_requests,
        arrival_rate_rps=arrival_rate,
        vocab_size=model.vocab_size,
        workload_profiles=cfg.resolved_request_workload_profiles(),
        seed=cfg.seed,
    )


def _reset_run_state(model, cfg: SchedulingExperimentConfig) -> None:
    model.reset_paged_cache_pools()
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _run_dynamic(model, cfg: SchedulingExperimentConfig, arrival_rate: float, max_batch_size: int, timeout_ms: float, label: str) -> dict:
    _reset_run_state(model, cfg)
    requests = _make_requests(model, cfg, arrival_rate)
    scheduler = DynamicBatchingScheduler(max_batch_size=max_batch_size, batch_timeout_ms=timeout_ms)

    def batch_executor(batch):
        return run_batch_generate(model=model, requests=batch, device=cfg.device)

    with torch.no_grad():
        completed, batch_records = scheduler.run(
            requests,
            batch_executor,
            release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
        )

    summary = summarize_run(
        completed_requests=completed,
        batch_records=batch_records,
        arrival_rate_rps=arrival_rate,
        max_batch_size=max_batch_size,
        batch_timeout_ms=timeout_ms,
        scheduler_mode="dynamic",
        scheduling_policy_value=timeout_ms,
        run_idx=0,
    )
    summary["comparison_label"] = label
    return summary


def _run_static(model, cfg: SchedulingExperimentConfig, arrival_rate: float, max_batch_size: int) -> dict:
    _reset_run_state(model, cfg)
    requests = _make_requests(model, cfg, arrival_rate)
    scheduler = StaticBatchingScheduler(max_batch_size=max_batch_size)

    def batch_executor(batch):
        return run_batch_generate(model=model, requests=batch, device=cfg.device)

    with torch.no_grad():
        completed, batch_records = scheduler.run(
            requests,
            batch_executor,
            release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
        )

    summary = summarize_run(
        completed_requests=completed,
        batch_records=batch_records,
        arrival_rate_rps=arrival_rate,
        max_batch_size=max_batch_size,
        batch_timeout_ms=-1.0,
        scheduler_mode="static",
        scheduling_policy_value=0.0,
        run_idx=0,
    )
    summary["comparison_label"] = f"static(batch={max_batch_size})"
    return summary


def _run_continuous(model, cfg: SchedulingExperimentConfig, arrival_rate: float, max_batch_size: int, prefill_chunk_size: int, max_tokens_per_iteration: int) -> dict:
    _reset_run_state(model, cfg)
    requests = _make_requests(model, cfg, arrival_rate)
    scheduler = ContinuousBatchingScheduler(
        max_batch_size=max_batch_size,
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
            release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
        )

    summary = summarize_run(
        completed_requests=completed,
        batch_records=batch_records,
        arrival_rate_rps=arrival_rate,
        max_batch_size=max_batch_size,
        batch_timeout_ms=-1.0,
        scheduler_mode="continuous",
        scheduling_policy_value=prefill_chunk_size,
        run_idx=0,
    )
    summary["comparison_label"] = f"continuous(batch={max_batch_size},chunk={prefill_chunk_size})"
    return summary


def run():
    cfg = SchedulingExperimentConfig(
        arrival_rates=[44.0],
        num_requests=100,
        max_batch_sizes=[8],
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
    arrival_rate = cfg.arrival_rates[0]
    rows.append(_run_dynamic(model, cfg, arrival_rate, max_batch_size=1, timeout_ms=0.0, label="baseline_no_batching"))
    rows.append(_run_static(model, cfg, arrival_rate, max_batch_size=8))
    rows.append(_run_dynamic(model, cfg, arrival_rate, max_batch_size=8, timeout_ms=10.0, label="dynamic"))
    rows.append(_run_continuous(model, cfg, arrival_rate, max_batch_size=8, prefill_chunk_size=256, max_tokens_per_iteration=1536))

    df = pd.DataFrame(rows)
    cols = [
        "comparison_label",
        "throughput_rps",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_first_token_latency_ms",
        "mean_active_requests",
        "mean_tokens_scheduled",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    run()
