import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, SchedulingExperimentConfig
from src.inference.generate_with_cache import generate_with_cache
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
        attention_backend=mcfg.attention_backend,
        kv_block_size=mcfg.kv_block_size,
        kv_pool_initial_blocks=mcfg.kv_pool_initial_blocks,
        kv_pool_growth_factor=mcfg.kv_pool_growth_factor,
        enable_attention_correctness_checks=mcfg.enable_attention_correctness_checks,
    ).to(device)
    model.eval()
    return model, mcfg


def warmup_paged_backend(model, mcfg: ModelConfig, cfg: SchedulingExperimentConfig) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available():
        return

    max_batch_size = max(cfg.max_batch_sizes) if cfg.max_batch_sizes else 1
    warmup_prompt_len = min(mcfg.max_seq_len // 2, 512)
    prompt = torch.randint(
        0,
        mcfg.vocab_size,
        (max_batch_size, warmup_prompt_len),
        device=cfg.device,
    )
    with torch.no_grad():
        generate_with_cache(
            model,
            prompt,
            max_new_tokens=[2] * max_batch_size,
        )
    torch.cuda.synchronize()
    model.reset_paged_cache_pools()
    torch.cuda.reset_peak_memory_stats()


def _reset_model_measurement_state(model, device: str) -> None:
    model.reset_paged_cache_pools()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _run_scheduler_once(
    *,
    model,
    cfg: SchedulingExperimentConfig,
    requests: list,
    scheduler_mode: str,
    max_batch_size: int,
    policy_value: float,
):
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
                release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
            )
        return completed, batch_records, policy_value, policy_value

    if scheduler_mode == "static":
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
                release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
            )
        return completed, batch_records, -1.0, 0.0

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
            release_request_cache=lambda req: model.release_request_caches(req.kv_caches),
        )
    return completed, batch_records, -1.0, policy_value


def run_with_config(cfg: SchedulingExperimentConfig):
    set_seed(cfg.seed)

    raw_dir = f"{cfg.output_dir}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    model, mcfg = build_model(cfg.device)
    model.set_attention_backend(cfg.attention_backend)
    model.kv_block_size = cfg.kv_block_size
    model.kv_pool_initial_blocks = cfg.kv_pool_initial_blocks
    model.kv_pool_growth_factor = cfg.kv_pool_growth_factor
    model.enable_attention_correctness_checks = cfg.enable_attention_correctness_checks
    warmup_paged_backend(model, mcfg, cfg)

    summary_rows = []
    request_rows = []
    event_rows = []

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
                    _reset_model_measurement_state(model, cfg.device)
                    dry_run_requests = generate_requests(
                        num_requests=cfg.num_requests,
                        arrival_rate_rps=arrival_rate_rps,
                        vocab_size=mcfg.vocab_size,
                        workload_profiles=cfg.resolved_request_workload_profiles(),
                        seed=cfg.seed + 10_000,
                    )
                    print(
                        f"dry_run scheduler_mode={scheduler_mode}, "
                        f"arrival_rate={arrival_rate_rps}, "
                        f"max_batch_size={max_batch_size}, "
                        f"policy={policy_value}"
                    )
                    _run_scheduler_once(
                        model=model,
                        cfg=cfg,
                        requests=dry_run_requests,
                        scheduler_mode=scheduler_mode,
                        max_batch_size=max_batch_size,
                        policy_value=policy_value,
                    )
                    _reset_model_measurement_state(model, cfg.device)

                    for run_idx in range(cfg.repeats):
                        _reset_model_measurement_state(model, cfg.device)
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
                        completed, batch_records, batch_timeout_ms, scheduling_policy_value = _run_scheduler_once(
                            model=model,
                            cfg=cfg,
                            requests=requests,
                            scheduler_mode=scheduler_mode,
                            max_batch_size=max_batch_size,
                            policy_value=policy_value,
                        )

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

                        event_rows.extend(
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
    event_df = pd.DataFrame(event_rows)

    summary_path = f"{raw_dir}/summary.csv"
    request_path = f"{raw_dir}/requests.csv"
    event_path = f"{raw_dir}/events.csv"

    summary_df.to_csv(summary_path, index=False)
    request_df.to_csv(request_path, index=False)
    event_df.to_csv(event_path, index=False)

    print(summary_df.head())
    print(f"Saved summary to {summary_path}")
    print(f"Saved request-level results to {request_path}")
    print(f"Saved event-level results to {event_path}")


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
