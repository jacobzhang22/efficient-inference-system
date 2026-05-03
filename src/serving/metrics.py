from src.utils.metrics import mean


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_run(
    completed_requests: list,
    batch_records: list[dict],
    arrival_rate_rps: float,
    max_batch_size: int,
    batch_timeout_ms: float,
    scheduler_mode: str,
    scheduling_policy_value: float,
    run_idx: int,
) -> dict:
    latencies = [r.latency_ms for r in completed_requests]
    waits = [r.wait_time_ms for r in completed_requests]
    services = [r.service_time_ms for r in completed_requests]
    first_token_latencies = [r.first_token_latency_ms for r in completed_requests if r.first_token_time_ms is not None]
    prompt_lengths = [r.prompt_len for r in completed_requests]
    decode_lengths = [r.max_new_tokens for r in completed_requests]

    if completed_requests:
        makespan_ms = max(r.finish_time_ms for r in completed_requests) - min(
            r.arrival_time_ms for r in completed_requests
        )
    else:
        makespan_ms = 0.0

    total_generated_tokens = sum(r.max_new_tokens for r in completed_requests)
    num_completed = len(completed_requests)

    throughput_rps = (num_completed / (makespan_ms / 1000.0)) if makespan_ms > 0 else 0.0
    throughput_tokens_per_s = (
        total_generated_tokens / (makespan_ms / 1000.0) if makespan_ms > 0 else 0.0
    )

    realized_batch_sizes = [b["batch_size"] for b in batch_records]
    batch_runtimes = [b["batch_runtime_ms"] for b in batch_records]
    tokens_scheduled = [b.get("tokens_scheduled", 0) for b in batch_records]
    active_requests = [b.get("active_requests", 0) for b in batch_records]
    padding_waste_tokens = [b.get("padding_waste_tokens", 0) for b in batch_records]
    padding_waste_bytes = [b.get("padding_waste_bytes_est", 0) for b in batch_records]
    padding_waste_pcts = [b.get("padding_waste_pct", 0.0) for b in batch_records]
    tokens_per_s = [b.get("tokens_per_s", 0.0) for b in batch_records]
    live_kv_bytes = [b.get("live_kv_bytes", 0) for b in batch_records]
    reserved_kv_bytes = [b.get("reserved_kv_bytes", 0) for b in batch_records]
    fragmentation_bytes = [b.get("fragmentation_bytes", 0) for b in batch_records]
    workspace_bytes = [b.get("workspace_bytes", 0) for b in batch_records]
    gpu_allocated_bytes = [b.get("gpu_allocated_bytes", 0) for b in batch_records]
    gpu_peak_allocated_bytes = [b.get("gpu_peak_allocated_bytes", 0) for b in batch_records]
    prefill_tokens_total = sum(b.get("prefill_tokens", 0) for b in batch_records)
    decode_tokens_total = sum(b.get("decode_tokens", 0) for b in batch_records)
    decode_kernel_tokens_total = sum(b.get("decode_kernel_tokens", 0) for b in batch_records)
    prefill_runtime_total = sum(b.get("prefill_runtime_ms", 0.0) for b in batch_records)
    decode_runtime_total = sum(b.get("decode_runtime_ms", 0.0) for b in batch_records)
    total_runtime_ms = sum(batch_runtimes)

    return {
        "run_idx": run_idx,
        "scheduler_mode": scheduler_mode,
        "scheduling_policy_value": scheduling_policy_value,
        "arrival_rate_rps": arrival_rate_rps,
        "max_batch_size": max_batch_size,
        "batch_timeout_ms": batch_timeout_ms,
        "num_requests": num_completed,
        "num_batches": len(batch_records),
        "throughput_rps": throughput_rps,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "mean_event_tokens_per_s": mean(tokens_per_s),
        "mean_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "mean_wait_ms": mean(waits),
        "p95_wait_ms": percentile(waits, 95),
        "mean_service_ms": mean(services),
        "mean_first_token_latency_ms": mean(first_token_latencies),
        "p95_first_token_latency_ms": percentile(first_token_latencies, 95),
        "mean_prompt_len": mean(prompt_lengths),
        "min_prompt_len": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_len": max(prompt_lengths) if prompt_lengths else 0,
        "mean_max_new_tokens": mean(decode_lengths),
        "min_max_new_tokens": min(decode_lengths) if decode_lengths else 0,
        "max_max_new_tokens": max(decode_lengths) if decode_lengths else 0,
        "mean_batch_size": mean(realized_batch_sizes),
        "max_batch_size_realized": max(realized_batch_sizes) if realized_batch_sizes else 0,
        "mean_batch_runtime_ms": mean(batch_runtimes),
        "mean_tokens_scheduled": mean(tokens_scheduled),
        "mean_active_requests": mean(active_requests),
        "mean_decode_ms_per_token": (
            decode_runtime_total / decode_kernel_tokens_total
            if decode_kernel_tokens_total > 0
            else 0.0
        ),
        "prefill_runtime_ms_total": prefill_runtime_total,
        "decode_runtime_ms_total": decode_runtime_total,
        "prefill_runtime_share": (prefill_runtime_total / total_runtime_ms) if total_runtime_ms > 0 else 0.0,
        "decode_runtime_share": (decode_runtime_total / total_runtime_ms) if total_runtime_ms > 0 else 0.0,
        "mean_padding_waste_tokens": mean(padding_waste_tokens),
        "total_padding_waste_tokens": sum(padding_waste_tokens),
        "mean_padding_waste_bytes_est": mean(padding_waste_bytes),
        "total_padding_waste_bytes_est": sum(padding_waste_bytes),
        "mean_padding_waste_pct": mean(padding_waste_pcts),
        "mean_live_kv_bytes": mean(live_kv_bytes),
        "max_live_kv_bytes": max(live_kv_bytes) if live_kv_bytes else 0,
        "mean_reserved_kv_bytes": mean(reserved_kv_bytes),
        "max_reserved_kv_bytes": max(reserved_kv_bytes) if reserved_kv_bytes else 0,
        "mean_fragmentation_bytes": mean(fragmentation_bytes),
        "max_fragmentation_bytes": max(fragmentation_bytes) if fragmentation_bytes else 0,
        "mean_workspace_bytes": mean(workspace_bytes),
        "max_workspace_bytes": max(workspace_bytes) if workspace_bytes else 0,
        "mean_gpu_allocated_bytes": mean(gpu_allocated_bytes),
        "max_gpu_allocated_bytes": max(gpu_allocated_bytes) if gpu_allocated_bytes else 0,
        "max_gpu_peak_allocated_bytes": max(gpu_peak_allocated_bytes) if gpu_peak_allocated_bytes else 0,
        "prefill_tokens_total": prefill_tokens_total,
        "decode_tokens_total": decode_tokens_total,
        "decode_kernel_tokens_total": decode_kernel_tokens_total,
        "backend_name": batch_records[0].get("backend_name", "triton_paged") if batch_records else "triton_paged",
    }


def requests_to_rows(
    completed_requests: list,
    arrival_rate_rps: float,
    max_batch_size: int,
    batch_timeout_ms: float,
    scheduler_mode: str,
    scheduling_policy_value: float,
    run_idx: int,
) -> list[dict]:
    rows = []
    for r in completed_requests:
        rows.append(
            {
                "run_idx": run_idx,
                "scheduler_mode": scheduler_mode,
                "scheduling_policy_value": scheduling_policy_value,
                "arrival_rate_rps": arrival_rate_rps,
                "max_batch_size": max_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                "request_id": r.request_id,
                "batch_id": r.batch_id,
                "workload_name": r.workload_name,
                "prompt_len": r.prompt_len,
                "max_new_tokens": r.max_new_tokens,
                "arrival_time_ms": r.arrival_time_ms,
                "start_time_ms": r.start_time_ms,
                "finish_time_ms": r.finish_time_ms,
                "first_token_time_ms": r.first_token_time_ms,
                "first_token_latency_ms": r.first_token_latency_ms,
                "wait_time_ms": r.wait_time_ms,
                "service_time_ms": r.service_time_ms,
                "latency_ms": r.latency_ms,
            }
        )
    return rows


def batches_to_rows(
    batch_records: list[dict],
    arrival_rate_rps: float,
    max_batch_size: int,
    batch_timeout_ms: float,
    scheduler_mode: str,
    scheduling_policy_value: float,
    run_idx: int,
) -> list[dict]:
    rows = []
    for record in batch_records:
        rows.append(
            {
                "run_idx": run_idx,
                "scheduler_mode": scheduler_mode,
                "scheduling_policy_value": scheduling_policy_value,
                "arrival_rate_rps": arrival_rate_rps,
                "max_batch_size": max_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                **record,
            }
        )
    return rows
