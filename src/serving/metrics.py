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
    run_idx: int,
) -> dict:
    latencies = [r.latency_ms for r in completed_requests]
    waits = [r.wait_time_ms for r in completed_requests]
    services = [r.service_time_ms for r in completed_requests]

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

    return {
        "run_idx": run_idx,
        "arrival_rate_rps": arrival_rate_rps,
        "max_batch_size": max_batch_size,
        "batch_timeout_ms": batch_timeout_ms,
        "num_requests": num_completed,
        "num_batches": len(batch_records),
        "throughput_rps": throughput_rps,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "mean_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "mean_wait_ms": mean(waits),
        "p95_wait_ms": percentile(waits, 95),
        "mean_service_ms": mean(services),
        "mean_batch_size": mean(realized_batch_sizes),
        "max_batch_size_realized": max(realized_batch_sizes) if realized_batch_sizes else 0,
        "mean_batch_runtime_ms": mean(batch_runtimes),
    }


def requests_to_rows(completed_requests: list, arrival_rate_rps: float, max_batch_size: int, batch_timeout_ms: float, run_idx: int) -> list[dict]:
    rows = []
    for r in completed_requests:
        rows.append(
            {
                "run_idx": run_idx,
                "arrival_rate_rps": arrival_rate_rps,
                "max_batch_size": max_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                "request_id": r.request_id,
                "batch_id": r.batch_id,
                "prompt_len": r.prompt_len,
                "max_new_tokens": r.max_new_tokens,
                "arrival_time_ms": r.arrival_time_ms,
                "start_time_ms": r.start_time_ms,
                "finish_time_ms": r.finish_time_ms,
                "wait_time_ms": r.wait_time_ms,
                "service_time_ms": r.service_time_ms,
                "latency_ms": r.latency_ms,
            }
        )
    return rows


def batches_to_rows(batch_records: list[dict], arrival_rate_rps: float, max_batch_size: int, batch_timeout_ms: float, run_idx: int) -> list[dict]:
    rows = []
    for record in batch_records:
        rows.append(
            {
                "run_idx": run_idx,
                "arrival_rate_rps": arrival_rate_rps,
                "max_batch_size": max_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                **record,
            }
        )
    return rows