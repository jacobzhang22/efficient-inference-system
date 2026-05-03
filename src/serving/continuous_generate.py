import torch
from torch.nn.utils.rnn import pad_sequence

from src.cache.paged_kv import BatchedPagedKVCache, PagedKVCacheState
from src.utils.timing import timed_section


def _cache_bytes_per_token(model) -> int:
    param = next(model.parameters())
    bytes_per_element = param.element_size()
    num_layers = len(model.blocks)
    return num_layers * 2 * model.blocks[0].attn.d_model * bytes_per_element


def _stack_layer_caches(requests: list) -> list[BatchedPagedKVCache | PagedKVCacheState | None] | None:
    if not requests or requests[0].kv_caches is None:
        return None

    num_layers = len(requests[0].kv_caches)
    batched_caches: list[BatchedPagedKVCache | PagedKVCacheState | None] = []

    for layer_idx in range(num_layers):
        layer_caches = [req.kv_caches[layer_idx] for req in requests]
        sample = next((cache for cache in layer_caches if cache is not None), None)
        if sample is None:
            raise ValueError("Paged request caches must be initialized before batching.")
        states = [
            cache if isinstance(cache, PagedKVCacheState) else PagedKVCacheState(pool=sample.pool)
            for cache in layer_caches
        ]
        batched_caches.append(BatchedPagedKVCache(states=states, pool=sample.pool))

    return batched_caches


def _scatter_layer_caches(requests: list, batched_caches: list[BatchedPagedKVCache | PagedKVCacheState | None] | None) -> None:
    if batched_caches is None:
        return

    for request_idx, req in enumerate(requests):
        req.kv_caches = []
        for cache in batched_caches:
            if cache is None:
                req.kv_caches.append(None)
                continue
            if isinstance(cache, BatchedPagedKVCache):
                req.kv_caches.append(cache.states[request_idx])
            else:
                req.kv_caches.append(cache)


def _ensure_request_kv_caches(model, requests: list) -> None:
    missing = [idx for idx, req in enumerate(requests) if req.kv_caches is None]
    if not missing:
        return

    created = model.create_request_kv_caches(len(missing))
    for local_idx, request_idx in enumerate(missing):
        requests[request_idx].kv_caches = created[local_idx]


def _cache_totals(model, requests: list) -> tuple[int, int, int]:
    live = 0
    for req in requests:
        if req.kv_caches is None:
            continue
        for cache in req.kv_caches:
            if isinstance(cache, PagedKVCacheState):
                live += cache.live_bytes()
    pool_stats = model.paged_memory_stats() if hasattr(model, "paged_memory_stats") else {}
    reserved = pool_stats.get("reserved_kv_bytes", live)
    fragmentation = max(reserved - live, 0)
    return live, reserved, fragmentation


def _gpu_bytes(device: str) -> tuple[int, int]:
    if device == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())
    return 0, 0


@torch.no_grad()
def run_prefill_chunk(
    model,
    requests: list,
    chunk_size: int,
    device: str,
    start_time_ms: float,
    event_id: int,
) -> dict:
    if not requests:
        return {"batch_runtime_ms": 0.0, "requests_completed": []}

    chunk_start = requests[0].prompt_tokens_processed
    if any(req.prompt_tokens_processed != chunk_start for req in requests):
        raise ValueError("run_prefill_chunk expects requests with the same prompt progress.")

    chunk_lengths = [
        max(min(chunk_size, req.prompt_len - chunk_start), 0)
        for req in requests
    ]
    max_chunk_len = max(chunk_lengths)
    if max_chunk_len <= 0:
        return {"batch_runtime_ms": 0.0, "requests_completed": []}

    prompt_batch = pad_sequence(
        [
            req.prompt_ids[chunk_start:chunk_start + chunk_len].to(device)
            for req, chunk_len in zip(requests, chunk_lengths)
        ],
        batch_first=True,
        padding_value=0,
    )
    current_lengths = torch.tensor(chunk_lengths, device=device, dtype=torch.long)

    _ensure_request_kv_caches(model, requests)
    batched_caches = _stack_layer_caches(requests)

    with timed_section(device=device) as timer:
        logits, new_caches = model(
            prompt_batch,
            kv_caches=batched_caches,
            use_cache=True,
            position_offset=chunk_start,
            current_lengths=current_lengths,
        )

    finish_time_ms = start_time_ms + timer.elapsed_ms
    completed_requests = []
    first_tokens_emitted = 0
    last_token_logits = logits[
        torch.arange(len(requests), device=device),
        current_lengths - 1,
        :,
    ]

    for req_idx, req in enumerate(requests):
        if req.start_time_ms is None:
            req.start_time_ms = start_time_ms
            req.batch_id = event_id

        req.scheduler_mode = "continuous"
        req.prompt_tokens_processed = chunk_start + chunk_lengths[req_idx]

        if req.prompt_tokens_processed >= req.prompt_len:
            next_token = int(torch.argmax(last_token_logits[req_idx]).item())
            req.generated_token_ids.append(next_token)
            first_tokens_emitted += 1
            if req.first_token_time_ms is None:
                req.first_token_time_ms = finish_time_ms

            if req.num_generated_tokens >= req.max_new_tokens:
                req.phase = "finished"
                req.finish_time_ms = finish_time_ms
                completed_requests.append(req)
            else:
                req.phase = "decode"
        else:
            req.phase = "prefill"

    _scatter_layer_caches(requests, new_caches)
    live_kv_bytes, reserved_kv_bytes, fragmentation_bytes = _cache_totals(model, requests)
    gpu_allocated_bytes, gpu_peak_allocated_bytes = _gpu_bytes(device)
    padding_waste_tokens = len(requests) * max_chunk_len - sum(chunk_lengths)

    return {
        "batch_runtime_ms": timer.elapsed_ms,
        "requests_completed": completed_requests,
        "batch_size": len(requests),
        "tokens_scheduled": sum(chunk_lengths),
        "phase": "prefill",
        "prompt_len": max(req.prompt_len for req in requests),
        "max_new_tokens": max(req.max_new_tokens for req in requests),
        "prefill_tokens": sum(chunk_lengths),
        "decode_tokens": first_tokens_emitted,
        "decode_kernel_tokens": 0,
        "prefill_runtime_ms": timer.elapsed_ms,
        "decode_runtime_ms": 0.0,
        "decode_ms_per_token": 0.0,
        "live_kv_bytes": live_kv_bytes,
        "reserved_kv_bytes": reserved_kv_bytes,
        "fragmentation_bytes": fragmentation_bytes,
        "workspace_bytes": 0,
        "gpu_allocated_bytes": gpu_allocated_bytes,
        "gpu_peak_allocated_bytes": gpu_peak_allocated_bytes,
        "backend_name": getattr(model, "attention_backend", "triton_paged"),
        "padding_waste_tokens": padding_waste_tokens,
        "padding_waste_bytes_est": padding_waste_tokens * _cache_bytes_per_token(model),
        "padding_waste_pct": (
            100.0 * padding_waste_tokens / (len(requests) * max_chunk_len)
            if max_chunk_len > 0
            else 0.0
        ),
    }


@torch.no_grad()
def run_decode_step(
    model,
    requests: list,
    device: str,
    start_time_ms: float,
    event_id: int,
) -> dict:
    if not requests:
        return {"batch_runtime_ms": 0.0, "requests_completed": []}

    decode_input = torch.tensor(
        [[req.generated_token_ids[-1]] for req in requests],
        device=device,
        dtype=torch.long,
    )
    position_offset = torch.tensor(
        [req.prompt_len + req.num_generated_tokens - 1 for req in requests],
        device=device,
        dtype=torch.long,
    )
    _ensure_request_kv_caches(model, requests)
    batched_caches = _stack_layer_caches(requests)

    with timed_section(device=device) as timer:
        logits, new_caches = model(
            decode_input,
            kv_caches=batched_caches,
            use_cache=True,
            position_offset=position_offset,
        )

    finish_time_ms = start_time_ms + timer.elapsed_ms
    completed_requests = []

    for req_idx, req in enumerate(requests):
        if req.start_time_ms is None:
            req.start_time_ms = start_time_ms
            req.batch_id = event_id

        req.scheduler_mode = "continuous"
        next_token = int(torch.argmax(logits[req_idx, -1, :]).item())
        req.generated_token_ids.append(next_token)

        if req.num_generated_tokens >= req.max_new_tokens:
            req.phase = "finished"
            req.finish_time_ms = finish_time_ms
            completed_requests.append(req)
        else:
            req.phase = "decode"

    _scatter_layer_caches(requests, new_caches)
    live_kv_bytes, reserved_kv_bytes, fragmentation_bytes = _cache_totals(model, requests)
    gpu_allocated_bytes, gpu_peak_allocated_bytes = _gpu_bytes(device)

    return {
        "batch_runtime_ms": timer.elapsed_ms,
        "requests_completed": completed_requests,
        "batch_size": len(requests),
        "tokens_scheduled": len(requests),
        "phase": "decode",
        "prompt_len": max(req.prompt_len for req in requests),
        "max_new_tokens": max(req.max_new_tokens for req in requests),
        "prefill_tokens": 0,
        "decode_tokens": len(requests),
        "decode_kernel_tokens": len(requests),
        "prefill_runtime_ms": 0.0,
        "decode_runtime_ms": timer.elapsed_ms,
        "decode_ms_per_token": timer.elapsed_ms / max(len(requests), 1),
        "live_kv_bytes": live_kv_bytes,
        "reserved_kv_bytes": reserved_kv_bytes,
        "fragmentation_bytes": fragmentation_bytes,
        "workspace_bytes": 0,
        "gpu_allocated_bytes": gpu_allocated_bytes,
        "gpu_peak_allocated_bytes": gpu_peak_allocated_bytes,
        "backend_name": getattr(model, "attention_backend", "triton_paged"),
        "padding_waste_tokens": 0,
        "padding_waste_bytes_est": 0,
        "padding_waste_pct": 0.0,
    }
