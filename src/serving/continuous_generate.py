import torch
from torch.nn.utils.rnn import pad_sequence

from src.cache.kv_cache import BatchedKVCache, KVCache
from src.utils.timing import timed_section


def _cache_bytes_per_token(model) -> int:
    param = next(model.parameters())
    bytes_per_element = param.element_size()
    num_layers = len(model.blocks)
    return num_layers * 2 * model.blocks[0].attn.d_model * bytes_per_element


def _stack_layer_caches(requests: list) -> list[KVCache | BatchedKVCache | None] | None:
    if not requests or requests[0].kv_caches is None:
        return None

    num_layers = len(requests[0].kv_caches)
    batched_caches: list[KVCache | BatchedKVCache | None] = []

    for layer_idx in range(num_layers):
        layer_caches = [req.kv_caches[layer_idx] for req in requests]
        if any(cache is None for cache in layer_caches):
            if all(cache is None for cache in layer_caches):
                batched_caches.append(BatchedKVCache([None] * len(requests)))
            else:
                batched_caches.append(BatchedKVCache(layer_caches))
            continue

        batched_caches.append(BatchedKVCache(layer_caches))

    return batched_caches


def _scatter_layer_caches(requests: list, batched_caches: list[KVCache | BatchedKVCache | None] | None) -> None:
    if batched_caches is None:
        return

    for request_idx, req in enumerate(requests):
        req.kv_caches = []
        for cache in batched_caches:
            if cache is None:
                req.kv_caches.append(None)
                continue
            if isinstance(cache, BatchedKVCache):
                req.kv_caches.append(cache.caches[request_idx])
            else:
                req.kv_caches.append(cache)


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
    padding_waste_tokens = len(requests) * max_chunk_len - sum(chunk_lengths)

    return {
        "batch_runtime_ms": timer.elapsed_ms,
        "requests_completed": completed_requests,
        "batch_size": len(requests),
        "tokens_scheduled": sum(chunk_lengths),
        "phase": "prefill",
        "prompt_len": max(req.prompt_len for req in requests),
        "max_new_tokens": max(req.max_new_tokens for req in requests),
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

    return {
        "batch_runtime_ms": timer.elapsed_ms,
        "requests_completed": completed_requests,
        "batch_size": len(requests),
        "tokens_scheduled": len(requests),
        "phase": "decode",
        "prompt_len": max(req.prompt_len for req in requests),
        "max_new_tokens": max(req.max_new_tokens for req in requests),
        "padding_waste_tokens": 0,
        "padding_waste_bytes_est": 0,
        "padding_waste_pct": 0.0,
    }
