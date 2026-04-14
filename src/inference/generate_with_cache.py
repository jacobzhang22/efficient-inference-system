import torch

from src.cache.kv_cache import BatchedKVCache, KVCache
from src.utils.timing import timed_section


def _normalize_prompt_lengths(prompt_ids: torch.Tensor, prompt_lengths: torch.Tensor | None) -> torch.Tensor:
    if prompt_lengths is None:
        return torch.full(
            (prompt_ids.shape[0],),
            prompt_ids.shape[1],
            device=prompt_ids.device,
            dtype=torch.long,
        )
    return prompt_lengths.to(device=prompt_ids.device, dtype=torch.long)


def _normalize_decode_limits(
    batch_size: int,
    device: torch.device,
    max_new_tokens: int | list[int] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(max_new_tokens, int):
        return torch.full((batch_size,), max_new_tokens, device=device, dtype=torch.long)
    if isinstance(max_new_tokens, torch.Tensor):
        return max_new_tokens.to(device=device, dtype=torch.long)
    return torch.tensor(max_new_tokens, device=device, dtype=torch.long)


def _extract_request_caches(
    kv_caches: list[KVCache | BatchedKVCache | None] | None,
    batch_size: int,
) -> list[list[KVCache | None]] | None:
    if kv_caches is None:
        return None

    request_caches = [[] for _ in range(batch_size)]
    for layer_cache in kv_caches:
        if isinstance(layer_cache, BatchedKVCache):
            for request_idx, cache in enumerate(layer_cache.caches):
                request_caches[request_idx].append(cache)
        else:
            for request_idx in range(batch_size):
                request_caches[request_idx].append(layer_cache)
    return request_caches


def _stack_request_caches(
    request_caches: list[list[KVCache | None]] | None,
    request_indices: list[int],
) -> list[KVCache | BatchedKVCache | None] | None:
    if request_caches is None or not request_indices:
        return None

    num_layers = len(request_caches[request_indices[0]])
    return [
        BatchedKVCache([request_caches[request_idx][layer_idx] for request_idx in request_indices])
        for layer_idx in range(num_layers)
    ]


def _scatter_request_caches(
    request_caches: list[list[KVCache | None]] | None,
    request_indices: list[int],
    kv_caches: list[KVCache | BatchedKVCache | None] | None,
) -> None:
    if request_caches is None or kv_caches is None:
        return

    for layer_idx, layer_cache in enumerate(kv_caches):
        if isinstance(layer_cache, BatchedKVCache):
            for local_idx, request_idx in enumerate(request_indices):
                request_caches[request_idx][layer_idx] = layer_cache.caches[local_idx]
        else:
            for request_idx in request_indices:
                request_caches[request_idx][layer_idx] = layer_cache


@torch.no_grad()
def generate_with_cache(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int | list[int] | torch.Tensor,
    prompt_lengths: torch.Tensor | None = None,
):
    device = prompt_ids.device
    batch_size = prompt_ids.shape[0]
    prompt_lengths = _normalize_prompt_lengths(prompt_ids, prompt_lengths)
    decode_limits = _normalize_decode_limits(batch_size, device, max_new_tokens)

    if int(decode_limits.max().item()) <= 0:
        return {
            "generated_ids": [
                prompt_ids[idx, : int(prompt_lengths[idx].item())].clone()
                for idx in range(batch_size)
            ],
            "prefill_time_ms": 0.0,
            "decode_times_ms": [],
            "per_generated_token_times_ms": [],
            "total_time_ms": 0.0,
            "avg_generated_token_time_ms": 0.0,
            "avg_decode_only_token_time_ms": 0.0,
            "cache_bytes": 0,
            "kv_caches": None,
            "num_generated_tokens": 0,
            "generated_tokens_per_request": [0] * batch_size,
        }

    with timed_section(device=device) as prefill_timer:
        logits, kv_caches = model(
            prompt_ids,
            use_cache=True,
            position_offset=0,
            current_lengths=prompt_lengths,
        )
        gather_idx = prompt_lengths - 1
        last_prompt_logits = logits[torch.arange(batch_size, device=device), gather_idx, :]
        next_tokens = torch.argmax(last_prompt_logits, dim=-1)

    prefill_time_ms = prefill_timer.elapsed_ms
    decode_times_ms: list[float] = []
    per_generated_token_times_ms = [prefill_time_ms]

    generated_tokens_per_request = [0] * batch_size
    generated_token_ids: list[list[int]] = [[] for _ in range(batch_size)]
    last_tokens_by_request: list[int | None] = [None] * batch_size

    for request_idx in range(batch_size):
        if int(decode_limits[request_idx].item()) > 0:
            token = int(next_tokens[request_idx].item())
            generated_token_ids[request_idx].append(token)
            generated_tokens_per_request[request_idx] = 1
            last_tokens_by_request[request_idx] = token

    request_caches = _extract_request_caches(kv_caches, batch_size)
    max_decode_steps = int(decode_limits.max().item())

    for _ in range(1, max_decode_steps):
        active_request_indices = [
            request_idx
            for request_idx in range(batch_size)
            if generated_tokens_per_request[request_idx] < int(decode_limits[request_idx].item())
        ]
        if not active_request_indices:
            break

        decode_input = torch.tensor(
            [[last_tokens_by_request[request_idx]] for request_idx in active_request_indices],
            device=device,
            dtype=torch.long,
        )
        current_pos = torch.tensor(
            [
                int(prompt_lengths[request_idx].item()) + generated_tokens_per_request[request_idx] - 1
                for request_idx in active_request_indices
            ],
            device=device,
            dtype=torch.long,
        )
        active_caches = _stack_request_caches(request_caches, active_request_indices)

        with timed_section(device=device) as timer:
            logits, active_caches = model(
                decode_input,
                kv_caches=active_caches,
                use_cache=True,
                position_offset=current_pos,
                current_lengths=1,
            )
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

        decode_times_ms.append(timer.elapsed_ms)
        per_generated_token_times_ms.append(timer.elapsed_ms)
        _scatter_request_caches(request_caches, active_request_indices, active_caches)

        for local_idx, request_idx in enumerate(active_request_indices):
            token = int(next_tokens[local_idx].item())
            generated_token_ids[request_idx].append(token)
            generated_tokens_per_request[request_idx] += 1
            last_tokens_by_request[request_idx] = token

    final_request_caches = _stack_request_caches(request_caches, list(range(batch_size)))
    cache_bytes = (
        sum(0 if cache is None else cache.bytes_used() for cache in final_request_caches)
        if final_request_caches is not None
        else 0
    )
    total_time_ms = sum(per_generated_token_times_ms)

    generated_ids = [
        torch.cat(
            [
                prompt_ids[idx, : int(prompt_lengths[idx].item())].clone(),
                torch.tensor(generated_token_ids[idx], device=device, dtype=prompt_ids.dtype),
            ]
        )
        for idx in range(batch_size)
    ]

    return {
        "generated_ids": generated_ids,
        "prefill_time_ms": prefill_time_ms,
        "decode_times_ms": decode_times_ms,
        "per_generated_token_times_ms": per_generated_token_times_ms,
        "total_time_ms": total_time_ms,
        "avg_generated_token_time_ms": total_time_ms / len(per_generated_token_times_ms),
        "avg_decode_only_token_time_ms": (
            sum(decode_times_ms) / len(decode_times_ms) if decode_times_ms else 0.0
        ),
        "cache_bytes": cache_bytes,
        "kv_caches": final_request_caches,
        "num_generated_tokens": max(generated_tokens_per_request),
        "generated_tokens_per_request": generated_tokens_per_request,
    }
