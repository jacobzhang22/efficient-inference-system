import torch

from src.cache.paged_kv import BatchedPagedKVCache, PagedKVCacheState
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
    kv_caches: list[PagedKVCacheState | BatchedPagedKVCache | None] | None,
    batch_size: int,
) -> list[list[PagedKVCacheState | None]] | None:
    if kv_caches is None:
        return None

    request_caches = [[] for _ in range(batch_size)]
    for layer_cache in kv_caches:
        if isinstance(layer_cache, BatchedPagedKVCache):
            for request_idx, state in enumerate(layer_cache.states):
                request_caches[request_idx].append(state)
        else:
            for request_idx in range(batch_size):
                request_caches[request_idx].append(layer_cache)
    return request_caches


def _stack_request_caches(
    request_caches: list[list[PagedKVCacheState | None]] | None,
    request_indices: list[int],
) -> list[PagedKVCacheState | BatchedPagedKVCache | None] | None:
    if request_caches is None or not request_indices:
        return None

    num_layers = len(request_caches[request_indices[0]])
    stacked = []
    for layer_idx in range(num_layers):
        layer_entries = [request_caches[request_idx][layer_idx] for request_idx in request_indices]
        sample = next((entry for entry in layer_entries if entry is not None), None)
        if sample is None:
            raise ValueError("Paged request caches must be initialized before stacking.")
        states = [
            entry if isinstance(entry, PagedKVCacheState) else PagedKVCacheState(pool=sample.pool)
            for entry in layer_entries
        ]
        stacked.append(BatchedPagedKVCache(states=states, pool=sample.pool))
    return stacked


def _scatter_request_caches(
    request_caches: list[list[PagedKVCacheState | None]] | None,
    request_indices: list[int],
    kv_caches: list[PagedKVCacheState | BatchedPagedKVCache | None] | None,
) -> None:
    if request_caches is None or kv_caches is None:
        return

    for layer_idx, layer_cache in enumerate(kv_caches):
        if isinstance(layer_cache, BatchedPagedKVCache):
            for local_idx, request_idx in enumerate(request_indices):
                request_caches[request_idx][layer_idx] = layer_cache.states[local_idx]
        else:
            for request_idx in request_indices:
                request_caches[request_idx][layer_idx] = layer_cache


def _cache_metric_totals(
    request_caches: list[list[PagedKVCacheState | None]] | None,
    model=None,
) -> tuple[int, int, int]:
    if request_caches is None:
        return 0, 0, 0

    live = 0
    for caches in request_caches:
        for cache in caches:
            if isinstance(cache, PagedKVCacheState):
                live += cache.live_bytes()

    pool_stats = model.paged_memory_stats() if model is not None and hasattr(model, "paged_memory_stats") else {}
    reserved = pool_stats.get("reserved_kv_bytes", live)
    fragmentation = max(reserved - live, 0)
    return live, reserved, fragmentation


def _gpu_bytes(device: torch.device) -> tuple[int, int]:
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated(device)), int(torch.cuda.max_memory_allocated(device))
    return 0, 0


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
    decode_tokens_per_step: list[int] = []
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
        decode_tokens_per_step.append(len(active_request_indices))
        per_generated_token_times_ms.append(timer.elapsed_ms)
        _scatter_request_caches(request_caches, active_request_indices, active_caches)

        for local_idx, request_idx in enumerate(active_request_indices):
            token = int(next_tokens[local_idx].item())
            generated_token_ids[request_idx].append(token)
            generated_tokens_per_request[request_idx] += 1
            last_tokens_by_request[request_idx] = token

    final_request_caches = _stack_request_caches(request_caches, list(range(batch_size)))
    live_kv_bytes, reserved_kv_bytes, fragmentation_bytes = _cache_metric_totals(request_caches, model=model)
    gpu_allocated_bytes, gpu_peak_allocated_bytes = _gpu_bytes(device)
    cache_bytes = reserved_kv_bytes if reserved_kv_bytes > 0 else live_kv_bytes
    total_time_ms = sum(per_generated_token_times_ms)
    total_generated_tokens = sum(generated_tokens_per_request)
    decode_kernel_tokens = sum(decode_tokens_per_step)

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
        "decode_tokens_per_step": decode_tokens_per_step,
        "per_generated_token_times_ms": per_generated_token_times_ms,
        "total_time_ms": total_time_ms,
        "avg_generated_token_time_ms": (
            total_time_ms / total_generated_tokens if total_generated_tokens > 0 else 0.0
        ),
        "avg_decode_only_token_time_ms": (
            sum(decode_times_ms) / decode_kernel_tokens if decode_kernel_tokens > 0 else 0.0
        ),
        "cache_bytes": cache_bytes,
        "live_kv_bytes": live_kv_bytes,
        "reserved_kv_bytes": reserved_kv_bytes,
        "fragmentation_bytes": fragmentation_bytes,
        "workspace_bytes": 0,
        "gpu_allocated_bytes": gpu_allocated_bytes,
        "gpu_peak_allocated_bytes": gpu_peak_allocated_bytes,
        "kv_caches": final_request_caches,
        "num_generated_tokens": max(generated_tokens_per_request),
        "generated_tokens_per_request": generated_tokens_per_request,
        "decode_kernel_tokens": decode_kernel_tokens,
        "backend_name": getattr(model, "attention_backend", "triton_paged"),
    }
