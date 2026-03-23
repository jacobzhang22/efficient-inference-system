import torch

from src.utils.timing import timed_section


@torch.no_grad()
def generate_with_cache(model, prompt_ids: torch.Tensor, max_new_tokens: int):
    device = prompt_ids.device
    generated = prompt_ids.clone()

    if max_new_tokens <= 0:
        return {
            "generated_ids": generated,
            "prefill_time_ms": 0.0,
            "decode_times_ms": [],
            "per_generated_token_times_ms": [],
            "total_time_ms": 0.0,
            "avg_generated_token_time_ms": 0.0,
            "avg_decode_only_token_time_ms": 0.0,
            "cache_bytes": 0,
            "kv_caches": None,
            "num_generated_tokens": 0,
        }

    # Prefill: process the whole prompt once and generate the first new token.
    with timed_section(device=device) as prefill_timer:
        logits, kv_caches = model(prompt_ids, use_cache=True, position_offset=0)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    prefill_time_ms = prefill_timer.elapsed_ms
    generated = torch.cat([generated, next_token], dim=1)

    decode_times_ms = []
    per_generated_token_times_ms = [prefill_time_ms]

    # Remaining decode steps: one token at a time using cache.
    for step in range(1, max_new_tokens):
        last_token = generated[:, -1:]
        current_pos = prompt_ids.shape[1] + (step - 1)

        with timed_section(device=device) as timer:
            logits, kv_caches = model(
                last_token,
                kv_caches=kv_caches,
                use_cache=True,
                position_offset=current_pos,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        decode_times_ms.append(timer.elapsed_ms)
        per_generated_token_times_ms.append(timer.elapsed_ms)
        generated = torch.cat([generated, next_token], dim=1)

    cache_bytes = sum(cache.bytes_used() for cache in kv_caches if cache is not None)
    total_time_ms = sum(per_generated_token_times_ms)

    return {
        "generated_ids": generated,
        "prefill_time_ms": prefill_time_ms,
        "decode_times_ms": decode_times_ms,
        "per_generated_token_times_ms": per_generated_token_times_ms,
        "total_time_ms": total_time_ms,
        "avg_generated_token_time_ms": total_time_ms / len(per_generated_token_times_ms),
        "avg_decode_only_token_time_ms": (
            sum(decode_times_ms) / len(decode_times_ms) if decode_times_ms else 0.0
        ),
        "cache_bytes": cache_bytes,
        "kv_caches": kv_caches,
        "num_generated_tokens": len(per_generated_token_times_ms),
    }