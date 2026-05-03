import torch
from torch.nn.utils.rnn import pad_sequence

from src.inference.generate_with_cache import generate_with_cache
from src.utils.metrics import mean


def _cache_bytes_per_token(model) -> int:
    param = next(model.parameters())
    bytes_per_element = param.element_size()
    num_layers = len(model.blocks)
    return num_layers * 2 * model.blocks[0].attn.d_model * bytes_per_element


def run_batch_generate(
    model,
    requests: list,
    device: str,
) -> dict:
    """
    Runs one full batch to completion using the existing KV-cache generation path.

    Whole-request execution remains FIFO: requests in the same batch are padded
    to a common prompt length but still run to completion as one non-preemptive
    batch.
    """
    if not requests:
        return {
            "batch_runtime_ms": 0.0,
            "batch_size": 0,
            "prompt_len": 0,
            "max_new_tokens": 0,
            "tokens_generated_total": 0,
            "avg_request_generated_token_ms": 0.0,
        }

    prompt_lengths = torch.tensor([req.prompt_len for req in requests], dtype=torch.long, device=device)
    decode_limits = [req.max_new_tokens for req in requests]
    prompt_batch = pad_sequence(
        [req.prompt_ids.to(device) for req in requests],
        batch_first=True,
        padding_value=0,
    )

    result = generate_with_cache(
        model,
        prompt_batch,
        max_new_tokens=decode_limits,
        prompt_lengths=prompt_lengths,
    )
    max_prompt_len = int(prompt_lengths.max().item())
    wasted_prompt_tokens = len(requests) * max_prompt_len - int(prompt_lengths.sum().item())
    padding_waste_bytes_est = wasted_prompt_tokens * _cache_bytes_per_token(model)

    batch_result = {
        "batch_runtime_ms": result["total_time_ms"],
        "batch_size": len(requests),
        "prompt_len": max_prompt_len,
        "max_new_tokens": max(decode_limits),
        "tokens_generated_total": sum(result["generated_tokens_per_request"]),
        "prefill_tokens": int(prompt_lengths.sum().item()),
        "decode_tokens": sum(result["generated_tokens_per_request"]),
        "decode_kernel_tokens": result.get("decode_kernel_tokens", 0),
        "avg_request_generated_token_ms": mean(result["per_generated_token_times_ms"]),
        "first_token_time_ms": result["prefill_time_ms"],
        "prefill_runtime_ms": result["prefill_time_ms"],
        "decode_runtime_ms": sum(result["decode_times_ms"]),
        "decode_ms_per_token": result["avg_decode_only_token_time_ms"],
        "live_kv_bytes": result.get("live_kv_bytes", 0),
        "reserved_kv_bytes": result.get("reserved_kv_bytes", result.get("cache_bytes", 0)),
        "fragmentation_bytes": result.get("fragmentation_bytes", 0),
        "workspace_bytes": result.get("workspace_bytes", 0),
        "gpu_allocated_bytes": result.get("gpu_allocated_bytes", 0),
        "gpu_peak_allocated_bytes": result.get("gpu_peak_allocated_bytes", 0),
        "backend_name": result.get("backend_name", getattr(model, "attention_backend", "triton_paged")),
        "padding_waste_tokens": wasted_prompt_tokens,
        "padding_waste_bytes_est": padding_waste_bytes_est,
        "padding_waste_pct": (
            100.0 * wasted_prompt_tokens / (len(requests) * max_prompt_len)
            if max_prompt_len > 0
            else 0.0
        ),
    }
    if hasattr(model, "release_kv_caches"):
        model.release_kv_caches(result.get("kv_caches"))
    return batch_result
