import torch

from src.inference.generate_with_cache import generate_with_cache
from src.utils.metrics import mean


def run_batch_generate(
    model,
    requests: list,
    device: str,
) -> dict:
    """
    Runs one full batch to completion using the existing KV-cache generation path.

    Assumption:
        All requests in the batch have the same prompt length and max_new_tokens.
        That is enough for this week's scheduler study.
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

    prompt_batch = torch.stack([req.prompt_ids for req in requests], dim=0).to(device)
    max_new_tokens = requests[0].max_new_tokens

    result = generate_with_cache(model, prompt_batch, max_new_tokens=max_new_tokens)

    return {
        "batch_runtime_ms": result["total_time_ms"],
        "batch_size": len(requests),
        "prompt_len": requests[0].prompt_len,
        "max_new_tokens": max_new_tokens,
        "tokens_generated_total": len(requests) * result["num_generated_tokens"],
        "avg_request_generated_token_ms": mean(result["per_generated_token_times_ms"]),
    }