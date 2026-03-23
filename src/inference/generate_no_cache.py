import torch

from src.utils.timing import timed_section


@torch.no_grad()
def generate_no_cache(model, prompt_ids: torch.Tensor, max_new_tokens: int):
    device = prompt_ids.device
    generated = prompt_ids.clone()
    per_token_times_ms = []

    for _ in range(max_new_tokens):
        current_input = generated[:, -model.max_seq_len :]

        with timed_section(device=device) as timer:
            logits, _ = model(current_input, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        per_token_times_ms.append(timer.elapsed_ms)
        generated = torch.cat([generated, next_token], dim=1)

    total_time_ms = sum(per_token_times_ms)
    avg_generated_token_time_ms = total_time_ms / len(per_token_times_ms) if per_token_times_ms else 0.0

    return {
        "generated_ids": generated,
        "per_token_times_ms": per_token_times_ms,
        "total_time_ms": total_time_ms,
        "avg_generated_token_time_ms": avg_generated_token_time_ms,
        "num_generated_tokens": len(per_token_times_ms),
    }