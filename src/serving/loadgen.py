import random
import torch

from src.serving.request import InferenceRequest


def generate_requests(
    num_requests: int,
    arrival_rate_rps: float,
    prompt_len: int,
    max_new_tokens: int,
    vocab_size: int,
    seed: int = 42,
) -> list[InferenceRequest]:
    """
    Generates a simple synthetic request stream.

    Arrival process:
        Exponential inter-arrival times (Poisson arrivals)

    Prompt shape:
        Fixed prompt length for now, which keeps the serving experiment focused
        on scheduling rather than padding/attention-mask issues.
    """
    rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    requests: list[InferenceRequest] = []
    current_time_ms = 0.0
    mean_interarrival_ms = 1000.0 / arrival_rate_rps

    for request_id in range(num_requests):
        if request_id > 0:
            interarrival_ms = rng.expovariate(1.0 / mean_interarrival_ms)
            current_time_ms += interarrival_ms

        prompt_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(prompt_len,),
            generator=torch_gen,
        )

        requests.append(
            InferenceRequest(
                request_id=request_id,
                arrival_time_ms=current_time_ms,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                prompt_ids=prompt_ids,
            )
        )

    return requests