import random
import torch

from src.config import RequestWorkloadProfile
from src.serving.request import InferenceRequest


def _sample_request_shape(rng: random.Random, profile: RequestWorkloadProfile) -> tuple[int, int]:
    prompt_len = rng.randint(profile.prompt_len_min, profile.prompt_len_max)
    max_new_tokens = rng.randint(profile.max_new_tokens_min, profile.max_new_tokens_max)
    return prompt_len, max_new_tokens


def generate_requests(
    num_requests: int,
    arrival_rate_rps: float,
    vocab_size: int,
    prompt_len: int | None = None,
    max_new_tokens: int | None = None,
    workload_profiles: list[RequestWorkloadProfile] | None = None,
    seed: int = 42,
) -> list[InferenceRequest]:
    """
    Generates a simple synthetic request stream.

    Arrival process:
        Exponential inter-arrival times (Poisson arrivals)

    Prompt / decode shape:
        Either a single fixed workload or a weighted workload mixture. Each
        profile samples prompt and decode lengths from a range so the stream
        contains realistic shape variation instead of a few exact request sizes.
    """
    rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    if workload_profiles:
        profiles = workload_profiles
    elif prompt_len is not None and max_new_tokens is not None:
        profiles = [
            RequestWorkloadProfile(
                name="fixed",
                prompt_len_min=prompt_len,
                prompt_len_max=prompt_len,
                max_new_tokens_min=max_new_tokens,
                max_new_tokens_max=max_new_tokens,
                weight=1.0,
            )
        ]
    else:
        raise ValueError("Either fixed prompt/max_new_tokens or workload_profiles must be provided.")

    weights = [profile.weight for profile in profiles]

    requests: list[InferenceRequest] = []
    current_time_ms = 0.0
    mean_interarrival_ms = 1000.0 / arrival_rate_rps

    for request_id in range(num_requests):
        if request_id > 0:
            interarrival_ms = rng.expovariate(1.0 / mean_interarrival_ms)
            current_time_ms += interarrival_ms

        profile = rng.choices(profiles, weights=weights, k=1)[0]
        prompt_len_i, max_new_tokens_i = _sample_request_shape(rng, profile)

        prompt_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(prompt_len_i,),
            generator=torch_gen,
        )

        requests.append(
            InferenceRequest(
                request_id=request_id,
                arrival_time_ms=current_time_ms,
                prompt_len=prompt_len_i,
                max_new_tokens=max_new_tokens_i,
                prompt_ids=prompt_ids,
                workload_name=profile.name,
            )
        )

    return requests
