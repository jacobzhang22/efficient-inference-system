from dataclasses import dataclass, field
import torch

from src.cache.kv_cache import KVCache


@dataclass
class InferenceRequest:
    request_id: int
    arrival_time_ms: float
    prompt_len: int
    max_new_tokens: int
    prompt_ids: torch.Tensor
    workload_name: str = "fixed"

    start_time_ms: float | None = None
    finish_time_ms: float | None = None
    batch_id: int | None = None
    scheduler_mode: str | None = None
    phase: str = "prefill"
    prompt_tokens_processed: int = 0
    generated_token_ids: list[int] = field(default_factory=list)
    kv_caches: list[KVCache | None] | None = None
    first_token_time_ms: float | None = None

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_context_len(self) -> int:
        return self.prompt_len + self.num_generated_tokens

    @property
    def prefill_complete(self) -> bool:
        return self.prompt_tokens_processed >= self.prompt_len

    @property
    def first_token_latency_ms(self) -> float:
        if self.first_token_time_ms is None:
            return 0.0
        return self.first_token_time_ms - self.arrival_time_ms

    @property
    def wait_time_ms(self) -> float:
        if self.start_time_ms is None:
            return 0.0
        return self.start_time_ms - self.arrival_time_ms

    @property
    def service_time_ms(self) -> float:
        if self.start_time_ms is None or self.finish_time_ms is None:
            return 0.0
        return self.finish_time_ms - self.start_time_ms

    @property
    def latency_ms(self) -> float:
        if self.finish_time_ms is None:
            return 0.0
        return self.finish_time_ms - self.arrival_time_ms
