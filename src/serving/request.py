from dataclasses import dataclass
import torch


@dataclass
class InferenceRequest:
    request_id: int
    arrival_time_ms: float
    prompt_len: int
    max_new_tokens: int
    prompt_ids: torch.Tensor

    start_time_ms: float | None = None
    finish_time_ms: float | None = None
    batch_id: int | None = None

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