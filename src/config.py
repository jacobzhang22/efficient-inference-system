from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class RequestWorkloadProfile:
    name: str
    prompt_len_min: int
    prompt_len_max: int
    max_new_tokens_min: int
    max_new_tokens_max: int
    weight: float

    @property
    def prompt_len(self) -> int:
        return self.prompt_len_min

    @property
    def max_new_tokens(self) -> int:
        return self.max_new_tokens_min


def default_serving_request_mix() -> list[RequestWorkloadProfile]:
    """
    A simple but realistic chat-serving mixture.

    The distribution intentionally mixes short Q&A turns, medium chat turns,
    and a smaller number of longer retrieval/summarization-style requests.
    """
    return [
        RequestWorkloadProfile(
            name="short_qa",
            prompt_len_min=48,
            prompt_len_max=160,
            max_new_tokens_min=16,
            max_new_tokens_max=48,
            weight=0.35,
        ),
        RequestWorkloadProfile(
            name="chat_turn",
            prompt_len_min=128,
            prompt_len_max=320,
            max_new_tokens_min=32,
            max_new_tokens_max=96,
            weight=0.35,
        ),
        RequestWorkloadProfile(
            name="rag_answer",
            prompt_len_min=256,
            prompt_len_max=640,
            max_new_tokens_min=64,
            max_new_tokens_max=160,
            weight=0.20,
        ),
        RequestWorkloadProfile(
            name="long_summary",
            prompt_len_min=512,
            prompt_len_max=768,
            max_new_tokens_min=96,
            max_new_tokens_max=256,
            weight=0.10,
        ),
    ]


@dataclass
class ModelConfig:
    vocab_size: int = 5000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0


@dataclass
class ExperimentConfig:
    prompt_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 768])
    max_new_tokens: int = 128
    repeats: int = 3
    seed: int = 42
    batch_size: int = 1
    output_dir: str = "results/kv_cache_analysis"
    warmup_runs: int = 1

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SchedulingExperimentConfig:
    arrival_rates: list[float] = field(default_factory=lambda: [4.0, 8.0, 16.0, 24.0, 32.0, 36.0, 44.0, 52.0])
    max_batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    scheduler_modes: list[str] = field(default_factory=lambda: ["dynamic", "static", "continuous"])
    batch_timeouts_ms: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0])
    prefill_chunk_sizes: list[int] = field(default_factory=lambda: [128, 256])
    max_tokens_per_iteration: int = 1536
    num_requests: int = 200
    prompt_len: int = 256
    max_new_tokens: int = 16
    heterogeneous_requests: bool = True
    request_workload_profiles: list[RequestWorkloadProfile] = field(default_factory=list)
    repeats: int = 3
    seed: int = 42
    output_dir: str = "results/dynamic_batching"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_request_workload_profiles(self) -> list[RequestWorkloadProfile]:
        if self.heterogeneous_requests:
            return (
                self.request_workload_profiles
                if self.request_workload_profiles
                else default_serving_request_mix()
            )

        return [
            RequestWorkloadProfile(
                name="fixed",
                prompt_len_min=self.prompt_len,
                prompt_len_max=self.prompt_len,
                max_new_tokens_min=self.max_new_tokens,
                max_new_tokens_max=self.max_new_tokens,
                weight=1.0,
            )
        ]
