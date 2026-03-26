from dataclasses import dataclass, field
import torch


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
    arrival_rates: list[float] = field(default_factory=lambda: [20.0, 28.0, 36.0])
    max_batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    batch_timeouts_ms: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0])
    num_requests: int = 200
    prompt_len: int = 128
    max_new_tokens: int = 32
    repeats: int = 3
    seed: int = 42
    output_dir: str = "results/dynamic_batching"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"