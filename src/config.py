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