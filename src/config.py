from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    vocab_size: int = 5000
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 512
    dropout: float = 0.0


@dataclass
class ExperimentConfig:
    prompt_lengths: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    max_new_tokens: int = 32
    repeats: int = 5
    seed: int = 42
    batch_size: int = 1
    output_dir: str = "results/kv_cache_analysis"
    warmup_runs: int = 1

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"