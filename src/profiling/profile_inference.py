import torch

from src.config import ModelConfig
from src.inference.generate_no_cache import generate_no_cache
from src.inference.generate_with_cache import generate_with_cache
from src.model.transformer import TinyTransformerLM
from src.utils.device import get_device
from src.utils.seed import set_seed


def profile_once(use_cache: bool = False, prompt_len: int = 128, max_new_tokens: int = 16):
    set_seed(42)
    device = get_device()

    cfg = ModelConfig()
    model = TinyTransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(device)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        if use_cache:
            generate_with_cache(model, prompt, max_new_tokens=max_new_tokens)
        else:
            generate_no_cache(model, prompt, max_new_tokens=max_new_tokens)

    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    print("=== no cache ===")
    profile_once(use_cache=False)

    print("=== with cache ===")
    profile_once(use_cache=True)