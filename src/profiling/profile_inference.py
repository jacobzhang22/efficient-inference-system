import torch

from src.config import ModelConfig
from src.inference.generate_with_cache import generate_with_cache
from src.model.transformer import TinyTransformerLM
from src.utils.device import get_device
from src.utils.seed import set_seed


def build_model(device: str, backend_name: str) -> TinyTransformerLM:
    cfg = ModelConfig()
    model = TinyTransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        attention_backend=backend_name,
        kv_block_size=cfg.kv_block_size,
        kv_pool_initial_blocks=cfg.kv_pool_initial_blocks,
        kv_pool_growth_factor=cfg.kv_pool_growth_factor,
        enable_attention_correctness_checks=cfg.enable_attention_correctness_checks,
    ).to(device)
    model.set_attention_backend(backend_name)
    model.eval()
    return model


def profile_once(backend_name: str, prompt_len: int = 128, max_new_tokens: int = 16, batch_size: int = 8):
    set_seed(42)
    device = get_device()
    cfg = ModelConfig()
    model = build_model(device, backend_name)
    prompt = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len), device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        generate_with_cache(model, prompt, max_new_tokens=[max_new_tokens] * batch_size)

    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    for backend_name in ["paged_reference", "triton_paged"]:
        print(f"=== {backend_name} ===")
        profile_once(backend_name=backend_name)
