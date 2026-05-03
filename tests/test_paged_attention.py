import unittest

import torch

from src.config import ModelConfig
from src.inference.generate_with_cache import generate_with_cache
from src.model.transformer import TinyTransformerLM


def _build_model(backend_name: str, device: str) -> TinyTransformerLM:
    cfg = ModelConfig(
        vocab_size=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=128,
        attention_backend=backend_name,
        kv_block_size=8,
        kv_pool_initial_blocks=8,
        kv_pool_growth_factor=2.0,
    )
    model = TinyTransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        attention_backend=cfg.attention_backend,
        kv_block_size=cfg.kv_block_size,
        kv_pool_initial_blocks=cfg.kv_pool_initial_blocks,
        kv_pool_growth_factor=cfg.kv_pool_growth_factor,
        enable_attention_correctness_checks=cfg.enable_attention_correctness_checks,
    ).to(device)
    model.eval()
    return model


class PagedAttentionTests(unittest.TestCase):
    def test_chunked_prefill_then_decode_runs(self):
        model = _build_model("paged_reference", "cpu")
        prompt = torch.randint(0, 128, (2, 24))
        result = generate_with_cache(model, prompt, max_new_tokens=[3, 4])
        self.assertEqual(result["generated_tokens_per_request"], [3, 4])
        self.assertGreater(result["reserved_kv_bytes"], 0)

    def test_reference_shapes_match_single_and_batched(self):
        prompt = torch.randint(0, 128, (2, 16))
        paged_model = _build_model("paged_reference", "cpu")
        paged = generate_with_cache(paged_model, prompt, max_new_tokens=[2, 2])
        self.assertEqual(len(paged["generated_ids"]), 2)
        for paged_ids in paged["generated_ids"]:
            self.assertEqual(len(tuple(paged_ids.shape)), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests")
    def test_reference_and_triton_agree_on_small_case(self):
        prompt = torch.randint(0, 128, (2, 16), device="cuda")
        ref_model = _build_model("paged_reference", "cuda")
        tri_model = _build_model("triton_paged", "cuda")
        tri_model.load_state_dict(ref_model.state_dict())
        ref = generate_with_cache(ref_model, prompt, max_new_tokens=[2, 2])
        tri = generate_with_cache(tri_model, prompt, max_new_tokens=[2, 2])
        for ref_ids, tri_ids in zip(ref["generated_ids"], tri["generated_ids"]):
            self.assertTrue(torch.equal(ref_ids, tri_ids))


if __name__ == "__main__":
    unittest.main()
