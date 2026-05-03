import unittest

import torch

from src.cache.paged_kv import BatchedPagedKVCache, LayerBlockPool, PagedKVCacheState


class PagedKVTests(unittest.TestCase):
    def test_allocator_growth_and_reuse(self):
        pool = LayerBlockPool(block_size=4, initial_blocks=1, growth_factor=2.0)
        state = PagedKVCacheState(pool=pool)
        k = torch.randn(1, 2, 6, 8)
        v = torch.randn(1, 2, 6, 8)
        state.append(k, v, length=6)
        self.assertGreaterEqual(pool.capacity_blocks, 2)
        allocated = list(state.block_ids)
        state.release()
        reused = pool.allocate_block()
        self.assertIn(reused, allocated)

    def test_block_boundary_append(self):
        pool = LayerBlockPool(block_size=4, initial_blocks=2, growth_factor=2.0)
        state = PagedKVCacheState(pool=pool)
        k = torch.randn(1, 2, 5, 8)
        v = torch.randn(1, 2, 5, 8)
        state.append(k, v, length=5)
        self.assertEqual(state.seq_len, 5)
        self.assertEqual(len(state.block_ids), 2)

    def test_batched_page_table_handles_heterogeneous_lengths(self):
        pool = LayerBlockPool(block_size=4, initial_blocks=4, growth_factor=2.0)
        states = [PagedKVCacheState(pool=pool) for _ in range(2)]
        batch = BatchedPagedKVCache(states=states, pool=pool)
        k = torch.randn(2, 2, 4, 8)
        v = torch.randn(2, 2, 4, 8)
        batch.append_batch(k, v, current_lengths=torch.tensor([4, 2]))
        page_table = batch.page_table_tensor(device=k.device)
        seq_lens = batch.seq_lens_tensor(device=k.device)
        self.assertEqual(tuple(page_table.shape), (2, 1))
        self.assertEqual(seq_lens.tolist(), [4, 2])


if __name__ == "__main__":
    unittest.main()
