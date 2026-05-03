from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


class LayerBlockPool:
    def __init__(
        self,
        block_size: int,
        initial_blocks: int,
        growth_factor: float,
    ):
        self.block_size = block_size
        self.initial_blocks = initial_blocks
        self.growth_factor = growth_factor
        self.k_blocks: torch.Tensor | None = None
        self.v_blocks: torch.Tensor | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None
        self.dtype: torch.dtype | None = None
        self.device: torch.device | None = None
        self.free_block_ids: list[int] = []
        self.capacity_blocks = 0

    def ensure_initialized(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.k_blocks is not None:
            return

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._grow(self.initial_blocks)

    def _grow(self, min_extra_blocks: int) -> None:
        if self.num_heads is None or self.head_dim is None or self.dtype is None or self.device is None:
            raise ValueError("Pool must be initialized before growth.")

        grow_by = max(
            min_extra_blocks,
            self.initial_blocks if self.capacity_blocks == 0 else int(math.ceil(self.capacity_blocks * (self.growth_factor - 1.0))),
        )
        grow_by = max(grow_by, 1)

        new_k = torch.zeros(
            grow_by,
            self.num_heads,
            self.block_size,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        new_v = torch.zeros_like(new_k)

        if self.k_blocks is None:
            self.k_blocks = new_k
            self.v_blocks = new_v
        else:
            self.k_blocks = torch.cat([self.k_blocks, new_k], dim=0)
            self.v_blocks = torch.cat([self.v_blocks, new_v], dim=0)

        start = self.capacity_blocks
        self.capacity_blocks += grow_by
        self.free_block_ids.extend(range(start, self.capacity_blocks))

    def allocate_block(self) -> int:
        if not self.free_block_ids:
            self._grow(1)
        return self.free_block_ids.pop()

    def release_block(self, block_id: int) -> None:
        if self.k_blocks is None or self.v_blocks is None:
            return
        self.k_blocks[block_id].zero_()
        self.v_blocks[block_id].zero_()
        self.free_block_ids.append(block_id)

    def bytes_per_block(self) -> int:
        if self.k_blocks is None:
            return 0
        return 2 * self.k_blocks[0].numel() * self.k_blocks.element_size()

    def reserved_bytes(self) -> int:
        return self.capacity_blocks * self.bytes_per_block()


@dataclass
class PagedKVCacheState:
    pool: LayerBlockPool
    block_ids: list[int] = field(default_factory=list)
    seq_len: int = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor, length: int) -> None:
        if length <= 0:
            return

        _, num_heads, _, head_dim = k_new.shape
        self.pool.ensure_initialized(num_heads, head_dim, k_new.dtype, k_new.device)

        offset = 0
        while offset < length:
            block_offset = self.seq_len % self.pool.block_size
            if block_offset == 0:
                self.block_ids.append(self.pool.allocate_block())

            block_id = self.block_ids[-1]
            take = min(self.pool.block_size - block_offset, length - offset)
            self.pool.k_blocks[block_id, :, block_offset:block_offset + take, :] = k_new[0, :, offset:offset + take, :]
            self.pool.v_blocks[block_id, :, block_offset:block_offset + take, :] = v_new[0, :, offset:offset + take, :]
            self.seq_len += take
            offset += take

    def release(self) -> None:
        for block_id in self.block_ids:
            self.pool.release_block(block_id)
        self.block_ids.clear()
        self.seq_len = 0

    def live_bytes(self) -> int:
        if self.pool.k_blocks is None:
            return 0
        return self.seq_len * self.pool.k_blocks.shape[1] * self.pool.k_blocks.shape[3] * 2 * self.pool.k_blocks.element_size()

    def reserved_bytes(self) -> int:
        return len(self.block_ids) * self.pool.bytes_per_block()

    def fragmentation_bytes(self) -> int:
        return self.reserved_bytes() - self.live_bytes()


class BatchedPagedKVCache:
    def __init__(self, states: list[PagedKVCacheState], pool: LayerBlockPool | None = None):
        self.states = states
        if pool is not None:
            self.pool = pool
        else:
            non_null = next((state for state in states if state is not None), None)
            if non_null is None:
                raise ValueError("BatchedPagedKVCache needs a pool when all states are uninitialized.")
            self.pool = non_null.pool

    def append_batch(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        current_lengths: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_lengths is None:
            lengths = [k_new.shape[2]] * len(self.states)
        elif isinstance(current_lengths, int):
            lengths = [current_lengths] * len(self.states)
        else:
            lengths = [int(length.item()) for length in current_lengths]

        updated = []
        for batch_idx, state in enumerate(self.states):
            state.append(
                k_new[batch_idx : batch_idx + 1],
                v_new[batch_idx : batch_idx + 1],
                max(min(lengths[batch_idx], k_new.shape[2]), 0),
            )
            updated.append(state.seq_len)
        return torch.tensor(updated, dtype=torch.long, device=k_new.device)

    def page_table_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        max_pages = max((len(state.block_ids) for state in self.states), default=0)
        device = device or self.pool.device
        if max_pages == 0:
            return torch.empty((len(self.states), 0), dtype=torch.int32, device=device)

        page_table = torch.full(
            (len(self.states), max_pages),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for row_idx, state in enumerate(self.states):
            if state.block_ids:
                page_table[row_idx, : len(state.block_ids)] = torch.tensor(
                    state.block_ids,
                    dtype=torch.int32,
                    device=device,
                )
        return page_table

    def seq_lens_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        device = device or self.pool.device
        return torch.tensor([state.seq_len for state in self.states], dtype=torch.int32, device=device)

    def live_bytes(self) -> int:
        return sum(state.live_bytes() for state in self.states)

    def reserved_bytes(self) -> int:
        return sum(state.reserved_bytes() for state in self.states)

    def fragmentation_bytes(self) -> int:
        return sum(state.fragmentation_bytes() for state in self.states)

