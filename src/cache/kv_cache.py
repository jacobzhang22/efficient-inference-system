import torch


class KVCache:
    """
    Block-managed per-request KV cache.

    Keys/values are stored in fixed-size blocks so appends do not require
    reallocation of one large contiguous tensor.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.k_blocks: list[torch.Tensor] = []
        self.v_blocks: list[torch.Tensor] = []
        self._seq_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        _, num_heads, new_seq_len, head_dim = k_new.shape
        offset = 0

        while offset < new_seq_len:
            block_offset = self._seq_len % self.block_size
            if block_offset == 0:
                self.k_blocks.append(
                    torch.zeros(
                        1,
                        num_heads,
                        self.block_size,
                        head_dim,
                        dtype=k_new.dtype,
                        device=k_new.device,
                    )
                )
                self.v_blocks.append(torch.zeros_like(self.k_blocks[-1]))

            take = min(self.block_size - block_offset, new_seq_len - offset)
            self.k_blocks[-1][:, :, block_offset:block_offset + take, :] = k_new[:, :, offset:offset + take, :]
            self.v_blocks[-1][:, :, block_offset:block_offset + take, :] = v_new[:, :, offset:offset + take, :]
            self._seq_len += take
            offset += take

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def k(self) -> torch.Tensor | None:
        if not self.k_blocks:
            return None
        blocks = [block for block in self.k_blocks[:-1]]
        last_block_len = self._seq_len - self.block_size * (len(self.k_blocks) - 1)
        blocks.append(self.k_blocks[-1][:, :, :last_block_len, :])
        return torch.cat(blocks, dim=2)

    @property
    def v(self) -> torch.Tensor | None:
        if not self.v_blocks:
            return None
        blocks = [block for block in self.v_blocks[:-1]]
        last_block_len = self._seq_len - self.block_size * (len(self.v_blocks) - 1)
        blocks.append(self.v_blocks[-1][:, :, :last_block_len, :])
        return torch.cat(blocks, dim=2)

    def bytes_used(self) -> int:
        return sum(block.numel() * block.element_size() for block in self.k_blocks + self.v_blocks)


class BatchedKVCache:
    """
    Wrapper for a batch of per-request paged caches.

    This avoids building one padded dense cache tensor across requests.
    """

    def __init__(self, caches: list[KVCache | None], block_size: int = 16):
        self.caches = caches
        self.block_size = block_size

    def _infer_device(self) -> torch.device | None:
        for cache in self.caches:
            if cache is not None and cache.k_blocks:
                return cache.k_blocks[0].device
        return None

    def _infer_layout(self):
        for cache in self.caches:
            if cache is not None and cache.k_blocks:
                block = cache.k_blocks[0]
                return block.dtype, block.device, block.shape[1], block.shape[3]
        return None

    @property
    def past_lengths(self) -> torch.Tensor | None:
        if not self.caches:
            return None
        device = self._infer_device()
        return torch.tensor(
            [0 if cache is None else cache.seq_len for cache in self.caches],
            dtype=torch.long,
            device=device,
        )

    def append_batch(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        current_lengths: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_lengths is None:
            lengths = [k_new.shape[2]] * len(self.caches)
        elif isinstance(current_lengths, int):
            lengths = [current_lengths] * len(self.caches)
        else:
            lengths = [int(length.item()) for length in current_lengths]

        updated_lengths = []
        for batch_idx, request_cache in enumerate(self.caches):
            cache_i = request_cache if request_cache is not None else KVCache(block_size=self.block_size)
            take = max(min(lengths[batch_idx], k_new.shape[2]), 0)
            if take > 0:
                cache_i.append(
                    k_new[batch_idx : batch_idx + 1, :, :take, :],
                    v_new[batch_idx : batch_idx + 1, :, :take, :],
                )
            self.caches[batch_idx] = cache_i
            updated_lengths.append(cache_i.seq_len)
        return torch.tensor(updated_lengths, dtype=torch.long, device=k_new.device)

    def to_padded_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self._infer_layout()
        if layout is None:
            raise ValueError("Cannot materialize a batched cache without any initialized cache blocks.")

        dtype, device, num_heads, head_dim = layout
        seq_lens = torch.tensor(
            [0 if cache is None else cache.seq_len for cache in self.caches],
            dtype=torch.long,
            device=device,
        )
        max_seq_len = int(seq_lens.max().item()) if len(self.caches) > 0 else 0

        batch_size = len(self.caches)
        k_padded = torch.zeros(
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        v_padded = torch.zeros_like(k_padded)

        for batch_idx, cache in enumerate(self.caches):
            if cache is None or cache.seq_len == 0:
                continue
            k_padded[batch_idx, :, : cache.seq_len, :] = cache.k.squeeze(0)
            v_padded[batch_idx, :, : cache.seq_len, :] = cache.v.squeeze(0)

        return k_padded, v_padded, seq_lens

    def bytes_used(self) -> int:
        return sum(0 if cache is None else cache.bytes_used() for cache in self.caches)
