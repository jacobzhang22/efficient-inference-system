import torch


class KVCache:
    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

    @property
    def seq_len(self) -> int:
        if self.k is None:
            return 0
        return self.k.shape[2]

    def bytes_used(self) -> int:
        if self.k is None or self.v is None:
            return 0
        return (self.k.numel() * self.k.element_size()) + (self.v.numel() * self.v.element_size())