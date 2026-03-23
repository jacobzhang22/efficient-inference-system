import time
from contextlib import contextmanager
import torch


class _TimerResult:
    def __init__(self):
        self.elapsed_ms = 0.0


@contextmanager
def timed_section(device: str = "cpu"):
    result = _TimerResult()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield result
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    result.elapsed_ms = (end - start) * 1000.0