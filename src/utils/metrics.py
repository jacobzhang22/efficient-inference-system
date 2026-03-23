def mean(values):
    return sum(values) / len(values) if values else 0.0


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)