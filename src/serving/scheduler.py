class DynamicBatchingScheduler:
    """
    Minimal FIFO dynamic batching scheduler.

    Policy:
    - Maintain a FIFO queue of arrived requests.
    - Dispatch immediately if queue size reaches max_batch_size.
    - Otherwise, wait until oldest request has waited batch_timeout_ms.
    - Service is non-preemptive: once a batch starts, the worker is busy until it finishes.
    """

    def __init__(self, max_batch_size: int, batch_timeout_ms: float):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

    def run(self, requests: list, batch_executor) -> tuple[list, list[dict]]:
        requests = sorted(requests, key=lambda r: r.arrival_time_ms)

        pending: list = []
        completed: list = []
        batch_records: list[dict] = []

        current_time_ms = 0.0
        next_request_idx = 0
        batch_id = 0
        total_requests = len(requests)

        while len(completed) < total_requests:
            # If idle with no pending work, jump to next arrival.
            if not pending and next_request_idx < total_requests:
                current_time_ms = max(current_time_ms, requests[next_request_idx].arrival_time_ms)

            # Pull in all requests that have already arrived.
            while (
                next_request_idx < total_requests
                and requests[next_request_idx].arrival_time_ms <= current_time_ms
            ):
                pending.append(requests[next_request_idx])
                next_request_idx += 1

            if not pending:
                continue

            oldest_arrival_ms = pending[0].arrival_time_ms
            oldest_deadline_ms = oldest_arrival_ms + self.batch_timeout_ms

            can_fill_batch_now = len(pending) >= self.max_batch_size

            if not can_fill_batch_now:
                next_arrival_ms = (
                    requests[next_request_idx].arrival_time_ms
                    if next_request_idx < total_requests
                    else None
                )

                if next_arrival_ms is not None and next_arrival_ms < oldest_deadline_ms:
                    current_time_ms = next_arrival_ms
                    continue

                current_time_ms = max(current_time_ms, oldest_deadline_ms)

                while (
                    next_request_idx < total_requests
                    and requests[next_request_idx].arrival_time_ms <= current_time_ms
                ):
                    pending.append(requests[next_request_idx])
                    next_request_idx += 1

            batch = pending[: self.max_batch_size]
            pending = pending[self.max_batch_size :]

            dispatch_time_ms = current_time_ms
            for req in batch:
                req.start_time_ms = dispatch_time_ms
                req.batch_id = batch_id

            exec_result = batch_executor(batch)
            batch_runtime_ms = exec_result["batch_runtime_ms"]
            finish_time_ms = dispatch_time_ms + batch_runtime_ms

            for req in batch:
                req.finish_time_ms = finish_time_ms
                completed.append(req)

            batch_records.append(
                {
                    "batch_id": batch_id,
                    "dispatch_time_ms": dispatch_time_ms,
                    "finish_time_ms": finish_time_ms,
                    "batch_size": len(batch),
                    "batch_runtime_ms": batch_runtime_ms,
                    "prompt_len": exec_result["prompt_len"],
                    "max_new_tokens": exec_result["max_new_tokens"],
                    "tokens_generated_total": exec_result["tokens_generated_total"],
                }
            )

            current_time_ms = finish_time_ms
            batch_id += 1

        completed.sort(key=lambda r: r.request_id)
        return completed, batch_records