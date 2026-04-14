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

    def _pop_fifo_batch(self, pending: list):
        batch = pending[: self.max_batch_size]
        remaining = pending[self.max_batch_size :]
        return batch, remaining

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

            batch, pending = self._pop_fifo_batch(pending)

            dispatch_time_ms = current_time_ms
            for req in batch:
                req.start_time_ms = dispatch_time_ms
                req.batch_id = batch_id
                req.scheduler_mode = "dynamic"

            exec_result = batch_executor(batch)
            batch_runtime_ms = exec_result["batch_runtime_ms"]
            finish_time_ms = dispatch_time_ms + batch_runtime_ms

            for req in batch:
                req.first_token_time_ms = dispatch_time_ms + exec_result.get("first_token_time_ms", 0.0)
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
                    "scheduler_mode": "dynamic",
                    "scheduling_policy_value": self.batch_timeout_ms,
                    "phase": "dynamic",
                    "tokens_scheduled": exec_result["tokens_generated_total"],
                    "active_requests": len(batch),
                    "padding_waste_tokens": exec_result.get("padding_waste_tokens", 0),
                    "padding_waste_bytes_est": exec_result.get("padding_waste_bytes_est", 0),
                    "padding_waste_pct": exec_result.get("padding_waste_pct", 0.0),
                }
            )

            current_time_ms = finish_time_ms
            batch_id += 1

        completed.sort(key=lambda r: r.request_id)
        return completed, batch_records


class StaticBatchingScheduler:
    """
    FIFO fixed-size whole-request batching scheduler.

    Policy:
    - Maintain a FIFO queue of arrived requests.
    - Dispatch once the queue reaches max_batch_size requests.
    - If the arrival stream ends with a partial batch, flush the remainder once
      no more requests can arrive.
    - Service remains non-preemptive and whole-request.
    """

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size

    def _pop_fifo_batch(self, pending: list):
        batch = pending[: self.max_batch_size]
        remaining = pending[self.max_batch_size :]
        return batch, remaining

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
            if not pending and next_request_idx < total_requests:
                current_time_ms = max(current_time_ms, requests[next_request_idx].arrival_time_ms)

            while (
                next_request_idx < total_requests
                and requests[next_request_idx].arrival_time_ms <= current_time_ms
            ):
                pending.append(requests[next_request_idx])
                next_request_idx += 1

            if not pending:
                continue

            if len(pending) < self.max_batch_size and next_request_idx < total_requests:
                current_time_ms = max(current_time_ms, requests[next_request_idx].arrival_time_ms)
                continue

            batch, pending = self._pop_fifo_batch(pending)

            dispatch_time_ms = current_time_ms
            for req in batch:
                req.start_time_ms = dispatch_time_ms
                req.batch_id = batch_id
                req.scheduler_mode = "static"

            exec_result = batch_executor(batch)
            batch_runtime_ms = exec_result["batch_runtime_ms"]
            finish_time_ms = dispatch_time_ms + batch_runtime_ms

            for req in batch:
                req.first_token_time_ms = dispatch_time_ms + exec_result.get("first_token_time_ms", 0.0)
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
                    "scheduler_mode": "static",
                    "scheduling_policy_value": 0.0,
                    "phase": "whole_request",
                    "tokens_scheduled": exec_result["tokens_generated_total"],
                    "active_requests": len(batch),
                    "padding_waste_tokens": exec_result.get("padding_waste_tokens", 0),
                    "padding_waste_bytes_est": exec_result.get("padding_waste_bytes_est", 0),
                    "padding_waste_pct": exec_result.get("padding_waste_pct", 0.0),
                }
            )

            current_time_ms = finish_time_ms
            batch_id += 1

        completed.sort(key=lambda r: r.request_id)
        return completed, batch_records


class ContinuousBatchingScheduler:
    """
    Unified continuous batching scheduler with strict decode priority.

    Policy:
    - Requests arrive into a FIFO waiting queue using simulated arrival timestamps.
    - Up to max_batch_size requests may be active at once.
    - All active decode requests advance first, one token each.
    - Remaining token budget is used for chunked prefill work.
    - Requests are grouped by equal progress so the existing batched KV-cache path remains valid.
    """

    def __init__(self, max_batch_size: int, prefill_chunk_size: int, max_tokens_per_iteration: int):
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.max_tokens_per_iteration = max_tokens_per_iteration

    def _enqueue_arrivals(self, requests: list, next_request_idx: int, current_time_ms: float, waiting: list):
        total_requests = len(requests)
        while next_request_idx < total_requests and requests[next_request_idx].arrival_time_ms <= current_time_ms:
            waiting.append(requests[next_request_idx])
            next_request_idx += 1
        return next_request_idx

    def _admit_waiting(self, waiting: list, active: list) -> None:
        while waiting and len(active) < self.max_batch_size:
            req = waiting.pop(0)
            active.append(req)

    def _build_prefill_groups(self, active: list) -> dict[int, list]:
        prefill_groups: dict[int, list] = {}
        for req in active:
            if req.phase == "prefill":
                group_key = req.prompt_tokens_processed
                prefill_groups.setdefault(group_key, []).append(req)
        return prefill_groups

    def _select_prefill_group(self, prefill_groups: dict[tuple[int, int], list]):
        return max(
            prefill_groups,
            key=lambda group_key: (
                len(prefill_groups[group_key]),
                -min(req.arrival_time_ms for req in prefill_groups[group_key]),
            ),
        )

    def run(self, requests: list, prefill_executor, decode_executor) -> tuple[list, list[dict]]:
        requests = sorted(requests, key=lambda r: r.arrival_time_ms)

        waiting: list = []
        active: list = []
        completed: list = []
        event_records: list[dict] = []

        current_time_ms = 0.0
        next_request_idx = 0
        event_id = 0
        total_requests = len(requests)

        while len(completed) < total_requests:
            if not waiting and not active and next_request_idx < total_requests:
                current_time_ms = max(current_time_ms, requests[next_request_idx].arrival_time_ms)

            next_request_idx = self._enqueue_arrivals(
                requests=requests,
                next_request_idx=next_request_idx,
                current_time_ms=current_time_ms,
                waiting=waiting,
            )
            self._admit_waiting(waiting=waiting, active=active)

            if not active:
                continue

            iteration_tokens_used = 0

            decode_group = [req for req in active if req.phase == "decode"]
            if decode_group:
                group = decode_group
                exec_result = decode_executor(group, current_time_ms, event_id)
                current_time_ms += exec_result["batch_runtime_ms"]
                iteration_tokens_used += exec_result["tokens_scheduled"]

                for done_req in exec_result["requests_completed"]:
                    active.remove(done_req)
                    completed.append(done_req)

                event_records.append(
                    {
                        "batch_id": event_id,
                        "dispatch_time_ms": current_time_ms - exec_result["batch_runtime_ms"],
                        "finish_time_ms": current_time_ms,
                        "batch_size": exec_result["batch_size"],
                        "batch_runtime_ms": exec_result["batch_runtime_ms"],
                        "prompt_len": exec_result["prompt_len"],
                        "max_new_tokens": exec_result["max_new_tokens"],
                        "tokens_generated_total": exec_result["tokens_scheduled"],
                        "scheduler_mode": "continuous",
                        "scheduling_policy_value": self.prefill_chunk_size,
                        "phase": exec_result["phase"],
                        "tokens_scheduled": exec_result["tokens_scheduled"],
                        "active_requests": len(active),
                        "padding_waste_tokens": exec_result.get("padding_waste_tokens", 0),
                        "padding_waste_bytes_est": exec_result.get("padding_waste_bytes_est", 0),
                        "padding_waste_pct": exec_result.get("padding_waste_pct", 0.0),
                    }
                )
                event_id += 1

                next_request_idx = self._enqueue_arrivals(
                    requests=requests,
                    next_request_idx=next_request_idx,
                    current_time_ms=current_time_ms,
                    waiting=waiting,
                )
                self._admit_waiting(waiting=waiting, active=active)

            prefill_budget = max(self.max_tokens_per_iteration - iteration_tokens_used, 0)
            if prefill_budget <= 0:
                continue

            while prefill_budget > 0:
                prefill_groups = self._build_prefill_groups(active)
                if not prefill_groups:
                    break

                group_key = self._select_prefill_group(prefill_groups)
                group = prefill_groups[group_key]
                group_budget = min(self.prefill_chunk_size, max(prefill_budget // max(len(group), 1), 0))
                if group_budget <= 0:
                    break

                exec_result = prefill_executor(group, group_budget, current_time_ms, event_id)
                current_time_ms += exec_result["batch_runtime_ms"]
                prefill_budget -= exec_result["tokens_scheduled"]

                for done_req in exec_result["requests_completed"]:
                    active.remove(done_req)
                    completed.append(done_req)

                event_records.append(
                    {
                        "batch_id": event_id,
                        "dispatch_time_ms": current_time_ms - exec_result["batch_runtime_ms"],
                        "finish_time_ms": current_time_ms,
                        "batch_size": exec_result["batch_size"],
                        "batch_runtime_ms": exec_result["batch_runtime_ms"],
                        "prompt_len": exec_result["prompt_len"],
                        "max_new_tokens": exec_result["max_new_tokens"],
                        "tokens_generated_total": exec_result["tokens_scheduled"],
                        "scheduler_mode": "continuous",
                        "scheduling_policy_value": self.prefill_chunk_size,
                        "phase": exec_result["phase"],
                        "tokens_scheduled": exec_result["tokens_scheduled"],
                        "active_requests": len(active),
                        "padding_waste_tokens": exec_result.get("padding_waste_tokens", 0),
                        "padding_waste_bytes_est": exec_result.get("padding_waste_bytes_est", 0),
                        "padding_waste_pct": exec_result.get("padding_waste_pct", 0.0),
                    }
                )
                event_id += 1

                next_request_idx = self._enqueue_arrivals(
                    requests=requests,
                    next_request_idx=next_request_idx,
                    current_time_ms=current_time_ms,
                    waiting=waiting,
                )
                self._admit_waiting(waiting=waiting, active=active)

        completed.sort(key=lambda r: r.request_id)
        return completed, event_records
