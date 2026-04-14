# Efficient Inference Systems: KV Cache and Batching Scheduler Tradeoffs in a Controlled Transformer Serving Benchmark

## Overview

This project isolates two core transformer serving mechanisms: **KV caching** and **batching scheduler design**. The system includes a decoder-only transformer, two inference paths (with and without KV cache), a synthetic request generator, and benchmarking code for latency, throughput, memory, and scheduling tradeoffs under load.

The benchmark targets two questions:

- How does KV caching change decode-time scaling as prompt length grows?
- How do baseline, static, dynamic, and continuous batching behave under heterogeneous traffic?

---

## System

### Model configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 5000 |
| Hidden size (`d_model`) | 512 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Feed-forward size (`d_ff`) | 2048 |
| Max sequence length | 1024 |

### Environment

| Component | Value |
|---|---|
| Instance | AWS `g4dn.xlarge` |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 14.56 GB |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu130 |
| CUDA | 13.0 |
| cuDNN | 91300 |
| Precision | FP32 |

### Inference paths

Two generation paths were benchmarked:

1. **No-cache generation**: recomputes attention over the full sequence at every decode step.
2. **KV-cache generation**: performs one prompt prefill pass, stores keys and values, and reuses them during autoregressive decoding.

### Scheduler

The serving simulator compares four scheduling configurations under a shared synthetic request stream:

1. **Baseline (no batching)**: represented by `dynamic` with `max_batch_size = 1`, which executes requests one at a time without any batching benefit.
2. **Static batching**: FIFO whole-request batching that waits until `max_batch_size` requests are queued, then dispatches a batch.
3. **Dynamic batching**: FIFO whole-request batching that dispatches when either the batch fills or the oldest waiting request exceeds a timeout.
4. **Continuous batching**: decode-priority token-level batching with chunked prefill, where active decode requests are scheduled first and remaining token budget is used for prompt prefill.

All scheduler runs used the **KV-cache generation path** for execution. The baseline, static, and dynamic configurations use whole-request execution, while static and dynamic batching use padded whole-request batches. Continuous batching instead uses chunked prefill and incremental decode to reduce padding overhead and improve responsiveness under heterogeneous traffic.

---

## Experimental Configuration

### KV cache experiment

| Parameter | Value |
|---|---|
| Prompt lengths | `[128, 256, 512, 768]` |
| Max new tokens | `128` |
| Repeats | `3` |
| Warmup runs | `1` |
| Batch size | `1` |
| Seed | `42` |

### Scheduler experiment

| Parameter | Value |
|---|---|
| Arrival rates (req/s) | `[4.0, 8.0, 16.0, 24.0, 32.0, 36.0, 44.0, 52.0]` |
| Max batch sizes | `[1, 4, 8]` |
| Dynamic timeouts (ms) | `[0.0, 10.0, 20.0]` |
| Static policy | dispatch on full batch |
| Continuous prefill chunk sizes | `[128, 256]` |
| Requests per run | `200` |
| Repeats | `3` |
| Seed | `42` |
| Workload classes (weight, prompt range / decode range) | short_qa (`0.35`, `48–160` / `16–48`), chat_turn (`0.35`, `128–320` / `32–96`), rag_answer (`0.20`, `256–640` / `64–160`), long_summary (`0.10`, `512–768` / `96–256`) |



Requests arrive as a Poisson process and are generated from a weighted mix of workload classes; within each class, prompt length and decode length are sampled uniformly from configured ranges rather than fixed exact shapes, so the scheduler comparison reflects heterogeneous traffic rather than artificially uniform batches.



---

## Results

### 1. KV Cache

KV caching changes the scaling behavior of decode. Over the tested prompt lengths, the no-cache path became increasingly expensive, while the cached path kept generated-token latency much flatter once prompt length was large enough to overcome cache overhead, at the cost of additional memory.

### Latency behavior

<table>
  <tr>
    <td align="center">
      <img src="results/kv_cache_analysis/plots/total_generation_time.png" alt="Total generation time vs prompt length" width="420"/>
    </td>
    <td align="center">
      <img src="results/kv_cache_analysis/plots/avg_generated_token_latency.png" alt="Average generated-token latency vs prompt length" width="420"/>
    </td>
  </tr>
</table>

Together, these plots show both the end-to-end and per-token effect of caching. Total generation time diverged quickly as prompt length increased, and by prompt length `768`, the cached path was `2.32x` faster overall. At the same time, generated-token latency rose from about `4.07 ms` to `16.51 ms` without caching, while staying roughly in the `6.69–7.11 ms` range with caching.

### Memory behavior

<p align="center">
  <img src="results/kv_cache_analysis/plots/cache_memory_vs_prompt.png" alt="KV cache memory vs prompt length" width="700"/>
</p>

Cache memory grew approximately linearly over the tested range, from `6.0 MB` at prompt length `128` to `21.0 MB` at `768`. The decode-time improvement therefore comes with a direct memory tradeoff: caching reduces repeated computation, but requires storing more keys and values as context grows.

### Summary table

| Prompt Length | No Cache Total (ms) | With Cache Total (ms) | Cache Prefill (ms) | Cached Token Latency (ms) | Speedup | Cache Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 521.16 | 856.48 | 8.05 | 6.69 | 0.61x | 6.0 |
| 256 | 688.12 | 869.64 | 11.23 | 6.79 | 0.79x | 9.0 |
| 512 | 1273.93 | 889.94 | 17.19 | 6.95 | 1.43x | 15.0 |
| 768 | 2112.83 | 909.78 | 22.73 | 7.11 | 2.32x | 21.0 |

Across the tested range, the no-cache path scaled poorly with context length, while KV caching kept decode cost much flatter once prompt length was large enough to overcome cache overhead. End-to-end speedup then increased with prompt length, while memory usage rose roughly linearly.

---

## 2. Scheduler Comparison

The scheduler benchmark compares baseline, static, dynamic, and continuous batching under heterogeneous traffic. The main question is not just whether batching helps, but how different scheduler types trade off throughput, first-token latency, tail latency, and padding waste as offered load increases.

### Throughput and tail-latency behavior

<table>
  <tr>
    <td align="center">
      <img src="results/dynamic_batching/plots/throughput_mode_comparison_final.png" alt="Best-policy throughput vs arrival rate" width="420"/>
    </td>
    <td align="center">
      <img src="results/dynamic_batching/plots/p99_latency_mode_comparison_final.png" alt="Best-policy p99 latency vs arrival rate" width="420"/>
    </td>
  </tr>
</table>

Together, these plots show how the scheduler families diverged as arrival rate increased. The no-batching baseline remained capacity-limited across all tested arrival rates, sustaining only about `1.87–1.89 req/s` with p99 latency growing from roughly `54.48 s` to `101.64 s`. Among the batching schedulers, continuous batching delivered the strongest overall throughput and best tail behavior. Its best policy sustained about `3.87–6.40 req/s` across the tested range, compared with `3.82–4.75 req/s` for dynamic batching and `3.80–4.73 req/s` for static batching. At the highest tested load (`52 req/s`), continuous reached `6.30 req/s` with p99 latency around `27.53 s`, while dynamic and static remained near `4.74 req/s` and `4.70 req/s` with p99 latency around `38.35 s` and `38.75 s`.

### First-token latency behavior

<p align="center">
  <img src="results/dynamic_batching/plots/mean_first_token_latency_mode_comparison_final.png" alt="Best-policy first-token latency vs arrival rate" width="700"/>
</p>

First-token latency showed the clearest separation between scheduler types. The baseline and whole-request schedulers accumulated substantial waiting and service delay as load increased, while continuous batching consistently returned first tokens earlier because it prioritized decode and chunked prefill instead of executing large whole-request batches non-preemptively. Across the best policy for each scheduler type, continuous batching ranged from about `45.68 ms` to `13.00 s`, compared with `1.11–18.23 s` for dynamic batching, `1.45–18.38 s` for static batching, and `27.24–51.18 s` for the no-batching baseline.

### Padding behavior

<p align="center">
  <img src="results/dynamic_batching/plots/padding_waste_mode_comparison.png" alt="Padding waste vs arrival rate" width="700"/>
</p>

Padding waste helps explain the scheduler ranking. Static and dynamic whole-request batching padded each batch to the longest request actually dispatched, which produced substantial waste once the workload became heterogeneous. Under the best-throughput policies, dynamic batching wasted roughly `37.6–50.0%` of padded prompt capacity, and static batching wasted about `41.7%` at batch size `4` and `51.2%` at batch size `8`. Continuous batching, by contrast, stayed near zero padding waste, roughly `0.01–0.10%`, because it used chunked prefill and incremental decode rather than padding full prompts together.

### Summary table

| Scheduler | Best Throughput Range (req/s) | Best P99 Latency Range | Best First-Token Latency Range | Padding Waste |
|---|---:|---:|---:|---:|
| Baseline (`dynamic`, batch `1`) | `1.87–1.89` | `54.48–101.64 s` | `27.24–51.18 s` | `0%` |
| Dynamic batching | `3.82–4.75` | `4.73–38.35 s` | `1.11–18.23 s` | `37.6–50.0%` |
| Static batching | `3.80–4.73` | `4.98–38.75 s` | `1.45–18.38 s` | `41.7–51.2%` |
| Continuous batching | `3.87–6.40` | `2.81–27.53 s` | `45.68 ms–13.00 s` | `0.01–0.10%` |

In this workload, continuous batching was the strongest overall scheduler under realistic prompt-length heterogeneity. Static and dynamic whole-request batching still improved substantially over the no-batching baseline, but both paid a large padding penalty that limited throughput and worsened latency as load increased.

---


## Limitations

This benchmark is intentionally controlled and omits several production concerns:

- synthetic workload classes rather than real production request traces
- minimal scheduler implementations rather than production serving stacks
- single model size
- single GPU type
- shared simulator assumptions across all scheduler families

These constraints keep the benchmark controlled and isolated, but they also limit direct comparability to production serving systems.

---

## Conclusion

This benchmark isolates two core serving behaviors. KV caching reduced decode-time growth as context length increased, while the scheduler comparison showed that batching policy strongly affects throughput, latency, and padding efficiency under heterogeneous traffic. In this implementation, continuous batching delivered the strongest overall performance by combining higher throughput with better first-token and tail-latency behavior, while static and dynamic whole-request batching remained limited by substantial padding waste. Although the benchmark is intentionally controlled, the underlying tradeoffs of cache reuse, queue buildup, utilization, and memory growth are the same ones that shape larger transformer serving systems.

