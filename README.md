# Efficient Inference Systems: KV Cache and Dynamic Batching in a Tiny Transformer

## Overview

This project is a controlled inference-systems study built to develop intuition for modern AI infrastructure. I implemented a small decoder-only transformer and used it to isolate and measure two core serving ideas that appear in real LLM inference systems:

1. **KV caching**, which avoids recomputing attention over the full prefix at every decode step
2. **Dynamic batching**, which groups incoming requests to improve GPU utilization and throughput

My goal was not to reproduce a production stack, but to build a minimal system where the latency, throughput, and memory tradeoffs are easy to see and reason about. This project was part of a broader effort to build a stronger systems foundation for inference engineering and contribute to projects such as vLLM.

---

## Why I Built This

Modern LLM serving is fundamentally a systems problem. Good model quality is not enough by itself; real deployments depend on how efficiently the system handles long contexts, many concurrent users, and GPU memory limits.

I built this project to answer two practical questions:

- **How much does KV caching help as prompt length grows?**
- **How does dynamic batching change throughput, wait time, and tail latency under load?**

These are small experiments, but they target the same core pressures that show up in real inference systems: recomputation, memory growth, queueing, and batching tradeoffs.

---

## System Design

### Model

I implemented a small decoder-only transformer language model with the following configuration:

- Vocabulary size: `5000`
- Hidden size (`d_model`): `512`
- Number of attention heads: `8`
- Number of layers: `6`
- Feed-forward size (`d_ff`): `2048`
- Maximum sequence length: `1024`
- Dropout: `0.0`

The model is intentionally small so that the serving behavior is easy to instrument and analyze.

### Inference Paths

I implemented two generation modes:

#### 1. No-cache generation

At every decode step, the model recomputes attention over the full sequence seen so far. This is simple but increasingly expensive as context grows.

#### 2. KV-cache generation

The model performs one full prefill pass on the prompt, stores keys and values, and then reuses them during autoregressive decoding. This reduces repeated work during token generation, but consumes additional memory.

### Request Simulation and Scheduler

To study batching behavior, I implemented a small serving simulator with:

- synthetic request arrivals generated from a **Poisson process**
- a **FIFO dynamic batching scheduler**
- a **non-preemptive batch worker**

The scheduler follows a simple policy:

- dispatch immediately when queue size reaches `max_batch_size`
- otherwise wait until the oldest request has waited `batch_timeout_ms`
- once a batch starts, the worker stays busy until the batch finishes

This is intentionally simpler than production token-level continuous batching, but it is enough to expose the key throughput/latency tradeoffs.

---

## Experimental Setup

### Hardware and Software

All experiments were run on:

- **AWS instance:** `g4dn.xlarge`
- **GPU:** NVIDIA Tesla T4
- **GPU memory:** 14.56 GB
- **Python:** 3.12.3
- **PyTorch:** 2.9.1+cu130
- **CUDA:** 13.0
- **cuDNN:** 91300

### Precision

Experiments were run in standard:

- **FP32**
- default dtype: `torch.float32`
- autocast disabled

### KV Cache Experiment Configuration

- Prompt lengths: `[128, 256, 512, 768]`
- Max new tokens: `128`
- Repeats: `3`
- Warmup runs: `1`
- Batch size: `1`
- Seed: `42`

### Dynamic Batching Experiment Configuration

- Arrival rates: `[20.0, 28.0, 36.0]` requests/sec
- Max batch sizes: `[1, 4, 8]`
- Batch timeouts: `[0.0, 10.0, 20.0]` ms
- Requests per run: `200`
- Prompt length: `128`
- Max new tokens: `32`
- Repeats: `3`
- Seed: `42`

---

## Methodology

### Part 1: KV Cache Study

The first experiment compares generation with and without a KV cache as prompt length increases.

For each prompt length, I measured:

- total generation time
- average generated-token latency
- prefill time
- cache decode-only latency
- KV cache memory usage

The main question was whether caching keeps decode cost flatter as prompt length grows.

### Part 2: Dynamic Batching Study

The second experiment simulates a stream of arriving inference requests and runs them through the batching scheduler.

For each scheduler configuration, I measured:

- throughput in requests/sec
- throughput in tokens/sec
- mean latency
- p50, p95, and p99 latency
- mean wait time
- mean service time
- realized batch size

The main question was how batching changes system efficiency and tail latency under load.

---

## Results

# 1) KV Cache Results

### Total generation time improves sharply with cache at longer prompts

At short prompts, caching helps only a little. At long prompts, it matters a lot.

Observed total generation times:

| Prompt Length | No Cache (ms) | With Cache (ms) | Approx. Speedup |
| ------------- | ------------: | --------------: | --------------: |
| 128           |        538.07 |          531.98 |           1.01x |
| 256           |        751.24 |          614.43 |           1.22x |
| 512           |       1359.68 |          642.54 |           2.12x |
| 768           |       2292.91 |          700.23 |           3.27x |

The pattern is clear: as prompt length grows, no-cache decoding becomes increasingly expensive because the model keeps revisiting more and more prior context. With KV caching, that repeated work is largely removed from the decode path.

### Per-token generation cost stays much flatter with cache

Average generated-token latency shows the same story more directly.

- **No cache:** rises from about `4.16 ms` to `16.12 ms`
- **With cache:** stays around `4.2–4.9 ms`

This is one of the strongest results in the project. It shows that KV caching is not just a minor optimization; it changes how decode cost scales with context length.

### Prefill still grows with prompt length

Caching does not make the entire problem disappear. The model still has to process the full prompt once up front during prefill, so prefill time continues to rise with prompt length.

That distinction matters:

- **Prefill:** still scales with the input sequence
- **Decode:** becomes much cheaper with cache reuse

This matches the intuition behind practical LLM serving systems, where prefill and decode are often treated as distinct phases with different performance characteristics.

### Cache memory grows roughly linearly with prompt length

The main tradeoff is memory.

Observed KV cache memory:

| Prompt Length | Cache Memory (MB) |
| ------------- | ----------------: |
| 128           |              5.98 |
| 256           |             10.98 |
| 512           |             15.98 |
| 768           |             20.98 |

This is expected. Longer prompts require storing more keys and values across layers and heads. In other words, KV caching saves compute during decode by spending more GPU memory.

### KV Cache Takeaway

The KV-cache experiments show a clean systems tradeoff:

- without cache, decode latency grows quickly with context length
- with cache, decode latency stays much flatter
- the benefit becomes more dramatic at longer prompts
- the cost is increased memory usage proportional to context length

This is exactly why KV caching is foundational in modern inference engines.

---

# 2) Dynamic Batching Results

### Batch size 1 performs poorly under all tested loads

With `max_batch_size = 1`, the system is effectively serving requests one at a time. In this setting, throughput stayed near:

- `~7.37 req/s`

even as arrival rate increased from `20` to `36 req/s`

That means the system was overloaded across the entire tested range. Once arrival rate exceeds service capacity, queueing dominates, and latency explodes.

This shows up clearly in p99 latency:

- roughly `17s` to `21.6s`

That is the classic shape of an overloaded queueing system: the server cannot keep up, so requests spend most of their lifetime waiting.

### Batch size 4 is much better, but still shows tail growth at high load

With `max_batch_size = 4`, throughput improved substantially:

- about `20.5 req/s` at lower load
- up to `28.8 req/s` at higher load

This is a major improvement over single-request serving because batching allows the GPU to process several requests together more efficiently.

However, tail latency still grows at higher arrival rates. At `36 req/s`, p99 latency was around:

- `~1.68s`

So batching at 4 helps a lot, but the configuration still approaches saturation in the heaviest tested case.

### Batch size 8 gives the best tradeoff in this experiment

With `max_batch_size = 8`, the system achieved both the highest throughput and the best tail behavior among the tested settings.

Observed throughput:

- `~20.4 req/s` at low load
- up to `~36.17 req/s` at high load

Observed p99 latency:

- roughly `272–331 ms`

This is the most important dynamic batching result in the project. At the tested arrival rates, larger batches gave the GPU enough work to stay productive without creating the catastrophic queue buildup seen in the `batch=1` case.

### Realized batch size explains the throughput gains

The mean realized batch size increased with arrival rate, especially for larger batch caps. This is expected: when requests arrive more frequently, the scheduler has more opportunities to form fuller batches before timing out.

This explains why throughput improves with load up to a point. More arrivals make it easier to amortize batch overhead and keep the GPU busier.

### Timeout matters less than batch cap in this setup

I tested timeout values of `0 ms`, `10 ms`, and `20 ms`.

In these experiments, timeout had only a modest effect relative to the choice of `max_batch_size`. The largest performance differences came from allowing the scheduler to build bigger batches, not from small changes in waiting policy.

That does not mean timeout is unimportant in general. In a more realistic serving system with more varied request sizes or lower arrival rates, timeout policy could matter much more. But in this controlled setup, **batch cap dominated timeout**.

### Throughput vs. tail latency shows the core serving tradeoff

The throughput-versus-p99 scatter plot is the best summary figure for the batching study because it shows the frontier directly.

The key pattern is:

- very small batches underutilize the system and suffer extreme tail latency under load
- larger batches improve throughput substantially
- in this tested range, larger batches also improve tail latency because they prevent severe backlog

This last point is important. Batching is often described as a tradeoff where bigger batches increase waiting. That is true in some regimes. But when the unbatched system is badly underutilized and overloaded, batching can actually improve both throughput and latency by avoiding collapse.

### Dynamic Batching Takeaway

In this project, dynamic batching was not just a throughput optimization. It was the difference between:

- an overloaded system with multi-second to tens-of-seconds tail latency
- a stable system that handled the offered load with sub-second p99 latency

The scheduler is simple, but it still exposes a real serving lesson: **queueing and utilization matter as much as raw model compute**.

---

## Recommended Figures for the Writeup

For a clean portfolio-style presentation, I would include the following figures:

### KV Cache Section

1. **Total generation time vs prompt length**
2. **Average generated-token latency vs prompt length**
3. **Prefill time vs prompt length**
4. **KV cache memory vs prompt length**

### Dynamic Batching Section

5. **Throughput vs arrival rate**
6. **P99 latency vs arrival rate**
7. **Throughput vs p99 latency scatter**
8. **Latency CDF for one representative high-load case** (optional but useful)

These are the most decision-relevant plots. They support the main claims without overwhelming the reader.

---

## What This Project Demonstrates

This project shows several things that are relevant to inference and systems work:

### 1. I can reduce a broad infrastructure topic into measurable subproblems

Instead of trying to understand all of LLM serving at once, I broke the problem into two fundamental mechanisms: cache reuse and batching.

### 2. I can build controlled systems experiments

The value of this project is not just that it runs, but that it isolates variables and produces interpretable measurements.

### 3. I can reason about tradeoffs, not just code paths

The main lessons here are about balancing:

- compute vs memory
- throughput vs tail latency
- batching gains vs queueing delay

### 4. I am developing the right foundation for inference engineering

KV caching, batching, prefill/decode separation, queueing behavior, and GPU utilization are all core ideas in modern inference stacks. This project gave me a concrete foundation for understanding systems such as vLLM at a deeper level.

---

## Limitations

This is a controlled toy system, not a production LLM serving stack.

Important limitations:

- the model is small and synthetic
- requests use fixed prompt lengths within a run
- all requests in a batch share the same `max_new_tokens`
- the scheduler is FIFO and non-preemptive
- this is not token-level continuous batching
- only one GPU type was tested
- the request stream is synthetic rather than based on real production traces

These limitations are acceptable for the goal of the project, which was to build intuition and measure first-order serving effects in a clean environment.

---

## Future Work

There are several natural next steps:

### 1. Heterogeneous request sizes

Allow mixed prompt lengths and mixed decode lengths in the same run. This would introduce padding and scheduling complexity closer to real serving systems.

### 2. Continuous/token-level batching

Move beyond whole-request batching toward a more production-like scheduler that can admit work at decode granularity.

### 3. Better memory accounting

Extend the measurements to include allocator behavior, fragmentation effects, and longer-lived cache pressure.

### 4. Precision experiments

Compare FP32 with FP16 or BF16 to study the interaction between precision, throughput, and memory headroom.

### 5. Multi-request fairness

Track whether some requests systematically suffer worse latency due to queue position or batching policy.

---

## Conclusion

This project was a focused study of two foundational inference-serving techniques: KV caching and dynamic batching.

The results were clear:

- **KV caching** keeps decode latency much flatter as prompt length grows, with the tradeoff of increased memory usage
- **Dynamic batching** substantially improves throughput and, in this setup, prevents queueing collapse and dramatically reduces tail latency

Even though the system is intentionally minimal, the underlying lessons are directly relevant to real AI infrastructure work. Building and measuring this project gave me a stronger mental model for how modern inference stacks behave and why systems like vLLM are designed the way they are.

---

## Resume-Friendly Summary

Built a small transformer inference benchmark in PyTorch on AWS T4 GPU to study KV caching and dynamic batching. Measured latency, throughput, and memory tradeoffs across varying prompt lengths and request loads; showed that KV caching kept per-token decode latency nearly flat as context grew, while dynamic batching improved throughput from ~7 req/s to ~36 req/s and reduced p99 latency from ~20s to ~300ms in the tested regime.

---

## Short GitHub Description

A controlled transformer inference systems project that studies KV caching and dynamic batching on GPU. Includes a tiny decoder-only model, synthetic request simulator, FIFO batching scheduler, and benchmarks for latency, throughput, and memory tradeoffs.
