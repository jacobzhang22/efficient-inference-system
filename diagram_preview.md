# Final Writeup Diagram Preview

These are the three diagrams I would keep for the final Markdown writeup.

## 1. System Architecture

```mermaid
flowchart TD
  subgraph ENTRY["Entry Points"]
    Bench["experiments/batching/benchmark_scheduler.py"]
    KVExp["experiments/kv_cache_analysis/*.py"]
    Profile["src/profiling/profile_inference.py"]
  end

  subgraph SERVE["Serving Layer"]
    Sched["Schedulers + executors"]
    Gen["generate_with_cache()"]
    Metrics["metrics + CSV outputs"]
  end

  subgraph MODEL["Model + Cache"]
    LM["TinyTransformerLM"]
    KV["Paged KV cache"]
    Attn["Paged attention dispatch"]
  end

  subgraph BACKEND["Backends"]
    Ref["reference backend"]
    Triton["Triton paged backend"]
  end

  Bench --> Sched
  KVExp --> Gen
  Profile --> Gen

  Sched --> Gen
  Gen --> LM
  LM --> KV
  LM --> Attn

  Attn --> Ref
  Attn --> Triton

  Sched --> Metrics
  KVExp --> Metrics
```

## 2. Continuous Serving Flow

```mermaid
flowchart TD
  Stream["Request stream"] --> WaitQ["waiting queue"]
  WaitQ --> Admit["admit up to max_batch_size"]
  Admit --> Active["active requests"]

  Active --> CheckDecode{"decode-phase requests exist?"}

  CheckDecode -- yes --> Decode["run_decode_step()"]
  Decode --> ScatterD["scatter updated caches\nback into requests"]
  ScatterD --> DoneD{"finished?"}
  DoneD -- yes --> ReleaseD["release_request_caches()"]
  DoneD -- no --> Active

  CheckDecode -- no --> Budget["prefill token budget"]
  ReleaseD --> Active
  Active --> Budget

  Budget --> Group["group by prompt_tokens_processed"]
  Group --> Pick["pick largest/oldest group"]
  Pick --> Prefill["run_prefill_chunk()"]
  Prefill --> ScatterP["scatter updated caches\nback into requests"]
  ScatterP --> DoneP{"prefill complete?"}
  DoneP -- no --> Active
  DoneP -- yes --> First["emit first token\nswitch to decode"]
  First --> FinishP{"finished?"}
  FinishP -- yes --> ReleaseP["release_request_caches()"]
  FinishP -- no --> Active

  ReleaseP --> Active
```

## 3. Paged Attention Execution

```mermaid
flowchart TD
  X["hidden states x"] --> Proj["q_proj / k_proj / v_proj"]
  Proj --> Split["split heads -> q, k_new, v_new"]
  Split --> Append["kv_cache.append_batch(k_new, v_new, current_lengths)"]
  Append --> Dispatch["paged_attention(backend, q, kv_cache, current_lengths)"]

  Append --> State["PagedKVCacheState"]
  State --> Pool["LayerBlockPool\nshared k_blocks / v_blocks"]
  State --> Batch["BatchedPagedKVCache"]

  Batch --> PT["page_table_tensor()"]
  Batch --> SL["seq_lens_tensor()"]

  Dispatch --> Choose{"backend + device"}
  Choose -- "paged_reference or CPU" --> Ref["reference_paged_attention"]
  Choose -- "CUDA + query_len == 1" --> TriD["triton decode kernel"]
  Choose -- "CUDA + query_len > 1" --> TriP["triton prefill kernel"]

  PT --> TriD
  PT --> TriP
  SL --> TriD
  SL --> TriP
  Pool --> Ref
  Pool --> TriD
  Pool --> TriP

  Ref --> Merge["merge heads + out_proj"]
  TriD --> Merge
  TriP --> Merge
  Merge --> Out["attention output"]
```
