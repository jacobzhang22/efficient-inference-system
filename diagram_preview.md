# Engine Diagram Preview

These diagrams are intentionally split by concern so each one stays readable in Markdown preview.

## 1. Top-Level Map

```mermaid
flowchart TD
  subgraph ENTRY["Entry Points"]
    B1["experiments/batching/benchmark_scheduler.py"]
    B2["experiments/kv_cache_analysis/run_all.py"]
    B3["src/profiling/profile_inference.py"]
  end

  subgraph ENGINE["Serving / Inference Engine"]
    S1["Serving schedulers + executors"]
    S2["generate_with_cache()"]
    S3["TinyTransformerLM"]
    S4["Paged KV cache"]
    S5["Paged attention backends"]
  end

  subgraph OUT["Outputs"]
    O1["summary.csv / requests.csv / events.csv"]
    O2["KV benchmark CSVs"]
    O3["plots"]
  end

  B1 --> S1
  B2 --> S2
  B3 --> S2

  S1 --> S2
  S2 --> S3
  S3 --> S4
  S3 --> S5

  S1 --> O1
  B2 --> O2
  O1 --> O3
  O2 --> O3
```

## 2. Whole-Request Batching Path

This is the static/dynamic batching path.

```mermaid
flowchart TD
  Req["InferenceRequest list"] --> Sched["DynamicBatchingScheduler\nor StaticBatchingScheduler"]
  Sched --> Batch["run_batch_generate()"]
  Batch --> Pad["pad prompt batch + prompt_lengths"]
  Pad --> Gen["generate_with_cache()"]

  Gen --> Prefill["prefill full prompt batch"]
  Prefill --> DecodeLoop["decode loop over active requests"]
  DecodeLoop --> Result["timings + KV metrics + generated ids"]

  Result --> Release["model.release_kv_caches()"]
  Release --> Record["scheduler batch record"]
```

## 3. Continuous Scheduling Path

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

## 4. `generate_with_cache()` Runtime

```mermaid
flowchart TD
  Input["prompt_ids + max_new_tokens"] --> Norm["normalize prompt lengths\nand decode limits"]
  Norm --> Prefill["model(... use_cache=True)\non full prompt batch"]
  Prefill --> First["take last prompt logits\nemit first next-token"]

  First --> Extract["extract per-request cache handles"]
  Extract --> Loop{"more decode steps?"}

  Loop -- yes --> Select["select active requests"]
  Select --> Stack["stack request caches into\nbatched layer caches"]
  Stack --> Decode["model(... one token per\nactive request)"]
  Decode --> Scatter["scatter updated caches\nback to request slots"]
  Scatter --> Loop

  Loop -- no --> Metrics["compute live/reserved/\nfragmentation/GPU stats"]
  Metrics --> Output["return generated ids,\ntimings, kv_caches, metrics"]
```

## 5. Model Internals

```mermaid
flowchart TD
  In["input_ids"] --> Emb["token_emb + pos_emb"]
  Emb --> Blocks["Transformer blocks"]

  subgraph BLOCK["Per TransformerBlock"]
    LN1["LayerNorm"]
    Attn["CausalSelfAttention"]
    Res1["residual add"]
    LN2["LayerNorm"]
    MLP["MLP"]
    Res2["residual add"]
    LN1 --> Attn --> Res1 --> LN2 --> MLP --> Res2
  end

  Blocks --> Final["ln_f + lm_head"]
  Final --> Logits["logits"]
```

## 6. Paged KV Cache Design

```mermaid
flowchart TD
  LM["TinyTransformerLM"] --> Pools["one LayerBlockPool per layer"]
  LM --> ReqCache["create_request_kv_caches()"]

  ReqCache --> State["PagedKVCacheState per request\nper layer"]
  State --> IDs["block_ids + seq_len"]
  State --> PoolUse["allocate/release blocks\nfrom layer pool"]

  Pools --> Storage["shared k_blocks / v_blocks"]
  PoolUse --> Storage

  State --> BatchState["BatchedPagedKVCache"]
  BatchState --> PT["page_table_tensor()"]
  BatchState --> SL["seq_lens_tensor()"]
```

## 7. Cached Attention Execution

```mermaid
flowchart TD
  X["hidden states x"] --> Proj["q_proj / k_proj / v_proj"]
  Proj --> Split["split heads -> q, k_new, v_new"]
  Split --> Append["kv_cache.append_batch(k_new, v_new, current_lengths)"]
  Append --> Dispatch["paged_attention(backend, q, kv_cache, current_lengths)"]

  Dispatch --> Choose{"backend + device"}
  Choose -- "paged_reference or CPU" --> Ref["reference_paged_attention"]
  Choose -- "CUDA + query_len == 1" --> TriD["triton decode kernel"]
  Choose -- "CUDA + query_len > 1" --> TriP["triton prefill kernel"]

  Dispatch --> PT["page table"]
  Dispatch --> SL["seq lens"]
  Dispatch --> Pool["k_blocks / v_blocks"]

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

## 8. Scheduler Benchmark Data Flow

```mermaid
flowchart TD
  Cfg["SchedulingExperimentConfig"] --> Build["build_model()"]
  Cfg --> Load["generate_requests()"]
  Build --> Run["run scheduler once"]
  Load --> Run

  Run --> Completed["completed requests"]
  Run --> Events["batch/event records"]

  Completed --> Summ["summarize_run()"]
  Completed --> ReqRows["requests_to_rows()"]
  Events --> EventRows["batches_to_rows()"]

  Summ --> CSV1["summary.csv"]
  ReqRows --> CSV2["requests.csv"]
  EventRows --> CSV3["events.csv"]
```
