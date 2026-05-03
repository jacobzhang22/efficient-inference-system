# Final Writeup Diagram Preview

These are the two diagrams I would keep for the final Markdown writeup.

## 1. Continuous Serving Flow

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
  DoneD -- no --> Budget["remaining iteration budget\nfor prefill"]

  CheckDecode -- no --> Budget
  ReleaseD --> Budget

  Budget --> Pick["select prefill group\nsame prompt progress,\nlargest/oldest first"]
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

## 2. Paged Attention Execution

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
