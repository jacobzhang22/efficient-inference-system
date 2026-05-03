# Final Writeup Diagram Preview

These are the two diagrams I would keep for the final Markdown writeup.

## 1. Continuous Serving Flow

```mermaid
flowchart TD
  Stream["Incoming request stream"] --> WaitQ["Waiting queue"]
  WaitQ --> Admit["Admit requests\nup to max_batch_size"]
  Admit --> Active["Active request set"]

  Active --> CheckDecode{"Any active requests\nin decode phase?"}

  CheckDecode -- yes --> Decode["Decode one token for\ndecode-ready requests\n(run_decode_step)"]
  Decode --> ScatterD["Write updated KV state\nback to each request"]
  ScatterD --> DoneD{"Did any request finish\ngeneration?"}
  DoneD -- yes --> ReleaseD["Return completed request\nKV pages to pool\n(release_request_caches)"]
  DoneD -- no --> Budget["Use remaining iteration\nbudget for prefill"]

  CheckDecode -- no --> Budget
  ReleaseD --> Budget

  Budget --> Pick["Select a prefill batch:\nsame prompt progress,\nlargest/oldest first"]
  Pick --> Prefill["Prefill next prompt chunk\nfor selected batch\n(run_prefill_chunk)"]
  Prefill --> ScatterP["Write updated KV state\nback to each request"]
  ScatterP --> DoneP{"Did any request finish\nprefill?"}
  DoneP -- no --> Active
  DoneP -- yes --> First["Emit first output token\nand move request to decode"]
  First --> FinishP{"Did any request already\nreach max_new_tokens?"}
  FinishP -- yes --> ReleaseP["Return completed request\nKV pages to pool\n(release_request_caches)"]
  FinishP -- no --> Active

  ReleaseP --> Active
```

## 2. CUDA Paged Attention Execution

```mermaid
flowchart TD
  X["Input hidden states"] --> Proj["Project Q, K, V"]
  Proj --> Split["Split into attention heads"]
  Split --> Append["Append new K/V vectors\nto paged KV cache\n(append_batch)"]
  Append --> Dispatch["Execute paged attention\nover cached KV pages\n(paged_attention)"]

  Append --> State["For each request, track:\nwhich cache pages belong to it\nand how many tokens it has cached"]
  State --> Pool["Shared cache-page storage\nfor this layer's K/V blocks"]
  State --> Batch["Combine all active requests\ninto one batched cache view"]

  Batch --> PT["List which cache pages\nbelong to each request"]
  Batch --> SL["List how many cached tokens\nare valid for each request"]

  Dispatch --> Choose{"CUDA execution mode"}
  Choose -- "Decode\n(query_len = 1)" --> TriD["Fast Triton decode path"]
  Choose -- "Prefill\n(query_len > 1)" --> TriP["Fast Triton prefill path"]

  PT --> TriD
  PT --> TriP
  SL --> TriD
  SL --> TriP
  Pool --> TriD
  Pool --> TriP

  TriD --> Merge
  TriP --> Merge
  Merge --> Out["Attention output"]
```
