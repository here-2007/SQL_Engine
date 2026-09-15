# Text-to-SQL: End-to-End Architecture & Pipeline

This document presents the complete **End-to-End Architecture & Pipeline** for the Text-to-SQL system, structured specifically for explaining the project on a whiteboard or during a technical interview.

---

## High-Level Mental Model: Two Distinct Halves

When explaining this project, divide it into two major parts:

1. **The Offline Pipeline:** How raw relational data was transformed, trained, and rigorously evaluated.
2. **The Online Production Architecture:** How user queries travel from the edge to the model and execute safely on live databases.

---

# 1. The Offline Engineering Pipeline

```text
┌─────────────────────────┐
│   Yale Spider Benchmark │ (10,181 queries, 200 databases)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 10-Stage Data Pipeline  │ • AST Hardness Audit (Easy, Med, Hard, Extra-Hard)
│  (data/processing/)     │ • DDL Serialization (Extracts PKs & FKs)
│                         │ • Token Distribution & 95th Percentile Cutoff (1,024 tokens)
└────────────┬────────────┘
             │
             ▼ (Apache Arrow memory-mapped dataset)
┌─────────────────────────┐
│ Distributed QLoRA Run   │ • Base Model: Qwen2.5-Coder-1.5B
│  (Dual NVIDIA T4 GPUs)  │ • 4-bit NF4 Double Quantization
│                         │ • All 7 Projections Adapted (q, k, v, o, gate, up, down)
│                         │ • 8-bit Paged AdamW + PyTorch DDP
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Standalone FP16 Merge   │ • merge_and_unload() fused weights into 2.45GB artifact
│   (Published to Kaggle) │ • low_cpu_mem_usage=True (<3GB RAM on CPU)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Evaluation Suite        │ • Exact Match (EM) with column-order sorting
│ (src/evaluation/)       │ • Execution Accuracy (EX) with multiset column permutation
│                         │ • 3-second timeout guard + AST read-only sandbox
└─────────────────────────┘
```

## Offline Pipeline Breakdown

### 1. Yale Spider Benchmark

- **10,181 queries**
- **200 relational databases**

The benchmark provides the raw question/SQL/database examples used to construct the training and evaluation pipeline.

### 2. 10-Stage Data Pipeline

The processing layer performs:

- **AST Hardness Audit:** Classifies queries into `Easy`, `Med`, `Hard`, and `Extra-Hard`.
- **DDL Serialization:** Converts database schemas into serialized SQLite DDL while preserving primary keys (PKs) and foreign keys (FKs).
- **Token Distribution Analysis:** Determines the sequence-length cutoff using the **95th percentile**, resulting in a **1,024-token cutoff**.

The processed dataset is stored as an **Apache Arrow memory-mapped dataset**.

### 3. Distributed QLoRA Training

Training configuration:

- **Base model:** `Qwen2.5-Coder-1.5B`
- **Hardware:** Dual NVIDIA T4 GPUs
- **Quantization:** 4-bit NF4 with double quantization
- **Adapted projections:** `q`, `k`, `v`, `o`, `gate`, `up`, `down`
- **Optimizer:** 8-bit Paged AdamW
- **Distributed training:** PyTorch DDP

### 4. Standalone FP16 Merge

After adapter training:

- `merge_and_unload()` fuses the LoRA weights into the base model.
- The resulting standalone FP16 artifact is approximately **2.45 GB**.
- `low_cpu_mem_usage=True` is used so the merge process stays below approximately **3 GB RAM on CPU**.
- The merged model is published to Kaggle.

### 5. Evaluation Suite

The evaluation layer contains:

- **Exact Match (EM):** SQL comparison with column-order sorting.
- **Execution Accuracy (EX):** Compares execution results with multiset column permutation.
- **3-second timeout guard:** Prevents indefinitely running queries.
- **AST read-only sandbox:** Prevents destructive SQL from being executed.

---

# 2. The Online Serving Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                       1. Edge Routing (Cloudflare Workers)                    │
│                      https://text-to-sql.here-2007.workers.dev                │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ HTTPS
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                 Hugging Face Spaces ZeroGPU / Container Runtime               │
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐   │
│   │                      2. Gradio Workstation (:7860)                    │   │
│   │  • 3-Column Terminal Layout with Dark/Light Theme                     │   │
│   │  • Connection Manager: SQLite, PostgreSQL, MySQL, Supabase API        │   │
│   │  • ZERO CREDENTIAL LEAKAGE: DB passwords never leave this layer       │   │
│   │  • Dialect Harmonizer: AST rewrites (e.g. IFNULL ➔ COALESCE)          │   │
│   └──────────────────────────────────┬──────────────────▲─────────────────┘   │
│                                      │                  │                     │
│         POST /v1/tosql               │                  │ Returns             │
│        {question, schema}            ▼                  │ Formatted SQL       │
│   ┌─────────────────────────────────────────────────────┴─────────────────┐   │
│   │                    3. FastAPI Inference Gateway (:8000)               │   │
│   │  • SlowAPI Rate Limiter (10 req/min, burst 3 req/10s per IP)          │   │
│   │  • asyncio.Lock Concurrency Mutex (prevents GPU/CPU memory thrashing) │   │
│   │  • 30-Second Bounded Queue Timeout (fails fast with HTTP 503)          │   │
│   │  • Telemetry: /health queue depth and latency profiling               │   │
│   └──────────────────────────────────┬────────────────────────────────────┘   │
│                                      │ asyncio.to_thread()                    │
│                                      ▼                                        │
│   ┌───────────────────────────────────────────────────────────────────────┐   │
│   │                  4. Neural Engine (PyTorch Singleton)                 │   │
│   │  • Fine-Tuned Qwen2.5-Coder-1.5B (Auto-resolved via KaggleHub)        │   │
│   │  • Single-pass autoregressive generation (<500ms latency)             │   │
│   │  • Multi-Device Support: CUDA, Apple MPS, or CPU                      │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ Direct Query Execution (from UI)
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                      5. Live Relational Database Targets                      │
│      ┌────────────┐     ┌────────────┐     ┌───────────┐     ┌──────────────┐ │
│      │   SQLite   │     │ PostgreSQL │     │   MySQL   │     │ Supabase API │ │
│      │  (Local)   │     │(SQLAlchemy)│     │(SQLAlchemy│     │ (REST/HTTPS) │ │
│      └────────────┘     └────────────┘     └───────────┘     └──────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Online Architecture Breakdown

### 1. Edge Routing — Cloudflare Workers

The public entry point is the Cloudflare Workers route:

`https://text-to-sql.here-2007.workers.dev`

Requests are forwarded over HTTPS into the Hugging Face Spaces runtime.

### 2. Gradio Workstation — Port `7860`

The Gradio layer acts as the user-facing workstation.

Responsibilities include:

- **3-column terminal layout** with dark/light theme support.
- **Connection Manager** supporting:
  - SQLite
  - PostgreSQL
  - MySQL
  - Supabase API
- **Zero credential leakage:** Database passwords remain inside this layer and are never transmitted to the inference backend.
- **Dialect Harmonizer:** Applies AST-level SQL rewrites such as `IFNULL → COALESCE`.

The Gradio workstation submits:

```json
{
  "question": "...",
  "schema": "..."
}
```

through:

```text
POST /v1/tosql
```

### 3. FastAPI Inference Gateway — Port `8000`

The FastAPI layer protects and controls access to model inference.

Key responsibilities:

- **SlowAPI rate limiter**
  - `10 req/min`
  - burst of `3 req/10s` per IP
- **`asyncio.Lock` concurrency mutex**
  - serializes model inference
  - avoids GPU/CPU memory thrashing
- **30-second bounded queue timeout**
  - fails fast with HTTP `503 Service Unavailable`
- **Telemetry**
  - `/health`
  - queue depth
  - latency profiling

The request is passed into the model execution layer using:

```python
asyncio.to_thread()
```

### 4. Neural Engine — PyTorch Singleton

The inference engine contains the fine-tuned model:

- **Model:** Fine-Tuned `Qwen2.5-Coder-1.5B`
- **Model loading:** Auto-resolved through KaggleHub
- **Generation:** Single-pass autoregressive generation
- **Target latency:** `<500ms`
- **Device support:**
  - CUDA
  - Apple MPS
  - CPU

### 5. Live Relational Database Targets

Generated SQL can be executed against:

- **SQLite** — local database
- **PostgreSQL** — SQLAlchemy
- **MySQL** — SQLAlchemy
- **Supabase API** — REST/HTTPS

The important architectural boundary is that **model inference and live database execution are decoupled**.

---

# 3. The 4 Core Architectural Principles

These are the four principles to emphasize during an interview.

## 3.1 Zero Credential Exposure — Security

### Why it matters

Database connection strings and passwords exist **exclusively inside the client/Gradio process**.

The FastAPI inference gateway receives only:

- the user's question
- sanitized schema DDL

Database credentials are never sent to the model-serving backend.

### Security boundary

```text
Client / Gradio
    │
    │ question + sanitized schema
    ▼
FastAPI Inference Gateway
    │
    ▼
Model
```

The model therefore has no direct access to:

- database passwords
- raw connection strings
- authentication credentials

This prevents a compromised model or prompt-injection attempt from using inference access to extract database credentials.

---

## 3.2 Deterministic Concurrency Serialization — `asyncio.Lock`

### Why it matters

LLM inference is GPU/CPU compute-bound.

If several users trigger parallel forward passes, the process can experience:

- GPU memory spikes
- CPU memory spikes
- resource contention
- container crashes
- OOM failures

The solution is an `asyncio.Lock` that serializes inference requests.

```text
Request A ──┐
Request B ──┼──► asyncio.Lock ──► Model ──► Response
Request C ──┤
Request D ──┘
```

A bounded queue is used alongside the lock:

- **Queue timeout:** 30 seconds
- If the request cannot acquire capacity within that bound:
  - return HTTP `503 Service Unavailable`

This gives predictable behavior under load instead of allowing uncontrolled concurrent inference.

---

## 3.3 Multi-Dialect AST Harmonization

### Why it matters

The model is fine-tuned primarily around **SQLite syntax**, while the production system needs to support:

- SQLite
- PostgreSQL
- MySQL
- Supabase

Instead of forcing the model to perfectly reproduce every target dialect, the architecture uses a **post-generation AST transformation layer**.

Example:

```text
SQLite-style SQL
     │
     ▼
AST Parser
     │
     ▼
Dialect Harmonizer
     │
     ├── IFNULL()  ─────► COALESCE()
     ├── strftime() ────► EXTRACT()
     └── other rewrites
     │
     ▼
Target Database SQL
```

This separates **generation** from **dialect adaptation**.

---

## 3.4 Execution Sandboxing & Safety

### Why it matters

Generated SQL should never be allowed to freely mutate production databases.

The execution layer therefore enforces two key controls:

### AST Read-Only Guardrail

Statements such as:

- `DROP`
- `DELETE`
- `UPDATE`
- `ALTER`

are rejected.

The guard operates at the AST level so destructive operations cannot simply be hidden inside:

- subqueries
- CTEs
- nested statements

### 3-Second Execution Timeout

Every execution is bounded by a **3.0-second timeout**.

This protects against expensive or pathological queries such as runaway Cartesian products.

The result is:

```text
Generated SQL
     │
     ▼
AST Safety Check
     │
     ├── destructive → REJECT
     │
     └── read-only
           │
           ▼
      3-second timeout
           │
           ▼
      Execute safely
```

---

# 4. End-to-End Request Flow

The complete online request lifecycle can be summarized as:

```text
User
 │
 ▼
Cloudflare Workers
 │ HTTPS
 ▼
Gradio Workstation
 │
 │ 1. Connect to database
 │ 2. Introspect schema
 │ 3. Sanitize credentials away
 │ 4. Collect user question
 │
 ▼
POST /v1/tosql
{question, schema}
 │
 ▼
FastAPI Gateway
 │
 ├── Rate limiting
 ├── Concurrency lock
 ├── Queue timeout
 └── Telemetry
 │
 ▼
Qwen2.5-Coder-1.5B
 │
 │ Autoregressive SQL generation
 ▼
Generated SQL
 │
 ▼
Dialect Harmonizer
 │
 │ AST-level rewrites
 ▼
Target-Dialect SQL
 │
 ▼
Read-Only AST Guard
 │
 ├── unsafe → reject
 │
 └── safe
      │
      ▼
3-second execution timeout
 │
 ▼
SQLite / PostgreSQL / MySQL / Supabase
 │
 ▼
Formatted Result
 │
 ▼
Gradio UI
```

---

# 5. Offline-to-Online Relationship

The offline and online halves are directly connected:

```text
                 OFFLINE
┌──────────────────────────────────────────┐
│ Spider Benchmark                         │
│      ↓                                   │
│ Data Processing                          │
│      ↓                                   │
│ QLoRA Training                           │
│      ↓                                   │
│ FP16 Merge                               │
│      ↓                                   │
│ Evaluation                               │
└──────────────────┬───────────────────────┘
                   │
                   │ Fine-tuned model artifact
                   ▼
              ONLINE
┌──────────────────────────────────────────┐
│ Cloudflare Edge                          │
│      ↓                                   │
│ Gradio                                   │
│      ↓                                   │
│ FastAPI Gateway                          │
│      ↓                                   │
│ Qwen2.5-Coder-1.5B                       │
│      ↓                                   │
│ Dialect Harmonizer                       │
│      ↓                                   │
│ Safety Sandbox                           │
│      ↓                                   │
│ Live Database                            │
└──────────────────────────────────────────┘
```

The core idea is:

> **Offline work produces a model artifact; online infrastructure turns that artifact into a safe, multi-database production service.**

---

# 6. 90-Second Interview Script

> **"Our project has two distinct halves: the offline ML pipeline and the online production serving architecture.**
>
> In the offline pipeline, we took Yale's Spider benchmark of 10,181 queries across 200 databases, classified them into 4 AST hardness tiers, and serialized schemas into clean SQLite DDL preserving primary and foreign keys. We filtered sequences at a 1,024-token cutoff and trained Qwen2.5-Coder-1.5B using 4-bit NF4 QLoRA on Dual NVIDIA T4 GPUs with PyTorch DDP and Paged AdamW. We then merged the LoRA weights into a standalone FP16 checkpoint that streams under 3GB of RAM on CPU.
>
> For the online serving architecture, we strictly decoupled model inference from database execution. A user connects via a Cloudflare Workers edge route to our Gradio workstation. The workstation introspects the connected database—whether SQLite, PostgreSQL, MySQL, or Supabase—and sends only the schema DDL and question to our FastAPI backend.
>
> The FastAPI gateway protects the model using SlowAPI rate limiting and an `asyncio.Lock` queue to serialize inferences, preventing GPU memory thrashing or OOM crashes under concurrent load. Once the model outputs SQL in sub-500ms, our dialect harmonizer adapts the query to the target database, and our sandbox executes it within a 3-second safety window.
>
> Everything is verified by 236 automated unit and integration tests."**

---

# 7. Interview Whiteboard Version

For a fast whiteboard explanation, draw the system as five blocks:

```text
       OFFLINE
┌───────────────┐
│ Spider Data   │
└───────┬───────┘
        ▼
┌───────────────┐
│ Data Pipeline │
└───────┬───────┘
        ▼
┌───────────────┐
│ QLoRA Train   │
└───────┬───────┘
        ▼
┌───────────────┐
│ FP16 Model    │
└───────┬───────┘
        │
        │ deploy
        ▼
        ONLINE
┌───────────────────────┐
│ Cloudflare + Gradio   │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ FastAPI Gateway       │
│ Rate Limit + Lock     │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ Qwen2.5-Coder-1.5B    │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ AST Harmonizer        │
│ + Safety Sandbox      │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ Live SQL Databases    │
└───────────────────────┘
```

The four words to emphasize verbally are:

**Security → Concurrency → Dialects → Safety**

---

# 8. Key Numbers to Memorize

| Category | Number / Value |
|---|---:|
| Spider queries | **10,181** |
| Spider databases | **200** |
| AST hardness tiers | **4** |
| Token cutoff | **1,024** |
| Base model | **Qwen2.5-Coder-1.5B** |
| Training GPUs | **2 × NVIDIA T4** |
| Quantization | **4-bit NF4** |
| Adapted projections | **7** |
| Optimizer | **8-bit Paged AdamW** |
| Merged artifact | **2.45 GB** |
| CPU merge target | **<3 GB RAM** |
| Rate limit | **10 req/min** |
| Burst limit | **3 req/10s/IP** |
| Queue timeout | **30 sec** |
| Target inference latency | **<500 ms** |
| SQL execution timeout | **3.0 sec** |
| Automated tests | **236** |

---

# 9. Core Takeaway

The project is not simply:

> **"Fine-tune an LLM to generate SQL."**

The actual engineering contribution is the combination of:

```text
Benchmark Engineering
        +
Data Processing
        +
QLoRA Fine-Tuning
        +
Model Packaging
        +
Evaluation
        +
API Architecture
        +
Concurrency Control
        +
Credential Isolation
        +
Dialect Translation
        +
SQL Safety
        =
Production-Oriented Text-to-SQL System
```

The strongest way to describe the project in an interview is therefore:

> **A fine-tuned Text-to-SQL model wrapped in a production-style architecture that deliberately separates model inference from database access, controls concurrency, prevents credential exposure, translates SQL across dialects, and sandboxes generated queries before execution.**
