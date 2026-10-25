# EvoAgent: A Self-Learning Autonomous Coding Agent

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/PrathameshTawar/self-learning-/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathameshTawar/self-learning-/actions)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Monolith-orange.svg)](#system-architecture)

> **EvoAgent** is an agentic software engineering system with a test-verified learning loop. EvoAgent observes test failures, plans hypotheses, patches code in isolated disposable environments, independently evaluates test outcomes, and persists verified lessons into BM25 episodic memory. A fixed train/held-out benchmark split measures whether that memory actually helps on unseen bugs (see [Results](#results)).

---

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Three Learning Layers](#three-learning-layers)
3. [System Architecture](#system-architecture)
4. [Task State Machine](#task-state-machine)
5. [Hardware-Conscious Design (GTX 1650 / 8GB RAM)](#hardware-conscious-design)
6. [Quickstart Guide](#quickstart-guide)
7. [Running Benchmarks & Held-Out Ablations](#running-benchmarks--held-out-ablations)
8. [Results](#results)
9. [Fine-Tuning a Q-LoRA Adapter (Layer C)](#fine-tuning-a-q-lora-adapter-layer-c)
10. [Production Deployment & Remote Workers](#production-deployment--remote-workers)
11. [Repository Structure](#repository-structure)
12. [Roadmap & Milestones](#roadmap--milestones)

---

## Core Concepts

A self-learning coding agent executes a disciplined lifecycle:

```mermaid
graph TD
    A[Observe] -->|Read code, stack traces, test results| B[Plan]
    B -->|Form hypotheses & retry feedback| C[Act]
    C -->|Apply SEARCH/REPLACE in throwaway tempdir| D[Evaluate]
    D -->|Independent regression & unit tests| E{Success?}
    E -->|Yes| F[Learn]
    E -->|No| G[Retry with Feedback]
    G --> B
    F -->|Store verified repair & real hypothesis| H[Improve]
    H -->|BM25 retrieval over error signatures| B
```

### Core Design Principles
1. **The evaluator is the sole authority**: The coding agent cannot declare its own fix successful. Only independent, reproducible test executions can validate patches.
2. **Reproduction gate**: Tasks where the targeted test already passes pre-patch, or never runs (bad path or `-k` filter), are rejected as `INVALID_TASK`.
3. **The sandbox runs in throwaway copies**: Tasks execute inside isolated temporary directories (`tempfile.TemporaryDirectory`). Accepted patches and failed attempts never pollute or corrupt the original benchmark or host repository.
4. **Reliable patch format for small models**: Prefers exact `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks. A SEARCH must match whole lines in exactly one place; ambiguous or missing matches are rejected and fed back to the next attempt. Unified diffs are applied hunk by hunk using their context lines.
5. **Evidence-based learning**: Memory is built from the train split only and frozen while the held-out split (`benchmarks/split.json`) is evaluated, so a task can never retrieve its own solution. Strategy champions are promoted on held-out fitness, not the score they were selected with.
6. **Fault localization**: The defective file is located from the failing test's traceback (deepest non-test frame) or, for assertion failures, from the modules the test imports. Benchmarks can be run with `--no-file-hints` to drop the `target_file` hint every task carries.
7. **Traceable telemetry**: Every episode records the strategy ID, failure signature, patch, model hypothesis, outcome, and run telemetry (duration, attempts, patch validity, regression results). Every run also writes a JSON trace to `logs/traces/` with each attempt's full prompt, raw model output, patch, evaluation, and the feedback given to the next attempt. Prompts are deterministic: per-run noise such as pytest timings is stripped.

---

## Three Learning Layers

| Layer | Type | Mechanism | Status |
|---|---|---|---|
| **Layer A** | **Episodic Memory** | Stores failure signatures, root causes, SEARCH/REPLACE patches, and verified outcomes. Injects only verified lessons into future planning sessions via Okapi BM25 ranking. | **Implemented & Tested**; no measured benefit yet (see Results) |
| **Layer B** | **Strategy Optimization** | Genetic evolution of hyperparameters (`prompt_style`, `retrieval_policy`, `retrieval_top_k`, `max_repair_attempts`, `max_patch_lines`, `temperature`, `enable_multi_file_context`), evaluated against frozen memory on the train split and promoted only if held-out fitness beats the active strategy by a fixed margin. | **Implemented & Tested** (no LLM results yet) |
| **Layer C** | **Model Adaptation** | Builds SFT examples from evaluator-accepted attempts (with the exact prompt the model saw) and DPO pairs from accepted vs. rejected outputs for identical prompts, using run traces. Splits are assigned per task, so no task appears in both train and test, and benchmark held-out tasks are excluded by default. | Export and Q-LoRA training implemented (`export-dataset`, `train-adapter`); training loop verified on CPU only, no GPU results yet |

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                                  USER / CLI                                      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        Orchestration & Task State Machine                         |
|  - Task lifecycle management (QUEUED -> ANALYZING -> PLANNED -> PATCHING ...)    |
|  - Isolated execution in throwaway temp directories (no repo corruption)          |
|  - Pre-patch test verification gate (rejects pre-passing tests as INVALID_TASK)    |
|  - Retry feedback injection (supplies failed patches and errors to next attempt)  |
+-----------------------------------------------------------------------------------+
        |                           |                           |
        v                           v                           v
+--------------------+      +--------------------+      +--------------------+
|   Analysis Tools   |      |  Pluggable Models  |      |   Memory & Store   |
| - Git inspection   |      | - OpenAI-compatible|      | - Okapi BM25 search|
| - AST symbols (ast)|      |   (Ollama/vLLM)    |      | - SQLite persistent|
| - Code search      |      | - Test Mock        |      | - In-memory test   |
| - SEARCH/REPLACE   |      |   (Heuristic Model)|      |   fixtures         |
+--------------------+      +--------------------+      +--------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                            Isolated Execution Sandbox                             |
|  - Strict environment variable allowlist (blocks host secret leaking)             |
|  - Argument vector execution without shell=True                                   |
|  - POSIX resource boundaries on CPU/memory where supported                        |
|  - Timeouts kill the command's whole process tree                                 |
|  - Optional Docker backend: no network, read-only root, only the workspace mounted |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                               Independent Evaluator                               |
|  - Pre-patch test verification (must fail initially)                              |
|  - Post-patch test verification (must pass targeted test)                         |
|  - Regression suite validation (must not break existing tests)                     |
+-----------------------------------------------------------------------------------+
```

---

## Hardware-Conscious Design

Configured specifically for local developer machines (e.g. **NVIDIA GeForce GTX 1650 4 GB VRAM / 8–16 GB RAM**):
- **Zero Heavy Daemons by default**: Standalone Python architecture using standard library `ast`, SQLite, and subprocess execution. PostgreSQL, Redis and the API are opt-in (Milestone 5).
- **Small Model Optimized**: Tailored for 1.5B–7B coding models (e.g. `qwen2.5-coder:1.5b` via Ollama) using exact SEARCH/REPLACE blocks.
- **Offline Test Mock**: Includes `HeuristicCodingModel` strictly as a fast, deterministic mock for unit tests and CI without needing an LLM.

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.11+ (Python 3.12 recommended)
- Git

### 2. Setup Virtual Environment
```powershell
# In PowerShell:
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### 3. Run Autonomous Repair
```powershell
# Run repair on a specific benchmark task
python -m evoagent.cli run --task benchmarks/tasks/bug_001_parser/task.json
```

### 4. Launch Live Dashboard
```powershell
# Open browser-based telemetry dashboard
python -m evoagent.cli dashboard --port 8080
```

---

## Running Benchmarks & Held-Out Ablations

### Run Full Benchmark Suite
```powershell
python benchmarks/run_benchmarks.py --split all
```

### Run Train vs. Held-Out Split
```powershell
# Run only train tasks
python benchmarks/run_benchmarks.py --split train

# Run only held-out tasks
python benchmarks/run_benchmarks.py --split held_out
```

### Run 3-Way Self-Learning Ablation Table
Evaluate held-out tasks under 3 controlled conditions to measure learning transfer:
```powershell
# With a real model, repeated 5 times (recommended)
python benchmarks/run_benchmarks.py --eval-comparison --provider ollama --model qwen2.5-coder:1.5b --trials 5

# With the offline heuristic mock (pipeline check only)
python benchmarks/run_benchmarks.py --eval-comparison
```

Outputs the comparative ablation table and writes `data/ablation_results.json` (shown in the dashboard):
- **Condition A (Zero-Shot Baseline)**: No memory, 1 attempt.
- **Condition B (Feedback Loop)**: No memory, 3 attempts with retry feedback.
- **Condition C (Train-Split Memory)**: Memory learned on the train split only, frozen during held-out evaluation, plus retry feedback.

The memory effect is **C - B**; A to B isolates retry feedback. With `--trials N` the whole protocol runs N times and the table shows mean ± sd. B and C are compared per task and trial with an exact McNemar test, and the runner says when the difference is not statistically significant. Per-attempt traces go to `logs/traces/` (`--trace-dir ""` disables them).

### Benchmark Design
Every held-out task has a train task with the same defect class but different code, so episodic memory has something to transfer:

| Defect class | Train task | Held-out task |
|---|---|---|
| empty-collection-guard | `bug-001-parser` | `bug-009-latest-event` |
| none-guard | `bug-002-multifile` | `bug-004-dict-merge` |
| off-by-one-bound | `bug-003-binary-search` | `bug-010-moving-average` |
| boundary-initialization | `bug-007-max-subarray` | `bug-005-levenshtein` |
| inverted-comparison | `bug-008-cache-ttl` | `bug-006-rate-limiter` |

Issue texts describe symptoms only, and task sources contain no hints about the fix.

### QuixBugs Benchmark (40 tasks)
`benchmarks/quixbugs/` holds the 40 [QuixBugs](https://github.com/jkoppel/QuixBugs) Python programs (MIT license) converted into EvoAgent tasks by `benchmarks/import_quixbugs.py`. Each task was checked to fail before and pass after QuixBugs' reference fix, which is not included. Defect classes are derived from the buggy-vs-correct diff, and the 24/16 split gives every held-out task a same-class train task.
```powershell
python benchmarks/run_benchmarks.py --benchmark-root benchmarks/quixbugs --eval-comparison --provider ollama --model qwen2.5-coder:1.5b --trials 3
```

### Useful Runner Flags
- `--no-file-hints`: the agent must find the defective file itself.
- `--sandbox docker`: run all candidate code in the container sandbox (see below).
- `--request-timeout 300`: allow slow CPU-only inference (default 120 s, or `EVOAGENT_REQUEST_TIMEOUT_SECONDS`).
- `--trace-dir DIR`: where per-attempt traces go (`""` disables them).

### Container Sandbox
For model-generated code, run tests in throwaway containers with no network and only the task workspace mounted:
```powershell
docker build -t evoagent-sandbox -f docker/sandbox.Dockerfile .
python benchmarks/run_benchmarks.py --eval-comparison --sandbox docker   # or EVOAGENT_SANDBOX_BACKEND=docker
```

---

## Results

### Real model: Qwen2.5-3B-Instruct on CPU
Measured with `python benchmarks/run_benchmarks.py --eval-comparison --provider openai_compatible --trials 3` on the bundled 5 train / 5 held-out split, with file hints on.

- **Model and runtime:** `ai/qwen2.5:3B-Q4_K_M` (GGUF from Docker Hub's model catalog), served by llama.cpp (`llama-cpp-python` 0.3.16) on 4 CPU cores at temperature 0.2.
- **Not the recommended model:** the recommended `qwen2.5-coder:1.5b` could not be downloaded in the environment used.

| Condition | Held-out resolution (mean ± sd) | Per trial | Avg attempts |
|---|---|---|---|
| A: Zero-shot, no memory | 26.7% ± 23.1 | 40% / 0% / 40% | 1.0 |
| B: Retry feedback, no memory | 26.7% ± 11.5 | 20% / 40% / 20% | 2.6 |
| C: Train-split memory + feedback | 20.0% ± 20.0 | 40% / 20% / 0% | 2.7 |

**Memory effect (C - B): -6.7 points.** Pairing B and C task by task and trial by trial, 2 tasks were solved only by C and 3 only by B; the exact McNemar p-value is 1.0. **There is no evidence that memory or retry feedback helps this model on this benchmark.** Raw numbers and two example traces are in `benchmarks/results/qwen2.5-3b-cpu/`.

**What the traces show** (60 runs, 128 attempts):

| Attempt outcome | Share |
|---|---|
| SEARCH text not in the file | 27% |
| Targeted test still fails | 27% |
| No patch extracted | 21% |
| Accepted | 14% |
| Patch broke syntax | 6% |
| Fixed the target, broke the regression suite | 5% |

- **The bottleneck is producing a patch at all.** About half of all attempts never applied (no patch extracted, or SEARCH text not in the file).
- **Memory sometimes helped.** When a held-out task's same-class train task had been solved, retrieval usually ranked its lesson first. In trial 1 this preceded fixes for `bug-004` and `bug-009` (see `trace_memory_guided_success_bug-004.json`), but trials 2 and 3 did not repeat it.
- **Memory also added noise.** Retrieval also returned unrelated lessons, and "caution" entries from failed train runs that contain no fix. Since this run, only verified lessons are retrieved.
- **The training data is real but small.** These traces yield 7 SFT rows from 4 train tasks and 1 DPO pair, with benchmark held-out tasks excluded. Examples are 562 to 1060 tokens long, so none fits the old 512-token training limit.

With 5 held-out tasks, a single task moves the rate by 20 points. The next meaningful run is the 16-task QuixBugs held-out split with the coder model on a GPU, over several trials.

### Offline mock (pipeline check)
The heuristic mock scores 0/5 on held-out tasks in every condition. That is expected: its hard-coded rules only match two seeded train bugs. A positive-control test (`tests/e2e/test_ablation_runner.py`) checks that the runner does report a memory effect when one exists.

---

## Fine-Tuning a Q-LoRA Adapter (Layer C)

The adapter is trained on evaluator-accepted repairs only, with the exact prompts the model saw.

```powershell
# 1. Collect traces (train split only; held-out tasks are excluded from the export anyway)
python benchmarks/run_benchmarks.py --split train --provider ollama --model qwen2.5-coder:1.5b

# 2. Export SFT/DPO data plus manifest.json (rows and tasks per split, file hashes)
evoagent export-dataset

# 3. Train (install the CUDA build of torch first, then: pip install -e ".[training]")
evoagent train-adapter

# 4. Merge, convert to GGUF, and serve it like any other model
evoagent merge-adapter
python llama.cpp/convert_hf_to_gguf.py model_weights/evoagent_merged --outfile evoagent-ft.gguf --outtype q8_0
llama-server -m evoagent-ft.gguf --port 8081

# 5. Measure it: same held-out ablation, fine-tuned model vs. the base model
python benchmarks/run_benchmarks.py --eval-comparison --provider openai_compatible --model evoagent-ft --trials 5
```
To serve the merged model with Ollama instead, print the base model's Modelfile (`ollama show qwen2.5-coder:1.5b --modelfile`), point its `FROM` line at the GGUF file, and run `ollama create evoagent-ft -f Modelfile`, so the chat template stays the same.

**Choices made for a 4 GB GTX 1650 (Turing, compute capability 7.5):**

| Choice | Why |
|---|---|
| Base model `Qwen2.5-Coder-1.5B-Instruct` | The repair loop is served the instruct model (Ollama's `qwen2.5-coder` tags), so training must use the same weights and chat template. The previous default was the base (non-instruct) model. |
| Chat template, loss on the reply only | Prompts are rendered with `apply_chat_template` exactly as at inference. The old `### Instruction:` format matched nothing the model sees when served. |
| NF4 4-bit + double quantization, fp16 compute | Turing has no bf16. If fp16 losses turn NaN, use `--precision fp32`. |
| LoRA r=16, alpha=32 on all 7 linear projections | QLoRA results show that adapting every linear layer, not only q/v, is needed to match full fine-tuning. |
| Logits only at reply positions | Qwen's 152k-token vocabulary makes full-sequence fp32 logits about 0.9 GB at 1536 tokens. Projecting only the ~100-300 reply positions removes most of that. A test checks that the loss equals the standard one. |
| `max_seq_length` 1536, over-long examples skipped | Measured trace prompts are 562-1060 tokens, so the old 512 limit dropped every example. Truncation would cut the reply, so long examples are skipped and reported. |
| Batch 1 × 8 gradient accumulation, gradient checkpointing, paged 8-bit AdamW, SDPA attention | Keeps activations and optimizer state small. FlashAttention 2 needs compute capability 8.0+. |
| Splits by task, val loss each epoch | `val` tasks are never trained on. Held-out benchmark tasks are excluded at export. |

The export scrubs credential-looking strings (API keys, tokens, private keys) and removes duplicate rows. Repeated trials with deterministic prompts would otherwise over-weight the same example. `training_summary.json` records the config, dataset hash, token-length distribution, losses, GPU and library versions.

**Status:** the training loop, loss masking, adapter saving and merging are verified on CPU with a tiny random Qwen2 model (`tests/unit/test_finetune.py`, run in CI). It has not been run on a GTX 1650 yet, and no fine-tuned model has been evaluated. The last measured run yielded only 7 SFT rows, far too few for a measurable effect. Collect a few hundred accepted repairs first (the QuixBugs train split over several trials, or remote workers). DPO is not implemented: the same run produced 1 preference pair.

---

## Production Deployment & Remote Workers

```
 client ──HTTPS──> FastAPI (evoagent serve) ──XADD job id──> Redis stream ──XREADGROUP──> worker 1..N
                        │                                                                   │
                        └──────────── PostgreSQL: repair_jobs, memory_episodes (pgvector) ──┘
                                                                         workers ──> model server (Ollama / llama.cpp)
                                                                         workers ──> docker sandbox per test run
```

- **Postgres holds job state; Redis only delivers job ids.** The API writes the job row before enqueueing, so a worker never gets an unknown id. A job is acknowledged only after its result is stored.
- **Crash recovery.** Redis Streams keep unacknowledged deliveries pending. While a job runs, its worker sends heartbeats. If the worker dies, `XAUTOCLAIM` hands the job to another worker once `EVOAGENT_QUEUE_VISIBILITY_TIMEOUT_SECONDS` passes without one. The heartbeat is a Lua script that checks ownership atomically, so a slow worker cannot take back a job that was already reclaimed. A job that has killed its worker `EVOAGENT_QUEUE_MAX_DELIVERIES` times is marked `error` and not retried again. Redeliveries of finished or cancelled jobs are acknowledged without running.
- **Shared memory.** Every worker reads and writes the same `memory_episodes` table, so a repair verified on one worker becomes a lesson for all of them. Retrieval first takes a shortlist from a partial HNSW index over verified episodes, then BM25 ranks it exactly as with SQLite. Switching backends changes how candidates are found, not how lessons are ranked. The default embedder is dependency-free feature hashing; `EVOAGENT_EMBEDDING_PROVIDER=openai_compatible` uses an `/embeddings` endpoint such as Ollama's `nomic-embed-text`. Jobs with `"learn": false` (held-out evaluation) do not write to memory.
- **Traces feed Layer C.** Each job stores its run trace. `evoagent export-traces` writes them to disk for `export-dataset`, so remote workers produce training data.

### Run it
```bash
# Single machine, no Redis/Postgres: in-process queue and an embedded worker (local development)
evoagent serve --embedded-worker --insecure-no-auth

# Full stack: API, workers, Postgres + pgvector, Redis (see the header of docker-compose.yml)
docker build -t evoagent-sandbox -f docker/sandbox.Dockerfile .
export EVOAGENT_API_TOKENS=$(openssl rand -hex 24) POSTGRES_PASSWORD=$(openssl rand -hex 16)
export DOCKER_GID=$(stat -c %g /var/run/docker.sock)
mkdir -p /tmp/evoagent-work && sudo chown 1000:1000 /tmp/evoagent-work
docker compose up -d --build --scale worker=2
```
Workers on other machines need the same image, `EVOAGENT_DATABASE_URL`, `EVOAGENT_REDIS_URL` and a model endpoint, and run `evoagent worker`. With one GPU, one worker per model server is the right number: Ollama on a GTX 1650 serves one request at a time.

### API
All `/v1` routes need `Authorization: Bearer <token>`; `/healthz` and `/readyz` (checks Postgres and Redis) do not.

| Method & path | Purpose |
|---|---|
| `POST /v1/jobs` | Submit a job: `files` (path → content, up to `EVOAGENT_MAX_UPLOAD_BYTES`) plus `targeted_test`, or `benchmark_task` (a task on the workers). Returns `201` with the job id. |
| `GET /v1/jobs/{id}` | Status (`queued`, `running`, `succeeded`, `failed`, `error`, `cancelled`) and the result: final state, accepted patch, per-attempt outcome. |
| `GET /v1/jobs/{id}/trace` | Full trace: every prompt, raw model output, and evaluation. |
| `GET /v1/jobs?status=…` | Recent jobs. |
| `POST /v1/jobs/{id}/cancel` | Cancel a queued job (`409` once running). |
| `GET /v1/stats` | Job counts, queue depth, memory size. |
| `GET /v1/memory/search?q=…` | Lessons the planner would retrieve for a failure text. |

```bash
curl -X POST localhost:8000/v1/jobs -H "Authorization: Bearer $EVOAGENT_API_TOKENS" -H "Content-Type: application/json" \
  -d '{"task_id": "bug-001", "benchmark_task": "tasks/bug_001_parser"}'
```

### Security
A job makes a worker run the submitted repository's tests, which means running arbitrary code. Therefore:
- The API refuses to start without `EVOAGENT_API_TOKENS` (tokens are compared in constant time). Compose binds it to `127.0.0.1`; expose it only behind a TLS reverse proxy.
- Workers refuse uploaded repositories unless `EVOAGENT_SANDBOX_BACKEND=docker`: a throwaway container per test run, with no network, a read-only root filesystem, and only the workspace mounted. `EVOAGENT_WORKER_ALLOW_LOCAL_SANDBOX=true` overrides this for trusted callers only.
- Uploaded paths must be relative with no `..`. `benchmark_task` must resolve, after following symlinks, inside `EVOAGENT_WORKER_BENCHMARK_ROOTS`.
- Mounting the Docker socket gives the worker container root-equivalent control of its host. Run workers on dedicated machines, or use a rootless Docker daemon or a gVisor (`runsc`) runtime for sandbox containers.

**Status:** the API, Postgres job store, pgvector memory and Redis queue are tested against real PostgreSQL 16 + pgvector and Redis 7 (`tests/integration`, run in CI with service containers). A multi-process run was also checked by hand: API, two workers, one worker killed with SIGKILL mid-job and its job reclaimed by the other, and graceful SIGTERM shutdown. Still missing: a production load test, metrics export (Prometheus), and schema migrations beyond create-if-missing.

---

## Running Tests & Linting

```powershell
# Install test & lint tools, then run unit & end-to-end tests
pip install -e ".[dev,server]"
pytest -v -p no:deepeval

# Integration tests need disposable PostgreSQL (with pgvector) and Redis instances
$env:EVOAGENT_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/evoagent_test"
$env:EVOAGENT_TEST_REDIS_URL="redis://localhost:6379/0"
pytest -v -p no:deepeval tests/integration

# Run linter
ruff check src tests benchmarks
```

---

## Repository Structure

```
├── .gitignore
├── LICENSE                       # MIT License
├── pyproject.toml
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions test matrix (Python 3.11, 3.12)
├── src/
│   └── evoagent/
│       ├── config.py             # Configuration & environment variables
│       ├── cli.py                # Command-line interface
│       ├── domain/               # Domain models & states (Pydantic v2)
│       │   ├── models.py
│       │   └── states.py
│       ├── tools/                # Analysis tools
│       │   ├── git_tools.py      # Git status & diff inspection
│       │   ├── search_tools.py   # Code search & file navigation
│       │   ├── patch_tools.py    # SEARCH/REPLACE & unified diff applicator
│       │   └── test_tools.py     # Pytest execution & error extraction
│       ├── sandbox/              # Process isolation & execution runners
│       │   ├── interface.py
│       │   ├── local_runner.py   # Environment allowlisting & process sandbox
│       │   └── docker_runner.py  # Container sandbox (no network, workspace-only mount)
│       ├── models/               # Pluggable inference backends
│       │   ├── base.py
│       │   ├── open_model.py     # OpenAI / Ollama / vLLM SEARCH/REPLACE backend
│       │   ├── factory.py        # Provider selection shared by CLI and benchmarks
│       │   └── heuristic_model.py# Offline test mock fixture
│       ├── evaluation/           # Independent evaluation authority
│       │   ├── evaluator.py
│       │   └── metrics.py
│       ├── memory/               # Episodic memory & retrieval (Layer A)
│       │   ├── bm25.py           # Okapi BM25 ranking algorithm
│       │   ├── schemas.py
│       │   ├── store.py          # SQLite & in-memory stores
│       │   ├── postgres_store.py # Shared PostgreSQL + pgvector store
│       │   ├── embeddings.py     # Feature-hashing & OpenAI-compatible embedders
│       │   └── retrieval.py      # Experience retriever with policies
│       ├── storage/
│       │   └── postgres.py       # Connection pool & advisory-locked schema setup
│       ├── service/              # Production service (Milestone 5)
│       │   ├── api.py            # FastAPI job API
│       │   ├── jobs.py           # Job request/record models & path validation
│       │   ├── job_store.py      # In-memory & PostgreSQL job stores
│       │   ├── queue.py          # In-memory & Redis Streams queues with reclaim
│       │   ├── worker.py         # Remote repair worker
│       │   └── runtime.py        # Builds stores, queue & worker from settings
│       ├── intelligence/         # AST symbol intelligence
│       │   ├── ast_symbols.py    # Python AST symbol extractor
│       │   ├── symbol_graph.py   # Cross-file symbol reference graph
│       │   └── context_builder.py# Multi-file AST context assembly
│       ├── dashboard/            # Local web telemetry dashboard
│       │   ├── server.py         # HTTP API & static server
│       │   └── static/
│       │       └── index.html    # Single-page monitoring UI
│       ├── orchestration/        # Orchestration engine
│       │   ├── factory.py        # State machine wiring shared by CLI & workers
│       │   ├── planner.py        # Task planner with retry feedback & genes
│       │   ├── repair_agent.py   # Candidate proposal coordinator
│       │   └── state_machine.py  # FSM lifecycle in throwaway tempdirs
│       └── learning/             # Self-learning systems (Layers B & C)
│           ├── dataset.py        # SFT / DPO training split curator
│           ├── finetune.py       # Q-LoRA adapter training & merging (GTX 1650)
│           ├── genetic.py        # Strategy chromosome & genetic operators
│           ├── evolution.py      # Tournament & generational strategy evaluation
│           └── registry.py       # Versioned strategy registry with promotion gate
├── docker-compose.yml            # API, workers, Postgres + pgvector, Redis
├── docker/
│   ├── app.Dockerfile            # Image for the API and workers
│   └── sandbox.Dockerfile        # Image for the container sandbox
├── training/                     # Script wrappers for export & adapter training
├── benchmarks/                   # Benchmark suite & evaluation splits
│   ├── split.json                # Fixed train vs. held-out task partition
│   ├── run_benchmarks.py         # Multi-condition evaluation runner
│   ├── import_quixbugs.py        # Converts QuixBugs into EvoAgent tasks
│   ├── quixbugs/                 # 40 QuixBugs tasks, split, LICENSE (MIT)
│   └── tasks/
│       ├── bug_001_parser/       # [train] IndexError on empty input
│       ├── bug_002_multifile/    # [train] AttributeError on None across service/validator
│       ├── bug_003_binary_search/# [train] Off-by-one boundary index defect
│       ├── bug_004_dict_merge/   # [held-out] NoneType attribute error in dict merge
│       ├── bug_005_levenshtein/  # [held-out] Dynamic programming boundary condition
│       ├── bug_006_rate_limiter/ # [held-out] Sliding window filter inequality bug
│       ├── bug_007_max_subarray/ # [train] Accumulator initialized to the wrong value
│       ├── bug_008_cache_ttl/    # [train] Inverted freshness comparison
│       ├── bug_009_latest_event/ # [held-out] IndexError on empty event list
│       └── bug_010_moving_average/# [held-out] Off-by-one window range
└── tests/                        # Comprehensive unit & E2E tests
    ├── unit/
    ├── e2e/
    └── integration/              # Real PostgreSQL/pgvector & Redis (opt-in via env vars)
```

---

## Roadmap & Milestones

- [x] **Milestone 1: Controlled Repair Loop & Foundation** (Task runner, sandbox execution, patch application, independent evaluation, episodic memory).
- [x] **Milestone 2: Multi-File Context & AST Symbol Navigation** (Python AST symbol extraction, dependency graph, cross-file reference assembly).
- [x] **Milestone 3: Strategy Optimization Engine** (Genetic strategy evolution, prompt exploration, held-out promotion gate).
- [x] **Milestone 4: PEFT/Q-LoRA 4-bit Fine-Tuning** (Curated dataset export, adapter training on GeForce GTX 1650). Training loop verified on CPU; GPU run and evaluation of a fine-tuned model still to do.
- [x] **Milestone 5: Production Deployment & Remote Workers** (FastAPI API, PostgreSQL/pgvector, Redis queues).
