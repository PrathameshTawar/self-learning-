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
9. [Repository Structure](#repository-structure)
10. [Roadmap & Milestones](#roadmap--milestones)

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
| **Layer C** | **Model Adaptation** | Builds SFT examples from evaluator-accepted attempts (with the exact prompt the model saw) and DPO pairs from accepted vs. rejected outputs for identical prompts, using run traces. Splits are assigned per task, so no task appears in both train and test, and benchmark held-out tasks are excluded by default. | Dataset export ready (`export-dataset`); LoRA training not validated yet |

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
- **Zero Heavy Daemons**: Standalone Python architecture using standard library `ast`, SQLite, and subprocess execution.
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

## Running Tests & Linting

```powershell
# Install test & lint tools, then run unit & end-to-end tests
pip install -e ".[dev]"
pytest -v -p no:deepeval

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
│       │   └── retrieval.py      # Experience retriever with policies
│       ├── intelligence/         # AST symbol intelligence
│       │   ├── ast_symbols.py    # Python AST symbol extractor
│       │   ├── symbol_graph.py   # Cross-file symbol reference graph
│       │   └── context_builder.py# Multi-file AST context assembly
│       ├── dashboard/            # Local web telemetry dashboard
│       │   ├── server.py         # HTTP API & static server
│       │   └── static/
│       │       └── index.html    # Single-page monitoring UI
│       ├── orchestration/        # Orchestration engine
│       │   ├── planner.py        # Task planner with retry feedback & genes
│       │   ├── repair_agent.py   # Candidate proposal coordinator
│       │   └── state_machine.py  # FSM lifecycle in throwaway tempdirs
│       └── learning/             # Self-learning systems (Layers B & C)
│           ├── dataset.py        # SFT / DPO training split curator
│           ├── genetic.py        # Strategy chromosome & genetic operators
│           ├── evolution.py      # Tournament & generational strategy evaluation
│           └── registry.py       # Versioned strategy registry with promotion gate
├── docker/
│   └── sandbox.Dockerfile        # Image for the container sandbox
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
    └── e2e/
```

---

## Roadmap & Milestones

- [x] **Milestone 1: Controlled Repair Loop & Foundation** (Task runner, sandbox execution, patch application, independent evaluation, episodic memory).
- [x] **Milestone 2: Multi-File Context & AST Symbol Navigation** (Python AST symbol extraction, dependency graph, cross-file reference assembly).
- [x] **Milestone 3: Strategy Optimization Engine** (Genetic strategy evolution, prompt exploration, held-out promotion gate).
- [ ] **Milestone 4: PEFT/Q-LoRA 4-bit Fine-Tuning** (Curated dataset export, adapter training on GeForce GTX 1650).
- [ ] **Milestone 5: Production Deployment & Remote Workers** (FastAPI API, PostgreSQL/pgvector, Redis queues).
