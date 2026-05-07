<img width="1408" height="768" alt="Generated Image May 05, 2026 - 11_25PM" src="https://github.com/user-attachments/assets/c86f538e-5c4c-4578-883e-b37d3da9214f" />

# SetupX

**Experience-Driven Automated Environment Configuration with LLM Agents**


## Overview

**SetupX** is an LLM-powered multi-agent system that automatically configures software repository build environments inside Docker containers. Given a GitHub repository URL, SetupX spins up a container, iteratively installs dependencies, resolves errors, and configures the environment until the project's test suite can be executed successfully.

Unlike prior approaches that start each configuration from scratch, SetupX features three mutually reinforcing mechanisms:

-  **XPU (eXPerience Unit) Knowledge System** — A vector database (PostgreSQL + pgvector) that stores transferable configuration experiences. Successful fixes are extracted, deduplicated, and reused across repositories via two-layer semantic retrieval.
-  **Speculative Execution** — Docker container snapshots enable safe trial-and-rollback of past fixes, addressing the inherently irreversible nature of environment configuration.
-  **Adversarial Verification** — A Prosecutor–Judge pipeline structurally separates configuration and verification roles, preventing self-confirmation bias.

## Architecture

SetupX orchestrates repository configuration through three sequential phases:

```
Phase 1: Setup with In-Loop Verification
┌─────────────────────────────────────────────────────┐
│  Speculative Setup Agent (ReAct loop)               │
│    ├── Observe environment state                    │
│    ├── Retriever Agent → XPU two-layer retrieval    │
│    ├── LLM decision → Action selection              │
│    ├── Docker execution (with snapshot/rollback)     │
│    └── Verifier Agent → test suite verification     │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 2: Adversarial Verification
┌─────────────────────────────────────────────────────┐
│  Prosecutor Agent → investigate & file charges      │
│  Judge Agent      → verify each charge independently│
│  Verdict: guilty / not_guilty                       │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 3: Experience Extraction
┌─────────────────────────────────────────────────────┐
│  Extract transferable XPU from agent trajectory     │
│  Deduplicate & ingest into XPU library              │
└─────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SetupX** | Main orchestrator. Runs a ReAct loop with 6 action types: `SHELL_COMMAND`, `TRY_XPU_SUGGESTION`, `SET_ENV`, `ROLLBACK_ENV`, `VERIFY`, `FINISH`. |
| **RetrieverAgent** | Sub-agent for XPU knowledge retrieval. Layer 1: vector coarse filtering (pgvector cosine similarity, top-N). Layer 2: LLM re-ranking for precise matching. Also performs delayed audit of previously used XPUs. |
| **VerifierAgent** | Read-only sub-agent that runs the project's test suite (`pytest`) and distinguishes setup-induced failures from inherent project issues. |
| **ProsecutorAgent** | Adversarial investigator. Has container access for evidence gathering, files charges with concrete evidence. |
| **JudgeAgent** | Independent adjudicator. Verifies each charge with 1–2 targeted commands. Renders final verdict. |
| **EnvironmentManager** | Docker container lifecycle management: create, execute, snapshot (`docker commit`), rollback (stack-based LIFO). |
| **XPU Vector Store** | PostgreSQL + pgvector backend. Stores embeddings via `text-embedding-3-small`, supports composite scoring with telemetry-based tier boosting. |

## Prerequisites

- Python 3.10+
- Docker (with daemon running)
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- OpenAI-compatible LLM API access

## Installation

```bash
git clone https://github.com/Zi-hang-Zhou/setUpAgentOurs.git
cd setUpAgentOurs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | LLM API key |
| `OPENAI_BASE_URL` | LLM API base URL (OpenAI-compatible) |
| `MODEL_NAME` | LLM model name |
| `dns` | PostgreSQL connection string (for XPU vector store) |
| `EMBEDDING_API_KEY` | Embedding API key (optional, falls back to `OPENAI_API_KEY`) |
| `DOCKER_BASE_IMAGE` | Base Docker image (default: `python:3.10-bookworm`) |

## Usage

### Single Repository

```bash
.venv/bin/python -m src.main https://github.com/owner/repo --max-steps 50
```

Options:
- `--max-steps N` — Maximum number of agent steps (default: 50)
- `--phase1-timeout S` — Phase 1 timeout in seconds (default: 1800)
- `--no-xpu` — Disable XPU knowledge retrieval (ablation mode)

### Batch Benchmark

```bash
.venv/bin/python experiment/ours/run_benchmark_ours.py \
    --repo-list data/python329.jsonl \
    --output-dir experiment/results_benchmark329 \
    --parallelism 5 \
    --phase1-timeout 1800
```

### Import XPU Knowledge Base

```bash
.venv/bin/python scripts/import_xpu_jsonl.py xpu_final.jsonl --clear
```

## Project Structure

```
setUpAgentOurs/
├── src/
│   ├── main.py                 # Entry point (3-phase pipeline)
│   ├── agent.py                # SpeculativeSetupAgent
│   ├── retriever_agent.py      # RetrieverAgent (two-layer XPU retrieval)
│   ├── verifier_agent.py       # VerifierAgent (test suite runner)
│   ├── prosecutor_agent.py     # ProsecutorAgent (adversarial investigation)
│   ├── judge_agent.py          # JudgeAgent (charge verification)
│   ├── environment_manager.py  # Docker container management
│   ├── llm_engine.py           # LLM API interface
│   ├── xpu_client.py           # XPU client hierarchy
│   ├── config.py               # Centralized configuration
│   ├── models.py               # Data structures
│   └── xpu/
│       ├── xpu_vector_store.py # pgvector database operations
│       ├── xpu_adapter.py      # XPU data structures & atom rendering
│       ├── xpu_dedup.py        # LLM-driven experience deduplication
│       ├── extract_xpu_from_trajs_mvp.py  # Trajectory → XPU extraction
│       └── online_xpu_extractor.py        # Online extraction pipeline
├── data/
├── experiment/                 # Experiment results & comparative analysis
├── scripts/                    # Utility scripts
├── paper/                      # LaTeX source for the paper
└── log/                        # Runtime logs
```




## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
