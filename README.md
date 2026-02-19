# MA-LLM Refactored

This folder contains a structural rewrite of the original multi-agent LLM scripts while preserving the same workflow style:

- multiple agents answer a task,
- a coordinator aggregates responses,
- a judge scores the final answer,
- agent contribution updates credibility weights.

The original files remain untouched in the project root.

## What changed

- Shared logic moved into a package: `refactored/ma_llm/`.
- Task-specific scripts are now thin wrappers.
- Modules, classes, and functions include docstrings for Sphinx/autodoc.
- Sphinx documentation scaffold is included in `refactored/docs/`.

## Folder layout

- `ma_llm/clients.py`: OpenAI/Ollama/judge clients
- `ma_llm/templates.py`: prompt templates by task
- `ma_llm/data_io.py`: dataset loaders and result writer
- `ma_llm/parsing.py`: extraction/parsing helpers
- `ma_llm/scoring.py`: contribution and weight update logic
- `ma_llm/orchestration.py`: collaboration mechanics
- `ma_llm/pipelines.py`: dynamic graph + chain pipelines
- `ma_llm/judge_eval.py`: post-hoc judge evaluation
- `ma_llm/embedding_pipeline.py`: disconnected embedding-based aggregation

Compatibility runners:

- `multi-agent_coordinator.py` (QA)
- `multi_agent_coordinator_MATH.py`
- `multi_agent_coordinator_MMLU.py`
- `multi_agent_coordinator_HumanEval.py`
- `multi_agents_llm.py`
- `judge.py`

## Setup

1. Create the conda env from `environment.yml`:
   - `./.miniconda/bin/conda env create -f environment.yml`
2. Activate it:
   - `./.miniconda/bin/conda activate ma-llm-refactored`
3. Set `OPENAI_API_KEY` for OpenAI-backed judge/scoring paths.
4. Ensure local Ollama server is running on `localhost:11434`.

## Run

From `refactored/`:

- QA dynamic + chain:
  - `python multi-agent_coordinator.py`
- MATH/GSM8K dynamic + chain:
  - `python multi_agent_coordinator_MATH.py`
- MMLU dynamic + chain:
  - `python multi_agent_coordinator_MMLU.py`
- HumanEval dynamic + chain:
  - `python multi_agent_coordinator_HumanEval.py`
- Disconnected embedding mode:
  - `python multi_agents_llm.py`
- Post-hoc judge evaluation:
  - `python judge.py`

## Build docs

From `refactored/docs/`:

- `make html`

The generated HTML is in `refactored/docs/_build/html/`.
The docs use the `pydata-sphinx-theme` with structured pages for overview, architecture, workflows, operations, and full module API reference.
