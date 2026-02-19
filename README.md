# MA-LLM Refactored

This folder contains a structural rewrite of the original multi-agent LLM scripts while preserving the same workflow style:

- multiple agents answer a task,
- a coordinator aggregates responses,
- a judge scores the final answer,
- agent contribution updates credibility weights.

The original files remain untouched in the project root.

## Paper

This codebase is a refactored implementation of:

**An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring**  
Sana Ebrahimi, Mohsen Dehghankar, Abolfazl Asudeh (IJCNLP-AACL 2025)

- Anthology page: `https://aclanthology.org/2025.ijcnlp-long.90/`
- PDF: `https://aclanthology.org/2025.ijcnlp-long.90.pdf`

## Implementation mapping to paper

- Team formation and communication topology:
  - dynamic/random links and chain topology in `ma_llm/pipelines.py`
- CrS-aware aggregation (credibility-weighted coordination):
  - aggregation and coordinator prompts in `ma_llm/pipelines.py`
- Contribution score (CSc):
  - LLM-as-Judge contribution estimation in `ma_llm/scoring.py`
  - optional embedding/Shapley-style disconnected path in `ma_llm/embedding_pipeline.py`
- Credibility score updates:
  - iterative CrS update step in `ma_llm/scoring.py` (`update_weights`)
- Multi-task evaluation setup (MMLU, MATH/GSM8K, HumanEval, QA):
  - dataset readers in `ma_llm/data_io.py`
  - task runners in top-level scripts under `refactored/`

The refactor keeps the paper’s core iterative structure:
- query -> agent responses -> coordinator output -> reward
- contribution attribution -> credibility update -> next query

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

## Citation

```bibtex
@inproceedings{ebrahimi-etal-2025-adversary,
    title = "An Adversary-Resistant Multi-Agent {LLM} System via Credibility Scoring",
    author = "Ebrahimi, Sana and Dehghankar, Mohsen and Asudeh, Abolfazl",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    pages = "1676--1693",
    url = "https://aclanthology.org/2025.ijcnlp-long.90/",
}
```
