# README: `mmlu_label_free_pipeline.py`

## Purpose

`mmlu_label_free_pipeline.py` runs the MMLU workflow with **label-free trust updates**.

It is designed to avoid using adversary identity when updating agent weights.

## What makes it different

Compared to the existing MMLU pipeline, this script:

- does **not** use an `is_adversary` label in weight updates,
- randomizes agent order per question (reduces position bias),
- updates weights from observed behavior only:
  - per-agent correctness vs. gold label,
  - per-agent contribution to coordinator choice.

## Method summary

For each question:

1. Ask all agents for an answer (A/B/C/D).
2. Compute coordinator answer by weighted vote.
3. Compute per-agent signals:
   - `correctness`: 1 if agent answer equals gold answer, else 0.
   - `contribution`: agent vote share among agents that supported the coordinator winner.
4. Combine signals:

`signal_i = alpha * correctness_i + (1 - alpha) * contribution_i`

5. Update weights multiplicatively:

`w_i <- w_i * exp(lr * (signal_i - mean(signal)))`

6. Normalize weights to sum to 1.

## Configuration

The script uses `RunConfig`:

- `model`: LLM model name (default: `gpt-4o-mini`)
- `n_agents`: total agents (default: 5)
- `n_adversarial`: adversaries in simulation only (default: 3)
- `n_rows`: number of MMLU rows to run (default: 10)
- `lr`: update learning rate (default: 0.5)
- `alpha`: correctness weight in behavior signal (default: 0.7)
- `seed`: reproducibility seed (default: 42)

## Inputs

Default dataset path in script:

`/Users/sanaebrahimi/Desktop/MA-LLM/data/MMLU/high_school_statistics_test.csv`

## Output

Default output file in script:

`/Users/sanaebrahimi/Desktop/MA-LLM/code/refactored/mmlu_results_label_free_10.json`

Each record includes:

- question, choices, gold solution,
- randomized agent order,
- per-agent response/parsed answer,
- per-agent correctness and contribution,
- weight before and after update,
- coordinator answer and vote totals,
- reward (`+1` if coordinator answer is correct, else `-1`).

## Run

From `refactored/`:

```bash
export OPENAI_API_KEY="<your_key>"
PYTHONPATH=. ./.miniconda/bin/conda run -n ma-llm-refactored python mmlu_label_free_pipeline.py
```

## Notes

- This script is **standalone** and does not modify existing pipeline files.
- Adversarial prompts may still be used for simulation, but trust updates remain label-free.
- If you need no adversarial simulation at all, set `n_adversarial=0` in `RunConfig`.
