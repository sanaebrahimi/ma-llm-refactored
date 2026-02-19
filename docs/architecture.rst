Architecture
============

Package Layout
--------------

The refactor organizes logic under ``ma_llm``:

- ``ma_llm.clients``
  Client wrappers for OpenAI/Ollama and judge roles.
- ``ma_llm.templates``
  Prompt templates grouped by task.
- ``ma_llm.data_io``
  Dataset readers and JSON result writer.
- ``ma_llm.parsing``
  Output parsing helpers for rewards and multiple-choice extraction.
- ``ma_llm.scoring``
  Contribution and weight update logic.
- ``ma_llm.orchestration``
  Agent initialization, collaboration turns, and communication extraction.
- ``ma_llm.pipelines``
  End-to-end dynamic and chain pipelines.
- ``ma_llm.judge_eval``
  Post-hoc evaluation pipeline for saved solution logs.
- ``ma_llm.embedding_pipeline``
  Disconnected embedding-aggregation workflow.

Data Flow
---------

1. Read task dataset from ``ma_llm.data_io``.
2. Initialize regular/adversarial agents.
3. Collect initial responses and run communication rounds.
4. Aggregate responses through coordinator prompts.
5. Score aggregated output with judge rules.
6. Recompute credibility weights for subsequent rounds.
7. Persist result artifacts as JSON.

Extensibility Points
--------------------

Common extension points:

- add a new task template in ``ma_llm.templates``,
- define task-specific scoring rule in ``ma_llm.pipelines``,
- add a task runner script in ``refactored/``.
