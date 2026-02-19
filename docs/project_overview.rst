Project Overview
================

Purpose
-------

The MA-LLM refactor provides a maintainable implementation of a collaborative multi-agent LLM system while preserving the original workflow behavior.

Core capabilities:

- run multiple regular and adversarial agents,
- coordinate iterative communication between agents,
- synthesize final answers with a coordinator,
- evaluate output quality with a judge,
- update credibility weights based on contribution.

Design Goals
------------

- preserve existing workflow semantics,
- reduce duplication across task scripts,
- isolate task-specific prompts and logic,
- provide strong source-level documentation for Sphinx.

Task Coverage
-------------

The refactored project supports the same task families as the legacy scripts:

- factoid QA,
- mathematics/GSM8K,
- MMLU multiple-choice evaluation,
- HumanEval code completion,
- post-hoc judge-based evaluation on stored runs.

Execution Modes
---------------

Two collaboration modes are implemented:

- dynamic architecture: random agent pair communication over configurable rounds,
- chain architecture: sequential handoff where each agent sees prior output.
