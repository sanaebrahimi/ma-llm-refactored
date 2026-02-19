MA-LLM Refactored Documentation
===============================

A production-style documentation site for the refactored multi-agent LLM orchestration project.

The system coordinates multiple language-model agents to solve tasks, uses a coordinator to synthesize group output, and applies a judge to score quality and update agent credibility.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      Set up the local conda environment and build the project documentation.
      :link: getting_started
      :link-type: doc

   .. grid-item-card:: Project Overview
      Understand goals, design scope, and key concepts used in the system.
      :link: project_overview
      :link-type: doc

   .. grid-item-card:: Architecture
      Explore package structure and data flow across core components.
      :link: architecture
      :link-type: doc

   .. grid-item-card:: Workflows
      Review execution paths for QA, MATH, MMLU, HumanEval, and judge evaluation.
      :link: workflows
      :link-type: doc

   .. grid-item-card:: Operations
      Runbook-style commands for running, troubleshooting, and maintaining docs.
      :link: operations
      :link-type: doc

   .. grid-item-card:: API Reference
      Full module-level API docs generated from source docstrings.
      :link: api
      :link-type: doc

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   project_overview
   architecture
   usage
   workflows
   operations
   api
