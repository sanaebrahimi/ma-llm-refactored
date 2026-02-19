Workflows
=========

Coordinator Runners
-------------------

Run from the ``refactored`` directory.

.. code-block:: bash

   python multi-agent_coordinator.py
   python multi_agent_coordinator_MATH.py
   python multi_agent_coordinator_MMLU.py
   python multi_agent_coordinator_HumanEval.py

Each runner configures:

- number of agents and adversaries,
- communication graph style,
- task dataset loader,
- output JSON artifact naming.

Disconnected Embedding Workflow
-------------------------------

.. code-block:: bash

   python multi_agents_llm.py

This path uses embedding-based response aggregation rather than iterative inter-agent communication.

Post-hoc Judge Evaluation
-------------------------

.. code-block:: bash

   python judge.py

This workflow re-evaluates previously saved multi-agent outputs.

Output Artifacts
----------------

All workflows serialize per-sample records containing:

- problem and reference fields,
- per-agent response/history blocks,
- coordinator output,
- reward value,
- contribution/weight snapshots when applicable.
