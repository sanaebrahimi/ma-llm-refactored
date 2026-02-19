Getting Started
===============

This section provides the minimum steps to run the refactored project and render documentation locally.

Prerequisites
-------------

- macOS/Linux shell
- Local Ollama runtime (for agent/coordinator execution paths)
- OpenAI API key for OpenAI-backed judge flows

Environment Setup
-----------------

Create the environment from ``refactored/environment.yml``:

.. code-block:: bash

   cd refactored
   ./.miniconda/bin/conda env create -f environment.yml
   ./.miniconda/bin/conda activate ma-llm-refactored

Set environment variables as needed:

.. code-block:: bash

   export OPENAI_API_KEY="<your_key>"

Build Documentation
-------------------

.. code-block:: bash

   cd refactored/docs
   make html

The generated site is located at ``refactored/docs/_build/html``.

Serve Documentation Locally
---------------------------

.. code-block:: bash

   cd refactored/docs/_build/html
   python3 -m http.server 8000

Then open ``http://localhost:8000``.
