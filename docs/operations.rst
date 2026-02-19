Operations Guide
================

Common Commands
---------------

Build docs:

.. code-block:: bash

   cd refactored/docs
   make html

Clean docs:

.. code-block:: bash

   make clean

Run docs server:

.. code-block:: bash

   cd _build/html
   python3 -m http.server 8000

Troubleshooting
---------------

- If Sphinx imports fail, confirm environment activation and dependencies in ``environment.yml``.
- If runtime pipelines fail, verify Ollama is running and required models are available locally.
- For OpenAI-backed judge paths, confirm ``OPENAI_API_KEY`` is defined.

Maintenance Notes
-----------------

- Keep prompt templates centralized in ``ma_llm.templates``.
- Keep task runners thin and delegate behavior to package modules.
- Regenerate docs after changing docstrings or public APIs.
