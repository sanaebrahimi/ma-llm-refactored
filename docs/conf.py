"""Sphinx configuration for the refactored multi-agent project."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "MA-LLM Refactored"
copyright = f"{datetime.now():%Y}, MA-LLM"
author = "MA-LLM"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosectionlabel_prefix_document = True

# Keep docs build resilient in environments without optional runtime deps.
autodoc_mock_imports = [
    "openai",
    "ollama",
    "llmlingua",
    "tiktoken",
    "torch",
    "numpy",
    "pandas",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "MA-LLM Refactored"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "show_prev_next": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "icon_links": [
        {
            "name": "Repository",
            "url": "https://github.com/",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
