# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CoordAnalyst'
copyright = '2026, Claire Bachmann, Bertille Delloye, Victor Davril, Enea Drezet--Marcot'
author = 'Claire Bachmann, Bertille Delloye, Victor Davril, Enea Drezet--Marcot'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

extensions = [
    "sphinx.ext.autodoc",    # auto-generates docs from docstrings
    "sphinx.ext.napoleon",   # supports Google/NumPy style docstrings
    "sphinx.ext.viewcode",   # adds links to source code
]

html_theme = "sphinx_rtd_theme"   # Read the Docs theme

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
