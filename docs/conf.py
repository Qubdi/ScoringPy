# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Adjust the path as needed

# -- Project information -----------------------------------------------------

project = 'Timeline Manager'
copyright = '2024, Anri Tvalabeishvili'
author = 'Anri Tvalabeishvili'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
]

# templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
