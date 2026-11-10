# ruff: ignore[undocumented-public-module]
# Configuration file for the Sphinx documentation builder
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys

src_root = pathlib.Path("../..").resolve()
sys.path.insert(0, src_root)
project = "Gaussian Random Field"
company_copyright = "2026, Earthwave Ltd."
author = "Earthwave Ltd."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []

# autosummary and autodoc configuration
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
add_module_names = False

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
