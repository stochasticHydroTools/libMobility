# Configuration file for the Sphinx documentation builder.
project = "libMobility"
author = "RaulPPelaez"

import git
import sys
import os


def get_latest_git_tag(repo_path="."):
    repo = git.Repo(repo_path)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    return tags[-1].name if tags else None


current_tag = get_latest_git_tag("../../")
sys.path.append(os.path.abspath("./extensions"))
if current_tag is None:
    current_tag = "master"
release = current_tag
version = current_tag

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autoprogram",
    "customsig",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
autosummary_ignore_module_all = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "pybind11": ("https://pybind11.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

autoclass_content = "both"
autodoc_typehints = "none"
autodoc_inherit_docstrings = False
sphinx_autodoc_typehints = True
html_show_sourcelink = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
# Exclude all torchmdnet.datasets.*.rst files in source/generated/
html_static_path = ["../_static"]
html_css_files = [
    "style.css",
]
