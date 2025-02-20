# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("./src/torch_numopt"))
sys.path.insert(0, os.path.abspath("../src/torch_numopt"))

# -- Project information -----------------------------------------------------

project = 'torch_numopt'
author = 'Eugenio Lorente-Ramos'
copyright = f'2024, {author}'

# The full version, including alpha/beta/rc tags
release = '0.2.4'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon", "sphinx.ext.autosectionlabel"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False

autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

html_theme_options = {
    "nav_title": "torch_numopt docs",
    "color_primary": "indigo",
    "color_accent": "red",
    "globaltoc_collapse": True,
    "globaltoc_includehidden": False,
    "globaltoc_depth": 1,
    "nav_links": [
        {
            "href": "api_reference",
            "title": "API reference",
            "internal": "api_ref",
        },
        {
            "href": "torch_numopt",
            "title": "Documentation",
            "internal": "docs",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_css_files = ["custom.css"]