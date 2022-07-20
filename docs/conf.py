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
import os.path as op
import sys

sys.path.insert(0, os.path.abspath('..'))

import piscat
# include parent directory
pdir = op.dirname(op.dirname(op.abspath(__file__)))
# include extensions
sys.path.append(op.abspath('extensions'))

env_flag = os.getenv("status_flag")

# set device name
if env_flag != None:
    status_flag = 'HTML'
else:
    status_flag = 'HTML'  # default device name

# -- Project information -----------------------------------------------------

project = 'PiSCAT'
copyright = '2021, Houman Mirzaalian Dastjerdi, Reza Gholami Mahmoodabadi and et. al.'
author = 'Houman Mirzaalian Dastjerdi, Reza Gholami Mahmoodabadi and et. al.'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.autosummary',
                'sphinx.ext.intersphinx',
                'sphinx.ext.mathjax',
                'sphinx.ext.viewcode',
                'sphinx.ext.napoleon',
                'nbsphinx',
                'recommonmark',
                'sphinx_markdown_tables',
                'sphinxcontrib.bibtex',
                'sphinx_copybutton',

]
# The html index document.
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'
autodoc_member_order = 'groupwise'
autoclass_content = 'both'
nbsphinx_timeout = -1
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "./Fig/PiSCAT_logo_bg.png"

html_css_files = [
    'widestyle.css',
]

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'titles_only': False,
}

nbsphinx_allow_errors = True

