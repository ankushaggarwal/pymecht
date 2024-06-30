# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath('../..'))

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

############ ADDED to copy Examples to the source folder #####################

examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Examples"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "Examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

# Include examples in documentation
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)

            shutil.copyfile(source_filename, dest_filename)

####################################################################################

project = 'pyMechT'
copyright = '2023, Computational Biomechanics Research Group, University of Glasgow'
author = 'Computational Biomechanics Research Group, University of Glasgow'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode", "nbsphinx"]

templates_path = ['_templates']
exclude_patterns = []
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
