# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path Info ---------------------------------------------------------------
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../examples'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DPM_Tools'
copyright = '2024, Digital Porous Media Team'
author = 'Digital Porous Media Team'

shutil.copytree("../examples", "examples", dirs_exist_ok=True)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

myst_nb_execute_notebooks = "auto"
nb_execution_excludepatterns = ["**/3D_visualization.ipynb", "**/competent_subset.ipynb"]

add_module_names = False  # dpm_tools.visualization -> visualization
autosummary_generate = True
globaltoc_maxdepth = 2

templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['dpm_tools']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_domain_indices = True
html_use_index = True
html_split_index = False
html_show_sourcelink = False
html_show_sphinx = True
#
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/digital-porous-media/dpm_tools",
            "icon": "fab fa-github-square",
        },
    ],
    "external_links": [
        {
            "name": "Issue Tracker", "url": "https://github.com/digital-porous-media/dpm_tools/issues"
        },
    ],
    "navigation_with_keys": False,
    "show_prev_next": False,
    "icon_links_label": "Quick Links",
    "use_edit_page_button": False,
    "navbar_align": "left",
}
