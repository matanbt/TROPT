# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Text-Trigger-Opt Toolbox'
copyright = '2026, Matan Ben-Tov'
author = 'Matan Ben-Tov'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

autodoc_mock_imports = ['runner', 'sentence_transformers', 'wandb', 'livelossplot', 'openai', 'litellm', 'IPython', 'torch', 'tqdm', 'jaxtyping', 'numpy', 'transformers', 'accelerate', 'hydra', 'omegaconf']
autodoc_typehints = "description"


templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "logo": {
        "image_light": "_static/logo_light.png",
        "image_dark": "_static/logo.png",
    }
}
