# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TROPT'
copyright = '2026, Matan Ben-Tov'
author = 'Matan Ben-Tov'

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    release = _pkg_version("tropt")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

autodoc_mock_imports = ['runner', 'sentence_transformers', 'wandb', 'livelossplot', 'openai', 'litellm', 'IPython', 'tqdm', 'transformers', 'accelerate', 'hydra', 'omegaconf', 'huggingface_hub', 'datasets', 'PIL', 'diffusers', 'trackio']
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
    },
}


# -- LLM-friendly artifacts --------------------------------------------------
# Expose raw Markdown sources alongside rendered HTML so coding agents can
# fetch the source-of-truth Markdown without a lossy HTML→MD round-trip.
# Also publish docs/llms.txt at the docs root per https://llmstxt.org/.
def _copy_llm_artifacts(app, exception):
    if exception is not None:
        return
    import shutil
    # Raw guide markdown: docs/guides/*.md -> _build/html/guides/*.md
    guides_src = os.path.join(app.srcdir, 'guides')
    guides_dst = os.path.join(app.outdir, 'guides')
    if os.path.isdir(guides_src) and os.path.isdir(guides_dst):
        for fname in os.listdir(guides_src):
            if fname.endswith('.md'):
                shutil.copy2(os.path.join(guides_src, fname),
                             os.path.join(guides_dst, fname))
    # llms.txt at the docs root
    llms_src = os.path.join(app.srcdir, 'llms.txt')
    if os.path.isfile(llms_src):
        shutil.copy2(llms_src, os.path.join(app.outdir, 'llms.txt'))


def setup(app):
    app.connect('build-finished', _copy_llm_artifacts)
