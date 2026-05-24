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
    'sphinx_copybutton',
    'sphinx_codeautolink',
    'sphinx_design',
    'myst_parser',
    # SEO: emits sitemap.xml + per-page Open Graph / Twitter card meta tags.
    # Both extensions need `html_baseurl` (set below) to produce absolute URLs.
    'sphinx_sitemap',
    'sphinxext.opengraph',
]

# sphinx-copybutton: strip shell prompts and Python REPL prefixes so the
# clipboard receives runnable code only.
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True

# sphinx-codeautolink: turn identifiers in code blocks into links to the
# autodoc API reference. `concat_default=True` chains code blocks within a
# single page (like a notebook), so later blocks can resolve names imported
# in earlier ones.
codeautolink_concat_default = True

# MyST: enable colon-fence so sphinx-design directives (`:::{grid}` etc.) and
# admonitions work in .md guides.
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

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
html_js_files = ['skip_auto_theme.js']
html_favicon = '_static/favicon.svg'

# -- SEO ---------------------------------------------------------------------
# Canonical URL of the deployed docs. sphinx-sitemap and sphinxext-opengraph
# both read this; without it they silently produce relative URLs (which
# Twitter/LinkedIn/Slack won't accept for og:url and og:image).
html_baseurl = "https://matanbt.github.io/TROPT/"

# Crisper window title — strips the noisy "TROPT 0.0.1a1 documentation" suffix.
# Sphinx renders <title> as "{page_h1} — {html_title}", so setting this to the
# tagline gives "TROPT — Textual Trigger Optimization Toolbox" on the landing
# page and "<topic> — Textual Trigger Optimization Toolbox" elsewhere.
html_title = "Textual Trigger Optimization Toolbox"

# Pull robots.txt into the build output (extra_path copies files verbatim into
# the html dir). The file lives at docs/_static/robots.txt; the build script
# copies _static into the output anyway, but extra_path ensures it lands at
# the SITE ROOT (matanbt.github.io/TROPT/robots.txt) which is where crawlers
# look for it — placement inside _static/ wouldn't be discoverable.
html_extra_path = ['robots.txt']

# sphinx-sitemap config — emits sitemap.xml at the docs root.
# `sitemap_url_scheme` template: {link} = relative URL of each page.
sitemap_url_scheme = "{link}"

# sphinxext-opengraph — auto-generates og:title, og:description, og:url, plus
# Twitter Card tags. og:image is set globally; per-page overrides go via the
# MyST `myst.html_meta` front-matter (see index.md).
ogp_site_url = html_baseurl
ogp_site_name = "TROPT — Textual Trigger Optimization Toolbox"
ogp_image = html_baseurl + "_static/og-image.png"
ogp_image_alt = "TROPT — Textual Trigger Optimization Toolbox"
ogp_type = "website"
ogp_enable_meta_description = True
# Generate <meta name="twitter:card" content="summary_large_image"> so the
# preview is the wide card layout (not the small thumbnail variant).
ogp_social_cards = {"enable": False}  # don't auto-render; we ship our own.
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta name="twitter:site" content="@matanbentov">',
]

html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
    "show_toc_level": 2,
    "navbar_align": "left",
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page"],
        "index": [],
    },
    # New visitors land in light mode. Theme switcher is kept so users can
    # opt into dark mode; we just override the switcher's JS below to skip
    # the "auto" (system-selected) state — light ↔ dark only.
    "default_mode": "light",
    # Compact single-line footer — see docs/_templates/footer-tropt.html.
    # Drops the default copyright / sphinx-version / theme-version stack.
    "footer_start": [],
    "footer_center": ["footer-tropt"],
    "footer_end": [],
    # GitHub icon in the top-right of the navbar (GEPA-style).
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/matanbt/TROPT",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# Landing page: hide the primary (left) sidebar so the hero spans full width.
html_sidebars = {
    "index": [],
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
