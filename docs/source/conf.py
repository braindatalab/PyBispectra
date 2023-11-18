# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import pybispectra
from pybispectra.utils._docs import linkcode_resolve


project = "PyBispectra"
copyright = "2023, Thomas Samuel Binns"
author = "Thomas Samuel Binns"
release = "1.2.0dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]

bibtex_bibfiles = ["refs.bib"]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/braindatalab/PyBispectra",
            icon="fa-brands fa-square-github",
        )
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "switcher": {
        "json_url": "https://pybispectra.readthedocs.io/en/main/_static/versions.json",  # noqa E501
        "version_match": pybispectra.__version__,
    },
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark",
}


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/latest", None),
    "mne": ("https://mne.tools/stable", None),
}


# NumPyDoc configuration -----------------------------------------------------

# Define what extra methods numpydoc will document
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "bool": ":class:`python:bool`",
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
}
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "pyplot",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    "low",
    "high",
    # shapes
    "epochs",
    "channels",
    "nodes",
    "rank",
    "times",
    "components",
    "frequencies",
    "x",
    "n_vertices",
    "n_faces",
    "n_channels",
    "m",
    "n",
    "n_events",
    "n_colors",
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_dipoles",
    "n_ica_components",
    "n_pos",
    "n_node_names",
    "n_tapers",
    "n_signals",
    "n_step",
    "n_freqs",
    "wsize",
    "Tx",
    "M",
    "N",
    "p",
    "q",
    "r",
    "n_observations",
    "n_regressors",
    "n_cols",
    "n_frequencies",
    "n_tests",
    "n_samples",
    "n_permutations",
    "nchan",
    "n_points",
    "n_features",
    "n_parts",
    "n_features_new",
    "n_components",
    "n_labels",
    "n_events_in",
    "n_splits",
    "n_scores",
    "n_outputs",
    "n_trials",
    "n_estimators",
    "n_tasks",
    "nd_features",
    "n_classes",
    "n_targets",
    "n_slices",
    "n_hpi",
    "n_fids",
    "n_elp",
    "n_pts",
    "n_tris",
    "n_nodes",
    "n_nonzero",
    "n_events_out",
    "n_segments",
    "n_orient_inv",
    "n_orient_fwd",
    "n_orient",
    "n_dipoles_lcmv",
    "n_dipoles_fwd",
    "n_picks_ref",
    "n_coords",
    "n_meg",
    "n_good_meg",
    "n_moments",
    "n_patterns",
    "n_new_events",
}
