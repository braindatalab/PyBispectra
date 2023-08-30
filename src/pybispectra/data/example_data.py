"""Tools for fetching example data."""

import os
from pathlib import Path

DATASETS = {}
DATASETS["sim_data_aac"] = "sim_data_aac.npy"
DATASETS["sim_data_pac_bivariate"] = "sim_data_pac_bivariate.npy"
DATASETS["sim_data_pac_univariate"] = "sim_data_pac_univariate.npy"
DATASETS["sim_data_ppc"] = "sim_data_ppc.npy"
DATASETS["sim_data_tde_correlated_noise"] = "sim_data_tde_correlated_noise.npy"
DATASETS[
    "sim_data_tde_independent_noise"
] = "sim_data_tde_independent_noise.npy"
DATASETS["sim_data_waveshape_noisy"] = "sim_data_waveshape_noisy.npy"
DATASETS[
    "sim_data_waveshape_peaks_troughs"
] = "sim_data_waveshape_peaks_troughs.npy"
DATASETS["sim_data_waveshape_sawtooths"] = "sim_data_waveshape_sawtooths.npy"


def get_example_data_paths(name: str) -> str:
    """Return the path to the requested example data.

    Parameters
    ----------
    name : str
        Name of the example data.

    Returns
    -------
    path : str
        Path to the example data.
    """
    if name not in DATASETS.keys():
        raise ValueError(f"`name` must be one of: {list(DATASETS.keys())}")

    filepath_upper = Path(os.path.abspath(__file__)).parent
    return os.path.join(filepath_upper, "example_data", DATASETS[name])
