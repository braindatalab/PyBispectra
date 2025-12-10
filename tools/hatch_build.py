import platform

from hatchling.metadata.plugin.interface import MetadataHookInterface


class JSONMetaDataHook(MetadataHookInterface):
    def update(self, metadata):
        # Numba dropped official support for macOS using Intel CPUs from v0.63 (which
        # also brought support for Python 3.14), so we set different dependencies for
        # that platform.
        is_macos_intel = (
            platform.system() == "Darwin" and platform.machine().startswith("x86")
        )

        # requires-python
        requires_python = ">=3.10"
        if is_macos_intel:
            requires_python += ", <3.14"
        metadata["requires-python"] = requires_python

        # dependencies
        dependencies = [
            "joblib>=1.2",
            "matplotlib>=3.6",
            "mne>=1.7",
            "numpy>=1.22",
            "scikit-learn>=1.1",
            "scipy>=1.8",
            "numba>=0.56",
        ]
        if is_macos_intel:
            dependencies[-1] += ", <0.63"
        metadata["dependencies"] = dependencies
