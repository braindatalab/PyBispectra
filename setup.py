"""Install the PyBispectra package."""

from setuptools import setup

setup(
    name="pybispectra",
    version="1.0.0dev",
    package_dir={"": "src"},
    packages=[
        "pybispectra",
        "pybispectra.cfc",
        "pybispectra.data",
        "pybispectra.tde",
        "pybispectra.utils",
        "pybispectra.waveshape",
    ],
)
