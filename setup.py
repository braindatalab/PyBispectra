"""Install the pybispectra package."""

from setuptools import setup

setup(
    name="pybispectra",
    version="0.0.1",
    package_dir={"": "src/"},
    packages=[
        "pybispectra",
        "pybispectra.cfc",
        "pybispectra.tde",
        "pybispectra.utils",
        "pybispectra.waveshape"
    ],
)
