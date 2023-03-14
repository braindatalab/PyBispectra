"""Install the pybispectra package."""

from setuptools import setup

setup(
    name="pybispectra",
    python_requires='==3.10.9'
    version="0.0.1",
    package_dir={"": "src/"},
    packages=[
        "pybispectra",
        "pybispectra.cfc",
        "pybispectra.tde",
        "pybispectra.utils",
    ],
)
