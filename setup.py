"""Install the pybispectra package."""

from setuptools import setup, find_packages

setup(
    name="pybispectra",
    version="0.0.1",
    package_dir={"": "src/"},
    packages=find_packages(),
)
