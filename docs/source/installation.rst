Installation
============

To install PyBispectra, activate the desired environment in which you want the package,
then install it using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block::
    
    pip install pybispectra

or `conda <https://docs.conda.io/en/latest/>`_ from
`conda-forge <https://anaconda.org/conda-forge>`_ (for PyBispectra ≥ 1.2.2):

.. code-block::
    
    conda install -c conda-forge pybispectra

PyBispectra requires Python ≥ 3.10.

If you need to create an environment in which to install PyBispectra, you can do so
using `conda <https://docs.conda.io/en/latest/>`_,
`venv <https://docs.python.org/3/library/venv.html>`_, or
`uv <https://docs.astral.sh/uv/>`_.

With ``conda``
--------------

In a shell with ``conda`` available, run the following commands:

.. code-block::

    conda create -n pybispectra_env
    conda activate pybispectra_env
    conda install -c conda-forge pybispectra

With ``venv``
-------------

In a shell with Python available, navigate to your project location and create the
environment:

.. code-block::

    python -m venv pybispectra_env

Activate the environment using the
`appropriate venv command for your operating system and shell <https://docs.python.org/3/library/venv.html#how-venvs-work>`_,
then install the package:

.. code-block::

    pip install pybispectra

With ``uv``
-----------

In a shell with ``uv`` available, navigate to your project location and create the
environment:

.. code-block::

    uv venv pybispectra_env

Activate the environment using the
`appropriate uv command for your operating system and shell <https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment>`_,
then install the package:

.. code-block::

    uv pip install pybispectra
