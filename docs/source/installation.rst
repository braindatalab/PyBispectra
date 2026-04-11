Installation
============

PyBispectra is available on `PyPI <https://pypi.org/project/pybispectra/>`_, and
`conda-forge <https://anaconda.org/channels/conda-forge/packages/pybispectra/overview>`_
for version ≥ 1.2.2.

PyBispectra requires Python ≥ 3.10.


Package installation
--------------------

To install PyBispectra, activate the desired environment or project in which you want
the package, then install it using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block::
    
    pip install pybispectra

`uv <https://docs.astral.sh/uv/>`_:

.. code-block::
    
    uv pip install pybispectra

`conda <https://docs.conda.io/en/latest/>`_:

.. code-block::
    
    conda install -c conda-forge pybispectra

or `pixi <https://pixi.prefix.dev/latest/>`_:

.. code-block::
    
    pixi add pybispectra

|

.. dropdown:: Compatibility for newly released Python versions
    :icon: alert
    :color: info

    If you encounter issues installing PyBispectra in environments with newly released Python versions, this may be due to a lack of compatible ``numba`` releases, one of PyBispectra's core dependencies.
    
    ``numba`` is not always immediately compatible with new Python versions, and there may be a delay of several weeks before a compatible version is released.

    You can find the supported Python versions for ``numba`` in this `table <https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info>`_.

    If a compatible ``numba`` release is available for your environment's Python version are you are still encountering installation issues, please report this on the `PyBispectra issue page <https://github.com/braindatalab/PyBispectra/issues>`_.

.. dropdown:: Compatibility with Python ≥ 3.14 on macOS Intel systems
    :icon: alert
    :color: info

    Support for macOS Intel systems is limited to Python < 3.14 due to wheel availability limitations for ``llvmlite``, which can lead to installation issues using ``pip`` and ``uv``.
    
    If you have a macOS Intel system and need to use Python ≥ 3.14, consider using ``conda`` or ``pixi`` for installation.


Creating an environment or project for installation
---------------------------------------------------

If you need to create an environment or project in which to install PyBispectra, you can
do so using `venv <https://docs.python.org/3/library/venv.html>`_,
`uv <https://docs.astral.sh/uv/>`_, `pixi <https://pixi.prefix.dev/latest/>`_, or
`conda <https://docs.conda.io/en/latest/>`_.

With ``venv``
~~~~~~~~~~~~~

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
~~~~~~~~~~~

In a shell with ``uv`` available, navigate to your project location and create the
environment:

.. code-block::

    uv venv pybispectra_env

Activate the environment using the
`appropriate uv command for your operating system and shell <https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment>`_,
then install the package:

.. code-block::

    uv pip install pybispectra

With ``pixi``
~~~~~~~~~~~~~

In a shell with ``pixi`` available, run the following commands:

.. code-block::

    pixi init
    pixi shell-hook
    pixi add pybispectra

With ``conda``
~~~~~~~~~~~~~~

In a shell with ``conda`` available, run the following commands:

.. code-block::

    conda create -n pybispectra_env
    conda activate pybispectra_env
    conda install -c conda-forge pybispectra
