Development
===========

Changelog
---------
View the changelog for each PyBispectra version here: `version changelog
<https://braindatalab.github.io/PyBispectra/changelog>`_


Installing PyBispectra in editable mode
---------------------------------------

If you want to make changes to PyBispectra, you may wish to install it in editable mode.
To do so, first clone the `GitHub repository
<https://github.com/braindatalab/PyBispectra/tree/main>`_ to your desired location. Once
cloned, navigate to this location and install the package alongside its development
requirements in your desired environment using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block::
    
    pip install -e .
    pip install .[dev]


Contributing to PyBispectra
---------------------------

    This project and everyone participating in it is governed by our `Code of Conduct
    <https://github.com/braindatalab/PyBispectra/blob/main/CODE_OF_CONDUCT.md>`_. By
    participating, you are expected to uphold this code.

If you encounter any issues with the package or wish to suggest improvements, please
submit a report on the `issues page
<https://github.com/braindatalab/PyBispectra/issues>`_.

If you have made any changes which you would like to see officially added to the
package, consider submitting a `pull request
<https://github.com/braindatalab/PyBispectra/pulls>`_. A unit test suite is included.
Tests must be added for any new features, and adaptations to the existing tests must be
made where necessary. Checks for these tests are run when a pull request is submitted,
however these tests can also be run locally by calling `coverage
<https://coverage.readthedocs.io/en/>`_ with `pytest <https://docs.pytest.org/en/>`_ in
the base directory:

.. code-block::
    
    coverage run && coverage report

Please also check that the documentation can be built following any changes. The
documentation is built when a pull request is submitted, however the documentation can
also be built locally using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in the
``/docs`` directory (outputs are in the ``/docs/build/html`` directory):

.. code-block::
    
    make html

Finally, features of the code such as compliance with established styles and spelling
errors in the documentation are also checked. These checks are run when a pull request
is submitted, however they can also be run locally using `pre-commit
<https://pre-commit.com/>`_. To have these checks run automatically whenever you commit
changes, install ``pre-commit`` with the following command in the base directory:

.. code-block::
    
    pre-commit install
