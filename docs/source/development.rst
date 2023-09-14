Development
===========

Changelog
---------
View the changelog for each PyBispectra version here: `version changelog
<https://braindatalab.github.io/PyBispectra/changelog>`_


Installing PyBispectra in editable mode
---------------------------------------

If you want to make changes to PyBispectra, you may wish to install it in
editable mode. To do so, first clone the `GitHub repository
<https://github.com/braindatalab/PyBispectra/tree/main>`_ to your desired
location. Once cloned, navigate to this location and install the package
alongside its `development requirements
<https://github.com/braindatalab/PyBispectra/tree/main/requirements_dev.txt>`_
using pip:

.. code-block:: console
    
    $ pip install -e .
    $ pip install pybispectra[dev]


Contributing to PyBispectra
---------------------------

If you encounter any issues with the package or wish to suggest improvements,
please submit a report on the `issues page
<https://github.com/braindatalab/PyBispectra/issues>`_.

If you have made any changes which you would like to see officially added to
the package, consider submitting a `pull request
<https://github.com/braindatalab/PyBispectra/pulls>`_. When submitting a pull
request, please check that the existing test suite passes, and if you add new
features, please make sure that these are covered in the unit tests. The tests
can be run by calling `coverage <https://coverage.readthedocs.io/en/>`_ with
`pytest <https://docs.pytest.org/en/>`_ in the base directory:

.. code-block:: console
    
    $ coverage run --source=pybispectra -m pytest -v tests && coverage report -m

Please also check that the documentation can be built following any changes,
which can be done using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in
the ``/docs`` directory:

.. code-block:: console
    
    $ make html

Finally, features of the code such as compliance with established styles and
spelling errors in the documentation are also checked. Please ensure that the
code is formatted using `Black <https://black.readthedocs.io/en/stable/>`_, and
check that there are no egregious errors from the following commands:

.. code-block:: console
    
    $ pycodestyle
    $ pydocstyle
    $ codespell
