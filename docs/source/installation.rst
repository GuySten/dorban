Installation
============

The code
--------

In order to install the code, clone it and run::

	pip install .

The tests
---------

In order to run the tests run::

	conda env update --file tests_requirements.yml

Then in the tests folder run::

	pytest

To run the doctests, run the following command in the root folder::

        pytest --doctest-modules dorban

The documentation
-----------------

In order to build the documentation first run ::

	conda env update --file docs/conda_requirements.yml

Then in the docs folder run::

	make html

The docs will be created under docs/build/html/
