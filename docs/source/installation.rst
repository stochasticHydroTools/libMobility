Installation
============

We recommend working with a `conda <https://docs.conda.io/en/latest/>`_ environment. The file ``environment.yml`` contains the necessary dependencies to compile and use the library.

Getting dependencies
--------------------

You can create the environment with:

.. code-block:: shell

    $ conda env create -f environment.yml

Then, activate the environment with:

.. code-block:: shell

    $ conda activate libmobility

.. hint:: At the moment we offer several installation methods: via pip, conda or building from source. We recommend using pip or conda first, resorting to building from source only if the other methods do not work for you.

	  
Installing via pip
------------------

After installing the dependencies, you can install the library with pip. Go to the root of the repository and run:

.. code-block:: shell

    $ pip install .
    

Building a conda package
------------------------

Building the conda package only requires the conda-build package. You can install it with:

.. code-block:: shell

    $ conda install conda-build

You can build a conda package with the following command from the root of the repository:

.. code-block:: shell
		
    $ conda build conda

This will build and test the package, which you can install in any environment with:

.. code-block:: shell

    $ conda install --use-local libMobility

Conda will automatically install all the dependencies needed to run the library.


Building from source
--------------------


CMake is used for compilation, you can compile and install everything with:

.. code-block:: shell

    $ mkdir build && cd build
    $ cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    $ make all install

It is advisable to install the library in the conda environment, so that the python bindings are available. The environment variable ``$CONDA_PREFIX`` is set to the root of the conda environment.

After compilation, the python bindings will be available in the conda environment under the name ``libMobility``. See :ref:`usage` for more information.

The following variables are available to customize the compilation process:

- ``DOUBLEPRECISION``: If this variable is defined, libMobility is compiled in double precision (single by default).
