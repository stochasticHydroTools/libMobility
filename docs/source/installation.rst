Installation
============

We recommend working with a `conda <https://docs.conda.io/en/latest/>`_ environment. The file ``environment.yml`` contains the necessary dependencies to compile and use the library.

You can create the environment with:

.. code-block:: shell

    $ conda env create -f environment.yml

Then, activate the environment with:

.. code-block:: shell

    $ conda activate libmobility

CMake is used for compilation, you can compile and install everything with:

.. code-block:: shell

    $ mkdir build && cd build
    $ cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    $ make all install

It is advisable to install the library in the conda environment, so that the python bindings are available. The environment variable ``$CONDA_PREFIX`` is set to the root of the conda environment.

CMake will compile all modules under the ``solvers`` directory as long as they adhere to the conventions described in "Adding a new solver".

After compilation, the python bindings will be available in the conda environment under the name ``libMobility``.

The following variables are available to customize the compilation process:

- ``DOUBLEPRECISION``: If this variable is defined, libMobility is compiled in double precision (single by default).
