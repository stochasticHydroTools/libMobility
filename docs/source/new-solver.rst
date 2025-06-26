Adding a new solver
===================

Solvers must adhere to the ``libmobility::Mobility`` interface (see ``include/MobilityInterface``). This C++ base class unifies all solvers under a common interface. Once the C++ interface is prepared, Python bindings can be added automatically using ``pythonify.h`` (a tool under MobilityInterface).

Solvers are exposed to Python and C++ via a single class that inherits from ``libmobility::Mobility``.

Directory Structure
-------------------

A new solver must add a directory under the "solvers" folder, which must be the name of both the Python and C++ classes. Inside the solver directory, the following files (and only these files) must exist:

1. **mobility.h**
   
   This file must include ``MobilityInterface.h`` and define a single class, named the same as the directory for the solver, that inherits the ``libmobility::Mobility`` base class. For example, in ``solvers/NBody/mobility.h``:

   .. code-block:: cpp

      class NBody: public libmobility::Mobility {
          // ...
      };

   Functions that are not purely virtual offer default behavior that can be overridden. For example, :math:`\sqrt{\boldsymbol{\mathcal{M}}}d\boldsymbol{W}` defaults to using the iterative Lancozs algorithm and the thermal drift defaults to returning zero. Confined solvers that have a non-zero thermal drift, such as NBody with a bottom wall and DPStokes, override the function to provide a thermal drift.

2. **python_wrapper.cu**
   
   This file must provide the Python bindings for the class in ``mobility.h``. If the class follows the ``libmobility::Mobility`` interface correctly, this file can generally be quite simple, using the ``MOBILITY_PYTHONIFY`` or ``MOBILITY_PYTHONIFY_WITH_EXTRA_CODE`` utility in ``include/MobilityInterface/pythonify.h``. For example:

   .. code-block:: cpp

      MOBILITY_PYTHONIFY(MODULENAME, documentation)

   or

   .. code-block:: cpp

      MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA, documentation)

   See ``solvers/NBody/python_wrapper.cpp`` for an example.

3. **CMakeLists.txt**
   
   This must contain rules to create the shared library for the particular solver and its Python wrapper. The solver library should be called "lib[Solver].so", while the Python library should be called "[Solver].[Python_SOABI].so" with the correct extension suffix.

4. **README.md**
   
   The documentation for the specific module (anything relevant about the particular solver that does not fit the class docstring).

5. **tests**

   Add any relevant tests to the test folder using pytest.

Additional Considerations
-------------------------

- A new line should be added to ``solvers/CMakeLists.txt`` to include the new module in the compilation process.
- Some interface functions provide default behavior if not defined. For example, stochastic displacements will be computed using a Lanczos solver if the module does not override the corresponding function.
- The ``hydrodynamicVelocities`` function defaults to calling ``Mdot`` followed by ``sqrtMdotW`` and finally ``thermalDrift``.
- The ``clean`` function defaults to doing nothing.
- The ``initialize`` function of a new solver must call the ``libmobility::Mobility::initialize`` function at some point.

Python-only Modules
-------------------

In the case of a module being Python-only (or not providing a correct child of ``libmobility::Mobility``), ``python_wrapper.cu`` may be omitted. Instead, a file called ``[solver].py`` must exist, providing a Python class that is compatible with ``libmobility::Mobility``. This allows users to write ``from solver import *`` and get a class called "solver" that adheres to the libmobility requirements.

Additional Parameters
---------------------

When a module needs additional parameters beyond those provided to ``initialize``, an additional function called ``setParameters[SolverName]`` must be defined and exposed to Python. See ``solvers/PSE/mobility.h`` and ``solver/PSE/python_wrapper.cu`` for an example. Users of the library are responsible for calling ``setParameters`` before calling ``initialize`` with the required arguments.

Examples
--------

- See ``solvers/SelfMobility`` for a basic example.
- The NBody solver only provides an initialization, ``Mdot`` and ``thermalDrift`` functions, demonstrating the use of default implementations for other methods.
