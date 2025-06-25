.. _usage:

Usage
-----

.. hint:: See the :ref:`solvers` section for a list of available solvers.

.. hint:: libMobility can compute the thermal drift term, which in general can be approximated using Random Finite Differences (RFD)  

	.. math::

	   \boldsymbol{\partial}_{\boldsymbol{X}} \cdot \mathcal{M} = \lim_{\delta \to 0} \frac{1}{\delta} \left\langle \mathcal{M}\left(\boldsymbol{X} + \frac{\delta}{2} \boldsymbol{W}  \right) - \mathcal{M}\left(\boldsymbol{X} - \frac{\delta}{2} \boldsymbol{W}  \right) \right\rangle_{\boldsymbol{W}}, \hspace{1cm} \boldsymbol{W} \sim \mathcal{N}\left(0,1 \right)


The libMobility interface
~~~~~~~~~~~~~~~~~~~~~~~~~

All solvers present the same set of methods, which are used to set the parameters of the solver, initialize it, and compute the different terms in the equation above.

As an example, the :py:mod:`libMobility.SelfMobility` solver has the following API:

.. autoclass:: libMobility.SelfMobility
   :members:
   :inherited-members:
   :no-index:
  

   
Example
~~~~~~~

The following example shows how to use the :py:class:`libMobility.SelfMobility` solver to compute the different terms in the equation above.

A libMobility solver is initialized in three steps:

1. Creation of the solver object, specifying the periodicity of the system.
2. Setting the parameters proper to the solver.
3. Initialization of the solver with global the parameters.

.. literalinclude:: ../../examples/python/example.py
  :language: python