.. _solvers:
Available solvers
=================

The following solvers are available in libMobility.

Self Mobility
-------------
This module neglects hydrodynamic interactions and simply sets :math:`\boldsymbol{\mathcal{M} }= \frac{1}{6\pi\eta a} \mathbb{I}`.

.. autoclass:: libMobility.SelfMobility
   :members:
   :inherited-members:
   :no-index:

Positively Split Ewald (PSE)
----------------------------

This module computes the RPY mobility in triply periodic boundaries using Ewald splitting with the Positively Split Ewald method.


.. autoclass:: libMobility.PSE
   :members:
   :inherited-members:
   :no-index:


   

NBody
-----

This module computes the RPY mobility in open boundaries using an :math:`O(N^2)` algorithm.

.. autoclass:: libMobility.NBody
   :members:
   :inherited-members:
   :no-index:

Doubly Periodic Stokes (DPStokes)
---------------------------------

This module computes hydrodynamic interactions in a doubly periodic environment.

.. autoclass:: libMobility.DPStokes
   :members:
   :inherited-members:
   :no-index:
