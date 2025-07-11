.. _solvers:

Available solvers
=================
Solvers must be instantiated with the type of boundary in each direction (X, Y, and Z). Each periodicity condition can be one of the following:
	- open: No periodicity in the corresponding direction.
	- unspecified: The periodicity is not specified.
	- single_wall: The system is bounded by a single wall in the corresponding direction.
	- two_walls: The system is bounded by two walls in the corresponding direction.
	- periodic: The system is periodic in the corresponding direction.

All solvers support a subset of periodicity conditions. The description for each solver below will descript supported periodicities. The following solvers are available in libMobility.

Self Mobility
-------------
.. autoclass:: libMobility.SelfMobility
   :members:
   :inherited-members:
   :no-index:

Positively Split Ewald (PSE)
----------------------------

.. autoclass:: libMobility.PSE
   :members:
   :inherited-members:
   :no-index:


   

NBody
-----

.. autoclass:: libMobility.NBody
   :members:
   :inherited-members:
   :no-index:

Doubly Periodic Stokes (DPStokes)
---------------------------------

.. autoclass:: libMobility.DPStokes
   :members:
   :inherited-members:
   :no-index:
