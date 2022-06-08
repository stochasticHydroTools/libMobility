## Doubly Periodic Stokes

Solver for the Stokes equation in a Doubly Periodic environment.

Donev: We should consider moving DoublyPeriodicStokes repo here instead of just an interface to it, so that the only "interface" to Sachin's and your code is via libMobility's interface, instead of the common interface we came up with Sachin. After all, the two interfaces are super similar, and what has been gained by having two interfaces? But maybe there is no actual C++ interface, since I think Sachin+Zecheng used some python for the correction solve? In that case, keep as is.
Another possible reason to keep it as it is would be if we decide to not include torques in libMobility. Then the other interface is needed for that

