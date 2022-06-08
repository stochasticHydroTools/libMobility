# NBody solver

Computes the RPY mobility in open boundaries using the GPU with a dumb O(N^2) evaluation of the RPY tensor.

Donev: Is it really dumb? It is actually optimized for the GPU, so maybe say that nicer.
It seems this is not yet implemented with a wall?
Perhaps NBody_Wall should not be a separate solver, but rather, implemented here as a different kernel function
  
