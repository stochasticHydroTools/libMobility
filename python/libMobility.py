#Raul P. Pelaez 2021. This is the main python script for libMobility.
#It will load the python wrappers for all solvers in the "solvers" directory
""" Default documentation"""
import sys
import os
import importlib
for solver in os.listdir("../solvers"):
    try:
        sys.path.append('../solvers/'+solver)
        module = importlib.import_module(solver)

        globals().update(
        {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__') 
            else 
            {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')})
        
    except Exception as e:
        print("Could not load "+solver+" with error:")
        print(str(e))

