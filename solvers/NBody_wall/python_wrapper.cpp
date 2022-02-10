/*Raul P. Pelaez 2021. Python wrapper for the NBody_wall module
*/
#include"mobility.h"
#include <MobilityInterface/pythonify.h>

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(NBody_wall,
				   solver.def("setParametersNBody_wall",
					      &NBody_wall::setParametersNBody_wall,
					      "Lx"_a, "Ly"_a, "Lz"_a);,
				   "This module computes the RPY mobility in open boundaries in the presence of a wall. It uses an N^2 algorithm in the GPU.");

