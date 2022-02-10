/*Raul P. Pelaez 2021. Python wrapper for the SelfMobility module
*/
#include"mobility.h"
#include <MobilityInterface/pythonify.h>

MOBILITY_PYTHONIFY(SelfMobility, "This module ignores hydrodynamic interactions, AKA the mobility matrix is simply (1/(6*pi*eta*a))*I");
