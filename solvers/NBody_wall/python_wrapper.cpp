/*Raul P. Pelaez 2021. Python wrapper for the NBody_wall module
*/
#include"mobility.h"
#include <MobilityInterface/pythonify.h>

MOBILITY_PYTHONIFY(NBody_wall, "This module computes the RPY mobility in open boundaries in the presence of a wall. It uses an N^2 algorithm in the GPU.");

