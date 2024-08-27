Some of this data is generated from the original DoublyPeriodicStokes repository as a correctness check. Some of this data was generated from this repository as a consistency check.

The reason we've regenerated data is that default parameter selection has been changed to fix the grid parameters to the values that best preserve the hydrodynamic radius. When data has been regenerated, consistency was checked by first matching the original DoublyPeriodicStokes repository by using their same parameters, then switching to the new parameter selection and regenerating.

Abbreviations: 
bw- bottom wall
sc- slit channel

-------------- Files --------------
SELF:
self_mobility_bw_ref: from the DPStokes repository. Periodized NBody using kernels with wall corrections in double precision.
self_mobility_bw_ref_noimg: from the DPStokes repository. NBody kernels with wall corrections in double precision, but without using periodized images.
self_mobility_bw_w4, self_mobility_sc_w4: generated from this repository in double precision with default parameters. 
self_mobility_*_torques: generated from this repository in double precision with default parameters.

PAIR:
pair_mobility_bw_w4, pair_mobility_sc_w4: generated from this repository in double precision with default parameters. 
pair_mobility_bw_ref_noimg: from the DPStokes repository. NBody kernels with wall corrections in double precision, but without using periodized images.
pair_mobility_nbody_freespace: no-wall RPY pair mobility. Generated using mobility matrix numba code from the RigidMultiblob repository.
pair_mobility_*_torques: generated from this repository in double precision with default parameters.
