/* Raul P. Pelaez 2022. Hydrodynamic kernels for the NBody evaluator.

   New evaluators must be structs that define the following function:
       //Computes M(ri, rj)*vj
       __device__ real3 dotProduct(real3 pi, real3 pj, real3 vj);

Notes on notation:
Functions for RPY and dotProduct are defined as (resulting velocity)(input
force). Here, resulting velocity is either U or W, corresponding to linear and
angular velocities, and input force is either F or T, corresponding to linear
and angular forces (torques). For example, dotProduct_UF computes U = MF.
 */
#ifndef NBODY_HYDRODYNAMICKERNELS_CUH
#define NBODY_HYDRODYNAMICKERNELS_CUH
#include "vector.cuh"

namespace nbody_rpy {

struct mdot_result {
  real3 MF = real3(0, 0, 0);
  real3 MT = real3(0, 0, 0);
};

// RPY = (1/(6*pi*viscosity*rh))*(f*I + g* r\diadic r/r^2). rh is hydrodynamic
// radius. This function returns {f, g/r^2}
inline __device__ real2 RPY_UF(real r, real rh) {
  const real invrh = real(1.0) / rh;
  r *= invrh;
  if (r >= real(2.0)) {
    const real invr = real(1.0) / r;
    const real invr2 = invr * invr;
    const real f = (real(0.75) + real(0.5) * invr2) * invr;
    const real ginvr2 =
        (real(0.75) - real(1.5) * invr2) * invr * invr2 * invrh * invrh;
    return {f, ginvr2};
  } else {
    const real f = real(1.0) - real(0.28125) * r;
    const real ginvr2 =
        (r > real(0.0)) ? (real(0.09375) / (r * rh * rh)) : real(0);
    return {f, ginvr2};
  }
}

/*
 RPY_WT computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a**3
*/
__device__ real2 RPY_WT(real r, real rh) {
  const real invrh = real(1.0) / rh;
  r *= invrh;
  if (r >= real(2.0)) {
    const real invr = real(1.0) / r;
    const real invr2 = invr * invr;
    const real invr3 = invr * invr2;
    const real f = -0.5 * invr3;
    const real ginvr2 = 1.5 * invr2 * invr3 * invrh * invrh;
    return {f, ginvr2};
  } else {
    const real r3 = r * r * r;
    // const real c2 =  real(0.28125) * invr - real(0.046875) * r;    // 9/32 =
    // 0.28125, 3/64 = 0.046875
    const real f = (real(1.0) - real(0.84375) * r +
                    real(0.078125) * r3); // 27/32 = 0.84375, 5/64 = 0.078125
    const real ginvr2 =
        (r > real(0.0))
            ? ((real(0.28125) * real(1.0) / r - real(0.046875) * r) * invrh *
               invrh)
            : real(0);
    return {f, ginvr2};
  }
}

// returns (M_xy, M_xz, M_yz)
__device__ real3 RPY_UT(real3 rij, real r, real rh) {
  const real invrh = real(1.0) / rh;
  r *= invrh;
  rij *= invrh;
  if (r >= 2) {
    real invr3 = real(1.0) / (r * r * r);
    rij *= invr3;
    return {rij.z, -rij.y, rij.x};
  } else {
    real c1 = real(0.5) * (real(1.0) - real(0.375) * r); // 3/8 = 0.375
    rij *= c1;
    return {rij.z, -rij.y, rij.x};
  }
}

// returns (M_xy, M_xz, M_yz)
__device__ real3 RPY_WF(real3 rij, real r, real rh) {
  const real invrh = real(1.0) / rh;
  r *= invrh;
  rij *= invrh;
  if (r >= 2) {
    const real invr3 = real(1.0) / (r * r * r);
    rij *= invr3;
    return {rij.z, -rij.y, rij.x};
  } else {
    real c1 = real(0.5) * (real(1.0) - real(0.375) * r); // 3/8 = 0.375
    rij *= c1;
    return {rij.z, -rij.y, rij.x};
  }
}

// Evaluates the RPY tensor with open boundaries
class OpenBoundary {
  real rh;  // Hydrodynamic radius
  real t0;  // trans-trans mobility
  real r0;  // rot-rot mobility
  real rt0; // rot-trans & trans-rot mobility
  bool hasTorque;

public:
  __device__ mdot_result dotProduct(real3 pi, real3 pj, real3 fj, real3 tj) {
    mdot_result result;
    real3 rij = make_real3(pi) - make_real3(pj);
    const real r = sqrt(dot(rij, rij));

    result.MF += dotProduct_UF(rij, r, fj);
    if (hasTorque) {
      result.MF += dotProduct_UT(rij, r, tj);
      result.MT += dotProduct_WF(rij, r, fj);
      result.MT += dotProduct_WT(rij, r, tj);
    }

    return result;
  }

  // The constructor needs a self mobility and an hydrodynamic radius
  OpenBoundary(real t0, real r0, real rt0, real rh, bool hasTorque)
      : t0(t0), r0(r0), rt0(rt0), rh(rh), hasTorque(hasTorque) {}

  // Computes M(ri, rj)*vj
  __device__ real3 dotProduct_UF(real3 rij, real r, real3 vj) {
    const real2 c12 = RPY_UF(r, rh);
    const real f = c12.x;
    const real gdivr2 = c12.y;
    const real gv = gdivr2 * dot(rij, vj);
    const real3 Mv_t = f * vj + (r > real(0) ? gv * rij : real3());
    return t0 * Mv_t;
  }

  __device__ real3 dotProduct_WT(real3 rij, real r, real3 vj) {
    const real2 c12 = RPY_WT(r, rh);
    const real f = c12.x;
    const real gdivr2 = c12.y;
    const real gv = gdivr2 * dot(rij, vj);
    const real3 Mv_t = f * vj + (r > real(0) ? gv * rij : real3());
    return r0 * Mv_t;
  }

  __device__ real3 dotProduct_UT(real3 rij, real r, real3 vj) {
    const real3 m = RPY_UT(rij, r, rh); // (M_xy, M_xz, M_yz)
    const real3 Mv_t = {m.x * vj.y + m.y * vj.z, -m.x * vj.x + m.z * vj.z,
                        -m.y * vj.x - m.z * vj.y};
    return rt0 * Mv_t;
  }

  __device__ real3 dotProduct_WF(real3 rij, real r, real3 vj) {
    const real3 m = RPY_WF(rij, r, rh); // (M_xy, M_xz, M_yz)
    const real3 Mv_t = {m.x * vj.y + m.y * vj.z, -m.x * vj.x + m.z * vj.z,
                        -m.y * vj.x - m.z * vj.y};
    return rt0 * Mv_t;
  }
};

// Evaluates the RPY tensor with open boundaries in all boundaries except a wall
// at the bottom in Z=0
//  References:
//  [1] Simulation of hydrodynamically interacting particles near a no-slip
//  boundary, Swan & Brady 2007 [2] Brownian dynamics of confined suspensions of
//  active microrollers, Usabiaga et al. 2017
class BottomWall {
  real rh;  // Hydrodynamic radius
  real t0;  // trans-trans mobility
  real r0;  // rot-rot mobility
  real rt0; // rot-trans & trans-rot mobility (off-diagonal blocks)
  bool hasTorque;
  // Computes the correction to the open boundary RPY mobility due to a wall
  // located at z=0 rij: distance between particles rij.z: This component
  // contains ((pi.z-pj.z) + 2*pj.z)/rh self: self interaction hj: height of the
  // particle j vj: quantity (i.e force) of particle j
  __device__ real3 wallCorrection_UF(real3 rij, bool self, real hj, real3 vj) {
    real3 correction = real3(0, 0, 0);
    if (self) { // B1*vj in [1]
      real invZi = real(1.0) / hj;
      real invZi3 = invZi * invZi * invZi;
      real invZi5 = invZi3 * invZi * invZi;
      correction.x += -vj.x *
                      (real(9.0) * invZi - real(2.0) * invZi3 + invZi5) *
                      real(0.0625); // 1/16 = 0.0625
      correction.y += -vj.y *
                      (real(9.0) * invZi - real(2.0) * invZi3 + invZi5) *
                      real(0.0625); // 1/16 = 0.0625
      correction.z += -vj.z *
                      (real(9.0) * invZi - real(4.0) * invZi3 + invZi5) *
                      real(0.125); // 1/8 = 0.125
    } else {                       // C2*vj in [1]
      real h_hat = hj / rij.z;
      real invR = rsqrt(dot(rij, rij));
      real3 e = rij * invR;
      real invR3 = invR * invR * invR;
      real invR5 = invR3 * invR * invR;
      real fact1 = -(real(3.0) *
                         (real(1.0) +
                          real(2.0) * h_hat * (real(1.0) - h_hat) * e.z * e.z) *
                         invR +
                     real(2.0) * (real(1.0) - real(3.0) * e.z * e.z) * invR3 -
                     real(2.0) * (real(1.0) - real(5.0) * e.z * e.z) * invR5) *
                   real(0.25); // 1/4 = 0.25
      real fact2 = -(real(3.0) *
                         (real(1.0) -
                          real(6.0) * h_hat * (real(1.0) - h_hat) * e.z * e.z) *
                         invR -
                     real(6.0) * (real(1.0) - real(5.0) * e.z * e.z) * invR3 +
                     real(10.0) * (real(1.0) - real(7.0) * e.z * e.z) * invR5) *
                   real(0.25); // 1/4 = 0.25
      real fact3 =
          e.z *
          (real(3.0) * h_hat *
               (real(1.0) - real(6.0) * (real(1.0) - h_hat) * e.z * e.z) *
               invR -
           real(6.0) * (real(1.0) - real(5.0) * e.z * e.z) * invR3 +
           real(10.0) * (real(2.0) - real(7.0) * e.z * e.z) * invR5) *
          real(0.5); // 1/2 = 0.5
      real fact4 = e.z * (real(3.0) * h_hat * invR - real(10.0) * invR5) *
                   real(0.5); // 1/2 = 0.5
      real fact5 = -(real(3.0) * h_hat * h_hat * e.z * e.z * invR +
                     real(3.0) * e.z * e.z * invR3 +
                     (real(2.0) - real(15.0) * e.z * e.z) * invR5);
      correction.x += (fact1 + fact2 * e.x * e.x) * vj.x;
      correction.x += (fact2 * e.x * e.y) * vj.y;
      correction.x += (fact2 * e.x * e.z + fact3 * e.x) * vj.z;
      correction.y += (fact2 * e.y * e.x) * vj.x;
      correction.y += (fact1 + fact2 * e.y * e.y) * vj.y;
      correction.y += (fact2 * e.y * e.z + fact3 * e.y) * vj.z;
      correction.z += (fact2 * e.z * e.x + fact4 * e.x) * vj.x;
      correction.z += (fact2 * e.z * e.y + fact4 * e.y) * vj.y;
      correction.z +=
          (fact1 + fact2 * e.z * e.z + fact3 * e.z + fact4 * e.z + fact5) *
          vj.z;
    }
    return correction;
  }

  // NOTE: normalized by 8 pi eta a**3. [1] normalizes by 6 pi et a**3
  // so, all coeffs are multiplied by 4/3 if compared to [1]
  __device__ real3 wallCorrection_WT(real3 rij, bool self, real hj, real3 vj) {
    real3 correction = real3(0, 0, 0);
    if (self) { // B3*vj in [1]
      real invZi = real(1.0) / hj;
      real invZi3 = invZi * invZi * invZi;
      correction.x += vj.x * (-invZi3 * real(0.3125)); // 15/48 = 0.3125
      correction.y += vj.y * (-invZi3 * real(0.3125)); // 15/48 = 0.3125
      correction.z += vj.z * (-invZi3 * real(0.125));  // 3/24 = 0.125
    } else {                                           // C4*vj in [1].
      real h_hat = hj / rij.z;
      real invR = rsqrt(dot(rij, rij));
      real invR3 = invR * invR * invR;
      real3 e = rij * invR;
      real fact1 = ((1 - 6 * e.z * e.z) * invR3) * real(0.5); // 1/2 = 0.5
      real fact2 = -(invR3)*real(1.5);                        // 9/6 = 1.5
      real fact3 = (3 * invR3 * e.z);
      real fact4 = (3 * invR3);

      correction.x += (fact1 + fact2 * e.x * e.x + fact4 * e.y * e.y) * vj.x;
      correction.x += ((fact2 - fact4) * e.x * e.y) * vj.y;
      correction.x += (fact2 * e.x * e.z) * vj.z;
      correction.y += ((fact2 - fact4) * e.x * e.y) * vj.x;
      correction.y += (fact1 + fact2 * e.y * e.y + fact4 * e.x * e.x) * vj.y;
      correction.y += (fact2 * e.y * e.z) * vj.z;
      correction.z += (fact2 * e.z * e.x + fact3 * e.x) * vj.x;
      correction.z += (fact2 * e.z * e.y + fact3 * e.y) * vj.y;
      correction.z += (fact1 + fact2 * e.z * e.z + fact3 * e.z) * vj.z;
    }
    return correction;
  }

  // note: [1] seemingly uses the left-hand rule for the cross product, so the
  // signs are flipped on expressions in that paper that use the Levi-Civita
  // symbol
  __device__ real3 wallCorrection_UT(real3 rij, bool self, real h, real3 vj) {
    real3 correction = real3(0, 0, 0);
    if (self) { // B2^T*vj in [1]. ^T denotes transpose.
      real invZi = real(1.0) / h;
      real invZi4 = invZi * invZi * invZi * invZi;
      correction.x += (invZi4 * real(0.125)) * vj.y;  // 3/24 = 0.125
      correction.y += (-invZi4 * real(0.125)) * vj.x; // 3/24 = 0.125
    } else {                                          // C3^T*vj in [1].
      real h_hat = h / rij.z;
      real invR = rsqrt(dot(rij, rij));
      real invR2 = invR * invR;
      real invR4 = invR2 * invR2;
      real3 e = rij * invR;
      real fact1 = invR2;
      real fact2 = (real(6.0) * h_hat * e.z * e.z * invR2 +
                    (real(1.0) - real(10.0) * e.z * e.z) * invR4) *
                   real(2.0);
      real fact3 =
          -e.z * (real(3.0) * h_hat * invR2 - real(5.0) * invR4) * real(2.0);
      real fact4 = -e.z * (h_hat * invR2 - invR4) * real(2.0);

      correction.x -= (-fact3 * e.x * e.y) * vj.x;
      correction.x -= (-fact1 * e.z + fact3 * e.x * e.x - fact4) * vj.y;
      correction.x -= (fact1 * e.y) * vj.z;
      correction.y -= (fact1 * e.z - fact3 * e.y * e.y + fact4) * vj.x;
      correction.y -= (fact3 * e.x * e.y) * vj.y;
      correction.y -= (-fact1 * e.x) * vj.z;
      correction.z -= (-fact1 * e.y - fact2 * e.y - fact3 * e.y * e.z) * vj.x;
      correction.z -= (fact1 * e.x + fact2 * e.x + fact3 * e.x * e.z) * vj.y;
    }
    return correction;
  }

  // note: [1] seemingly uses the left-hand rule for the cross product, so the
  // signs are flipped on expressions in that paper that use the Levi-Civita
  // symbol
  __device__ real3 wallCorrection_WF(real3 rij, bool self, real h, real3 vj) {
    real3 correction = real3(0, 0, 0);
    if (self) { // B2*fj in [1].
      real invZi = real(1.0) / h;
      real invZi4 = invZi * invZi * invZi * invZi;
      correction.x += (-invZi4 * real(0.125)) * vj.y; // 3/24 = 0.125
      correction.y += (invZi4 * real(0.125)) * vj.x;  // 3/24 = 0.125
    } else {                                          // C3*fj in [1].
      real h_hat = h / rij.z;
      real invR = rsqrt(dot(rij, rij));
      real invR2 = invR * invR;
      real invR4 = invR2 * invR2;
      real3 e = rij * invR;

      real fact1 = invR2;
      real fact2 = (real(6.0) * h_hat * e.z * e.z * invR2 +
                    (real(1.0) - real(10.0) * e.z * e.z) * invR4) *
                   real(2.0);
      real fact3 =
          -e.z * (real(3.0) * h_hat * invR2 - real(5.0) * invR4) * real(2.0);
      real fact4 = -e.z * (h_hat * invR2 - invR4) * real(2.0);

      correction.x -= (-fact3 * e.x * e.y) * vj.x;                      // Mxx
      correction.x -= (fact1 * e.z - fact3 * e.y * e.y + fact4) * vj.y; // Mxy
      correction.x -=
          (-fact1 * e.y - fact2 * e.y - fact3 * e.y * e.z) * vj.z;       // Mxz
      correction.y -= (-fact1 * e.z + fact3 * e.x * e.x - fact4) * vj.x; // Myx
      correction.y -= (fact3 * e.x * e.y) * vj.y;                        // Myy
      correction.y -=
          (fact1 * e.x + fact2 * e.x + fact3 * e.x * e.z) * vj.z; // Myz
      correction.z -= (fact1 * e.y) * vj.x;                       // Mzx
      correction.z -= (-fact1 * e.x) * vj.y;                      // Mzy
    }
    return correction;
  }

public:
  // The constructor needs a mobility coefficient for each block and an
  // hydrodynamic radius
  BottomWall(real t0, real r0, real rt0, real rh, bool hasTorque)
      : t0(t0), r0(r0), rt0(rt0), rh(rh), hasTorque(hasTorque) {}

  __device__ mdot_result dotProduct(real3 pi, real3 pj, real3 fj, real3 tj) {
    mdot_result result;
    // implements damping from Appendix 1 in [2] so the matrix is positive
    // definite when a particle overlaps the wall
    real bi = min(pi.z / rh, real(1.0));
    bi = max(bi, real(0.0));
    real bj = min(pj.z / rh, real(1.0));
    bj = max(bj, real(0.0));
    real bij = bi * bj;

    pi.z = max(pi.z, rh);
    pj.z = max(pj.z, rh);

    real3 rij = make_real3(pi) - make_real3(pj);
    const real r = sqrt(dot(rij, rij));

    result.MF += bij * dotProduct_UF(rij, r, fj, pj.z);
    if (hasTorque) {
      result.MF +=
          bij *
          dotProduct_UT(
              rij, r, tj,
              pi.z); // this is correct with pi.z: see note on dotProduct_UT
      result.MT += bij * dotProduct_WF(rij, r, fj, pj.z);
      result.MT += bij * dotProduct_WT(rij, r, tj, pj.z);
    }

    return result;
  }

  __device__ real3 dotProduct_UF(real3 rij, real r, real3 vj, real hj) {
    const real2 c12 = RPY_UF(r, rh);
    const real f = c12.x;
    const real gdivr2 = c12.y;
    const real gv = gdivr2 * dot(rij, vj);
    real3 Mv_t = f * vj + (r > real(0) ? gv * rij : real3());
    rij.z += 2 * hj;
    Mv_t += wallCorrection_UF(rij / rh, (r == 0), hj / rh, vj);
    return t0 * Mv_t;
  }

  __device__ real3 dotProduct_WT(real3 rij, real r, real3 vj, real hj) {
    const real2 c12 = RPY_WT(r, rh);
    const real f = c12.x;
    const real gdivr2 = c12.y;
    const real gv = gdivr2 * dot(rij, vj);
    real3 Mv_t = f * vj + (r > real(0) ? gv * rij : real3());
    rij.z += 2 * hj;
    Mv_t += wallCorrection_WT(rij / rh, (r == 0), hj / rh, vj);
    return r0 * Mv_t;
  }

  // IMPORTANT: wallCorrection_UT is implemented as the transpose of
  // wallCorrection_WF. In the complete mobility matrix, M_{WF, ij}^T = M_{UT,
  // ji}. however, we want to compute M_{UT, ij} on this iteration of the
  // calling loop. so, we call wallCorrection_UT with R = -rij = pj - pi and h =
  // pi.z, i.e. we flip the order of the (ij) arguments so that we get M_{UT,
  // ij}
  __device__ real3 dotProduct_UT(real3 rij, real r, real3 vj, real hi) {
    const real3 m = RPY_UT(rij, r, rh); // (M_xy, M_xz, M_yz)
    real3 Mv_t = {m.x * vj.y + m.y * vj.z, -m.x * vj.x + m.z * vj.z,
                  -m.y * vj.x - m.z * vj.y};
    rij = -1 * rij;
    rij.z += 2 * hi;
    Mv_t += wallCorrection_UT(rij / rh, (r == 0), hi / rh, vj);
    return rt0 * Mv_t;
  }

  __device__ real3 dotProduct_WF(real3 rij, real r, real3 vj, real hj) {
    const real3 m = RPY_WF(rij, r, rh); // (M_xy, M_xz, M_yz)
    real3 Mv_t = {m.x * vj.y + m.y * vj.z, -m.x * vj.x + m.z * vj.z,
                  -m.y * vj.x - m.z * vj.y};
    rij.z += 2 * hj;
    Mv_t += wallCorrection_WF(rij / rh, (r == 0), hj / rh, vj);
    return rt0 * Mv_t;
  }
};

} // namespace nbody_rpy
#endif
