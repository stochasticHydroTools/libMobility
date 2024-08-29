/* Raul P. Pelaez 2022. Hydrodynamic kernels for the NBody evaluator.

   New evaluators must be structs that define the following function:
       //Computes M(ri, rj)*vj
       __device__ real3 dotProduct(real3 pi, real3 pj, real3 vj);

 */
#ifndef NBODY_HYDRODYNAMICKERNELS_CUH
#define NBODY_HYDRODYNAMICKERNELS_CUH
#include "vector.cuh"

namespace nbody_rpy{


  //RPY = (1/(6*pi*viscosity*rh))*(f*I + g* r\diadic r/r^2). rh is hydrodynamic radius. This function returns {f, g/r^2}
  inline  __device__  real2 RPY_UF(real r, real rh){
    const real invrh = real(1.0)/rh;
    r *= invrh;
    if(r >= real(2.0)){
      const real invr  = real(1.0)/r;
      const real invr2 = invr*invr;
      const real f = (real(0.75) + real(0.5)*invr2)*invr;
      const real ginvr2 = (real(0.75) - real(1.5)*invr2)*invr*invr2*invrh*invrh;
      return {f, ginvr2};
    }
    else{
      const real f = real(1.0)-real(0.28125)*r;
      const real ginvr2 = (r>real(0.0))?(real(0.09375)/(r*rh*rh)):real(0);
      return {f, ginvr2};
    }
  }

/*
 RPY_WT computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a**3
*/
__device__ real2 RPY_WT(real r, real rh){
    const real invrh = real(1.0)/rh;
    r *= invrh;
    if(r >= real(2.0)){
      const real invr  = real(1.0)/r;
      const real invr2 = invr*invr;
      const real invr3 = invr*invr2;
      const real f = -0.5*invr3;
      const real ginvr2 = 1.5*invr2*invr3*invrh*invrh;
      return {f, ginvr2};
    }
    else{
      const real r3 = r*r*r;
      // const real c2 =  real(0.28125) * invr - real(0.046875) * r;    // 9/32 = 0.28125, 3/64 = 0.046875
      const real f = (real(1.0) - real(0.84375) * r + real(0.078125) * r3); // 27/32 = 0.84375, 5/64 = 0.078125
      const real ginvr2 = (r>real(0.0))?((real(0.28125) * real(1.0)/r - real(0.046875) * r)*invrh*invrh):real(0);
      return {f, ginvr2};
    }
}
  
// returns (M_xy, M_xz, M_yz)
__device__ real3 RPY_UT(real3 rij, real rh){
  const real invrh = real(1.0)/rh;
  const real r = sqrt(dot(rij, rij))*invrh;
  rij *= invrh;
  if(r>=2){
    real invr3 = real(1.0)/(r*r*r);
    rij *= invr3;
    return {rij.z, -rij.y, rij.x};
  } 
  else{
    real c1 = real(0.5) * (real(1.0) - real(0.375) * r); // 3/8 = 0.375
    rij *= c1;
    return {rij.z, -rij.y, rij.x};
  }
}

// returns (M_xy, M_xz, M_yz)
__device__ real3 RPY_WF(real3 rij, real rh){
  const real invrh = real(1.0)/rh;
  const real r = sqrt(dot(rij, rij))*invrh;
  rij *= invrh;
  if(r>=2){
    const real invr3 = real(1.0)/(r*r*r);
    rij *= invr3;
    return {rij.z, -rij.y, rij.x};
  }
  else{
    real c1 =  real(0.5)*( real(1.0) - real(0.375) * r); // 3/8 = 0.375
    rij *= c1;
    return {rij.z, -rij.y, rij.x};
  }
}

  //Evaluates the RPY tensor with open boundaries
  class OpenBoundary{
    real rh; //Hydrodynamic radius
    real t0; //trans-trans mobility
    real r0; //rot-rot mobility
    real rt0; //rot-trans & trans-rot mobility
  public:

    //The constructor needs a self mobility and an hydrodynamic radius
    OpenBoundary(real t0, real r0, real rt0, real rh):t0(t0), r0(r0), rt0(rt0), rh(rh){}

    //Computes M(ri, rj)*vj
    __device__ real3 dotProduct_UF(real3 pi, real3 pj, real3 vj){
      const real3 rij = make_real3(pi)-make_real3(pj);
      const real r = sqrt(dot(rij, rij));
      const real2 c12 = RPY_UF(r, rh);
      const real f = c12.x;
      const real gdivr2 = c12.y;
      const real gv = gdivr2*dot(rij, vj);
      const real3 Mv_t = f*vj + (r>real(0)?gv*rij:real3());
      return t0*Mv_t;
    }

  __device__ real3 dotProduct_WT(real3 pi, real3 pj, real3 vj){
      const real3 rij = make_real3(pi)-make_real3(pj);
      const real r = sqrt(dot(rij, rij));
      const real2 c12 = RPY_WT(r, rh);
      const real f = c12.x;
      const real gdivr2 = c12.y;
      const real gv = gdivr2*dot(rij, vj);
      const real3 Mv_t = f*vj + (r>real(0)?gv*rij:real3());
      return r0*Mv_t;
  }

  __device__ real3 dotProduct_UT(real3 pi, real3 pj, real3 vj){
      const real3 rij = make_real3(pi)-make_real3(pj);
      const real3 m = RPY_UT(rij, rh); // (M_xy, M_xz, M_yz)
      const real3 Mv_t = {m.x*vj.y + m.y*vj.z,
                         -m.x*vj.x + m.z*vj.z, 
                         -m.y*vj.x - m.z*vj.y};
      return rt0*Mv_t;
  }

  __device__ real3 dotProduct_WF(real3 pi, real3 pj, real3 vj){
      const real3 rij = make_real3(pi)-make_real3(pj);
      const real3 m = RPY_WF(rij, rh); // (M_xy, M_xz, M_yz)
      const real3 Mv_t = {m.x*vj.y + m.y*vj.z,
                         -m.x*vj.x + m.z*vj.z, 
                         -m.y*vj.x - m.z*vj.y};
      return rt0*Mv_t;
  }
};


  //Evaluates the RPY tensor with open boundaries in all boundaries except a wall at the bottom in Z=0
  // References: 
    // [1] Simulation of hydrodynamically interacting particles near a no-slip boundary, Swan & Brady 2007
  class BottomWall{
    real rh; //Hydrodynamic radius
    real t0; //trans-trans mobility
    real r0; //rot-rot mobility
    real rt0; //rot-trans & trans-rot mobility

    //Computes the correction to the open boundary RPY mobility due to a wall located at z=0
    //rij: distance between particles
    //rij.z: This component contains ((pi.z-pj.z) + 2*pj.z)/rh
    //self: self interaction
    //hj: height of the particle j
    //vj: quantity (i.e force) of particle j
    __device__ real3 wallCorrection_UF(real3 rij, bool self, real hj, real3 fj){
      real3 correction = real3();
      if(self){ // B1*vj in [1]
	real invZi = real(1.0) / hj;
	real invZi3 = invZi * invZi * invZi;
	real invZi5 = invZi3 * invZi * invZi;
	correction.x += -fj.x*(real(9.0)*invZi - real(2.0)*invZi3 + invZi5 ) / real(16.0);
	correction.y += -fj.y*(real(9.0)*invZi - real(2.0)*invZi3 + invZi5 ) / real(16.0);
	correction.z += -fj.z*(real(9.0)*invZi - real(4.0)*invZi3 + invZi5 ) / real(8.0);
      }
      else{ // C2*vj in [1]
	real h_hat = hj / rij.z;
	real invR = rsqrt(dot(rij, rij));
	real3 e = rij*invR;
	real invR3 = invR * invR * invR;
	real invR5 = invR3 * invR * invR;
	real fact1 = -(real(3.0)*(real(1.0)+real(2.0)*h_hat*(real(1.0)-h_hat)*e.z*e.z) * invR + real(2.0)*(real(1.0)-real(3.0)*e.z*e.z) * invR3 - real(2.0)*(real(1.0)-real(5.0)*e.z*e.z) * invR5)  / real(4.0);
	real fact2 = -(real(3.0)*(real(1.0)-real(6.0)*h_hat*(real(1.0)-h_hat)*e.z*e.z) * invR - real(6.0)*(real(1.0)-real(5.0)*e.z*e.z) * invR3 + real(10.0)*(real(1.0)-real(7.0)*e.z*e.z) * invR5) / real(4.0);
	real fact3 =  e.z * (real(3.0)*h_hat*(real(1.0)-real(6.0)*(real(1.0)-h_hat)*e.z*e.z) * invR - real(6.0)*(real(1.0)-real(5.0)*e.z*e.z) * invR3 + real(10.0)*(real(2.0)-real(7.0)*e.z*e.z) * invR5) / real(2.0);
	real fact4 =  e.z * (real(3.0)*h_hat*invR - real(10.0)*invR5) / real(2.0);
	real fact5 = -(real(3.0)*h_hat*h_hat*e.z*e.z*invR + real(3.0)*e.z*e.z*invR3 + (real(2.0)-real(15.0)*e.z*e.z)*invR5);
	correction.x += (fact1 + fact2 * e.x*e.x)*fj.x;
	correction.x += (fact2 * e.x*e.y)*fj.y;
	correction.x += (fact2 * e.x*e.z + fact3 * e.x)*fj.z;
	correction.y += (fact2 * e.y*e.x)*fj.x;
	correction.y += (fact1 + fact2 * e.y*e.y)*fj.y;
	correction.y += (fact2 * e.y*e.z + fact3 * e.y)*fj.z;
	correction.z += (fact2 * e.z*e.x + fact4 * e.x)*fj.x;
	correction.z += (fact2 * e.z*e.y + fact4 * e.y)*fj.y;
	correction.z += (fact1 + fact2 * e.z*e.z + fact3 * e.z + fact4 * e.z + fact5)*fj.z;
      }
      return correction;
    }
    
    // NOTE: normalized by 8 pi eta a**3. [1] normalizes by 6 pi et a**3
    __device__ real3 wallCorrection_WT(real3 rij, bool self, real hj, real3 tj){
        real3 correction = real3();
        if(self){ // B3*tj in [1]
            real invZi = real(1.0) / hj;
            real invZi3 = invZi * invZi * invZi;
            correction.x += tj.x*(-invZi3 * real(0.3125)); // 15/48 = 0.3125
            correction.y += tj.y*(-invZi3 * real(0.3125)); // 15/48 = 0.3125
            correction.z += tj.z*(-invZi3 * real(0.125)); // 3/24 = 0.125
        }
    else{ // C4*tj in [1]. all coeffs should be multiplied by 4/3
    real h_hat = hj / rij.z;
    real invR = rsqrt(dot(rij, rij));
    real invR3 = invR * invR * invR;
    real3 e = rij*invR;
    real fact1 =  ((1-6*e.z*e.z) * invR3 ) / real(2.0);
    real fact2 = -(9 * invR3) / real(6.0);
    real fact3 =  (3 * invR3 * e.z);
    real fact4 =  (3 * invR3);

    correction.x += (fact1 + fact2 * e.x*e.x + fact4 * e.y*e.y)*tj.x;
    correction.x += ((fact2 - fact4)* e.x*e.y)*tj.y;
    correction.x += (fact2 * e.x*e.z)*tj.z;
    correction.y += ((fact2 - fact4)* e.x*e.y)*tj.x;
    correction.y += (fact1 + fact2 * e.y*e.y + fact4 * e.x*e.x)*tj.y;
    correction.y += (fact2 * e.y*e.z)*tj.z;
    correction.z += (fact2 * e.z*e.x + fact3 * e.x)*tj.x;
    correction.z += (fact2 * e.z*e.y + fact3 * e.y)*tj.y;
    correction.z += (fact1 + fact2 * e.z*e.z + fact3 * e.z)*tj.z;
    }
    return correction;
  }

  __device__ real3 wallCorrection_UT(real3 rij, bool self, real hj, real3 tj){
      real3 correction = real3();
      if(self){ // B2^T*tj in [1]. ^T denotes transpose.
        real invZi = real(1.0) / hj;
        real invZi4 = invZi * invZi * invZi * invZi;
        correction.x += (invZi4 * real(0.125))*tj.y; // 3/24 = 0.125
        correction.y += (-invZi4 * real(0.125))*tj.x; // 3/24 = 0.125
      }
  else{
    real h_hat = hj / rij.z;
    real invR = rsqrt(dot(rij, rij));
    real invR2 = invR * invR;
    real invR4 = invR2 * invR2;
    real3 e = rij*invR;
    real fact1 =  invR2;
    real fact2 = (real(6.0)*h_hat*e.z*e.z*invR2 + (real(1.0)-real(10.0)*e.z*e.z)*invR4) * real(2.0);
    real fact3 = -e.z*(real(3.0)*h_hat*invR2 - real(5.0)*invR4) * real(2.0);
    real fact4 = -e.z*(h_hat*invR2 - invR4) * real(2.0);

    correction.x -= (-fact3 * e.x * e.y) * tj.x;
    correction.x -= (-fact1 * e.z + fact3 * e.x * e.x - fact4) * tj.y;
    correction.x -= (fact1 * e.y) * tj.z;
    correction.y -= (fact1 * e.z - fact3 * e.y * e.y + fact4) * tj.x;
    correction.y -= (fact3 * e.x * e.y) * tj.y;
    correction.y -= (-fact1 * e.x) * tj.z;
    correction.z -= (-fact1 * e.y - fact2 * e.y - fact3 * e.y * e.z) * tj.x;
    correction.z -= (fact1 * e.x + fact2 * e.x + fact3 * e.x * e.z) * tj.y;
  }
  return correction;
}

  __device__ real3 wallCorrection_WF(real3 rij, bool self, real hj, real3 fj){
        real3 correction = real3();
        if(self){ // B2*tj in [1].
            real invZi = real(1.0) / hj;
            real invZi4 = invZi * invZi * invZi * invZi;
            correction.x += (-invZi4 * real(0.125))*fj.y; // 3/24 = 0.125
            correction.y += (invZi4 * real(0.125))*fj.x; // 3/24 = 0.125
        }
  else{
    real h_hat = hj / rij.z;
    real invR = rsqrt(dot(rij, rij));
    real invR2 = invR * invR;
    real invR4 = invR2 * invR2;
    real3 e = rij*invR;

    real fact1 =  invR2;
    real fact2 = (real(6.0)*h_hat*e.z*e.z*invR2 + (real(1.0)-real(10.0)*e.z*e.z)*invR4) * real(2.0);
    real fact3 = -e.z*(real(3.0)*h_hat*invR2 - real(5.0)*invR4) * real(2.0);
    real fact4 = -e.z*(h_hat*invR2 - invR4) * real(2.0);

    correction.x -= (-fact3*e.x*e.y) * fj.x;
    correction.x -= (fact1*e.z - fact3*e.y*e.y + fact4) * fj.y;
    correction.x -= (-fact1*e.y - fact2*e.y - fact3*e.y*e.z) * fj.z;
    correction.y -= (-fact1*e.z + fact3*e.x*e.x - fact4) * fj.x;
    correction.y -= (fact3*e.x*e.y) * fj.y;
    correction.y -= (fact1*e.x + fact2*e.x + fact3*e.x*e.z) * fj.z;
    correction.z -= (fact1*e.y) * fj.x;
    correction.z -= (-fact1*e.x) * fj.y;
  }
  return correction;
}

  public:

    //The constructor needs a mobility coefficient for each block and an hydrodynamic radius
    BottomWall(real t0, real r0, real rt0, real rh):t0(t0), r0(r0), rt0(rt0), rh(rh){}

    //Computes M(ri, rj)*vj
    __device__ real3 dotProduct_UF(real3 pi, real3 pj, real3 vj){
      real3 rij = make_real3(pi)-make_real3(pj);
      const real r = sqrt(dot(rij, rij));
      const real2 c12 = RPY_UF(r, rh);
      const real f = c12.x;
      const real gdivr2 = c12.y;
      const real gv = gdivr2*dot(rij, vj);
      real3 Mv_t = f*vj + (r>real(0)?gv*rij:real3());
      const real hj = pj.z;
      rij.z = rij.z +2*pj.z;
      Mv_t += wallCorrection_UF(rij/rh,(r==0), hj/rh, vj);
      return t0*Mv_t;
    }

    __device__ real3 dotProduct_WT(real3 pi, real3 pj, real3 tj){
      real3 rij = make_real3(pi)-make_real3(pj);
      const real r = sqrt(dot(rij, rij));
      const real2 c12 = RPY_WT(r, rh);
      const real f = c12.x;
      const real gdivr2 = c12.y;
      const real gv = gdivr2*dot(rij, tj);
      real3 Mv_t = f*tj + (r>real(0)?gv*rij:real3());
      const real hj = pj.z;
      rij.z = rij.z +2*pj.z;
      Mv_t += wallCorrection_WT(rij/rh,(r==0), hj/rh, tj);
      return r0*Mv_t;
    }

    __device__ real3 dotProduct_UT(real3 pi, real3 pj, real3 tj){
    real3 rij = make_real3(pi)-make_real3(pj);
    const real r = sqrt(dot(rij, rij));
    const real3 m = RPY_UT(rij, rh); // (M_xy, M_xz, M_yz)
    real3 Mv_t = {m.x*tj.y + m.y*tj.z,
                 -m.x*tj.x + m.z*tj.z, 
                 -m.y*tj.x - m.z*tj.y};
    const real hj = pi.z;
    rij.z = rij.z +2*pj.z;
    rij.x = -rij.x;
    rij.y = -rij.y;
    Mv_t += wallCorrection_UT(rij/rh,(r==0), hj/rh, tj);
    return rt0*Mv_t;
    }

    __device__ real3 dotProduct_WF(real3 pi, real3 pj, real3 fj){
    real3 rij = make_real3(pi)-make_real3(pj);
    const real r = sqrt(dot(rij, rij));
    const real3 m = RPY_WF(rij, rh); // (M_xy, M_xz, M_yz)
    real3 Mv_t = {m.x*fj.y + m.y*fj.z,
                 -m.x*fj.x + m.z*fj.z, 
                 -m.y*fj.x - m.z*fj.y};
    const real hj = pj.z;
    rij.z = rij.z +2*pj.z;
    Mv_t += wallCorrection_WF(rij/rh,(r==0), hj/rh, fj);
    return rt0*Mv_t;
    }
    
  };

}
#endif
