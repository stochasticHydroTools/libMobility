#pragma once

#include "algebra.h"
#include "memory/container.h"
#include <thrust/device_vector.h>

namespace lanczos {
using libmobility::device;
using libmobility::device_adapter;
using libmobility::device_span;
using libmobility::numeric;
template <typename T> using device_container = thrust::device_vector<T>;

namespace detail {
template <numeric T>
void steqr(T *matrix, T *eigenvalues, T *eigenvectors, int size) {
  auto info = LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'I', size, matrix, eigenvalues,
                             eigenvectors, size);
  if (info != 0) {
    throw std::runtime_error("[Lanczos] Could not diagonalize tridiagonal "
                             "krylov matrix, steqr failed with code " +
                             std::to_string(info));
  }
}

template <numeric T> void square_mv(int size, const T *A, const T *x, T *y) {
  T alpha = T(1.0);
  T beta = T(0.0);
  cblas_sgemv(CblasColMajor, CblasNoTrans, size, size, alpha, A, size, x, 1,
              beta, y, 1);
}

/*See algorithm I in [1]*/
template <numeric T> class KrylovSubspace {
  Algebra algebra;
  device_container<T> w; // size N, v in each iteration
  // size Nxmax_iter; Krylov subspace base transformation matrix
  device_container<T> V;
  // Mobility Matrix in the Krylov subspace
  // Transformation Matrix to diagonalize H, max_iter x max_iter
  std::vector<T> P;
  /*upper diagonal and diagonal of H*/
  std::vector<T> hdiag, hsup, htemp;
  int size;
  int subSpaceDimension;

  T normz;

  void diagonalizeSubSpace() {
    int size = getSubSpaceSize();
    /**************LAPACKE********************/
    /*The tridiagonal matrix is stored only with its diagonal and subdiagonal*/
    /*Store both in a temporal array*/
    std::copy(hdiag.begin(), hdiag.end(), htemp.begin());
    std::copy(hsup.begin(), hsup.end(), htemp.begin() + size);
    /*P = eigenvectors must be filled with zeros, I do not know why*/
    P.assign(size * size, 0);
    /*Compute eigenvalues and eigenvectors of a triangular symmetric matrix*/
    steqr(&htemp[0], &hdiag[0], &P[0], size);
  }

  device_container<T> computeSquareRoot() {
    int size = getSubSpaceSize();
    diagonalizeSubSpace();
    /***Hdiag_temp = Hdiag·P·e1****/
    for (int j = 0; j < size; j++) {
      htemp[j] = sqrt(htemp[j]) * P[size * j];
    }
    device_container<T> squareRoot(size);
    /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
    square_mv(size, P.data(), htemp.data(), htemp.data() + size);
    thrust::copy(htemp.begin() + size, htemp.begin() + 2 * size,
                 squareRoot.begin());
    return squareRoot;
  }

  device_container<T> &getTransformationMatrix() { return V; }

  void resize(int subSpaceSize) {
    try {
      w.resize((size + 1), T());
      V.resize(size * subSpaceSize, 0);
      P.resize(subSpaceSize * subSpaceSize, 0);
      hdiag.resize(subSpaceSize + 1, 0);
      hsup.resize(subSpaceSize + 1, 0);
      htemp.resize(2 * subSpaceSize, 0);
    } catch (...) {
      throw std::runtime_error("[KrylovSubspace] Could not allocate memory");
    }
  }

  void setFirstBasisVector(device_span<const T> z) {
    auto &Vm = getTransformationMatrix();
    this->normz = algebra.norm2(z);
    // v[0] = z/norm(z)
    copy(z, Vm);
    algebra.scal(Vm, 1.0 / normz);
  }

public:
  KrylovSubspace(int N, device_span<const T> firstBasisVector)
      : subSpaceDimension(0), size(N) {
    this->resize(1);
    this->setFirstBasisVector(firstBasisVector);
  }

  template <typename MatrixDot> void nextIteration(MatrixDot &dot) {
    int i = subSpaceDimension;
    this->subSpaceDimension++;
    resize(subSpaceDimension + 1);
    /*w = D·vi*/
    auto V_i = device_span<T>(V.data().get() + size * i, size, device::cuda);
    auto V_im1 =
        device_span<T>(V.data().get() + size * (i - 1), size, device::cuda);
    auto V_ip1 =
        device_span<T>(V.data().get() + size * (i + 1), size, device::cuda);
    dot(V_i, w);
    if (i > 0) {
      /*w = w-h[i-1][i]·vi*/
      algebra.axpy(-hsup[i - 1], V_im1, w);
    }
    /*h[i][i] = dot(w, vi)*/
    hdiag[i] = algebra.dot(w, V_i);
    /*w = w-h[i][i]·vi*/
    algebra.axpy(-hdiag[i], V_i, w);
    /*h[i+1][i] = h[i][i+1] = norm(w)*/
    hsup[i] = algebra.norm2(w);
    /*v_(i+1) = w·1/ norm(w)*/
    auto tol = 1e-3 * hdiag[i] / normz;
    if (hsup[i] < tol)
      hsup[i] = 0.0;
    if (hsup[i] > 0.0) {
      algebra.scal(w, 1.0 / hsup[i]);
    } else { /*If norm(w) = 0 that means all elements of w are zero, so set w =
                e1*/
      thrust::fill(w.begin(), w.end(), T());
      w[0] = 1;
    }
    copy(w, V_ip1);
  }

  int getSubSpaceSize() { return subSpaceDimension; }

  // Computes the current result guess sqrt(M)·v, stores in mv
  void computeCurrentResultEstimation(device_span<T> mv) {
    int m = getSubSpaceSize();
    /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
    /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
    auto HhalfDotE1 = computeSquareRoot();
    /*y = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
    // real beta = 0.0;
    // device_gemv(N, m, &this->normz, V, N, HhalfDotE1, 1, &beta, BdW, 1);
    device_span<T> HhalfDotE1_s(HhalfDotE1);
    device_span<T> V_s(V);
    algebra.gemv(size, m, normz, V_s, HhalfDotE1_s, mv);
  }
};
} // namespace detail
template <numeric T> struct Lanczos {

  Lanczos(int maxIterations = 100) : iterationHardLimit(maxIterations) {}

  template <typename Dot>
  int operator()(Dot &dot, device_span<const T> v, device_span<T> mv,
                 T tolerance) {
    const int size = v.size();
    oldBz.resize((size + 1), T());
    /*Lanczos iterations for Krylov decomposition*/
    detail::KrylovSubspace<T> solver(size, v);
    const int checkConvergenceSteps =
        std::min(check_convergence_steps, iterationHardLimit - 2);
    for (int i = 0; i < iterationHardLimit; i++) {
      solver.nextIteration(dot);
      if (i >= checkConvergenceSteps) {
        solver.computeCurrentResultEstimation(mv);
        if (i > 0) {
          auto currentResidual = computeError(mv);
          if (currentResidual <= tolerance) {
            registerRequiredStepsForConverge(i);
            return i;
          }
        }
        // Store current estimation
        copy(mv, oldBz);
      }
    }
    throw std::runtime_error("[Lanczos] Could not converge");
  }

private:
  T computeError(device_span<T> mv) {
    // Compute error as in eq 27 in [1]
    // Error = ||Bz_i - Bz_{i-1}||_2 / ||Bz_{i-1}||_2
    T normResult_prev = algebra.nrm2(oldBz);
    // oldBz = Bz-oldBz
    algebra.axpy(-1.0, mv, oldBz);
    // yy = ||Bz_i - Bz_{i-1}||_2
    T yy = algebra.nrm2(oldBz);
    // eq. 27 in [1]
    T Error = abs(yy / normResult_prev);
    if (std::isnan(Error)) {
      throw std::runtime_error(
          "[Lanczos] Unknown error (found NaN in result guess)");
    }
    return Error;
  }

  void registerRequiredStepsForConverge(int steps_needed) {
    if (steps_needed - 2 > check_convergence_steps) {
      check_convergence_steps += 1;
    }
    // Or check more often if I performed too many iterations
    else {
      check_convergence_steps = std::max(1, check_convergence_steps - 2);
    }
  }

  device_container<T> oldBz;
  int check_convergence_steps;
  int iterationHardLimit = 200;
  Algebra algebra;
};

} // namespace lanczos
