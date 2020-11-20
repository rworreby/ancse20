#ifndef FVMSCALAR1D_RECONSTRUCTION_HPP
#define FVMSCALAR1D_RECONSTRUCTION_HPP

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/rate_of_change.hpp>
#include <ancse/simulation_time.hpp>

inline double sign(double a) { return copysign(1.0, a); }


//-------------SlopeLimiterMinModBegin------------
inline double minmod(double a, double b) {
    return 0.5 * (sign(a) + sign(b)) * std::min(std::abs(a), std::abs(b));
}

struct MinMod {
    inline double operator()(double sL, double sR) const {
        return minmod(sL, sR);
    }
};
//------------SlopeLimiterMinModEnd---------------

//------------SlopeLimiterMinAbsBegin-------------
inline double minabs(double a, double b) {
    double abs_a{ std::abs(a) };
    double abs_b{ std::abs(b) };

    if(abs_a == abs_b){
        return (a + b) / 2;
    }
    return abs_a < abs_b ? a : b;
}

struct MinAbs {
    inline double operator()(double sL, double sR) const {
        return minabs(sL, sR);
    }
};
//-------------SlopeLimiterMinAbsEnd--------------


class PWConstantReconstruction {
  public:
    /// Compute the left and right trace at the interface i + 1/2.
    /** Note: This API is agnostic to the number of cell-averages required
     *        by the method. Therefore, reconstructions with different stencil
     *        sizes can implement this API; and this call can be used in parts
     *        of the code that do not need to know about the details of the
     *        reconstruction.
     */
    inline std::pair<double, double> operator()(const Eigen::VectorXd &u,
                                                int i) const {
        return (*this)(u[i], u[i + 1]);
    }

    /// Compute the left and right trace at the interface.
    /** Piecewise constant reconstruction of the left and right trace only
     *  requires the cell-average to the left and right of the interface.
     *
     *  Note: Compared to the other overload this reduces the assumption on
     *        how the cell-averages are stored. This is useful when testing and
     *        generally makes the function useful in more situations.
     */
    inline std::pair<double, double> operator()(double ua, double ub) const {
        return {ua, ub};
    }
};


//----------------LinearRCBegin----------------
template <class SlopeLimiter>
class PWLinearReconstruction {
  public:
    explicit PWLinearReconstruction(const SlopeLimiter &slope_limiter)
        : slope_limiter(slope_limiter) {}

    std::pair<double, double> operator()(const Eigen::VectorXd &u,
                                         int i) const {
        return (*this)(u[i - 1], u[i], u[i + 1], u[i + 2]);
    }

    std::pair<double, double>
    operator()(double ua, double ub, double uc, double ud) const {

        double uL = 0.0;
        double uR = 0.0;

        uL = ub + 0.5 * slope_limiter(uc-ub, ub-ua);
        uR = uc - 0.5 * slope_limiter(ud-uc, uc-ub);

        return {uL, uR};
    }

  private:
    SlopeLimiter slope_limiter;
};
//----------------LinearRCEnd----------------


#endif // FVMSCALAR1D_RATE_OF_CHANGE_HPP
