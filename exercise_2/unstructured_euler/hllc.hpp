#pragma once


double square(double x) { return x * x; }

//----------------RoeAverageBegin----------------
/// The Roe average of two states.
class RoeAverage {
  public:
    /// Construct a RoeAverage w.r.t. the densities `rho_L` and `rho_R`.
    RoeAverage(double rho_L, double rho_R)
        : roe_ratio(std::sqrt(rho_R / rho_L)) {}

    /// Compute the Roe average.
    double operator()(double qL, double qR) {
        return (qL + qR * roe_ratio) / (1.0 + roe_ratio);
    }

  private:
    double roe_ratio;
};
//----------------RoeAverageEnd----------------

//----------------HLLCSpeedsBegin----------------
static std::tuple<double, double, double>
batten_speeds(const EulerState &uL,
              const std::pair<double, double> &xvarL,
              const EulerState &uR,
              const std::pair<double, double> &xvarR) {

    RoeAverage roe_average(uL(0), uR(0));

    auto [pL, aL] = xvarL;
    auto [pR, aR] = xvarR;

    double vL = uL(1) / uL(0);
    double vR = uR(1) / uR(0);
    double v_tilda = roe_average(vL, vR);

    double HL = (uL(3) + pL) / uL(0);
    double HR = (uR(3) + pR) / uR(0);
    double H_tilda = roe_average(HL, HR);

    double vroe_square = square(roe_average(uL(1) / uL(0), uR(1) / uR(0)))
                         + square(roe_average(uL(2) / uL(0), uR(2) / uR(0)));

    double a_tilda = std::sqrt((GAMMA - 1.0) * (H_tilda - 0.5 * vroe_square));

    double sL = std::min(vL - aL, v_tilda - a_tilda);
    double sR = std::max(vR + aR, v_tilda + a_tilda);

    // Given the velocities of the left- and right-most waves this
    // is the choice of the middle wave speed.
    //
    double s_star = (uR(1) * (sR - vR) - uL(1) * (sL - vL) + pL - pR)
                    / (uR(0) * (sR - vR) - uL(0) * (sL - vL));

    return {sL, s_star, sR};
}
//----------------HLLCSpeedsEnd----------------

/// HLLC numerical flux with Einfeldt-Batten wavespeeds.
/** Reference: Batten, Wavespeed Estimates for the HLLC Riemann Solver, 1997
 *  @param euler
 *  @param uL    conserved variables 'inside' of the cell.
 *  @param uR    conserved variables 'outside' of the cell.
 */
EulerState hllc(const EulerState &uL, const EulerState &uR) {
    // implement the HLLC approximate Riemann solver.
    // tip, compute the three wave speeds sL, s_star & sR in
    // a separate function to keep things readable.

//----------------HLLCBegin----------------
    auto xvarL = std::pair{euler::pressure(uL), euler::speedOfSound(uL)};
    auto xvarR = std::pair{euler::pressure(uR), euler::speedOfSound(uR)};

    auto [sL, s_star, sR] = batten_speeds(uL, xvarL, uR, xvarR);

    // HLLC flux
    const EulerState &uK = (0.0 <= s_star ? uL : uR);
    double pK = (0.0 <= s_star ? xvarL.first : xvarR.first);
    auto nf = euler::flux(uK);

    if (sL < 0.0 && 0.0 <= sR) {
        double sK = (0.0 <= s_star ? sL : sR);
        double vK = (0.0 <= s_star ? uL(1) / uL(0) : uR(1) / uR(0));
        double cK = (sK - vK) / (sK - s_star);

        nf(0) += sK * (cK * uK(0) - uK(0));
        nf(1) += sK * (cK * uK(0) * s_star - uK(1));
        nf(2) += sK * (cK * uK(2) - uK(2));
        nf(3) += sK
                 * (cK
                        * (uK(3)
                           + (s_star - vK) * (uK(0) * s_star + pK / (sK - vK)))
                    - uK(3));
    }

    return nf;
//----------------HLLCEnd----------------
}
