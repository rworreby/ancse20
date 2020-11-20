#ifndef FVMSCALAR1D_NUMERICAL_FLUX_HPP
#define FVMSCALAR1D_NUMERICAL_FLUX_HPP

#include <memory>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/simulation_time.hpp>

/// Central flux.
/** This flux works does not depend on the model. It is also unconditionally a
 * bad choice.
 */
class CentralFlux {
  public:
    // Note: the interface for creating fluxes will give you access to the
    //       following:
    //         - model
    //         - grid
    //         - shared_ptr to simulation_time
    //       Therefore, try to only use a subset of those three in your
    //       constructors.
    explicit CentralFlux(const Model &model) : model(model) {}

    /// Compute the numerical flux given the left and right trace.
    double operator()(double uL, double uR) const {
        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        return 0.5 * (fL + fR);
    }

  private:
    Model model;
};


//---------------FluxLFBegin----------------
/// Lax-Friedrichs numerical flux.
/** This flux works for any model. */
class LaxFriedrichs {
  public:
    // Note: This version is a bit tricky. A numerical flux should be
    //       a function of the two trace values at the interface, i.e. what we
    //       call `uL`, `uR`. However, it requires 'dt' and 'dx'. Therefore,
    //       these need to be made available to the flux. This is one of the
    //       reasons why `SimulationTime`.
    LaxFriedrichs(const Grid &grid,
                  const Model &model,
                  std::shared_ptr<SimulationTime> simulation_time)
        : simulation_time(std::move(simulation_time)),
          grid(grid),
          model(model) {}

    double operator()(double uL, double uR) const {
        double dx = grid.dx;
        double dt = simulation_time->dt;

        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        double flux{ 0.5 * (fL + fR) };
        flux -= (dx / (2 * dt)) * (uR - uL);
        return flux;
    }

  private:
    std::shared_ptr<SimulationTime> simulation_time;
    Grid grid;
    Model model;
};
//----------------FluxLFEnd-----------------


//---------------FluxRusBegin---------------
/// Rusanov's numerical flux.
class Rusanov {
  public:
    Rusanov(const Model &model) : model(model) {}

    double operator()(double uL, double uR) const {
        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        double flux{ 0.5 * (fL + fR) };
        flux -= (0.5 * std::max(std::abs(model.max_eigenvalue(uL)),
                 std::abs(model.max_eigenvalue(uR)))
                ) * (uR - uL);
        return flux;
    }

  private:
    Model model;
};
//----------------FluxRusEnd----------------


//---------------FluxRoeBegin---------------
/// Roe numerical flux.
class Roe {
  public:
    Roe(const Model &model) : model(model) {}

    double operator()(double uL, double uR) const {
        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        double a_hat{ 0.0 };
        if(uL != uR){
            a_hat = (fR - fL) / (uR - uL);
        }
        else{
            a_hat = model.max_eigenvalue(uL);
        }

        return a_hat < 0 ? fR : fL;
    }

  private:
    Model model;
};
//----------------FluxRoeEnd----------------


//-------------FluxGodunovBegin-------------
/// Godunov numerical flux.
class Godunov {
  public:
    // Note: This version is a bit tricky. A numerical flux should be
    //       a function of the two trace values at the interface, i.e. what we
    //       call `uL`, `uR`. However, it requires 'dt' and 'dx'. Therefore,
    //       these need to be made available to the flux. This is one of the
    //       reasons why `SimulationTime`.
    Godunov(const Model &model) : model(model) {}

    double operator()(double uL, double uR) const {
        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        // TODO: implement Godunov Flux
    }

  private:
    Model model;
};
//--------------FluxGodunovEnd--------------


//---------------FluxEOBegin----------------
/// Engquist-Osher numerical flux.
class EngquistOsher {
  public:
    // Note: This version is a bit tricky. A numerical flux should be
    //       a function of the two trace values at the interface, i.e. what we
    //       call `uL`, `uR`. However, it requires 'dt' and 'dx'. Therefore,
    //       these need to be made available to the flux. This is one of the
    //       reasons why `SimulationTime`.
    EngquistOsher(const Model &model) : model(model) {}

    double operator()(double uL, double uR) const {
        auto fL = model.flux(uL);
        auto fR = model.flux(uR);

        // TODO: implement EO Flux
    }

  private:
    Model model;
};
//----------------FluxEOEnd-----------------

#endif // FVMSCALAR1D_NUMERICAL_FLUX_HPP
