#ifndef HYPSYS1D_NUMERICAL_FLUX_HPP
#define HYPSYS1D_NUMERICAL_FLUX_HPP

#include <memory>
#include <iostream>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/simulation_time.hpp>

/// Central flux.
/** This flux works does not depend on the model.
 * It is also unconditionally a bad choice.
 */
class CentralFlux {
  public:
    // Note: the interface for creating fluxes will give you access
    //       to the following:
    //         - model
    //         - grid
    //         - shared_ptr to simulation_time
    //       Therefore, try to only use a subset of those three in your
    //       constructors.
    explicit CentralFlux(const std::shared_ptr<Model> &model)
        : model(model) {}

    /// Compute the numerical flux given the left and right trace.
    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        auto fL = model->flux(uL);
        auto fR = model->flux(uR);

        return 0.5 * (fL + fR);
    }

  private:
    std::shared_ptr<Model> model;
};


/// Lax-Friedrichs numerical flux.
/** This flux works for any model. */
//----------------FluxLFBegin----------------
class LaxFriedrichs {
  public:
    // Note: This version is a bit tricky. A numerical flux should be
    //       a function of the two trace values at the interface,
    //       i.e. what we call `uL`, `uR`.
    //       However, it requires 'dt' and 'dx'. Therefore,
    //       these need to be made available to the flux.
    //       This is one of the reasons why `SimulationTime`.
    LaxFriedrichs(const Grid &grid,
                  const std::shared_ptr<Model> &model,
                  std::shared_ptr<SimulationTime> simulation_time)
        : simulation_time(std::move(simulation_time)),
          grid(grid),
          model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const {
        double dx = grid.dx;
        double dt = simulation_time->dt;
        return Eigen::VectorXd();
    }

  private:
    std::shared_ptr<SimulationTime> simulation_time;
    Grid grid;
    std::shared_ptr<Model> model;
};
//----------------FluxLFEnd----------------


/// Rusanov's flux (or local Lax-Friedrichs).
/** This flux works for any model. */
//----------------FluxRusanovBegin----------------
class Rusanov {
  public:
    explicit Rusanov(const std::shared_ptr<Model> &model)
        : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        return Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxRusanovEnd----------------

/// Roe flux.
/** This requires knowledge about the model.
 *  It is also well-known for generating unphysical weak solutions.
 */
//----------------FluxRoeBegin----------------
class Roe{
  public:
    explicit Roe(const std::shared_ptr<Model> &model)
        : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        return Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxRoeEnd----------------



/// HLL flux.
/** This requires knowledge about the model. */
//----------------FluxHLLBegin---------------- 
class HLL {
  public:
    explicit HLL(const std::shared_ptr<Model> &model) : model(model) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        return Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Model> model;
};
//----------------FluxHLLEnd---------------- 

/// HLLC flux
/** This requires knowledge about the model.
 *  This version is for the Euler equation.
 */
//----------------FluxHLLCEulerBegin----------------  
class HLLCEuler {
  public:
    explicit HLLCEuler(const std::shared_ptr<Euler> &model)
        : model(model) {
        n_vars = model->get_nvars();
    }

    Eigen::VectorXd operator()(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
    {
        return Eigen::VectorXd();
    }

  private:
    std::shared_ptr<Euler> model;
    int n_vars;
};
//----------------FluxHLLCEulerEnd----------------  



#endif // HYPSYS1D_NUMERICAL_FLUX_HPP
