#ifndef HYPSYS1D_FVM_RATE_OF_CHANGE_HPP
#define HYPSYS1D_FVM_RATE_OF_CHANGE_HPP

#include <memory>
#include <Eigen/Dense>

#include <ancse/config.hpp>
#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/rate_of_change.hpp>
#include <ancse/simulation_time.hpp>

/// Compute the rate of change due to FVM.
/** The semidiscrete approximation of a PDE using FVM is
 *      du_i/dt = - (F_{i+0.5} - F_{i-0.5}) / dx.
 *  This computes the right hand side of the ODE.
 *
 * @tparam NumericalFlux see e.g. `CentralFlux`.
 * @tparam Reconstruction see e.g. `PWConstantReconstruction`.
 */
template <class NumericalFlux, class Reconstruction>
class FVMRateOfChange : public RateOfChange {
  public:
    FVMRateOfChange(const Grid &grid,
                    const std::shared_ptr<Model> &model,
                    const NumericalFlux &numerical_flux,
                    const Reconstruction &reconstruction)
        : grid(grid),
          model(model),
          numerical_flux(numerical_flux),
          reconstruction(reconstruction) {}

    //----------------FVMRateOfChangeBegin----------------
    virtual void operator()(Eigen::MatrixXd &dudt,
                            const Eigen::MatrixXd &u0) const override {
        auto n_cells = grid.n_cells;
        auto n_ghost = grid.n_ghost;

        double dx = grid.dx;
        int const grid_lim{ grid.n_cells - grid.n_ghost };

        Eigen::VectorXd fL = Eigen::VectorXd::Zero(u0.rows());
        Eigen::VectorXd fR = Eigen::VectorXd::Zero(u0.rows());
        Eigen::VectorXd uL(u0.rows()), uR(u0.rows());

        reconstruction.set(u0);

        for (int i = grid.n_ghost-1; i < grid_lim; ++i) {
            for (size_t j = 0; j < fL.size(); j++) {
                fL = fR;
                std::tie(uL, uR) = reconstruction(i);
                fR = numerical_flux(uL, uR);
                // From -(fR - fL) = (fL - fR)
                dudt.col(i) = (fL - fR) / dx;
            }
        }
    }
    //----------------FVMRateOfChangeEnd----------------

  private:
    Grid grid;
    std::shared_ptr<Model> model;
    NumericalFlux numerical_flux;
    Reconstruction reconstruction;
};

std::shared_ptr<RateOfChange>
make_fvm_rate_of_change(const nlohmann::json &config,
                        const Grid &grid,
                        const std::shared_ptr<Model> &model,
                        const std::shared_ptr<SimulationTime> &simulation_time);

#endif // HYPSYS1D_FVM_RATE_OF_CHANGE_HPP
