#include <ancse/dg_rate_of_change.hpp>

#include <Eigen/Dense>
#include <ancse/config.hpp>
#include <ancse/polynomial_basis.hpp>
#include <ancse/dg_handler.hpp>
#include <ancse/numerical_flux.hpp>
#include <fmt/format.h>

//----------------DGRateOfChangeBegin----------------
/// DG numerical flux term
template <class NumericalFlux>
void DGRateOfChange<NumericalFlux>
:: eval_numerical_flux (Eigen::MatrixXd &dudt,
                        const Eigen::MatrixXd &u0) const
{

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;
    auto n_vars = model->get_nvars();
    int n_coeff = 1 + poly_basis.get_degree();

    Eigen::VectorXd fL = Eigen::VectorXd::Zero(n_vars);
    Eigen::VectorXd fR = Eigen::VectorXd::Zero(n_vars);
    Eigen::VectorXd uL(n_vars), uR(n_vars);

    // implement the loop for DG numerical flux term.
}

/// DG volume integral term
template <class NumericalFlux>
void DGRateOfChange<NumericalFlux>
:: eval_volume_integral(Eigen::MatrixXd &dudt,
                        const Eigen::MatrixXd &u0) const
{
    // implement the loop for DG volume integral.
}
//----------------DGRateOfChangeEnd----------------

#define REGISTER_NUMERICAL_FLUX(token, FluxType, flux)          \
    if (config["flux"] == (token)) {                            \
        return std::make_shared< DGRateOfChange<FluxType> >(    \
            grid, model, flux, poly_basis, dg_handler);                     \
    }


std::shared_ptr<RateOfChange> make_dg_rate_of_change(
    const nlohmann::json &config,
    const Grid &grid,
    const std::shared_ptr<Model> &model,
    const PolynomialBasis &poly_basis,
    const DGHandler &dg_handler,
    const std::shared_ptr<SimulationTime> &simulation_time)
{

    // Register the other numerical fluxes.

    throw std::runtime_error(
        fmt::format("Unknown numerical flux. {}",
                    std::string(config["flux"])));
}

#undef REGISTER_NUMERICAL_FLUX
