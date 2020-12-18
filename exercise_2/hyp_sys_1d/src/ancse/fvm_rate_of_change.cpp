#include <ancse/fvm_rate_of_change.hpp>

#include <Eigen/Dense>
#include <ancse/numerical_flux.hpp>
#include <ancse/reconstruction.hpp>
#include <fmt/format.h>

#define REGISTER_NUMERICAL_FLUX(token, FluxType, flux)                         \
    if (config["flux"] == (token)) {                                           \
        return std::make_shared<FVMRateOfChange<FluxType, Reconstruction>>(    \
            grid, model, flux, reconstruction);                                       \
    }

template <class Reconstruction>
std::shared_ptr<RateOfChange>
deduce_numerical_flux(const nlohmann::json &config,
                      const Grid &grid,
                      const std::shared_ptr<Model> &model,
                      const std::shared_ptr<SimulationTime> &simulation_time,
                      const Reconstruction &reconstruction)
{
    REGISTER_NUMERICAL_FLUX("central_flux", CentralFlux, CentralFlux(model))
    REGISTER_NUMERICAL_FLUX("lax_friedrichs",
                            LaxFriedrichs,
                            LaxFriedrichs(grid, model, simulation_time))
    REGISTER_NUMERICAL_FLUX("rusanov", Rusanov, Rusanov(model))
    REGISTER_NUMERICAL_FLUX("roe", Roe, Roe(model))
    REGISTER_NUMERICAL_FLUX("hll", HLL, HLL(model))

    if (model->get_name().compare("euler") == 0) {
        auto model_euler = std::dynamic_pointer_cast<Euler>(model);

        REGISTER_NUMERICAL_FLUX("hllc", HLLCEuler, HLLCEuler(model_euler))
    }


    throw std::runtime_error(
        fmt::format("Unknown numerical flux. {}", std::string(config["flux"])));
}
#undef REGISTER_NUMERICAL_FLUX

#define REGISTER_RECONSTRUCTION(token, reconstruction)                         \
    if (config["reconstruction"] == token) {                                   \
        return deduce_numerical_flux(                                          \
            config, grid, model, simulation_time, reconstruction);                     \
    }

std::shared_ptr<RateOfChange> make_fvm_rate_of_change(
    const nlohmann::json &config,
    const Grid &grid,
    const std::shared_ptr<Model> &model,
    const std::shared_ptr<SimulationTime> &simulation_time)
{
    REGISTER_RECONSTRUCTION("o1", PWConstantReconstruction{})

    if (config["reconstruction_variable"] == "primitive") {
        REGISTER_RECONSTRUCTION(
            "minmod",
            (PWLinearReconstruction<MinMod, Primitive>(model, MinMod{}))
        )
        REGISTER_RECONSTRUCTION(
            "superbee",
            (PWLinearReconstruction<SuperBee, Primitive>(model, SuperBee{}))
        )
        REGISTER_RECONSTRUCTION(
            "monotonized_central",
            (PWLinearReconstruction<MonotonizedCentral, Primitive>
                (model, MonotonizedCentral{})
            )
        )
    }
    else if (config["reconstruction_variable"] == "conserved") {
        REGISTER_RECONSTRUCTION(
            "minmod",
            (PWLinearReconstruction<MinMod, Conserved>(MinMod{}))
        )
        REGISTER_RECONSTRUCTION(
            "superbee",
            (PWLinearReconstruction<SuperBee, Conserved>(SuperBee{}))
        )
        REGISTER_RECONSTRUCTION(
            "monotonized_central",
            (PWLinearReconstruction<MonotonizedCentral, Conserved>
                (MonotonizedCentral{})
            )
        )
    }


    throw std::runtime_error(fmt::format(
        "Unknown reconstruction. [{}]", std::string(config["reconstruction"])));
}

#undef REGISTER_RECONSTRUCTION
