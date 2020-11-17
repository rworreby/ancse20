#ifndef FVMSCALAR1D_CFL_CONDITION_HPP
#define FVMSCALAR1D_CFL_CONDITION_HPP

#include <cmath>
#include <limits>
#include <memory>

#include <Eigen/Dense>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>

/// Interface for computing CFL restricted timesteps.
class CFLCondition {
  public:
    virtual ~CFLCondition() = default;

    /// Compute the largest time-step satisfying the CFL condition.
    virtual double operator()(const Eigen::VectorXd &u) const = 0;
};

//----------------StandardCFLConditionDeclBegin----------------
/// Compute the CFL condition for generic models.
/** Compute the maximum eigenvalue given all the data, and compute the
 *  CFL restricted time-step accordingly.
 *
 *  Note: this requires looping over the entire grid, and computing a reduction,
 *        i.e. max.
 */
class StandardCFLCondition : public CFLCondition {
  public:
    StandardCFLCondition(const Grid &grid,
                         const Model &model,
                         double cfl_number);

    virtual double operator()(const Eigen::VectorXd &u) const override;

  private:
    Grid grid;
    Model model;
    double cfl_number;
};
//----------------StandardCFLConditionDeclEnd----------------


std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid, const Model &model, double cfl_number);

#endif // FVMSCALAR1D_CFL_CONDITION_HPP
