#ifndef HYPSYS1D_CFL_CONDITION_HPP
#define HYPSYS1D_CFL_CONDITION_HPP

#include <cmath>
#include <limits>
#include <memory>

#include <Eigen/Dense>

#include <ancse/includes.hpp>
#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/dg_handler.hpp>

/// Interface for computing CFL restricted timesteps.
class CFLCondition {
  public:
    virtual ~CFLCondition() = default;

    /// Compute the largest time-step satisfying the CFL condition.
    virtual double operator()(const Eigen::MatrixXd &u) const = 0;
};


//----------------CFLBegin1----------------
/// Compute the CFL condition for generic models.
/** Compute the maximum eigenvalue given all the data, and compute the
 *  CFL restricted time-step accordingly.
 *
 *  Note: this requires looping over the entire grid, and computing a
 *  reduction, i.e. max.
 */
template <int DiscretizationType>
class StandardCFLCondition;

template <>
class StandardCFLCondition <FVM> : public CFLCondition {
  public:

    StandardCFLCondition(const Grid &grid,
                         const std::shared_ptr<Model> &model,
                         double cfl_number);

    virtual double operator()(const Eigen::MatrixXd &u) const override;

  private:
    Grid grid;
    std::shared_ptr<Model> model;
    double cfl_number;
};
//----------------CFLEnd1----------------

//----------------CFLBegin2----------------
template <>
class StandardCFLCondition <DG> : public CFLCondition {
  public:

    StandardCFLCondition(const Grid &grid,
                         const std::shared_ptr<Model> &model,
                         const DGHandler &dg_handler,
                         double cfl_number);

    virtual double operator()(const Eigen::MatrixXd &u) const override;

  private:
    Grid grid;
    std::shared_ptr<Model> model;
    DGHandler dg_handler;
    double cfl_number;
};
//----------------CFLEnd2----------------


/// make CFL condition for FVM
std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid,
                   const std::shared_ptr<Model> &model,
                   double cfl_number);

/// make CFL condition for DG
std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid,
                   const std::shared_ptr<Model> &model,
                   const DGHandler &dg_handler,
                   double cfl_number);

#endif // HYPSYS1D_CFL_CONDITION_HPP
