#ifndef FVMSCALAR1D_BOUNDARY_CONDITION_HPP
#define FVMSCALAR1D_BOUNDARY_CONDITION_HPP

#include <Eigen/Dense>
#include <memory>

/// Interface for enforcing boundary conditions through ghost-cells.
class BoundaryCondition {
  public:
    // BoundaryCondition is an abstract base class. It needs a virtual
    // destructor.
    virtual ~BoundaryCondition() = default;

    /// Set the value of the ghost-cells in `u`.
    virtual void operator()(Eigen::VectorXd &u) const = 0;
};

//----------------PeriodicBCDeclBegin----------------
/// Boundary conditions that mimic a periodic domain.
class PeriodicBC : public BoundaryCondition {
  public:
    explicit PeriodicBC(int n_ghost);

    virtual void operator()(Eigen::VectorXd &u) const override;

  private:
    int n_ghost;
};
//----------------PeriodicBCDeclEnd----------------

//----------------OutflowBCDeclBegin----------------
/// These boundary conditions mimic a permeable surface.
/** A very simple way of mimicing this is to say that the value of the
 *  ghost-cells is the same as the value just inside the physical domain.
 */
class OutflowBC : public BoundaryCondition {
  public:
    explicit OutflowBC(int n_ghost);

    virtual void operator()(Eigen::VectorXd &u) const override;

  private:
    int n_ghost;
};
//----------------OutflowBCDeclEnd----------------

/// Create the requested boundary conditions.
/** Note: This is a factory. The purpose is to select a boundary condition at
 *        runtime, e.g. through a configuration file such as `config.json`.
 *
 * @param n_ghost  number of ghost cell.
 * @param bc_key   name/identifies of the boundary condition
 */
std::shared_ptr<BoundaryCondition>
make_boundary_condition(int n_ghost, const std::string &bc_key);

#endif // FVMSCALAR1D_BOUNDARY_CONDITION_HPP
