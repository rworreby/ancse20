#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include "gradient.hpp"
#include "hllc.hpp"
#include "mesh.hpp"
#include "slope_limiter.hpp"

// Note: this class will compute the rate of change due to the fluxes.
// Note: the reason we made this a class is that it allows you to allocate
//       buffers, once at the beginning of the simulation. Add these buffers
//       as needed.
class FluxRateOfChange {
  public:
    explicit FluxRateOfChange(int n_cells) {

        dwdx = Eigen::MatrixXd(n_cells, 4);
        dwdy = Eigen::MatrixXd(n_cells, 4);
        pvars = Eigen::MatrixXd(n_cells, 4);

    }

    void operator()(Eigen::MatrixXd &dudt,
                    const Eigen::MatrixXd &u,
                    const Mesh &mesh) const {


        // Compute the rate of change of u.
        //// ANCSE_COMMENT
        // Note: Please use the method `computeFlux` to abstract
        // away the details of computing the flux through a
        // given interface.
        //// ANCSE_COMMENT
        // Note: You can use `assert_valid_flux` to check
        // if what `computeFlux` returns makes any sense.
        //// ANCSE_COMMENT
        // Note: Do not assume `dudt` is filled with zeros.

        dudt.resize(u.rows(), u.cols());
        dudt.setZero();

        int n_cells = mesh.getNumberOfTriangles();

        for (int i = 0; i < n_cells; ++i) {
            pvars.row(i) = euler::primitiveVars(u.row(i));
        }

        // Note: This will only serve for the linear recontruction.
        // if the computation of the gradients is not done in gradient.hpp, then
        // dwdx and dwdy contains only zeros

        compute_gradients(dwdx, dwdy, pvars, mesh);

    }

    void assert_valid_flux(const Mesh &mesh,
                           int i,
                           int k,
                           const EulerState &nF) const {
        // This is mostly for debugging (but also important to check in
        // real simulations!): Make sure our flux contribution is not
        // nan (ie. it is not not a number, ie it is a number)
        if (!euler::isValidFlux(nF)) {
            // clang-format off
            throw std::runtime_error(
                "invalid value detected in numerical flux, " + euler::to_string(nF)
                + "\nat triangle: " + std::to_string(i)
                + "\nedge:        " + std::to_string(k)
                + "\nis_boundary: " + std::to_string(!mesh.isValidNeighbour(i, k)));
            // clang-format on
        }
    }

    /// Compute the flux through the k-th interface of cell i.
    EulerState computeFlux(const Eigen::MatrixXd &U,
                           int i,
                           int k,
                           const Mesh &mesh) const {
        auto boundary_type = mesh.getBoundaryType(i, k);

        if (boundary_type == Mesh::BoundaryType::INTERIOR_EDGE) {
            return computeInteriorFlux(U, i, k, mesh);
        } else {
            if (boundary_type == Mesh::BoundaryType::OUTFLOW_EDGE) {
                return computeOutflowFlux(U, i, k, mesh);
            } else /* boundary_type == Mesh::BoundaryType::WING_EDGE */
            {
                return computeReflectiveFlux(U, i, k, mesh);
            }
        }
    }

//----------------OutflowBCBegin----------------
    /// Compute the outflow flux through the k-th interface of cell i.
    /** Note: you know that edge k is an outflow edge.
     */
    EulerState computeOutflowFlux(const Eigen::MatrixXd &U,
                                  int i,
                                  int k,
                                  const Mesh &mesh) const {
        // Implement the outflow flux boundary condition.

        auto normal = mesh.getUnitNormal(i, k);

        return EulerState{};
    }
//----------------OutflowBCEnd----------------

//----------------ReflectiveBCBegin----------------
    /// Compute the reflective boundary flux through the k-th edge of cell i.
    /** Note: you know that edge k is a reflective/wall boundary edge.
     */
    EulerState computeReflectiveFlux(const Eigen::MatrixXd &U,
                                     int i,
                                     int k,
                                     const Mesh &mesh) const {


        // Implement the reflective flux boundary condition.

        auto normal = mesh.getUnitNormal(i, k);

        return EulerState{};
    }
//----------------ReflectiveBCEnd----------------

//----------------InteriorFluxBegin----------------
    /// Compute the flux through the k-th interface of cell i.
    /** Note: This edge is an interior edge, therefore approximate the flux
     * through this edge with the appropriate FVM formulas.
     */
    EulerState computeInteriorFlux(const Eigen::MatrixXd &U,
                                   int i,
                                   int k,
                                   const Mesh &mesh) const {

        // Reconstruct the trace values of U and compute
        // the numerical flux through the k-th interface of
        // cell i.

        int j = mesh.getNeighbour(i, k);
        auto normal = mesh.getUnitNormal(i, k);

        auto xi = mesh.getCellCenter(i);
        auto xj = mesh.getCellCenter(j);
        auto x_ij = mesh.getEdgeCenter(i, k);

        auto grad_wi = getGradient(i);
        auto grad_wj = getGradient(j);

        // w are the primitive variables
        // u are the conservative variables

        EulerState uL, uR;
        EulerState wL, wR;

        return EulerState{};
    }
//----------------InteriorFluxEnd----------------


    double limited_slope(const Eigen::Vector2d &grad_L,
                         const Eigen::Vector2d &grad_R,
                         const Eigen::Vector2d &dx) const {

        return slope_limiter(grad_L.dot(dx), grad_R.dot(dx));
    }

    EulerStateGradient getGradient(int i) const {
        EulerStateGradient grad_w;

        // Compute here the gradients
        return grad_w;

    }


  private:

    mutable Eigen::MatrixXd dwdx;
    mutable Eigen::MatrixXd dwdy;
    mutable Eigen::MatrixXd pvars;
};
