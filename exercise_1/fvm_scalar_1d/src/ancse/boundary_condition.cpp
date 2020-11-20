#include <ancse/boundary_condition.hpp>

#include <fmt/format.h>

//----------------PeriodicBCDefnBegin----------------
PeriodicBC::PeriodicBC(int n_ghost) : n_ghost(n_ghost) {}

void PeriodicBC::operator()(Eigen::VectorXd &u) const {
    using index_t = Eigen::Index;
    index_t n_cells = u.size();

    for (index_t i = 0; i < n_ghost; ++i) {
        u[i] = u[n_cells - 2*n_ghost + i];
        u[n_cells - n_ghost + i] = u[n_ghost + i];
    }
}
//----------------PeriodicBCDefnEnd----------------

//----------------OutflowBCDefnBegin----------------
OutflowBC::OutflowBC(int n_ghost) : n_ghost(n_ghost) {}

void OutflowBC::operator()(Eigen::VectorXd &u) const {
    using index_t = Eigen::Index;
    index_t n_cells = u.size();

    for (index_t i = 0; i < n_ghost; ++i) {
        u[i] = u[n_ghost];
        u[n_cells - n_ghost + i] = u[n_cells - n_ghost - 1];
    }
}
//----------------OutflowBCDefnEnd----------------

std::shared_ptr<BoundaryCondition>
make_boundary_condition(int n_ghost, const std::string &bc_key) {


    // register you boundary conditions here.
    if (bc_key == "periodic") {
        return std::make_shared<PeriodicBC>(n_ghost);
    }

    if (bc_key == "outflow") {
        return std::make_shared<OutflowBC>(n_ghost);
    }

    throw std::runtime_error(
        fmt::format("Unknown boundary condition. [{}]", bc_key));
}
