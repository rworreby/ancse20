#include <ancse/cfl_condition.hpp>
#include <limits>

//----------------StandardCFLConditionDefnBegin----------------
StandardCFLCondition::StandardCFLCondition(const Grid &grid,
                                           const Model &model,
                                           double cfl_number)
    : grid(grid), model(model), cfl_number(cfl_number) {}

double StandardCFLCondition::operator()(const Eigen::VectorXd &u) const {
    assert(cfl_number < 1.0);

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;

    double max_fp{ std::numeric_limits<double>::lowest() };
    for (int i = n_ghost-1; i < n_cells - n_ghost; i++) {
        if(max_fp < model.max_eigenvalue(u(i))){
            max_fp = model.max_eigenvalue(u(i));
        }
    }

    double const dt{ cfl_number * grid.dx / max_fp };

    return dt;
}
//----------------StandardCFLConditionDefnEnd----------------


std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid, const Model &model, double cfl_number) {
    // implement this 'factory' for your CFL condition.
    return std::make_shared<StandardCFLCondition>(grid, model, cfl_number);
    return nullptr;
}
