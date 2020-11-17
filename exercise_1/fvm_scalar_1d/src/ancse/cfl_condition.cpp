#include <ancse/cfl_condition.hpp>


//----------------StandardCFLConditionDefnBegin----------------
StandardCFLCondition::StandardCFLCondition(const Grid &grid,
                                           const Model &model,
                                           double cfl_number)
    : grid(grid), model(model), cfl_number(cfl_number) {}

double StandardCFLCondition::operator()(const Eigen::VectorXd &u) const {

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;


    return 0.0;
}
//----------------StandardCFLConditionDefnEnd----------------


std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid, const Model &model, double cfl_number) {
    // implement this 'factory' for your CFL condition.
    return std::make_shared<StandardCFLCondition>(grid, model, cfl_number);
    return nullptr;
}
