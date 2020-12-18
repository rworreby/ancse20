#include <ancse/cfl_condition.hpp>

#include <Eigen/Dense>


StandardCFLCondition <FVM>
:: StandardCFLCondition(const Grid &grid,
                        const std::shared_ptr<Model> &model,
                        double cfl_number)
    : grid(grid), model(model), cfl_number(cfl_number) {}

double StandardCFLCondition <FVM>
:: operator()(const Eigen::MatrixXd &u) const {

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;

    double a_max = 0.0;

    return 0.;

}

StandardCFLCondition <DG>
:: StandardCFLCondition(const Grid &grid,
                        const std::shared_ptr<Model> &model,
                        const DGHandler &dg_handler,
                        double cfl_number)
    : grid(grid), model(model), dg_handler (dg_handler),
      cfl_number(cfl_number) {}

double StandardCFLCondition <DG>
:: operator()(const Eigen::MatrixXd &u) const {

    auto n_cells = grid.n_cells;
    auto n_ghost = grid.n_ghost;

    return 0.0;
}


/// make CFL condition for FVM
std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid,
                   const std::shared_ptr<Model> &model,
                   double cfl_number) {

    return std::make_shared<StandardCFLCondition <FVM>>(grid, model,
                                                        cfl_number);

}

/// make CFL condition for DG
std::shared_ptr<CFLCondition>
make_cfl_condition(const Grid &grid,
                   const std::shared_ptr<Model> &model,
                   const DGHandler &dg_handler,
                   double cfl_number) {

    return std::make_shared<StandardCFLCondition <DG>>
            (grid, model, dg_handler, cfl_number);
}
