#include <ancse/model.hpp>

#include <iostream>
#include <fmt/format.h>


///------------------///
/// Euler equations  ///
///------------------///

// For task 1g you can use the functions implemented in model.hpp
//----------------ModelEulerBegin----------------
Eigen::VectorXd Euler::flux(const Eigen::VectorXd &u) const
{
    return Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::eigenvalues(const Eigen::VectorXd &u) const
{
    return Eigen::VectorXd::Zero(n_vars);
}

Eigen::MatrixXd Euler::eigenvectors(const Eigen::VectorXd &u) const
{
    return Eigen::MatrixXd::Zero(n_vars, n_vars);
}

double Euler::max_eigenvalue(const Eigen::VectorXd &u) const
{
    return 0;
}


Eigen::VectorXd Euler::cons_to_prim(const Eigen::VectorXd &u_cons) const
{
    return Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::prim_to_cons(const Eigen::VectorXd &u_prim) const
{
    return Eigen::VectorXd::Zero(n_vars);
}

Eigen::VectorXd Euler::roe_avg(const Eigen::VectorXd &uL,
                               const Eigen::VectorXd &uR) const
{
    // left state
    double rhoL, vL, pL;
    std::tie (rhoL, vL, pL) = this->primitive(uL);
    double EL = this->energy(rhoL, vL, pL);
    double HL = this->enthalpy(rhoL, EL, pL);

    // right state
    double rhoR, vR, pR;
    std::tie (rhoR, vR, pR) = this->primitive(uR);
    double ER = this->energy(rhoR, vR, pR);
    double HR = this->enthalpy(rhoR, ER, pR);

    // Roe state
    double rhoS = 0.5*(rhoL + rhoR);
    double tmp = (sqrt(rhoL) + sqrt(rhoR));
    double vS  = (sqrt(rhoL)*vL + sqrt(rhoR)*vR)/tmp;
    double HS  = (sqrt(rhoL)*HL + sqrt(rhoR)*HR)/tmp;

    Eigen::VectorXd uS(n_vars);
    uS(0) = rhoS;
    uS(1) = uS(0)*vS;
    uS(2) = rhoS*HS/gamma + 0.5*(gamma-1)/gamma*rhoS*vS*vS;

    return uS;
}
//----------------ModelEulerEnd----------------


#define REGISTER_MODEL(token, ModelType)      \
    if (config["model"] == (token)) {         \
        return std::make_shared<ModelType>(); \
    }

std::shared_ptr<Model> make_model (const nlohmann::json &config)
{
    REGISTER_MODEL("burgers", Burgers)
    // implement and register your models here

    throw std::runtime_error(
        fmt::format("Unknown model. {}", std::string(config["flux"])));
}
