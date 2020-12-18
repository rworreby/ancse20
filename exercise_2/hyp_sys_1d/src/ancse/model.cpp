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
    double rho, p, v;
    std::tie(rho, p, v) = primitive(u);
    double E = energy(rho, v, p);

    Eigen::VectorXd flux(n_vars);
    flux << u(1), u(1)*v+p, v*(E + p);

    return flux;
}

Eigen::VectorXd Euler::eigenvalues(const Eigen::VectorXd &u) const
{
    double rho, v, p;
    std::tie(rho, v, p) = primitive(u);

    double c{ sound_speed(rho, p) };

    return eigenvalues(v, c);
}

Eigen::MatrixXd Euler::eigenvectors(const Eigen::VectorXd &u) const
{
    double rho, v, p;
    std::tie(rho, v, p) = primitive(u);

    double E = energy(rho, v, p);
    double H = enthalpy(rho, E, p);

    return eigenvectors(v, H);
}

double Euler::max_eigenvalue(const Eigen::VectorXd &u) const
{
    return (eigenvalues(u).cwiseAbs()).maxCoeff();
}

Eigen::VectorXd Euler::cons_to_prim(const Eigen::VectorXd &u_cons) const
{
    double rho, v, p;
    std::tie(rho, v, p) = primitive(u_cons);

    Eigen::VectorXd u_prim(n_vars);
    u_prim << rho, v, p;

    return u_prim;
}

Eigen::VectorXd Euler::prim_to_cons(const Eigen::VectorXd &u_prim) const
{
    double E = energy(u_prim(0), u_prim(1), u_prim(2));

    Eigen::VectorXd u_cons(n_vars);
    u_cons << u_prim(0), u_prim(0)*u_prim(1), E;
    return u_cons;
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
    REGISTER_MODEL("euler", Euler)

    throw std::runtime_error(
        fmt::format("Unknown model. {}", std::string(config["flux"])));
}
