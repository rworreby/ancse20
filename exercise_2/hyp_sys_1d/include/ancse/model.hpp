#ifndef HYPSYS1D_MODEL_HPP
#define HYPSYS1D_MODEL_HPP

#include <cmath>
#include <memory>

#include <Eigen/Dense>
#include <ancse/config.hpp>

/// Interface for implementing different models,
/// eg. Euler equations, Shallow-water equations
///
/// Add more functions to this interface if needed.
class Model {
  public:
    virtual ~Model() = default;

    virtual Eigen::VectorXd flux(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &u) const = 0;
    virtual double max_eigenvalue(const Eigen::VectorXd &u) const = 0;

    virtual Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u) const = 0;
    virtual Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u) const = 0;

    virtual Eigen::VectorXd roe_avg(const Eigen::VectorXd &uL,
                                    const Eigen::VectorXd &uR) const = 0;

    virtual int get_nvars() const = 0;
    virtual std::string get_name() const = 0;
};

class Burgers : public Model {
  public:

    Eigen::VectorXd flux(const Eigen::VectorXd &u) const override
    {
        Eigen::VectorXd f(n_vars);
        f(0) = 0.5*u(0)*u(0);

        return f;
    }

    Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const override
    {
        Eigen::VectorXd eigvals(n_vars);
        eigvals(0) = u(0);

        return eigvals;
    }

    Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &) const override
    {
        Eigen::MatrixXd eigvecs(n_vars, n_vars);
        eigvecs (0,0) = 1;

        return eigvecs;
    }

    double max_eigenvalue(const Eigen::VectorXd &u) const override {
        return (eigenvalues(u).cwiseAbs()).maxCoeff();
    }

    Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u) const override {
        return u;
    }

    Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u) const override {
        return u;
    }

    Eigen::VectorXd roe_avg(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const override
    {return Eigen::VectorXd::Zero(n_vars);}

    int get_nvars() const override
    {
        return n_vars;
    }

    std::string get_name() const override
    {
        return "burgers";
    }

  private:
    int n_vars = 1;
};

/// Euler equations
class Euler : public Model {
  public:

    Eigen::VectorXd flux(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd eigenvalues(const Eigen::VectorXd &u) const override;

    Eigen::MatrixXd eigenvectors(const Eigen::VectorXd &u) const override;

    double max_eigenvalue(const Eigen::VectorXd &u) const override;

    Eigen::VectorXd cons_to_prim(const Eigen::VectorXd &u_cons) const override;

    Eigen::VectorXd prim_to_cons(const Eigen::VectorXd &u_prim) const override;

    Eigen::VectorXd roe_avg(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const override;

    //----------------ModelEulerBegin----------------
    inline std::tuple<double, double, double>
    primitive(const Eigen::VectorXd &u_cons) const
    {
        double rho{ u_cons[0] };
        double v{ u_cons[1] / u_cons[0] };
        double p{ pressure(rho, v, u_cons[2]) };

        return std:make_tuple(rho, v, p);
    }

    inline double pressure(double rho, double v, double E) const
    {
        return (gamma-1)*(E - 0.5*rho*v*v);
    }

    inline double sound_speed(double rho, double p) const
    {
        return sqrt(gamma*p/rho);
    }

    inline double enthalpy(double rho, double E, double p) const
    {
        return (E+p)/rho;
    }

    inline double energy(double rho, double v, double p) const
    {
        return (p/(gamma-1) + 0.5*rho*v*v);
    }

    inline Eigen::VectorXd eigenvalues(double v, double c) const
    {

        Eigen::VectorXd eigvals(n_vars);
        ///  ANCSE_COMMENT Compute eigenvalues
        return eigvals;
    }

    inline Eigen::MatrixXd eigenvectors(double v, double H) const
    {
        Eigen::MatrixXd eigvecs(n_vars, n_vars);
        ///  ANCSE_COMMENT Compute eigenvectors

        return eigvecs;
    }
    //----------------ModelEulerEnd----------------


    void set_gamma(double gamma_)
    {
        gamma = gamma_;
    }

    double get_gamma() const
    {
        return gamma;
    }

    int get_nvars() const override
    {
        return n_vars;
    }

    std::string get_name() const override
    {
        return "euler";
    }

  private:
    int n_vars = 3;
    double gamma = 5./3.;
};


std::shared_ptr<Model> make_model (const nlohmann::json &config);

#endif // HYPSYS1D_MODEL_HPP
