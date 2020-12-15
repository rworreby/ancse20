#include "../../include/ancse/polynomial_basis.hpp"

//----------------PolyBasisBegin----------------
/// Computes the Legendre polynomial basis
/// at a given reference point xi \in [0,1]
Eigen::VectorXd PolynomialBasis :: operator() (double xi) const
{
    Eigen::VectorXd basis(p+1);


    return Eigen::VectorXd::Zero(p+1);
}

/// Computes the derivative of Legendre polynomial basis
/// at a given reference point xi \in [0,1]
Eigen::VectorXd PolynomialBasis :: deriv (double xi) const
{
    Eigen::VectorXd basis_deriv(p+1);

    return Eigen::VectorXd::Zero(p+1);
}
//----------------PolyBasisEnd----------------
