#ifndef HYPSYS1D_RECONSTRUCTION_HPP
#define HYPSYS1D_RECONSTRUCTION_HPP

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include <ancse/grid.hpp>
#include <ancse/model.hpp>
#include <ancse/limiters.hpp>
#include <ancse/rate_of_change.hpp>
#include <ancse/simulation_time.hpp>


class PWConstantReconstruction {
  public:
    void set(const Eigen::MatrixXd &u) const {
        up.resize(u.rows(),u.cols());
        up = u;
    }

    /// Compute the left and right trace at the interface i + 1/2.
    /** Note: This API is agnostic to the number of cell-averages required
     *        by the method. Therefore, reconstructions with different stencil
     *        sizes can implement this API; and this call can be used in parts
     *        of the code that do not need to know about the details of the
     *        reconstruction.
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(int i) const {
        return (*this)(up.col(i), up.col(i+1));
    }

    /// Compute the left and right trace at the interface.
    /** Piecewise constant reconstruction of the left and right trace only
     *  requires the cell-average to the left and right of the interface.
     *
     *  Note: Compared to the other overload this reduces the assumption on
     *        how the cell-averages are stored. This is useful when testing and
     *        generally makes the function useful in more situations.
     */
    inline
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(Eigen::VectorXd ua, Eigen::VectorXd ub) const {
        return {std::move(ua), std::move(ub)};
    }

private:
  mutable Eigen::MatrixXd up;
};


//----------------ReconstructionBegin----------------
enum {Conserved, Primitive};

template <class SlopeLimiter, int ReconstructionVars>
class PWLinearReconstruction;

template <class SlopeLimiter>
class PWLinearReconstruction<SlopeLimiter, Conserved> {
  public:
    explicit PWLinearReconstruction(const SlopeLimiter &slope_limiter)
        : slope_limiter(slope_limiter) {}

    void set(const Eigen::MatrixXd &u) const {
        up.resize(u.rows(),u.cols());
        up = u;
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(int i) const {
        return (*this)(up.col(i - 1), up.col(i), up.col(i + 1), up.col(i + 2));
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(const Eigen::VectorXd &ua,
               const Eigen::VectorXd &ub,
               const Eigen::VectorXd &uc,
               const Eigen::VectorXd &ud) const
    {
        auto sL{ ub - ua };
        auto sM{ uc - ub };
        auto sR{ ud - uc };

        Eigen::VectorXd uL(ua.size());
        Eigen::VectorXd uR(ua.size());

        for (size_t i = 0; i < ua.size(); i++) {
            uL(i) = ub(i) + 0.5 * slope_limiter(sL(i), sM(i));
            uR(i) = uc(i) + 0.5 * slope_limiter(sM(i), sR(i));
        }

        return {std::move(uL), std::move(uR)};
    }

  private:
    SlopeLimiter slope_limiter;
    mutable Eigen::MatrixXd up;
};

template <class SlopeLimiter>
class PWLinearReconstruction<SlopeLimiter, Primitive> {
  public:
    explicit PWLinearReconstruction(const std::shared_ptr<Model> &model,
                                    const SlopeLimiter &slope_limiter)
        : model(model), slope_limiter(slope_limiter) {}

    void set(const Eigen::MatrixXd &u) const
    {
        up.resize(u.rows(),u.cols());
        for (int i = 0; i < u.cols(); ++i) {
            up.col(i) = model->cons_to_prim(u.col(i));
        }
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(int i) const {

        auto [uLp, uRp] = (*this)(up.col(i - 1), up.col(i), up.col(i + 1), up.col(i + 2));

        return {std::move(model->prim_to_cons(uLp)), std::move(model->prim_to_cons(uRp))};
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    operator()(const Eigen::VectorXd &ua,
               const Eigen::VectorXd &ub,
               const Eigen::VectorXd &uc,
               const Eigen::VectorXd &ud) const
    {
        auto sL{ ub - ua };
        auto sM{ uc - ub };
        auto sR{ ud - uc };

        Eigen::VectorXd uL(ua.size());
        Eigen::VectorXd uR(ua.size());

        for (size_t i = 0; i < ua.size(); i++) {
            uL(i) = ub(i) + 0.5 * slope_limiter(sL(i), sM(i));
            uR(i) = uc(i) + 0.5 * slope_limiter(sM(i), sR(i));
        }

        return {std::move(uL), std::move(uR)};
    }

  private:
    std::shared_ptr<Model> model;
    SlopeLimiter slope_limiter;

    mutable Eigen::MatrixXd up;
};
//----------------ReconstructionEnd----------------



#endif // HYPSYS1D_RATE_OF_CHANGE_HPP
