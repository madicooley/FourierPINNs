#pragma once

/**
 * @class PDE
 * @brief Represents a partial differential equation (PDE) to be solved using neural networks.
 *
 * The PDE class encapsulates the details and parameters of a partial differential equation. This class serves as a base or utility class
 * within a neural network framework to integrate PDE constraints into the training process. It can be extended or used as is to represent
 * specific PDE problems in the context of physics-informed neural networks (PINNs).
 *
 * @extends torch::nn::Module
 */
class PDE : public torch::nn::Module {
public:
    /**
     * Constructs a PDE object with a specified problem identifier.
     *
     * This constructor initializes the PDE with a string that represents or describes the PDE problem. It's suitable for scenarios
     * where the PDE parameters are predefined or not directly involved in the neural network's computational graph.
     *
     * @param pde_problem A string that uniquely identifies or describes the PDE problem.
     */
    PDE(std::string pde_problem_name, bool verbose = false)
    : pde_problem(pde_problem_name),
      verbose_(verbose) {
          if (verbose_) {
              std::cout<<"PDE Constructor. pde_problem="<<pde_problem_name<<"\n";
          }
    }

    torch::Tensor forwardPDE(torch::Tensor& u, torch::Tensor& x) {
        torch::Tensor f_pred;
        if      (pde_problem=="poisson")   f_pred = PoissonPDE(u, x);
        else if (pde_problem=="allencahn") f_pred = AllenCahnPDE(u, x);
        else std::cerr<<"WARNING: PDE not implemented in (1D) PDE class.\n";
        return f_pred;
    }

    torch::Tensor forwardPDE(torch::Tensor& u, torch::Tensor& x, torch::Tensor& y) {
        torch::Tensor f_pred;
        if      (pde_problem=="poisson")         f_pred = PoissonPDE(u, x, y);
        else if (pde_problem=="allencahn")       f_pred = AllenCahnPDE(u, x, y);
        else if (pde_problem=="nonss_allencahn") f_pred = NonSSAllenCahnPDE(u, x, y);
        else if (pde_problem=="onewaywave_inhomo") f_pred = OneWayWavePDE(u, x, y);
        else std::cerr<<"WARNING: PDE not implemented in (2D) PDE class.\n";
        return f_pred;
    }

    torch::Tensor forwardBasisPDE(torch::Tensor& u, torch::Tensor& A) {
        torch::Tensor f_basis;
        if (pde_problem=="poisson" || pde_problem=="allencahn") {
            TORCH_CHECK(A.sizes().size() == 2, "forwardBasisPDE expects X as [N, dim].");
            if (A.size(1) == 1) {
                f_basis = BasisSecondDerivCols(u, A, 0);
            } else if (A.size(1) == 2) {
                f_basis = BasisSecondDerivCols(u, A, 0) + BasisSecondDerivCols(u, A, 1);
            } else {
                TORCH_CHECK(false, "forwardBasisPDE only supports dim 1 or 2 for poisson/allencahn.");
            }
        } else if (pde_problem=="onewaywave_inhomo") {
            TORCH_CHECK(A.sizes().size() == 2 && A.size(1) >= 2,
                        "onewaywave_inhomo forwardBasisPDE expects X with columns [t,x].");
            auto u_t = BasisGradCols(u, A, 0);
            auto u_x = BasisGradCols(u, A, 1);
            f_basis = u_t + 10.0 * u_x;
        } else if (pde_problem=="nonss_allencahn") {
            TORCH_CHECK(A.sizes().size() == 2 && A.size(1) >= 2,
                        "nonss_allencahn forwardBasisPDE expects X with columns [t,x].");
            auto u_t = BasisGradCols(u, A, 0);
            auto u_xx = BasisSecondDerivCols(u, A, 1);
            f_basis = u_t - 0.0001 * u_xx;
        } else {
            std::cerr<<"WARNING: forwardBasisPDE not implemented in PDE class for pde="<<pde_problem<<"\n";
        }
        return f_basis;
    }

    torch::Tensor getPDELoss(torch::Tensor& f_pred) {
        auto f_loss = torch::mean((f_pred).pow(2));
        return f_loss;
    }

    torch::Tensor getPDELoss(torch::Tensor& f_pred, torch::Tensor& f_true) {
        auto f_loss = torch::mean((f_pred - f_true).pow(2));
        return f_loss;
    }

    torch::Tensor getGrad(torch::Tensor& u, torch::Tensor& x, int dim_index) {
        (void)dim_index;
        torch::Tensor oneslikeu = torch::ones_like(u);
        torch::Tensor u_x = torch::autograd::grad({ u }, { x }, {oneslikeu}, true, true)[0];

        return u_x;
    }

    torch::Tensor NonSSAllenCahnPDE(torch::Tensor& u, torch::Tensor& t, torch::Tensor& x) { // non steady state allen-cahn
        // u_func:  given from simulation [Raissi et al. original pinn paper]
        // f_func:
        // ut − 0.0001uxx + 5u^3 − 5u = 0, t ∈ [0, 1], x ∈ [−1, 1],
        // u(0, x)   = x^2 cos(πx),
        // u(t, −1)  = u(t, 1),
        // ux(t, −1) = ux(t, 1).
        auto oneslikeu = torch::ones_like(u);

        auto u_t  = torch::autograd::grad({ u }, { t }, {oneslikeu}, true, true)[0];
        auto u_x  = torch::autograd::grad({ u }, { x }, {oneslikeu}, true, true)[0];
        auto u_xx = torch::autograd::grad({ u_x }, { x }, { oneslikeu }, true, true)[0];

        auto f = u_t - 0.0001*u_xx + 5*u*u*u - 5*u;
        return f;
    }

    torch::Tensor OneWayWavePDE(torch::Tensor& u, torch::Tensor& t, torch::Tensor& x) {
        auto oneslikeu = torch::ones_like(u);
        auto u_t = torch::autograd::grad({ u }, { t }, {oneslikeu}, true, true)[0];
        auto u_x = torch::autograd::grad({ u }, { x }, {oneslikeu}, true, true)[0];
        return u_t + 10.0 * u_x;
    }

    torch::Tensor BasisGradCols(torch::Tensor& u, torch::Tensor& X, int dim_index) {
        std::vector<torch::Tensor> der_cols;
        der_cols.reserve(static_cast<std::size_t>(u.size(1)));
        for (int64_t col = 0; col < u.size(1); ++col) {
            auto u_row = u.index({torch::indexing::Slice(), col});
            auto grad_u = torch::autograd::grad(
                {u_row}, {X},
                /*grad_outputs=*/{torch::ones_like(u_row)},
                /*retain_graph=*/true,
                /*create_graph=*/true,
                /*allow_unused=*/true
            )[0];
            if (!grad_u.defined()) {
                grad_u = torch::zeros_like(X);
            }
            auto u_x = grad_u.index({torch::indexing::Slice(), dim_index}).unsqueeze(1);
            der_cols.push_back(u_x);
        }
        return torch::cat(der_cols, /*dim=*/1);
    }

    torch::Tensor BasisSecondDerivCols(torch::Tensor& u, torch::Tensor& X, int dim_index) {
        std::vector<torch::Tensor> der_cols;
        der_cols.reserve(static_cast<std::size_t>(u.size(1)));
        for (int64_t col = 0; col < u.size(1); ++col) {
            auto u_row = u.index({torch::indexing::Slice(), col});
            auto grad_u = torch::autograd::grad(
                {u_row}, {X},
                /*grad_outputs=*/{torch::ones_like(u_row)},
                /*retain_graph=*/true,
                /*create_graph=*/true,
                /*allow_unused=*/true
            )[0];
            if (!grad_u.defined()) {
                grad_u = torch::zeros_like(X);
            }
            auto u_x = grad_u.index({torch::indexing::Slice(), dim_index}).unsqueeze(1);
            auto grad_ux = torch::autograd::grad(
                {u_x}, {X},
                /*grad_outputs=*/{torch::ones_like(u_x)},
                /*retain_graph=*/true,
                /*create_graph=*/true,
                /*allow_unused=*/true
            )[0];
            if (!grad_ux.defined()) {
                grad_ux = torch::zeros_like(X);
            }
            auto u_xx = grad_ux.index({torch::indexing::Slice(), dim_index}).unsqueeze(1);
            der_cols.push_back(u_xx);
        }
        return torch::cat(der_cols, /*dim=*/1);
    }

    torch::Tensor PoissonPDE(torch::Tensor& u, torch::Tensor& x) {
        torch::Tensor laplacian_u = torch::zeros_like(u);

        auto u_x  = getGrad(u,   x, 0);
        auto u_xx = getGrad(u_x, x, 0);

        laplacian_u += u_xx;

        return laplacian_u;
    }

    torch::Tensor PoissonPDE(torch::Tensor& u, torch::Tensor& x, torch::Tensor& y) {
        torch::Tensor laplacian_u = torch::zeros_like(u);

        auto u_x  = getGrad(u,   x, 0);
        auto u_xx = getGrad(u_x, x, 0);

        auto u_y  = getGrad(u,   y, 1);
        auto u_yy = getGrad(u_y, y, 1);

        laplacian_u += u_xx;
        laplacian_u += u_yy;

        return laplacian_u;
    }

    torch::Tensor AllenCahnPDE(torch::Tensor& u, torch::Tensor& x) {
        torch::Tensor laplacian_u = torch::zeros_like(u);
        auto oneslikeu            = torch::ones_like(u);

        auto u_x     = torch::autograd::grad({ u }, { x }, {oneslikeu}, true, true)[0];
        auto u_xx    = torch::autograd::grad({ u_x }, { x }, { oneslikeu }, true, true)[0];
        laplacian_u += u_xx;

        auto reaction = u*(u.pow(2) - 1.0);
        return laplacian_u+reaction; // [n x 1]
    }

    torch::Tensor AllenCahnPDE(torch::Tensor& u, torch::Tensor& x, torch::Tensor& y) {
        torch::Tensor laplacian_u = torch::zeros_like(u);
        auto oneslikeu            = torch::ones_like(u);

        auto u_x = torch::autograd::grad({ u }, { x }, {oneslikeu}, true, true)[0];
        auto u_y = torch::autograd::grad({ u }, { y }, {oneslikeu}, true, true)[0];

        laplacian_u += torch::autograd::grad({ u_x }, { x }, { oneslikeu }, true, true)[0];
        laplacian_u += torch::autograd::grad({ u_y }, { y }, { oneslikeu }, true, true)[0];

        auto reaction = u * (u.pow(2) - 1.0);
        return laplacian_u+reaction; // [n x 1]
    }

    std::string getPDEProblem() {return pde_problem;}



private:
    std::string pde_problem; ///< String describing the PDE problem.
    bool verbose_ = false;
    float pde_parameter;
};
