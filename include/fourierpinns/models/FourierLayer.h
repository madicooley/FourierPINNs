#ifndef VANILLAFOURIERLAYER_H
#define VANILLAFOURIERLAYER_H

#include <torch/torch.h>
#include <cmath>
#include <memory>
#include "CustomNet.h"
#include "fourierpinns/config/Args.h"
#include "fourierpinns/basis/FourierBasisBuilder.h"

// No custom backward
class VanillaFourierLayerImpl : public CustomNetwork {
public:
    /**
     * @brief Construct a Fourier layer from a ModelConfig.
     *
     * Initializes Fourier basis settings, builds tensor-product basis metadata,
     * and allocates the final linear layer.
     */
    VanillaFourierLayerImpl(const ModelConfig& args,
                             std::string pde_problem,
                             torch::Device device,
                             torch::Dtype dtype)
    : device_(device), dtype_(dtype), pde_problem_(pde_problem) {
        inputDim_  = args.input_dim;
        outputDim_ = args.output_dim;
        modelType_ = args.model_type;
        verbose_ = args.verbose;
        FourierLayerArgs fl_args = getFourierLayerArgs(args);
        initModelArgs(fl_args);
        initFourierBasis(fl_args, args.domain_bounds);
        if (verbose_) {
            printModelInfo();
        }
        initializeFinalLayer();
    }

    virtual ~VanillaFourierLayerImpl() {}

    /**
     * @brief Forward pass using input coordinates.
     *
     * Builds the Fourier tensor-product basis from X and applies the final linear layer.
     */
    torch::Tensor forward(const torch::Tensor& X) override {
        torch::Tensor P   = basis_->buildBasis(X);
        torch::Tensor out = finalLayer_->forward(P);
        return out;
    }

    /**
     * @brief Forward pass using a precomputed basis.
     *
     * This is used when basis features are supplied externally.
     */
    torch::Tensor forwardBasis(const torch::Tensor& P) {
        torch::Tensor out = finalLayer_->forward(P);
        return out;
    }

    /**
     * @brief Build Fourier basis from input coordinates.
     */
    torch::Tensor buildBasisFromData(const torch::Tensor& X) const {
        return basis_->buildBasis(X);
    }

    torch::Tensor buildEvaluationBasis(const torch::Tensor& X) override {
        return buildBasisFromData(X);
    }

    /**
     * @brief Build PDE-transformed Fourier basis from input coordinates.
     */
    torch::Tensor buildBasisPDEFromData(const torch::Tensor& X) const {
        return basis_->buildBasisPDE(X);
    }

    /**
     * @brief Build first-derivative Fourier basis from input coordinates.
     */
    torch::Tensor buildBasisDxFromData(const torch::Tensor& X, int deriv_dim) const {
        return basis_->buildBasisDx(X, deriv_dim);
    }

    /**
     * @brief Build second-derivative Fourier basis from input coordinates.
     */
    torch::Tensor buildBasisDxxFromData(const torch::Tensor& X, int deriv_dim) const {
        return basis_->buildBasisDxx(X, deriv_dim);
    }

    /**
     * @brief Forward pass for PDE operator using input coordinates.
     *
     * Builds the PDE-transformed basis (e.g., Laplacian) and applies the final linear layer.
     */
    torch::Tensor forwardPDEFromData(const torch::Tensor& X) {
        torch::Tensor P_PDE = basis_->buildBasisPDE(X);
        return forwardBasis(P_PDE);
    }

    /**
     * @brief Forward pass of first derivative basis in a specified dimension.
     *
     * Builds d/dx_d of the tensor-product Fourier basis and applies the final linear layer.
     */
    torch::Tensor forwardDxFromData(const torch::Tensor& X, int deriv_dim) {
        torch::Tensor P_dx = basis_->buildBasisDx(X, deriv_dim);
        return forwardBasis(P_dx);
    }

    /**
     * @brief Forward pass of second derivative basis in a specified dimension.
     *
     * Builds d^2/dx_d^2 of the tensor-product Fourier basis and applies the final linear layer.
     */
    torch::Tensor forwardDxxFromData(const torch::Tensor& X, int deriv_dim) {
        torch::Tensor P_dxx = basis_->buildBasisDxx(X, deriv_dim);
        return forwardBasis(P_dxx);
    }

    /**
     * @brief Expose which dimension is used for periodic boundary derivatives.
     */
    int getPeriodicDerivDimValue() const {
        return basis_->getPeriodicDerivDim();
    }

    /**
     * @brief Forward pass using structured inputs.
     *
     * Uses basis if provided; otherwise computes basis from X.
     */
    torch::Tensor forward(const DataInputs& inputs) override {
        if (inputs.P.basis != nullptr) {
            torch::Tensor P = *inputs.P.basis;
            return forwardBasis(P);
        }
        TORCH_CHECK(inputs.X != nullptr, "DataInputs requires X when Basis is not provided.");
        return forward(*inputs.X);
    }

    /**
     * @brief Compute PDE residual contribution for the Fourier layer.
     *
     * Supports Poisson and Allen-Cahn via analytic basis transforms. If a PDE
     * object is provided and no custom path exists, falls back to autograd.
     */
    torch::Tensor forward_pde(const PDEInputs& inputs, PDE* pde) override {
        /*
         * Poisson:
         *      f_pred = (bhat V_xx + bhat V_yy) = bhat(V_xx + V_yy)
         *
         * Allen-Cahn:
         *      f_pred = (bhat V_xx + bhat V_yy) + u_pred*(u_pred^2 - 1.0)
         */
        torch::Tensor u_pred;
        torch::Tensor f_pred;

        if (pde_problem_=="poisson") {
            if (inputs.P_pde != nullptr) {
                torch::Tensor P_PDE = *inputs.P_pde;
                f_pred = forwardBasis(P_PDE);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when P_pde is not provided.");
                torch::Tensor X = *inputs.data.X;
                torch::Tensor P_PDE = basis_->buildBasisPDE(X);
                f_pred = forwardBasis(P_PDE);
            }

        } else if (pde_problem_=="allencahn") {
            if (inputs.data.P.basis != nullptr) {
                torch::Tensor P = *inputs.data.P.basis;
                u_pred = forwardBasis(P);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when Basis is not provided.");
                u_pred = forward(*inputs.data.X);
            }

            if (inputs.P_pde != nullptr) {
                torch::Tensor P_PDE = *inputs.P_pde;
                f_pred = forwardBasis(P_PDE) + u_pred * (u_pred * u_pred - 1.0);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when P_pde is not provided.");
                torch::Tensor X = *inputs.data.X;
                torch::Tensor P_PDE = basis_->buildBasisPDE(X);
                f_pred = forwardBasis(P_PDE) + u_pred * (u_pred * u_pred - 1.0);
            }

        } else if (pde_problem_=="onewaywave_inhomo") {
            torch::Tensor f_t;
            torch::Tensor f_x;
            if (inputs.P_t != nullptr && inputs.P_x != nullptr) {
                torch::Tensor P_t = *inputs.P_t;
                torch::Tensor P_x = *inputs.P_x;
                f_t = forwardBasis(P_t);
                f_x = forwardBasis(P_x);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when P_t/P_x are not provided.");
                auto X = *inputs.data.X;
                f_t = forwardBasis(basis_->buildBasisDx(X, /*deriv_dim=*/0));
                f_x = forwardBasis(basis_->buildBasisDx(X, /*deriv_dim=*/1));
            }
            f_pred = f_t + 10.0 * f_x;

        } else if (pde_problem_=="nonss_allencahn") {
            torch::Tensor u_pred_local;
            if (inputs.data.P.basis != nullptr) {
                torch::Tensor P = *inputs.data.P.basis;
                u_pred_local = forwardBasis(P);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when Basis is not provided.");
                u_pred_local = forward(*inputs.data.X);
            }

            torch::Tensor f_t;
            torch::Tensor f_xx;
            if (inputs.P_t != nullptr && inputs.P_xx != nullptr) {
                torch::Tensor P_t = *inputs.P_t;
                torch::Tensor P_xx = *inputs.P_xx;
                f_t  = forwardBasis(P_t);
                f_xx = forwardBasis(P_xx);
            } else {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X when P_t/P_xx are not provided.");
                auto X = *inputs.data.X;
                f_t  = forwardBasis(basis_->buildBasisDx(X, /*deriv_dim=*/0));
                f_xx = forwardBasis(basis_->buildBasisDxx(X, /*deriv_dim=*/1));
            }
            f_pred = f_t - 0.0001 * f_xx + 5.0 * u_pred_local * u_pred_local * u_pred_local - 5.0 * u_pred_local;

        } else {
            if (pde) {
                TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X.");
                torch::Tensor X = *inputs.data.X;
                u_pred = forward(X);
                return pde->forwardPDE(u_pred, X);
            }
            std::cerr<<"WARNING: FourierLayer forward_pde() not implemented for pde="<<pde_problem_<<"\n";
        }

        return f_pred;
    }

    /**
     * @brief Periodic boundary loss (value and derivative matching).
     *
     * Uses basis if provided; otherwise computes basis (and derivative basis)
     * from boundary coordinates.
     */
    torch::Tensor getPeriodicBndLoss(const BndInputs& BND_DATA_A,
                                     const BndInputs& BND_DATA_B) override {
        torch::Tensor bnd_loss;
        torch::Tensor P_a;
        torch::Tensor P_b;
        if (BND_DATA_A.data.P.basis != nullptr && BND_DATA_B.data.P.basis != nullptr) {
            P_a = *BND_DATA_A.data.P.basis;
            P_b = *BND_DATA_B.data.P.basis;
        } else {
            TORCH_CHECK(BND_DATA_A.data.X != nullptr, "BndInputs A requires X.");
            TORCH_CHECK(BND_DATA_B.data.X != nullptr, "BndInputs B requires X.");
            TORCH_CHECK(BND_DATA_A.data.X->sizes().size() == 2 &&
                        BND_DATA_A.data.X->size(1) == 2,
                        "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
            TORCH_CHECK(BND_DATA_B.data.X->sizes().size() == 2 &&
                        BND_DATA_B.data.X->size(1) == 2,
                        "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
            P_a = basis_->buildBasis(*BND_DATA_A.data.X);
            P_b = basis_->buildBasis(*BND_DATA_B.data.X);
        }

        torch::Tensor u_a = this->forwardBasis(P_a);
        torch::Tensor u_b = this->forwardBasis(P_b);

        // u(t, −1) = u(t, 1),
        torch::Tensor loss_1 = torch::mean((u_a - u_b).pow(2));
        bnd_loss = loss_1;

        // ux(t, −1) = ux(t, 1).
        torch::Tensor P_a_x;
        torch::Tensor P_b_x;
        if (BND_DATA_A.P_x != nullptr && BND_DATA_B.P_x != nullptr) {
            P_a_x = *BND_DATA_A.P_x;
            P_b_x = *BND_DATA_B.P_x;
        } else {
            TORCH_CHECK(BND_DATA_A.data.X != nullptr, "BndInputs A requires X.");
            TORCH_CHECK(BND_DATA_B.data.X != nullptr, "BndInputs B requires X.");
            TORCH_CHECK(BND_DATA_A.data.X->sizes().size() == 2 &&
                        BND_DATA_A.data.X->size(1) == 2,
                        "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
            TORCH_CHECK(BND_DATA_B.data.X->sizes().size() == 2 &&
                        BND_DATA_B.data.X->size(1) == 2,
                        "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
            int deriv_dim = basis_->getPeriodicDerivDim();
            P_a_x = basis_->buildBasisDx(*BND_DATA_A.data.X, deriv_dim);
            P_b_x = basis_->buildBasisDx(*BND_DATA_B.data.X, deriv_dim);
        }

        torch::Tensor u_xa = this->forwardBasis(P_a_x);
        torch::Tensor u_xb = this->forwardBasis(P_b_x);

        torch::Tensor loss_2 = torch::mean((u_xa - u_xb).pow(2));
        bnd_loss += loss_2;

        return bnd_loss;
    }

    /**
     * @brief Regularization loss (L1 + L2) for Fourier coefficients.
     */
    torch::Tensor getRegularizationLoss(const DataInputs& inputs) override {
        (void)inputs;
        torch::Tensor totalLoss = torch::zeros({}, torch::TensorOptions().dtype(dtype_)).to(device_);
        if (l1RegCoeff_ > 0) { totalLoss += getL1RegLoss(); }
        if (l2RegCoeff_ > 0) { totalLoss += getL2RegLoss(); }
        return totalLoss;
    }

    /**
     * @brief L1 regularization on active coefficients (weights and bias).
     */
    torch::Tensor getL1RegLoss() {
        auto weights     = getBasisCoeffs();
        auto biasval     = getBasisBias();
        auto l1_reg_loss = weights.abs().sum();
        if (biasval.defined()) l1_reg_loss = l1_reg_loss + biasval.abs().sum();
        return l1RegCoeff_*l1_reg_loss;
    }

    /**
     * @brief L2 regularization on active coefficients (weights and bias).
     */
    torch::Tensor getL2RegLoss() {
        auto weights     = getBasisCoeffs();
        auto biasval     = getBasisBias();
        auto l2_reg_loss = weights.pow(2).sum();
        if (biasval.defined()) l2_reg_loss = l2_reg_loss + biasval.pow(2).sum();
        return 0.5*l2RegCoeff_*l2_reg_loss;
    }

    /**
     * @brief Boolean selection mask including bias and basis weights.
     */
    torch::Tensor getFullBasisSelection() {
        torch::Tensor basisSelection = getBasisSelection();
        torch::Tensor biasSelection  = getBiasSelection();
        if (!biasSelection.defined()) {
            biasSelection = torch::zeros({basisSelection.size(0)}, basisSelection.options());
        }

        torch::Tensor selection = torch::concat({biasSelection.unsqueeze(0), basisSelection}, 1); // [1 x width+1]
        selection = selection.to(torch::kBool).squeeze();
        return selection;
    }

    /**
     * @brief Build least-squares matrix A for Fourier basis coefficients.
     *
     * Assembles boundary and PDE rows using either provided bases or
     * on-the-fly basis computation.
     */
    torch::Tensor getLSTSQ_A(const PDEInputs& pde_data,
                             const BndInputs& bnd_data,
                             const ICInputs* ic_data,
                             PDE* pde) {
        (void)pde;

        TORCH_CHECK(bnd_data.data.X != nullptr, "BndInputs requires X.");
        TORCH_CHECK(pde_data.data.X != nullptr, "PDEInputs requires X.");
        torch::Tensor X_data = *bnd_data.data.X;
        torch::Tensor X_pde  = *pde_data.data.X;

        const auto nbnd = X_data.sizes()[0];
        const auto ncol = X_pde.sizes()[0];

        torch::Tensor A_ff_bnd_basis;
        torch::Tensor A_ff_col_basis;
        bool periodic_pair = (bnd_data.data_b.X != nullptr);

        if (bnd_data.data.P.basis != nullptr) {
            A_ff_bnd_basis = *bnd_data.data.P.basis;
        } else {
            A_ff_bnd_basis = basis_->buildBasis(X_data);
        }
        if (periodic_pair) {
            TORCH_CHECK(bnd_data.data_b.X != nullptr, "Periodic BndInputs requires paired X.");
            torch::Tensor X_b = *bnd_data.data_b.X;
            torch::Tensor A_ff_bnd_basis_b;
            if (bnd_data.data_b.P.basis != nullptr) {
                A_ff_bnd_basis_b = *bnd_data.data_b.P.basis;
            } else {
                A_ff_bnd_basis_b = basis_->buildBasis(X_b);
            }
            A_ff_bnd_basis = A_ff_bnd_basis - A_ff_bnd_basis_b;
        }

        if (pde_problem_ == "onewaywave_inhomo") {
            // Wave LS rows must use the same transport operator as forward_pde.
            auto P_t = pde_data.P_t != nullptr ? *pde_data.P_t : basis_->buildBasisDx(X_pde, 0);
            auto P_x = pde_data.P_x != nullptr ? *pde_data.P_x : basis_->buildBasisDx(X_pde, 1);
            A_ff_col_basis = P_t + 10.0 * P_x;
        } else if (pde_data.P_pde != nullptr) {
            A_ff_col_basis = *pde_data.P_pde;
        } else {
            A_ff_col_basis = basis_->buildBasisPDE(X_pde);
        }

        torch::Tensor A_ff_bnd_bias;
        if (periodic_pair) {
            A_ff_bnd_bias = torch::zeros({nbnd, 1}).to(device_);
        } else {
            A_ff_bnd_bias = torch::ones({nbnd, 1}).to(device_);
        }
        torch::Tensor A_ff_col_bias = torch::zeros({ncol, 1}).to(device_); // [ncol x 1]

        torch::Tensor A_ff_bnd = torch::cat({A_ff_bnd_bias, A_ff_bnd_basis}, 1); // [nbnd x npoly+1]
        if (periodic_pair) {
            TORCH_CHECK(bnd_data.P_x != nullptr, "Periodic BndInputs requires P_x for derivative constraints.");
            TORCH_CHECK(bnd_data.P_x_b != nullptr, "Periodic BndInputs requires P_x_b for derivative constraints.");
            torch::Tensor P_a_x = *bnd_data.P_x;
            torch::Tensor P_b_x = *bnd_data.P_x_b;
            torch::Tensor A_ff_bnd_dx = torch::cat({torch::zeros({nbnd, 1}).to(device_), P_a_x - P_b_x}, 1);
            A_ff_bnd = torch::cat({A_ff_bnd, A_ff_bnd_dx}, 0);
        }
        torch::Tensor A_ff_col = torch::cat({A_ff_col_bias, A_ff_col_basis}, 1); // [ncol x npoly+1]

        torch::Tensor A_ff = A_ff_bnd;
        if (ic_data != nullptr && ic_data->data.X != nullptr) {
            torch::Tensor Xic = *ic_data->data.X;
            torch::Tensor A_ff_ic_basis;
            if (ic_data->data.P.basis != nullptr) {
                A_ff_ic_basis = *ic_data->data.P.basis;
            } else {
                A_ff_ic_basis = basis_->buildBasis(Xic);
            }
            torch::Tensor A_ff_ic = torch::cat({torch::ones({Xic.size(0), 1}).to(device_), A_ff_ic_basis}, 1);
            A_ff = torch::cat({A_ff, A_ff_ic}, 0);
        }
        A_ff = torch::cat({A_ff, A_ff_col}, 0); // [nbnd+nic+ncol x npoly+1]

        torch::Tensor selection = getFullBasisSelection();
        A_ff = A_ff.index({torch::indexing::Slice(), selection});

        return A_ff;
    }

    /**
     * @brief Replace final layer weights with least-squares solution.
     */
    void lstsqUpdates(torch::Tensor& soln) override { finalLayer_->reassignLayerWeights(soln); }
    /**
     * @brief Apply coefficient truncation mask.
     */
    void truncateCoeffs() override     { finalLayer_->truncateCoeffs(); }
    /**
     * @brief Number of remaining (untruncated) bases.
     */
    int getNRemainingBases()           { return finalLayer_->getNRemainingBases(); }
    /**
     * @brief Mask for basis coefficients (weights).
     */
    torch::Tensor& getBasisSelection() { return finalLayer_->getBasisSelection(); }
    /**
     * @brief Mask for bias coefficients.
     */
    torch::Tensor& getBiasSelection()  { return finalLayer_->getBiasSelection(); }
    /**
     * @brief Active basis coefficients (weights).
     */
    torch::Tensor getBasisCoeffs()     { return finalLayer_->getBasisCoeffs(); }
    /**
     * @brief Active bias coefficients.
     */
    torch::Tensor getBasisBias()       { return finalLayer_->getBasisBias(); }
    /**
     * @brief Weight gradients from the final layer.
     */
    torch::Tensor getPolyNetGrad()     { return finalLayer_->getGrads(); }
    /**
     * @brief Bias gradients from the final layer.
     */
    torch::Tensor getPolyNetBiasGrad() { return finalLayer_->getBiasGrad(); }


    /**
     * @brief Print current coefficients to stdout.
     */
    std::vector<BasisCoefficientGroup> getBasisCoefficientGroups() override {
        std::vector<std::string> labels;
        if (inputDim_ == 1) {
            if (basis_->getIncludeZeroMode()) labels.push_back("constant");
            for (int k = 1; k <= basis_->getEll(); ++k) {
                labels.push_back("cos(k=" + std::to_string(k) + ")");
                labels.push_back("sin(k=" + std::to_string(k) + ")");
            }
        }
        return {{"Fourier", getBasisCoeffs(), getBasisSelection(), labels}};
    }

    void printTrainingInfo() override {
        if (!verbose_) {
            return;
        }
        auto coeffs = getBasisCoeffs();
        std::cout<<"Poly Layer Weights:\n";
        for (int i=0; i<coeffs.sizes()[1]; i++) { // weights -> [1xm]
            std::cout<<", "<<coeffs[0][i].item<float>();
        }
    }

    std::string modelType_;

private:
    /**
     * @brief Initialize model args from ModelConfig.
     */
    void initModelArgs(const FourierLayerArgs& net) {
        l1RegCoeff_   = net.pn_l1_reg_coeff;
        l2RegCoeff_   = net.pn_l2_reg_coeff;
        truncThresh_  = net.pn_trunc_thresh;
        pn_truncate_  = net.pn_truncate;
        truncate_     = pn_truncate_;
    }

    /**
     * @brief Build tensor-product basis metadata and PDE scaling.
     */
    void initFourierBasis(const FourierLayerArgs& net,
                          const std::vector<std::vector<double>>& domain_bounds) {
        basis_ = std::make_unique<FourierBasisBuilder>(inputDim_,
                                                       net.ell,
                                                       net.basis_gen_typ,
                                                       net.include_zero_mode,
                                                       net.scale_to_period,
                                                       net.periodic_deriv_dim,
                                                       domain_bounds,
                                                       device_,
                                                       dtype_,
                                                       net.fourier_periods);
    }

    /**
     * @brief Print model configuration details.
     */
    void printModelInfo() const {
        // Function to print the details of VanillaFourierLayer
        std::cout << "\n\n----------------------- VanillaFourierLayer Information: \n"
                  << "modelType_: " << modelType_ << "\n"
                  << "pde_problem: " << pde_problem_ << "\n"
                  << "inputDim: " << inputDim_ << "\n"
                  << "ell: " << basis_->getEll() << "\n"
                  << "includeZeroMode: " << basis_->getIncludeZeroMode() << "\n"
                  << "scaleToPeriod: " << basis_->getScaleToPeriod() << "\n"
                  << "effective Fourier periods: " << basis_->getPeriods() << "\n"
                  << "perDimSize: " << basis_->getPerDimSize() << "\n"
                  << "npoly: " << basis_->getNPoly() << "\n"
                  << "Output Dimension: " << outputDim_ << "\n"
                  << "l1_reg_coeff: " << l1RegCoeff_ << "\n"
                  << "l2_reg_coeff: " << l2RegCoeff_ << "\n"
                  << "truncThresh_: " << truncThresh_ << "\n"
                  << "modelType_: " << modelType_ << "\n"
                  << "truncate: " << (truncate_ ? "enabled" : "disabled") << "\n"
                  << "-----------------------\n";
    }

    /**
     * @brief Initialize the final linear layer for Fourier coefficients.
     */
    void initializeFinalLayer() {
        finalLayer_ = FinalLinearLayer(basis_->getNPoly(), outputDim_, truncThresh_, /*include_bias=*/false, /*init_typ=*/"xavier", device_);
        finalLayer_->to(device_, dtype_);
        finalLayer_->setVerbose(verbose_);
        register_module("trunc_neuralnet", finalLayer_);
    }

    FourierLayerArgs getFourierLayerArgs(const ModelConfig& args) const {
        if (const FourierLayerArgs* net = std::get_if<FourierLayerArgs>(&args.arch)) {
            return *net;
        }
        throw std::runtime_error("ModelConfig.arch is not FourierLayerArgs");
    }

    FinalLinearLayer finalLayer_ = nullptr;

    torch::Device device_;
    torch::Dtype dtype_;
    bool pn_truncate_;
    float truncThresh_;
    float l1RegCoeff_;
    float l2RegCoeff_;
    bool verbose_ = false;
    int inputDim_;
    int outputDim_;
    std::unique_ptr<FourierBasisBuilder> basis_;
    std::string pde_problem_;

};

TORCH_MODULE(VanillaFourierLayer);

#endif
