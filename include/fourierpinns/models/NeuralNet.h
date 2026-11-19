#ifndef VANILLANEURALNET_H
#define VANILLANEURALNET_H

#include <torch/torch.h>
#include "CustomNet.h"
#include "fourierpinns/config/Args.h"

class VanillaNeuralNetImpl : public CustomNetwork {
public:
    /**
     * @brief Constructs a VanillaNeuralNet.
     *
     * @param args         Network arguments.
     * @param layers       Vector containing the size of each layer.
     * @param device       The target torch::Device.
     * @param dtype        The target torch::Dtype.
     */
    VanillaNeuralNetImpl(const ModelConfig& args,
                         std::string pde_problem,
                         torch::Device device,
                         torch::Dtype dtype)
        : modelType_("neuralnet"),
          pdeProblem_(pde_problem),
          layers_(args.layers),
          device_(device),
          dtype_(dtype) {
        // Initialize model parameters from the args
        initModelArgs(args);
        if (verbose_) {
            printModelInfo();
        }

        // Initialize various network modules in separate functions.
        initializeLayers();
        initializeFinalLayer();
        initializeActivations();
    }

    virtual ~VanillaNeuralNetImpl() {}

    // -------------------------------------------------------------------------
    // Forward pass routines
    // -------------------------------------------------------------------------

    /**
     * @brief Computes the network basis features.
     *
     * @param X         Input tensor.
     * @param startInd  Starting index in the module list (for modified MLP).
     * @return torch::Tensor  Output after applying layers and activations.
     */
    torch::Tensor forwardBasis(const torch::Tensor& X, int startInd = 0) {
        torch::Tensor res = X;

        for (size_t i = static_cast<size_t>(startInd); i < networkBasis_->size(); ++i) {
            res = networkBasis_[i]->as<torch::nn::Linear>()->forward(res);
            res = applyActivation(res);
        }
        return res;
    }

    /**
     * @brief Applies the final linear transformation.
     *
     * @param basis  The output from the basis network.
     * @return torch::Tensor  Final network output.
     */
    torch::Tensor forwardLastLayer(const torch::Tensor& basis) {
        torch::Tensor out = finalLayer_->forward(basis);
        return out;
    }

    /**
     * @brief Complete forward pass.
     *
     * @param X  Input tensor.
     * @return torch::Tensor  Network output.
     */
    torch::Tensor forward(const torch::Tensor& X) {
        auto basis = forwardBasis(X);
        auto out = forwardLastLayer(basis);

        return out;
    }

    /**
     * @brief Overloaded forward function accepting structured inputs.
     */
    torch::Tensor forward(const DataInputs& inputs) override {
        TORCH_CHECK(inputs.X != nullptr, "DataInputs requires X.");
        torch::Tensor X = *inputs.X;
        return forward(X);
    }

    torch::Tensor getPeriodicBndLoss(const BndInputs& BND_DATA_A,
                                     const BndInputs& BND_DATA_B) override {
        torch::Tensor bnd_loss;
        TORCH_CHECK(BND_DATA_A.data.X != nullptr, "BndInputs A requires X.");
        TORCH_CHECK(BND_DATA_B.data.X != nullptr, "BndInputs B requires X.");
        TORCH_CHECK(BND_DATA_A.data.X->sizes().size() == 2 &&
                    BND_DATA_A.data.X->size(1) == 2,
                    "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
        TORCH_CHECK(BND_DATA_B.data.X->sizes().size() == 2 &&
                    BND_DATA_B.data.X->size(1) == 2,
                    "Periodic boundary loss expects X shape [N,2] with columns [t,x].");
        torch::Tensor ta  = (*BND_DATA_A.data.X).index({torch::indexing::Slice(), 0}).unsqueeze(1);
        torch::Tensor xa  = (*BND_DATA_A.data.X).index({torch::indexing::Slice(), 1}).unsqueeze(1);

        torch::Tensor tb  = (*BND_DATA_B.data.X).index({torch::indexing::Slice(), 0}).unsqueeze(1);
        torch::Tensor xb  = (*BND_DATA_B.data.X).index({torch::indexing::Slice(), 1}).unsqueeze(1);

        auto input_a = torch::cat({ta, xa}, 1);  // shape: (N, 2)
        auto input_b = torch::cat({tb, xb}, 1);  // shape: (N, 2)

        torch::Tensor u_a = this->forward(input_a);
        torch::Tensor u_b = this->forward(input_b);

        // u(t, −1) = u(t, 1),
        torch::Tensor loss_1 = torch::mean((u_a - u_b).pow(2));
        bnd_loss = loss_1;

        // ux(t, −1) = ux(t, 1).
        auto oneslikeu_a = torch::ones_like(u_a);
        auto u_xa = torch::autograd::grad({ u_a }, { xa }, {oneslikeu_a}, true, true)[0];

        auto oneslikeu_b = torch::ones_like(u_b);
        auto u_xb = torch::autograd::grad({ u_b }, { xb }, {oneslikeu_b}, true, true)[0];

        torch::Tensor loss_2 = torch::mean((u_xa - u_xb).pow(2));
        bnd_loss += loss_2;

        return bnd_loss;
    }

    // -------------------------------------------------------------------------
    // Regularization and other utility methods
    // -------------------------------------------------------------------------

    torch::Tensor getRegularizationLoss(const DataInputs& inputs) override {
        (void)inputs;
        torch::Tensor totalLoss = torch::zeros({}, torch::TensorOptions().dtype(dtype_).device(device_));
        if (l1RegCoeff_ > 0) { totalLoss += getL1RegLoss(); }
        if (l2RegCoeff_ > 0) { totalLoss += getL2RegLoss(); }
        return totalLoss;
    }

    torch::Tensor getL1RegLoss() {
        auto weights = getBasisCoeffs();
        auto biasVal = getBasisBias();
        auto l1Loss = torch::mean(torch::abs(weights) + torch::abs(biasVal));
        return l1RegCoeff_ * l1Loss;
    }

    torch::Tensor getL2RegLoss() {
        auto weights = getBasisCoeffs();
        auto biasVal = getBasisBias();
        auto l2Loss = weights.pow(2).sum() + biasVal.pow(2).sum();
        return 0.5 * l2RegCoeff_ * l2Loss;
    }

    torch::Tensor getFullBasisSelection() {
        torch::Tensor basisSelection = getBasisSelection();
        torch::Tensor biasSelection  = getBiasSelection();
        torch::Tensor selection = torch::cat({biasSelection.unsqueeze(0), basisSelection}, 1);
        selection = selection.to(torch::kBool).squeeze();
        return selection;
    }

    /**
     * @brief Computes the matrix for least-squares update.
     *
     * @param pdeData  PDE-related data.
     * @param bndData  Boundary data.
     * @param pde      A pointer to a PDE object.
     * @return torch::Tensor  The matrix A_nn.
     */
    torch::Tensor getLSTSQ_A(const PDEInputs& pdeData,
                             const BndInputs& bndData,
                             const ICInputs* icData,
                             PDE* pde) {
        TORCH_CHECK(bndData.data.X != nullptr, "BndInputs requires X.");
        TORCH_CHECK(pdeData.data.X != nullptr, "PDEInputs requires X.");
        torch::Tensor XData = *bndData.data.X;
        torch::Tensor XPDE  = *pdeData.data.X;

        const auto nbnd = XData.size(0);
        const auto ncol = XPDE.size(0);

        bool periodic_pair = (bndData.data_b.X != nullptr);
        torch::Tensor A_nn_bnd_bias;
        torch::Tensor A_nn_bnd_basis = forwardBasis(XData).detach();
        if (periodic_pair) {
            TORCH_CHECK(bndData.data_b.X != nullptr, "Periodic BndInputs requires paired X.");
            torch::Tensor Xb = *bndData.data_b.X;
            torch::Tensor A_nn_bnd_basis_b = forwardBasis(Xb).detach();
            A_nn_bnd_basis = A_nn_bnd_basis - A_nn_bnd_basis_b;
            A_nn_bnd_bias = torch::zeros({nbnd, 1}, device_);
        } else {
            A_nn_bnd_bias = torch::ones({nbnd, 1}, device_);
        }

        torch::Tensor A_nn_col_bias  = torch::ones({ncol, 1}, device_);
        torch::Tensor A_nn_col_basis = forwardBasis(XPDE);
        A_nn_col_basis = torch::cat({A_nn_col_bias, A_nn_col_basis}, 1);
        if (verbose_) {
            std::cout << "A_nn_col (before forwardBasisPDE) min="
                      << torch::min(A_nn_col_basis).item<float>()
                      << ", max=" << torch::max(A_nn_col_basis).item<float>() << "\n";
        }

        torch::Tensor A_nn_col = pde->forwardBasisPDE(A_nn_col_basis, XPDE).detach();
        torch::Tensor A_nn_bnd = torch::cat({A_nn_bnd_bias, A_nn_bnd_basis}, 1);
        if (periodic_pair) {
            TORCH_CHECK(bndData.data.X->sizes().size() == 2 && bndData.data.X->size(1) >= 1,
                        "Periodic BndInputs expects X shape [N,dim].");
            TORCH_CHECK(bndData.data_b.X->sizes().size() == 2 && bndData.data_b.X->size(1) >= 1,
                        "Periodic BndInputs expects paired X shape [N,dim].");
            int deriv_dim = static_cast<int>(bndData.data.X->size(1)) - 1;
            auto Xa = *bndData.data.X;
            auto Xb = *bndData.data_b.X;

            torch::Tensor basis_a = forwardBasis(Xa);
            torch::Tensor basis_b = forwardBasis(Xb);

            std::vector<torch::Tensor> grads_a;
            std::vector<torch::Tensor> grads_b;
            grads_a.reserve(static_cast<std::size_t>(basis_a.size(1)));
            grads_b.reserve(static_cast<std::size_t>(basis_b.size(1)));
            auto ones_a = torch::ones({basis_a.size(0), 1}, basis_a.options());
            auto ones_b = torch::ones({basis_b.size(0), 1}, basis_b.options());
            for (int j = 0; j < basis_a.size(1); ++j) {
                auto col_a = basis_a.index({torch::indexing::Slice(), j}).unsqueeze(1);
                auto col_b = basis_b.index({torch::indexing::Slice(), j}).unsqueeze(1);
                auto grad_a = torch::autograd::grad({col_a}, {Xa}, {ones_a}, true, true, /*allow_unused=*/true)[0];
                auto grad_b = torch::autograd::grad({col_b}, {Xb}, {ones_b}, true, true, /*allow_unused=*/true)[0];
                auto dphi_a = grad_a.defined()
                    ? grad_a.index({torch::indexing::Slice(), deriv_dim}).unsqueeze(1)
                    : torch::zeros({Xa.size(0), 1}, Xa.options());
                auto dphi_b = grad_b.defined()
                    ? grad_b.index({torch::indexing::Slice(), deriv_dim}).unsqueeze(1)
                    : torch::zeros({Xb.size(0), 1}, Xb.options());
                grads_a.push_back(dphi_a);
                grads_b.push_back(dphi_b);
            }
            auto dphi_a_mat = torch::cat(grads_a, 1);
            auto dphi_b_mat = torch::cat(grads_b, 1);
            torch::Tensor A_nn_bnd_dx = torch::cat({torch::zeros({nbnd, 1}, device_), dphi_a_mat - dphi_b_mat}, 1);
            A_nn_bnd = torch::cat({A_nn_bnd, A_nn_bnd_dx}, 0);
        }

        if (verbose_) {
            std::cout << "A_nn_bnd min=" << torch::min(A_nn_bnd).item<float>()
                      << ", max=" << torch::max(A_nn_bnd).item<float>() << "\n";
            std::cout << "A_nn_col min=" << torch::min(A_nn_col).item<float>()
                      << ", max=" << torch::max(A_nn_col).item<float>() << "\n";
        }

        torch::Tensor A_nn = A_nn_bnd;
        if (icData != nullptr && icData->data.X != nullptr) {
            torch::Tensor Xic = *icData->data.X;
            torch::Tensor A_nn_ic_basis = forwardBasis(Xic).detach();
            torch::Tensor A_nn_ic = torch::cat({torch::ones({Xic.size(0), 1}, device_), A_nn_ic_basis}, 1);
            A_nn = torch::cat({A_nn, A_nn_ic}, 0);
        }
        A_nn = torch::cat({A_nn, A_nn_col}, 0);
        if (verbose_) {
            std::cout << "A_nn: " << A_nn.sizes() << "\n";
        }

        torch::Tensor selection = getFullBasisSelection();
        A_nn = A_nn.index({torch::indexing::Slice(), selection});
        if (verbose_) {
            std::cout << "A_nn (after selection): " << A_nn.sizes() << "\n";
        }

        return A_nn;
    }

    void lstsqUpdates(torch::Tensor& soln) { finalLayer_->reassignLayerWeights(soln); }
    void truncateCoeffs() { finalLayer_->truncateCoeffs(); }
    torch::Tensor getBasisCoeffs()     { return finalLayer_->getBasisCoeffs(); }
    torch::Tensor getBasisBias()       { return finalLayer_->getBasisBias(); }
    torch::Tensor& getBasisSelection() { return finalLayer_->getBasisSelection(); }
    torch::Tensor& getBiasSelection()  { return finalLayer_->getBiasSelection(); }
    torch::Tensor getNNGrad()          { return finalLayer_->getGrads(); }
    torch::Tensor getNNBiasGrad()      { return finalLayer_->getBiasGrad(); }
    int getNRemainingBases()           { return finalLayer_->getNRemainingBases(); }

    std::vector<BasisCoefficientGroup> getBasisCoefficientGroups() override {
        return {{"NN bases", getBasisCoeffs(), getBasisSelection(), {}},
                {"NN bias", getBasisBias(), getBiasSelection(), {"bias"}}};
    }

    void printTrainingInfo() {
        if (!verbose_) {
            return;
        }
        auto coeffs = getBasisCoeffs();
        std::cout << "NN Layer Weights:\n";
        for (int i = 0; i < coeffs.size(1); i++) {
            std::cout << ", " << coeffs[0][i].item<float>();
        }
        coeffs = getBasisBias();
        std::cout << "\nNN Layer Bias: \n" << coeffs[0].item<float>() << "\n";
    }

    std::string modelType_;

private:
    // ======================================================================
    // Initialization helper functions
    // ======================================================================

    /**
     * @brief Initializes model parameters from the Args structure.
     */
    void initModelArgs(const ModelConfig& args) {
        if (const NeuralNetArgs* net = std::get_if<NeuralNetArgs>(&args.arch)) {
            l1RegCoeff_  = net->nn_l1_reg_coeff;
            l2RegCoeff_  = net->nn_l2_reg_coeff;
            truncThresh_ = net->nn_trunc_thresh;
            nn_truncate_ = net->nn_truncate;

            width_       = net->width;
            netAct_      = net->net_act;
        } else {
            throw std::runtime_error("ModelConfig.arch is not NeuralNetArgs");
        }

        inputDim_              = args.input_dim;
        outputDim_             = args.output_dim;

        truncate_ = nn_truncate_;
        verbose_ = args.verbose;

    }

    /**
     * @brief Prints network configuration information.
     */
    void printModelInfo() const {
        std::cout << "\n---------------------------- Neural Net Information: \n"
                    << "PDE: " << pdeProblem_ << "\n"
                  << "Input Dimension: " << inputDim_ << "\n"
                  << "Output Dimension: " << outputDim_ << "\n"
                  << "Network Activation: " << netAct_ << "\n"
                  << "Layer Width: " << width_ << "\n"
                  << "L1 Reg Coeff: " << l1RegCoeff_ << "\n"
                  << "L2 Reg Coeff: " << l2RegCoeff_ << "\n"
                  << "Truncation Threshold: " << truncThresh_ << "\n"
                  << "Truncate (nn): " << (nn_truncate_ ? "enabled" : "disabled") << "\n"
                  << "modelType_: " << modelType_ << "\n"
                  << "Truncate (parent): " << (truncate_ ? "enabled" : "disabled") << "\n";

        std::cout << "Layers: ";
        for (const auto& l : layers_) {
            std::cout << l << " ";
        }
        std::cout << "\n----------------------------\n";
    }

    /**
     * @brief Initializes all the hidden layers.
     *
     * If modifiedMLP_ is true, the first two layers are special (U and V).
     */
    void initializeLayers() {
        // Create an empty ModuleList for the network basis.
        networkBasis_ = torch::nn::ModuleList();

        // Add the remaining (hidden) layers.
        for (size_t i = 0; i < layers_.size() - 2; ++i) {
            int64_t inputFeatures  = layers_[i];
            int64_t outputFeatures = layers_[i + 1];
            addLayer(inputFeatures, outputFeatures, std::to_string(i));
        }
    }

    /**
     * @brief Helper function to add a single layer.
     *
     * @param inFeatures   Number of input features.
     * @param outFeatures  Number of output features.
     * @param layerName    Name suffix for the layer.
     */
    void addLayer(int64_t inFeatures, int64_t outFeatures, const std::string& layerName) {
        auto layer = torch::nn::Linear(inFeatures, outFeatures);
        torch::nn::init::xavier_normal_(layer->weight);
        torch::nn::init::zeros_(layer->bias);            // Initialize bias to zero
        layer->to(device_, dtype_);
        networkBasis_->push_back(layer);
        register_module("linear_" + layerName, layer);
    }

    /**
     * @brief Initializes the final linear layer.
     */
    void initializeFinalLayer() {
        finalLayer_ = FinalLinearLayer(width_, outputDim_, truncThresh_, /*include_bias=*/true, /*init_typ=*/"xavier", device_);
        finalLayer_->to(device_, dtype_);
        finalLayer_->setVerbose(verbose_);
        register_module("trunc_neuralnet", finalLayer_);
        nCoeffs_ = layers_.back();
    }

    /**
     * @brief Initializes activation modules that require state.
     */
    void initializeActivations() {
        if (netAct_ == "elu") {
            elu_ = register_module("elu", torch::nn::ELU());
        } else if (netAct_ == "selu") {
            selu_ = register_module("selu", torch::nn::SELU());
        }
        // For other activations, we use inline if/else in applyActivation.
    }

public:
    // ======================================================================
    // Helper functions
    // ======================================================================

    /**
     * @brief Applies the configured activation function.
     * @param res  The current layer output.
     * @return torch::Tensor  Activated tensor.
     */
    torch::Tensor applyActivation(const torch::Tensor& res) {
        if (netAct_ == "tanh") {
            return torch::tanh(res);
        } else if (netAct_ == "relu") {
            return torch::relu(res);
        } else if (netAct_ == "repu") {
            double p = 3.0;  // Consider making p configurable.
            return torch::relu(res.pow(p));
        } else if (netAct_ == "softplus") {
            return torch::softplus(res);
        } else if (netAct_ == "gelu") {
            return torch::nn::functional::gelu(res);
        } else if (netAct_ == "elu") {
            return elu_->forward(res);
        } else if (netAct_ == "selu") {
            return selu_->forward(res);
        } else if (netAct_ == "swish") {
            auto swish = [](const torch::Tensor& x) { return x * torch::sigmoid(x); };
            return swish(res);
        } else if (netAct_ == "sigmoid") {
            return torch::sigmoid(res);
        }
        // If no matching activation is found, return the unmodified tensor.
        return res;
    }

    // ======================================================================
    // Member variables
    // ======================================================================

    // Network configuration parameters.
    int inputDim_;
    int outputDim_;
    int nCoeffs_;
    int width_;
    float l1RegCoeff_;
    float l2RegCoeff_;
    float truncThresh_;
    bool nn_truncate_;
    bool verbose_ = false;
    std::string netAct_;
    std::string pdeProblem_;
    std::vector<int> layers_;

    // Device and data type.
    torch::Device device_;
    torch::Dtype dtype_;

    // Modules
    torch::nn::ModuleList networkBasis_ = torch::nn::ModuleList();
    FinalLinearLayer finalLayer_        = nullptr;
    torch::nn::ELU elu_                 = nullptr;
    torch::nn::SELU selu_               = nullptr;
};

// This macro creates a module holder type called VanillaNeuralNet that behaves similarly
// to the built-in modules (like torch::nn::Linear).
TORCH_MODULE(VanillaNeuralNet);

#endif // VANILLANEURALNET_H
