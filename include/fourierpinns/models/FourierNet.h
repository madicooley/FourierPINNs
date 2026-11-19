#ifndef VANILLAFOURIERNEURALNET_H
#define VANILLAFOURIERNEURALNET_H

#include <torch/torch.h>
#include "CustomNet.h"
#include "FourierLayer.h"
#include "fourierpinns/config/Args.h"

class VanillaFourierNeuralNetImpl : public CustomNetwork {
public:
    VanillaFourierNeuralNetImpl(const ModelConfig& args,
                                 std::string pde_problem_name,
                                 torch::Device device,
                                 torch::Dtype dtype)
    : modelType_("fouriernet"), pde_problem(pde_problem_name), device_(device), dtype_(dtype) {
        NeuralNetArgs neuralnet_args;
        FourierLayerArgs fourierlayer_args;

        if (const FourierNetArgs* net = std::get_if<FourierNetArgs>(&args.arch)) {
            neuralnet_args      = net->neuralnet_args;
            fourierlayer_args   = net->fourierlayer_args;
            this->input_dim     = args.input_dim;

            if (neuralnet_args.nn_truncate || fourierlayer_args.pn_truncate) {
                setTruncateModel();
            }
            verbose_ = args.verbose;

            neuralnet = VanillaNeuralNet(
                makeNeuralNetConfig(args, neuralnet_args),
                pde_problem_name,
                device,
                dtype);

            fouriernet = VanillaFourierLayer(
                makeFourierLayerConfig(args, fourierlayer_args),
                pde_problem_name,
                device,
                dtype);

            npoly = fouriernet->getNRemainingBases();

            if (verbose_) {
                printModelInfo();
            }
            register_module("neuralnetModule", neuralnet);
            register_module("fouriernetModule", fouriernet);
        } else {
            throw std::runtime_error("ModelConfig.arch is not FourierNetArgs");
        }
    }

    virtual ~VanillaFourierNeuralNetImpl() {}

    static ModelConfig makeNeuralNetConfig(const ModelConfig& base, const NeuralNetArgs& nn_args) {
        ModelConfig cfg;
        cfg.applyCommon(base);
        cfg.model_type = "neuralnet";
        cfg.arch = nn_args;
        cfg.layers = nn_args.layers;
        return cfg;
    }

    static ModelConfig makeFourierLayerConfig(const ModelConfig& base, const FourierLayerArgs& fl_args) {
        ModelConfig cfg;
        cfg.applyCommon(base);
        cfg.model_type = "fourierlayer";
        cfg.arch = fl_args;
        cfg.layers.clear();
        return cfg;
    }

    void lstsqUpdates(torch::Tensor& soln) override {
        int n_ff_coeffs = fouriernet->getNRemainingBases();
        torch::Tensor ff_soln = soln.index({torch::indexing::Slice(0,           n_ff_coeffs)});
        torch::Tensor nn_soln = soln.index({torch::indexing::Slice(n_ff_coeffs, torch::indexing::None)});

        if (verbose_) {
            std::cout<<"n_ff_coeffs: "<<n_ff_coeffs<<"\n";
            std::cout<<"soln="<<soln.sizes()<<", ff_soln="<<ff_soln.sizes()<<", nn_soln="<<nn_soln.sizes()<<"\n";
        }
        this->fouriernet->lstsqUpdates(ff_soln);
        this->neuralnet->lstsqUpdates(nn_soln);
    }

    torch::Tensor getLSTSQ_A(const PDEInputs& pde_data,
                             const BndInputs& bnd_data,
                             const ICInputs* ic_data,
                             PDE* pde) override {
        /*
         * A = [ A_ff  |  A_nn ] --> [nbnd+ncol x (npoly+1)+(w+1)]
         */

        torch::Tensor A_ff = fouriernet->getLSTSQ_A(pde_data, bnd_data, ic_data, pde);
        torch::Tensor A_nn = neuralnet->getLSTSQ_A(pde_data, bnd_data, ic_data, pde);

        if (verbose_) {
            if (A_ff.sizes()[1] > 0) std::cout<<"\nA_ff min="<<torch::min(A_ff).item<float>()<<", max="<<torch::max(A_ff).item<float>()<<"\n";
            if (A_nn.sizes()[1] > 0) std::cout<<"A_nn min="<<torch::min(A_nn).item<float>()<<", max="<<torch::max(A_nn).item<float>()<<"\n";
        }

        torch::Tensor A = torch::cat({A_ff, A_nn}, 1);
        return A;
    }

    torch::Tensor forward_pde(const PDEInputs& inputs, PDE* pde) override {
        /*
         * Poisson:
         *      f_pred = (bhat V_xx + bhat V_yy) + (f_pred_nn) = bhat(V_xx + V_yy) + (f_pred_nn)
         *
         * Allen-Cahn:
         *      f_pred = (bhat V_xx + bhat V_yy) + (f_pred_nn) = bhat(V_xx + V_yy) + (f_pred_nn) + u_pred*(u_pred^2 - 1.0)
         */
        if (verbose_) {
            std::cout<<"forward_pde. FourierNet.\n";
        }

        torch::Tensor x, y;

        torch::Tensor u_pred_an;
        torch::Tensor u_pred_nn;
        torch::Tensor u_pred;

        torch::Tensor f_pred_an; // augmented layer prediction
        torch::Tensor f_pred_nn; // neural net preddiction
        torch::Tensor f_pred;    // full pde prediction, ie f_pred = f_pred_an + f_pred_nn

        TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X.");
        torch::Tensor X = *inputs.data.X;
        int datadim = static_cast<int>(X.sizes()[1]);
        if (verbose_) {
            std::cout<<"forward_pde. FourierNet. X: "<<X.sizes()<<"\n";
        }

        if (datadim==1) {
            u_pred_nn = neuralnet->forward(X);

        } else if (datadim==2) {
            x = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
            y = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);
            u_pred_nn = neuralnet->forward(torch::cat({x, y}, 1));

        } else if (datadim==3) {
            TORCH_CHECK(false, "3D PDEs are not supported in PDE::forwardPDE()");
        } else std::cerr<<"WARNING: NeuralNet forward_pde() not implemented for datadim="<<datadim<<"\n";

        if (pde_problem=="poisson") {
            if (inputs.P_pde != nullptr) {
                torch::Tensor P_PDE = *inputs.P_pde;
                f_pred_an = fouriernet->forwardBasis(P_PDE);
            } else {
                f_pred_an = fouriernet->forwardPDEFromData(X);
            }

            if (input_dim==1) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, X);

            } else if (input_dim==2) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, x, y);

            } else if (input_dim==3) {
                TORCH_CHECK(false, "3D PDEs are not supported in PDE::forwardPDE()");
            }

            f_pred = f_pred_an + f_pred_nn;

        } else if (pde_problem=="allencahn") {
            if (inputs.data.P.basis != nullptr) {
                torch::Tensor P = *inputs.data.P.basis;
                u_pred_an = fouriernet->forwardBasis(P);
            } else {
                u_pred_an = fouriernet->forward(X);
            }

            if (inputs.P_pde != nullptr) {
                torch::Tensor P_PDE = *inputs.P_pde;
                f_pred_an = fouriernet->forwardBasis(P_PDE);
            } else {
                f_pred_an = fouriernet->forwardPDEFromData(X);
            }

            if (input_dim==1) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, X);

            } else if (input_dim==2) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, x, y);

            } else if (input_dim==3) {
                TORCH_CHECK(false, "3D PDEs are not supported in PDE::forwardPDE()");
            }

            u_pred = u_pred_an + u_pred_nn;
            f_pred = (f_pred_an + f_pred_nn) + u_pred * (u_pred * u_pred - 1.0);

        } else if (pde_problem=="onewaywave_inhomo") {
            if (inputs.data.P.basis != nullptr) {
                torch::Tensor P = *inputs.data.P.basis;
                u_pred_an = fouriernet->forwardBasis(P);
            } else {
                u_pred_an = fouriernet->forward(X);
            }

            if (inputs.P_t != nullptr && inputs.P_x != nullptr) {
                torch::Tensor P_t = *inputs.P_t;
                torch::Tensor P_x = *inputs.P_x;
                f_pred_an = fouriernet->forwardBasis(P_t) + 10.0 * fouriernet->forwardBasis(P_x);
            } else {
                f_pred_an = fouriernet->forwardDxFromData(X, /*deriv_dim=*/0)
                          + 10.0 * fouriernet->forwardDxFromData(X, /*deriv_dim=*/1);
            }

            if (input_dim==2) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, x, y);
            } else if (input_dim==1) {
                f_pred_nn = pde->forwardPDE(u_pred_nn, X);
            } else {
                TORCH_CHECK(false, "Unsupported input_dim for onewaywave_inhomo.");
            }

            f_pred = f_pred_an + f_pred_nn;

        } else if (pde_problem=="nonss_allencahn") {
            if (inputs.data.P.basis != nullptr) {
                torch::Tensor P = *inputs.data.P.basis;
                u_pred_an = fouriernet->forwardBasis(P);
            } else {
                u_pred_an = fouriernet->forward(X);
            }

            if (inputs.P_t != nullptr && inputs.P_xx != nullptr) {
                torch::Tensor P_t = *inputs.P_t;
                torch::Tensor P_xx = *inputs.P_xx;
                f_pred_an = fouriernet->forwardBasis(P_t) - 0.0001 * fouriernet->forwardBasis(P_xx);
            } else {
                f_pred_an = fouriernet->forwardDxFromData(X, /*deriv_dim=*/0)
                          - 0.0001 * fouriernet->forwardDxxFromData(X, /*deriv_dim=*/1);
            }

            // NN linear part via autograd.
            TORCH_CHECK(input_dim == 2, "nonss_allencahn expects input_dim=2 (t, x).");
            auto oneslike = torch::ones_like(u_pred_nn);
            auto u_t = torch::autograd::grad({ u_pred_nn }, { x }, {oneslike}, true, true)[0];
            auto u_x = torch::autograd::grad({ u_pred_nn }, { y }, {oneslike}, true, true)[0];
            auto u_xx = torch::autograd::grad({ u_x }, { y }, { oneslike }, true, true)[0];
            f_pred_nn = u_t - 0.0001 * u_xx;

            u_pred = u_pred_an + u_pred_nn;
            f_pred = f_pred_an + f_pred_nn + 5.0 * u_pred * u_pred * u_pred - 5.0 * u_pred;

        } else {
            std::cerr<<"WARNING: FourierNet forward_pde() not implemented for pde="<<pde_problem<<"\n";
        }

        return f_pred;
    }

    torch::Tensor getPeriodicBndLoss(const BndInputs& BND_DATA_A,
                                     const BndInputs& BND_DATA_B) override {
        torch::Tensor bnd_loss;
        if (verbose_) {
            std::cout<<"Getting periodic BND Loss for "<<this->modelType_<<"\n";
        }

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

        torch::Tensor u_a_an;
        torch::Tensor u_b_an;
        if (BND_DATA_A.data.P.basis != nullptr && BND_DATA_B.data.P.basis != nullptr) {
            torch::Tensor P_a = *BND_DATA_A.data.P.basis;
            torch::Tensor P_b = *BND_DATA_B.data.P.basis;
            u_a_an = this->fouriernet->forwardBasis(P_a);
            u_b_an = this->fouriernet->forwardBasis(P_b);
        } else {
            u_a_an = this->fouriernet->forward(*BND_DATA_A.data.X);
            u_b_an = this->fouriernet->forward(*BND_DATA_B.data.X);
        }
        torch::Tensor u_a_nn = this->neuralnet->forward(input_a);
        torch::Tensor u_b_nn = this->neuralnet->forward(input_b);

        torch::Tensor u_a = u_a_an + u_a_nn;
        torch::Tensor u_b = u_b_an + u_b_nn;

        // u(t, −1) = u(t, 1),
        torch::Tensor loss_1 = torch::mean((u_a - u_b).pow(2));
        bnd_loss = loss_1;
        if (verbose_) {
            std::cout<<"loss_1: "<<loss_1.item()<<"\n";
        }

        // ux(t, −1) = ux(t, 1).
        torch::Tensor u_xa_an;
        torch::Tensor u_xb_an;
        if (BND_DATA_A.P_x != nullptr && BND_DATA_B.P_x != nullptr) {
            torch::Tensor P_a_x = *BND_DATA_A.P_x;
            torch::Tensor P_b_x = *BND_DATA_B.P_x;
            u_xa_an = this->fouriernet->forwardBasis(P_a_x);
            u_xb_an = this->fouriernet->forwardBasis(P_b_x);
        } else {
            int deriv_dim = fouriernet->getPeriodicDerivDimValue();
            u_xa_an = this->fouriernet->forwardDxFromData(*BND_DATA_A.data.X, deriv_dim);
            u_xb_an = this->fouriernet->forwardDxFromData(*BND_DATA_B.data.X, deriv_dim);
        }

        auto ones_xa = torch::ones_like(u_a_nn);
        auto u_xa_nn = torch::autograd::grad({ u_a_nn }, { xa }, {ones_xa}, true, true)[0];

        auto ones_xb = torch::ones_like(u_b_nn);
        auto u_xb_nn = torch::autograd::grad({ u_b_nn }, { xb }, {ones_xb}, true, true)[0];

        torch::Tensor u_xa = u_xa_an + u_xa_nn;
        torch::Tensor u_xb = u_xb_an + u_xb_nn;

        torch::Tensor loss_2 = torch::mean((u_xa - u_xb).pow(2));
        bnd_loss += loss_2;
        if (verbose_) {
            std::cout<<"loss_2: "<<loss_2.item()<<"\n";
        }

        return bnd_loss;
    }

    torch::Tensor forward(const DataInputs& inputs) override {
        TORCH_CHECK(inputs.X != nullptr, "DataInputs requires X.");
        torch::Tensor X = *inputs.X;
        if (inputs.P.basis != nullptr) {
            torch::Tensor P = *inputs.P.basis;
            return forward(X, P);
        }

        bool outputs=false;
        torch::Tensor out;
        int n_nn_coeffs = neuralnet->getNRemainingBases();
        int n_pn_coeffs = fouriernet->getNRemainingBases();

        if (n_nn_coeffs>0) {
            torch::Tensor res_nn = neuralnet->forward(X);
            out=res_nn;
            outputs=true;
        }

        if (n_pn_coeffs>0) {
            torch::Tensor res_polyn = fouriernet->forward(X);
            if (outputs) out+=res_polyn;
            else out=res_polyn;
            outputs=true;
        }

        TORCH_CHECK(outputs, "VanillaFourierNeuralNet::forward(DataInputs): both branches inactive (no outputs).");

        return out;
    }

    torch::Tensor buildEvaluationBasis(const torch::Tensor& X) override {
        if (fouriernet->getNRemainingBases() == 0) return {};
        return fouriernet->buildBasisFromData(X);
    }

    torch::Tensor forward(const torch::Tensor& X, const torch::Tensor& P) {
        bool outputs=false;
        torch::Tensor out;
        int n_nn_coeffs = neuralnet->getNRemainingBases();
        int n_pn_coeffs = fouriernet->getNRemainingBases();

        if (n_nn_coeffs>0) {
            torch::Tensor res_nn = neuralnet->forward(X);
            out=res_nn;
            outputs=true;
        }

        if (n_pn_coeffs>0) {
            torch::Tensor res_polyn = fouriernet->forwardBasis(P);
            if (outputs) out+=res_polyn;
            else out=res_polyn;
            outputs=true;
        }

        TORCH_CHECK(outputs, "VanillaFourierNeuralNet::forward(X,P): both branches inactive (no outputs).");

        return out;
    }

    torch::Tensor forward(const torch::Tensor& X) {
        TORCH_CHECK(false, "This forward(X) overload is disabled. Use forward(DataInputs) with time input.");
        return X;
    }

    torch::Tensor getRegularizationLoss(const DataInputs& inputs) override {
        torch::Tensor totalLoss = torch::zeros({}, torch::TensorOptions().dtype(dtype_)).to(device_);
        totalLoss+=fouriernet->getRegularizationLoss(inputs);
        totalLoss+=neuralnet->getRegularizationLoss(inputs);
        return totalLoss;
    }

    void truncateCoeffs() override {
        neuralnet->truncateCoeffs();
        fouriernet->truncateCoeffs();
    }

    torch::Tensor getNNetBasis(const torch::Tensor& X) { return neuralnet->forwardBasis(X); }
    torch::Tensor getPolyBasisCoeffs() { return fouriernet->getBasisCoeffs();   }
    torch::Tensor getNNetBasisCoeffs() { return neuralnet->getBasisCoeffs(); }
    torch::Tensor getNNetBias()        { return neuralnet->getBasisBias();   }
    torch::Tensor getPolyNetGrad()     { return fouriernet->getPolyNetGrad(); }
    torch::Tensor getNeuralNetGrad()   { return neuralnet->getNNGrad();      }
    torch::Tensor getNNBiasGrad()      { return neuralnet->getNNBiasGrad();  }
    torch::Tensor& getPolyBasisSelection() { return fouriernet->getBasisSelection(); }
    torch::Tensor& getPolyBiasSelection()  { return fouriernet->getBiasSelection(); }
    torch::Tensor& getNNBasisSelection()   { return neuralnet->getBasisSelection(); }
    torch::Tensor& getNNBiasSelection()    { return neuralnet->getBiasSelection(); }


    std::vector<BasisCoefficientGroup> getBasisCoefficientGroups() override {
        auto result = fouriernet->getBasisCoefficientGroups();
        auto nn = neuralnet->getBasisCoefficientGroups();
        result.insert(result.end(), nn.begin(), nn.end());
        return result;
    }

    void printTrainingInfo() override {
        fouriernet->printTrainingInfo();
        neuralnet->printTrainingInfo();
    }

    VanillaFourierLayer fouriernet = nullptr;
    VanillaNeuralNet     neuralnet = nullptr;
    std::string modelType_;
    bool verbose_ = false;

private:
    std::string pde_problem;
    int npoly = 0;
    int input_dim;
    torch::Device device_; torch::Dtype dtype_;


    void printModelInfo() const {
        // Function to print the details of VanillaFourierNeuralNet
        std::cout << "\n\n----------------------- VanillaFourierNeuralNet Information: \n"
                  << "Input Dimension: " << input_dim << "\n"
                  << "pde_problem: " << pde_problem << "\n"
                  << "npoly: " << npoly << "\n"
                  << "modelType_: " << modelType_ << "\n"
                  << "truncate_: " << (truncate_ ? "enabled" : "disabled") << "\n"
                  << "-----------------------\n";
    }

};


TORCH_MODULE(VanillaFourierNeuralNet);

#endif
