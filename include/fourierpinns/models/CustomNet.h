#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <memory>
#include <vector>
#include <string>

#include "FinalLinearLayer.h"
#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/data/DataBundle.h"
#include "fourierpinns/training/BasisLogging.h"


/**
 * @brief Abstract base class for custom neural network models.
 *
 * This class extends torch::nn::Module and provides an interface for neural networks
 * with additional functionalities such as least-squares updates,
 * and regularization losses.
 */

class CustomNetwork : public torch::nn::Module {
public:
    /**
     * @brief Construct a new CustomNetwork object.
     */
    CustomNetwork() : modelType_("none"), truncate_(false) {}

    /**
     * @brief Virtual destructor.
     */
    virtual ~CustomNetwork() override = default;

    /**
     * @brief Forward pass using tensor of inputs.
     *
     * The map should include a key (e.g., "Data") that points to the input tensor.
     *
     * @param inputs Tensor pointer.
     * @return torch::Tensor The network's output.
     */
    virtual torch::Tensor forward(const torch::Tensor& inputs) = 0;


    /**
     * @brief Forward pass using structured inputs.
     *
     * Uses optional basis features when provided.
     */
    virtual torch::Tensor forward(const DataInputs& inputs) = 0;

    /**
     * @brief Compute the PDE residual using provided inputs.
     *
     * The default implementation obtains the network prediction via
     * `forward()` and then evaluates the PDE residual using the provided
     * `PDE` object.  Derived classes can override this to implement custom
     * residuals that do not rely on autograd.
     */
    virtual torch::Tensor forward_pde(const PDEInputs& inputs, PDE* pde) {
        torch::Tensor u_pred;
        TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X data.");
        torch::Tensor X = *inputs.data.X;
        if (X.sizes().size() == 1 || X.sizes()[1] == 1) {
            u_pred = forward(X);
            return pde->forwardPDE(u_pred, X);

        } else if (X.sizes()[1] == 2) {
            auto x = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
            auto y = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);
            auto input = torch::cat({x, y}, 1);  // shape: (N, 2)
            u_pred = forward(input);
            return pde->forwardPDE(u_pred, x, y);

        } else if (X.sizes()[1] == 3) {
            TORCH_CHECK(false, "3D PDEs are not supported in PDE::forwardPDE()");
        }
        std::cerr << "WARNING: CustomNetwork forward_pde() could not determine coordinates\n";
        return torch::Tensor();
    }

    /**
     * @brief Perform least-squares updates on the network weights.
     *
     * @param soln The solution tensor used for updating weights.
     */
    virtual void lstsqUpdates(torch::Tensor& soln) = 0;

    /**
     * @brief Construct the matrix A used for least-squares updates.
     *
     * @param pdeData Data associated with the PDE.
     * @param bndData Boundary condition data.
     * @param pde A pointer to a PDE object.
     * @return torch::Tensor The constructed matrix A.
     */
    virtual torch::Tensor getLSTSQ_A(const PDEInputs& pdeData,
                                     const BndInputs& bndData,
                                     const ICInputs* icData,
                                     PDE* pde) = 0;

    /**
     * @brief Print training information (e.g., weights and biases).
     */
    virtual void printTrainingInfo() = 0;
    virtual std::vector<BasisCoefficientGroup> getBasisCoefficientGroups() { return {}; }
    // Non-Fourier models do not need externally built evaluation features.
    virtual torch::Tensor buildEvaluationBasis(const torch::Tensor&) { return {}; }

    /**
     * @brief Truncate network coefficients.
     */
    virtual void truncateCoeffs() = 0;

    /**
     * @brief Compute the regularization loss for the network.
     *
     * @return torch::Tensor The regularization loss.
     */
    virtual torch::Tensor getRegularizationLoss(const DataInputs& inputs) = 0;

    virtual torch::Tensor getPeriodicBndLoss(const BndInputs& inputs_a,
                                             const BndInputs& inputs_b) = 0;

    void setTruncateModel() {
        this->truncate_=true;
    }

    bool getTruncateModel() const {
        return truncate_;
    }


    std::string modelType_;

protected:
    bool truncate_;

private:
};

#endif // CUSTOMNETWORK_H
