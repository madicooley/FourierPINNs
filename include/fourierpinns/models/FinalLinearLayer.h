#ifndef FINAL_LINEAR_LAYER_H
#define FINAL_LINEAR_LAYER_H

#include <torch/torch.h>
#include <iostream>


// Custom Linear Layer that truncates weights based on a threshold
class FinalLinearLayerImpl : public torch::nn::Module {
public:
    // Constructor
    FinalLinearLayerImpl(int64_t in_features_value,
                         int64_t out_features_value,
                         double threshold_value,
                         bool include_bias,
                         std::string init_typ,
                         torch::Device device)
        : in_features(in_features_value),
          out_features(out_features_value),
          threshold(threshold_value),
          device_(device),
          includeBias_(include_bias),
          initTyp_(init_typ) {

        // Initialize weights and bias on the provided device
        weight = register_parameter(
            "weight",
            torch::empty({out_features, in_features},
                         torch::TensorOptions().device(device_)));

        if (includeBias_) {
            bias   = register_parameter(
                "bias",
                torch::empty(out_features, torch::TensorOptions().device(device_)));
            torch::nn::init::zeros_(bias);            // Initialize bias to zero
        }

        if (initTyp_=="xavier") {
            torch::nn::init::xavier_uniform_(weight); // Xavier initialization for weights
        } else if (initTyp_=="constant") {
            torch::nn::init::constant_(weight, 0.1);
        } // else default is He init
        
        // Initialize mask (all weights active at the start) and register as buffers
        weight_mask = register_buffer(
            "weight_mask",
            torch::ones_like(weight, torch::dtype(torch::kInt).device(device_)));

        if (includeBias_) {
            bias_mask   = register_buffer(
                "bias_mask",
                torch::ones_like(bias, torch::dtype(torch::kInt).device(device_)));
        }
    }

    void setVerbose(bool verbose) { verbose_ = verbose; }

    void truncateCoeffs() {
        // Update the weight_mask: set 0 where weights are below threshold
        weight_mask = torch::where(weight.abs() < threshold, torch::zeros_like(weight), weight_mask);
        weight.data().copy_(weight*weight_mask);

        if (includeBias_) {
            bias_mask   = torch::where(bias.abs() < threshold, torch::zeros_like(bias), bias_mask);
            bias.data().copy_(bias*bias_mask);
            if (verbose_) {
                std::cout<<"truncated bias: "<<bias_mask.sum().item<int>()<<"/1, ";
            }
        }

        if (verbose_) {
            std::cout<<"truncated weights: "<<weight_mask.sum().item<int>()<<"/"<<weight_mask.sizes()[1]<<"\n";
        }
        zero_truncated_gradients();
    }

    void zero_truncated_gradients() {
        // Zero gradients of truncated weights
        if (weight.grad().defined()) {
            weight.grad().mul_(weight_mask); // Use in-place multiplication
        }

        if (includeBias_) {
            if (bias.grad().defined()) {
                bias.grad().mul_(bias_mask);
            }
        }
    }

    torch::Tensor forward(const torch::Tensor& input) {
        // Perform linear transformation: y = xW^T + b
        if (includeBias_) {
            torch::Tensor output = torch::matmul(input, (weight*weight_mask).t()) + (bias*bias_mask);
            return output;
        } else {
            torch::Tensor output = torch::matmul(input, (weight*weight_mask).t());
            return output;
        }
    }

    void reassignLayerWeights(torch::Tensor& soln) {
        int startind=0;
        int expected = weight_mask.sum().item<int>();
        if (includeBias_) {
            expected += bias_mask.sum().item<int>();
        }
        TORCH_CHECK(soln.numel() == expected,
                    "Expected soln length ", expected,
                    " but got ", soln.numel(), ".");

        // Update the bias if bias_mask has non-zero elements
        if (includeBias_) {
            if (bias_mask.sum().item<int>() > 0) {
                torch::Tensor soln_bias = soln.index({torch::indexing::Slice(0, 1)});
                bias.data().copy_(soln_bias * bias_mask);
                startind=1;
            }
        }

        // Extract weights from the solution
        torch::Tensor soln_weights = soln.index({torch::indexing::Slice(startind, torch::indexing::None)});
        torch::Tensor upd_weights  = torch::zeros_like(weight);
        upd_weights.index_put_({ weight_mask.to(torch::kBool) }, soln_weights);
        weight.data().copy_(upd_weights * weight_mask);
    }

    int getNRemainingBases() {
        if (includeBias_) {
            return bias_mask.sum().item<int>() + weight_mask.sum().item<int>();
        }
        return weight_mask.sum().item<int>();
    }

    torch::Tensor& getBasisSelection() { return weight_mask; }
    torch::Tensor& getBiasSelection()  { return bias_mask; }
    torch::Tensor getBasisCoeffs()     { return weight * weight_mask; }
    torch::Tensor getBasisBias()       { return includeBias_ ? bias * bias_mask : torch::Tensor(); }
    torch::Tensor getGrads()           { return weight.grad() * weight_mask; }
    torch::Tensor getBiasGrad()        { return bias.grad() * bias_mask; }

private:
    torch::Tensor weight;        // Weight parameter
    torch::Tensor bias;          // Bias parameter
    torch::Tensor weight_mask;   // Binary mask to track active weights
    torch::Tensor bias_mask;     // Binary mask to track active bias
    int64_t in_features;         // Input size
    int64_t out_features;        // Output size
    double threshold;            // Truncation threshold
    torch::Device device_;

    bool includeBias_;
    std::string initTyp_;
    bool verbose_ = false;
};

// This macro creates a module holder type called FactorizedLinear that behaves similarly
// to the built-in modules (like torch::nn::Linear).
TORCH_MODULE(FinalLinearLayer);

#endif // FINAL_LINEAR_LAYER_H
