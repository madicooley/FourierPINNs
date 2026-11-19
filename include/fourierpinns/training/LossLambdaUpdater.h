#pragma once
#include <algorithm>
#include <map>
#include <string>

#include <torch/torch.h>

#include "fourierpinns/training/utils_trainer.h"

/**
 * @brief Helper class for updating loss weighting lambda values.
 */
class LossLambdaUpdater {
public:
    explicit LossLambdaUpdater(const TrainingConfig& training_args)
        : args(training_args) {}

    void updateLossLambdas(std::map<std::string, torch::Tensor>& losses_dict,
                           torch::optim::Optimizer& optimizer,
                           float& lambda_bnd,
                           float& lambda_ic) {
        if (args.reglr_args.loss_reweight_scheme == "grad") {
            gradientAnnealingWeighting(losses_dict, optimizer, lambda_bnd, lambda_ic);
        }
    }

private:
    const TrainingConfig& args;

    void gradientAnnealingWeighting(std::map<std::string, torch::Tensor>& losses_dict,
                                    torch::optim::Optimizer& optimizer,
                                    float& lambda_bnd,
                                    float& lambda_ic) {
        std::string key1 = "pde_loss";
        torch::Tensor pde_loss = losses_dict[key1];

        float max_abs_grad = getMaxAbsNNGrad(optimizer, pde_loss);

        std::map<std::string, float> grad_means_dict;
        for (const auto& entry : losses_dict) {
            if (entry.first == "pde_loss") continue;
            if (entry.first == "reg") continue;
            torch::Tensor loss = entry.second;
            float mean_abs_grad = getMeanAbsGrad(optimizer, loss);
            grad_means_dict[entry.first] = mean_abs_grad;
        }

        std::map<std::string, float> updated_lambdas;
        constexpr float kGradEps = 1e-8f;
        for (const auto& entry : grad_means_dict) {
            float denom = std::max(entry.second, kGradEps);
            float weight = max_abs_grad / denom;
            updated_lambdas[entry.first] = weight;
        }

        float momentum = 0.9f;
        for (const auto& entry : updated_lambdas) {
            if (entry.first == "bnd_loss")
                lambda_bnd = lambda_bnd * momentum + (1.0f - momentum) * entry.second;
            else if (entry.first == "ic_loss")
                lambda_ic = lambda_ic * momentum + (1.0f - momentum) * entry.second;
        }
    }

};
