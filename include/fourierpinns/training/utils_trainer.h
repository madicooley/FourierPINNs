#pragma once

#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include <torch/torch.h>

#include "fourierpinns/config/Args.h"

class TrainingLearningRateScheduler {
public:
    static double computeLearningRate(const TrainingConfig& args,
                                      double min_lr,
                                      double max_lr,
                                      int epoch,
                                      int max_epochs,
                                      double current_lr) {
        const int safe_warmup_steps = (args.lr_warmup_steps > 0) ? args.lr_warmup_steps : 0;
        const int safe_decay_interval = (args.lr_decay_interval > 0) ? args.lr_decay_interval : 0;
        if (safe_warmup_steps > 0 && epoch < safe_warmup_steps) {
            current_lr = min_lr + (max_lr - min_lr) * epoch / safe_warmup_steps;
        } else if (safe_decay_interval > 0 &&
                   epoch % safe_decay_interval == 0 &&
                   epoch >= safe_warmup_steps &&
                   epoch < args.lr_decay_steps) {
            if (args.lr_decaytype == "cosine") {
                current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + std::cos(M_PI * epoch / max_epochs));
            } else if (args.lr_decaytype == "exp") {
                int decayepoch = epoch - safe_warmup_steps;
                current_lr = max_lr * std::pow(args.lr_decay_rate, decayepoch / safe_decay_interval);
            }
        }
        return current_lr;
    }

    static void printSchedule(const TrainingConfig& args, double min_lr, double max_lr, int max_epochs) {
        double current_lr = max_lr;
        int n_decay_steps = 0;
        const int safe_warmup_steps = (args.lr_warmup_steps > 0) ? args.lr_warmup_steps : 0;
        const int safe_decay_interval = (args.lr_decay_interval > 0) ? args.lr_decay_interval : 0;

        for (int epoch = 0; epoch <= max_epochs; ++epoch) {
            if (safe_warmup_steps > 0 && epoch < safe_warmup_steps) {
                current_lr = min_lr + (max_lr - min_lr) * epoch / safe_warmup_steps;
                if (epoch % 500 == 0) {
                    std::cout << "Warmup | Epoch " << epoch << ": Learning Rate = " << current_lr << std::endl;
                }
            } else if (safe_decay_interval > 0 &&
                       epoch % safe_decay_interval == 0 &&
                       epoch >= safe_warmup_steps &&
                       epoch < args.lr_decay_steps) {
                if (args.lr_decaytype == "cosine") {
                    current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + std::cos(M_PI * epoch / max_epochs));
                } else if (args.lr_decaytype == "exp") {
                    int decayepoch = epoch - safe_warmup_steps;
                    current_lr = max_lr * std::pow(args.lr_decay_rate, decayepoch / safe_decay_interval);
                }
                n_decay_steps += 1;
                std::cout << "Decay | Epoch " << epoch << ": Learning Rate = " << current_lr
                          << ", N decay steps: " << n_decay_steps << std::endl;
            } else if (epoch >= args.lr_decay_steps) {
                if (epoch % 1000 == 0) {
                    std::cout << "Constant | Epoch " << epoch << ": Learning Rate = " << current_lr << std::endl;
                }
            }
        }
    }

    static double updateOptimizerLearningRate(torch::optim::Optimizer& optimizer,
                                              const TrainingConfig& args,
                                              double min_lr,
                                              double max_lr,
                                              int epoch,
                                              int max_epochs,
                                              double current_lr) {
        const double updated_lr = computeLearningRate(args, min_lr, max_lr, epoch, max_epochs, current_lr);
        if (updated_lr != current_lr) {
            setAllGroupLrs(optimizer, updated_lr);
        }
        return updated_lr;
    }

private:
    static void setAllGroupLrs(torch::optim::Optimizer& optimizer, double lr) {
        for (auto& group : optimizer.param_groups()) {
            group.options().set_lr(lr);
        }
    }
};

template <typename TrainingConfig,
          typename AdamOptimizer,
          typename UpdateLrFn,
          typename MaybeLstsqFn,
          typename MaybeTruncateFn,
          typename PrepareBatchFn,
          typename ClearBatchFn,
          typename ComputeLossesFn,
          typename UpdateLambdasFn,
          typename PostEpochFn,
          typename IsNanFn>
inline void runAdamTrainingStage(const TrainingConfig& args,
                                 AdamOptimizer& adamoptimizer,
                                 int& curr_epoch,
                                 float& total_time,
                                 UpdateLrFn&& update_lr,
                                 MaybeLstsqFn&& maybe_lstsq,
                                 MaybeTruncateFn&& maybe_truncate,
                                 PrepareBatchFn&& prepare_batch,
                                 ClearBatchFn&& clear_batch,
                                 ComputeLossesFn&& compute_losses,
                                 UpdateLambdasFn&& update_lambdas,
                                 PostEpochFn&& post_epoch,
                                 IsNanFn&& is_nan) {
    for (int epoch = 1; epoch <= args.adam_args.adam_max_epochs; ++epoch) {
        const auto epoch_start_time = std::chrono::high_resolution_clock::now();
        update_lr(epoch - 1, args.adam_args.adam_max_epochs);
        prepare_batch();

        adamoptimizer.zero_grad();
        auto losses = compute_losses(adamoptimizer);
        torch::Tensor loss = losses.at("total_loss");
        update_lambdas(epoch, losses, adamoptimizer);
        loss.backward();
        adamoptimizer.step();
        curr_epoch++;
        clear_batch();
        // Alternate only after completed Adam steps, including the final block.
        maybe_lstsq(epoch);
        maybe_truncate(epoch);
        if (epoch % args.print_interval == 0 || epoch == args.adam_args.adam_max_epochs) {
            losses = compute_losses(adamoptimizer);
        }

        const auto epoch_end_time = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();
        total_time += static_cast<float>(elapsed);

        post_epoch(epoch, losses, total_time);
        if (is_nan(loss)) {
            std::cout << "Loss is NaN. Stopping training." << std::endl;
            break;
        }
    }
}

template <typename TrainingConfig,
          typename LBFGSOptimizer,
          typename ComputeLossesFn,
          typename UpdateLrFn,
          typename UpdateLambdasFn,
          typename MaybeTruncateFn,
          typename PrintClosureStatsFn,
          typename PrintEpochStatsFn,
          typename OnEpochAdvanceFn,
          typename IsNanFn>
inline void runLBFGSTrainingStage(const TrainingConfig& args,
                                  LBFGSOptimizer& lbfgsoptimizer,
                                  int& curr_epoch,
                                  float& total_time,
                                  ComputeLossesFn&& compute_losses,
                                  UpdateLrFn&& update_lr,
                                  UpdateLambdasFn&& update_lambdas,
                                  MaybeTruncateFn&& maybe_truncate,
                                  PrintClosureStatsFn&& print_closure_stats,
                                  PrintEpochStatsFn&& print_epoch_stats,
                                  OnEpochAdvanceFn&& on_epoch_advance,
                                  IsNanFn&& is_nan) {
    int total_lbfgs_epochs = 0;

    for (int epoch = 0; epoch < args.lbfgs_args.lbfgs_max_epochs; ++epoch) {
        const auto epoch_start_time = std::chrono::high_resolution_clock::now();
        // Keep the objective and optimizer settings fixed during line search.
        update_lr(epoch, args.lbfgs_args.lbfgs_max_epochs, lbfgsoptimizer);
        auto stage_losses = compute_losses(lbfgsoptimizer);
        update_lambdas(epoch, stage_losses, lbfgsoptimizer);
        maybe_truncate(epoch);

        auto closure = [&]() {
            lbfgsoptimizer.zero_grad();
            auto losses = compute_losses(lbfgsoptimizer);
            torch::Tensor loss = losses.at("total_loss");

            print_closure_stats(total_lbfgs_epochs, losses, total_time);

            if (is_nan(loss)) {
                std::cout << "Loss is NaN. Stopping training." << std::endl;
                return loss;
            }
            total_lbfgs_epochs++;
            loss.backward();
            return loss;
        };

        lbfgsoptimizer.step(closure);

        const auto epoch_end_time = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();
        total_time += static_cast<float>(elapsed);

        auto losses = compute_losses(lbfgsoptimizer);
        torch::Tensor loss = losses.at("total_loss");
        print_epoch_stats(epoch, total_lbfgs_epochs, losses, total_time);
        if (is_nan(loss)) {
            std::cout << "Loss is NaN. Stopping training." << std::endl;
            return;
        }
        on_epoch_advance();
        curr_epoch++;
    }
}

template <typename TrainingConfig, typename PerUpdateFn>
inline void runLstsqUpdateLoop(int& lstsq_upd_count, const TrainingConfig& args, PerUpdateFn&& per_update) {
    int lstsq_upds_per_epoch_count = 0;
    if (lstsq_upd_count >= args.n_lstsq_upds) return;

    while (lstsq_upds_per_epoch_count < args.n_lstsq_upds_per_epoch
           && lstsq_upd_count < args.n_lstsq_upds) {
        std::cout << "\n[LS " << lstsq_upd_count + 1 << "/" << args.n_lstsq_upds
                  << " | block " << lstsq_upds_per_epoch_count + 1 << "/" << args.n_lstsq_upds_per_epoch
                  << "]\n";

        per_update();
        lstsq_upd_count++;
        lstsq_upds_per_epoch_count++;
    }
}

struct LstsqBacktrackingResult {
    bool accepted = false;
    double step = 0.0;
    double loss = 0.0;
};

// apply(0) must restore the original state exactly, including after a non-finite trial.
template <typename ApplyFn, typename EvaluateFn>
inline LstsqBacktrackingResult backtrackLstsqUpdate(double original_loss, ApplyFn&& apply, EvaluateFn&& evaluate) {
    if (std::isfinite(original_loss)) {
        double step = 1.0;
        for (int reduction = 0; reduction <= 20; ++reduction, step *= 0.5) {
            double loss;
            try {
                apply(step);
                loss = evaluate();
            } catch (...) {
                apply(0.0);
                throw;
            }
            if (std::isfinite(loss) && loss <= original_loss) return {true, step, loss};
        }
    }
    apply(0.0);
    return {false, 0.0, original_loss};
}

inline bool shouldRunLstsqAfterAdam(const TrainingConfig& args, int completed_steps) {
    return args.lstsq_updates && args.lstsq_upd_interval > 0 && completed_steps > 0
        && completed_steps % args.lstsq_upd_interval == 0
        && (args.lstsq_stop_epoch < 0 || completed_steps <= args.lstsq_stop_epoch);
}

inline double leastSquaresRankCutoff(const torch::Tensor& A, double configured_rcond) {
    if (configured_rcond >= 0.0) return configured_rcond;
    const double eps = A.scalar_type() == torch::kFloat32
        ? std::numeric_limits<float>::epsilon() : std::numeric_limits<double>::epsilon();
    return eps * static_cast<double>(std::max(A.size(0), A.size(1)));
}

inline std::pair<torch::Tensor, int64_t> solveLeastSquaresOnCpu(torch::Tensor A,
                                                                 torch::Tensor b,
                                                                 torch::Device target_device,
                                                                 torch::Dtype target_dtype,
                                                                 double configured_rcond = -1.0) {
    const double rcond = leastSquaresRankCutoff(A, configured_rcond);
    A = A.to(torch::kFloat64);
    b = b.to(torch::kFloat64);
    auto A_cpu = A.to(torch::kCPU);
    auto b_cpu = b.to(torch::kCPU);
    auto result = torch::linalg::lstsq(A_cpu, b_cpu, rcond, "gelsd");

    torch::Tensor soln = std::get<0>(result).squeeze(1).to(target_device);
    soln = soln.to(target_dtype);
    const int64_t rank = std::get<2>(result).item<int64_t>();
    return {soln, rank};
}

inline torch::Tensor accumulateTotalLoss(const std::map<std::string, torch::Tensor>& losses,
                                         const torch::Tensor& seed_loss) {
    torch::Tensor total_loss = seed_loss.clone();
    for (const auto& entry : losses) {
        total_loss += entry.second;
    }
    return total_loss;
}

inline void saveTensorMap(const std::map<std::string, torch::Tensor>& tensor_map, const std::string& filename) {
    torch::serialize::OutputArchive archive;

    // Iterate over each entry in the map and write it to the archive
    for (const auto& pair : tensor_map) {
        archive.write(pair.first, pair.second);
    }

    // Save the archive to the specified file
    archive.save_to(filename);
}

inline void zeroGrads(torch::optim::Optimizer& optimizer) {
    for (const auto& param_group : optimizer.param_groups()) {
        for (const auto& param : param_group.params()) {
            if (param.grad().defined()) param.grad().zero_();  // Zero-out any existing gradients
        }
    }
}

inline torch::Tensor getAllFlatGrads(torch::optim::Optimizer& optimizer) {
    std::vector<torch::Tensor> grads; // Collect gradients for each parameter
    for (const auto& param_group : optimizer.param_groups()) {
        for (const auto& param : param_group.params()) {
            if (param.grad().defined()) {
                torch::Tensor grad = param.grad();
                torch::Tensor flattened_grad = grad.view(-1);
                grads.push_back(flattened_grad);
            }
        }
    }
    torch::Tensor allflatgrads = torch::cat(grads, 0);
    return allflatgrads;
}

inline float getMaxAbsNNGrad(torch::optim::Optimizer& optimizer, torch::Tensor& loss) {
    zeroGrads(optimizer);
    loss.backward({}, true); // Compute gradients w.r.t. the current loss while retaining the graph
    torch::Tensor grads = getAllFlatGrads(optimizer);
    float max_abs_grad = torch::max(torch::abs(grads)).item<float>();
    return max_abs_grad;
}

inline float getMeanAbsGrad(torch::optim::Optimizer& optimizer, torch::Tensor& loss) {
    zeroGrads(optimizer);
    loss.backward({}, true);  // Compute gradients w.r.t. the current loss while retaining the graph
    torch::Tensor grads = getAllFlatGrads(optimizer);
    float mean_abs_grad = torch::mean(torch::abs(grads)).item<float>();
    return mean_abs_grad;
}
