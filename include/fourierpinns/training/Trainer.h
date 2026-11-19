#pragma once 
#include "fourierpinns/models/CustomNet.h"
#include "fourierpinns/training/utils_trainer.h"
#include "fourierpinns/training/LossLambdaUpdater.h"
#include "fourierpinns/training/StatsLogger.h"
#include <sstream>


/**
 * @class ModelTrainer
 * @brief Manages the training process of models, providing flexibility in the training strategy.
 *
 * The ModelTrainer class is designed to abstract the training process, enabling the use of different training strategies.
 * It supports general training strategies as well as strategies specifically tailored for solving PDEs using neural networks.
 * The class allows for easy switching of strategies and is adaptable to various training paradigms by leveraging polymorphism.
 */

template <typename ModelType>
class ModelTrainer {
public:
    ModelTrainer(const TrainingConfig& args,
                 ModelType* model,
                 LoadedPDEDomain* pde_domain,
                 PDE* pde);

    void printModelInfo();
    void trainModel();
    void runLstsqUpdatesForTesting() { lstsqUpdates(); }
    BndInputs getPeriodicBndInputsForTesting() { return pde_domain->getPeriodicBndDataPair(); }
    torch::Tensor getRegularizationLossForTesting() { return getRegularizationLoss(); }
    torch::Tensor getPDELossForTesting() { return getPDELoss(); }
    torch::Tensor getBndLossForTesting() { return getBndLoss(); }

    StatsLogger<ModelType> stats_logger;

protected:
    TrainingConfig args;
    LoadedPDEDomain* pde_domain;
    PDE* pde;
    ModelType* model_;

    // float current_min_time_window;
    float lambda_bnd; ///< Coefficient for boundary/data loss.
    float lambda_res; ///< Coefficient for residual loss.
    float lambda_ic;  ///< Coefficient for initial condition/data loss.

private:
    LossLambdaUpdater lambda_updater;
    bool solvingpde;   ///< is this solving a pde or no
    bool verbose;      ///< Flag to control the verbosity of the training output.
    int forward_count; ///< Count of forward passes.
    int curr_epoch;
    int total_epochs;
    int lstsq_upd_count;
    double current_lr;



    void trainAdam(torch::optim::Adam& adamoptimizer, float& total_time) {
        std::cout<<"\n_____________________________________________Starting Adam training. \n";
        this->current_lr=args.adam_args.adam_max_lr;
        runAdamTrainingStage(
            args,
            adamoptimizer,
            curr_epoch,
            total_time,
            [&](int epoch, int max_epochs) {
                updateLearningRate(adamoptimizer, args.adam_args.adam_min_lr, args.adam_args.adam_max_lr, epoch, max_epochs);
            },
            [&](int epoch) {
                if (shouldRunLstsqAfterAdam(args, epoch)) {
                    std::cout<<"------------------------------------------LSTSQ Updates epoch="<<epoch<<"\n";
                    lstsqUpdates();
                }
            },
            [&](int epoch) {
                if (!args.lstsq_updates && model_->getTruncateModel()
                    && args.truncate_interval > 0 && epoch%args.truncate_interval==0 && epoch>0) {
                    std::cout<<"\n------------------------------------------Truncating coefficients. epoch="<<epoch<<"\n";
                    model_->truncateCoeffs();
                }
            },
            [&]() {
                if (args.col_batch_size > 0) pde_domain->setRandomCollocationBatch(args.col_batch_size, args.col_batch_shuffle);
                else pde_domain->clearBatch();
            },
            [&]() { pde_domain->clearBatch(); },
            [&](torch::optim::Optimizer& optimizer) { return getPDELosses(optimizer); },
            [&](int epoch, std::map<std::string, torch::Tensor>& losses, torch::optim::Optimizer& optimizer) {
                if (args.reglr_args.loss_reweight_scheme != "none" &&
                    epoch % args.reglr_args.lambda_upd_interval == 0) {
                    lambda_updater.updateLossLambdas(losses, optimizer, lambda_bnd, lambda_ic);
                }
            },
            [&](int epoch, std::map<std::string, torch::Tensor>& losses, float running_total_time) {
                if (args.pred_save_interval!=0 && epoch%args.pred_save_interval==0) {
                    stats_logger.addTestPrediction("adam-epoch-"+std::to_string(epoch));
                }
                if (epoch % args.print_interval == 0 || epoch==args.adam_args.adam_max_epochs) {
                    stats_logger.printStats(losses, epoch, running_total_time, lambda_res, lambda_bnd, lambda_ic, current_lr,
                                            solvingpde,
                                            verbose, args.print_model_trn_info);
                }
            },
            [](const torch::Tensor& loss) { return torch::isnan(loss).any().item<bool>(); });
    }

    void trainLBFGS(torch::optim::LBFGS& lbfgsoptimizer, float& total_time) {
        std::cout<<"\n_____________________________________________Starting LBFGS training. \n";
        this->current_lr=args.lbfgs_args.lbfgs_max_lr;
        runLBFGSTrainingStage(
            args,
            lbfgsoptimizer,
            curr_epoch,
            total_time,
            [&](torch::optim::Optimizer& optimizer) { return getPDELosses(optimizer); },
            [&](int epoch, int max_epochs, torch::optim::Optimizer& optimizer) {
                updateLearningRate(optimizer, args.lbfgs_args.lbfgs_min_lr, args.lbfgs_args.lbfgs_max_lr, epoch, max_epochs);
            },
            [&](int epoch, std::map<std::string, torch::Tensor>& losses, torch::optim::Optimizer& optimizer) {
                if (args.reglr_args.loss_reweight_scheme != "none" &&
                    epoch % args.reglr_args.lambda_upd_interval == 0) {
                    lambda_updater.updateLossLambdas(losses, optimizer, lambda_bnd, lambda_ic);
                }
            },
            [&](int epoch) {
                if (!args.lstsq_updates && model_->getTruncateModel()
                    && args.truncate_interval > 0 && epoch%args.truncate_interval==0 && epoch>0) {
                    std::cout<<"\n------------------------------------------Truncating coefficients. totallbfgs_epochs="<<epoch<<"\n";
                    model_->truncateCoeffs();
                }
            },
            [&](int epoch, std::map<std::string, torch::Tensor>& losses, float running_total_time) {
                if (epoch % args.print_interval == 0) {
                    stats_logger.printStats(losses, epoch, running_total_time, lambda_res, lambda_bnd, lambda_ic, current_lr,
                                            solvingpde, verbose, args.print_model_trn_info);
                }
            },
            [&](int epoch, int total_lbfgs_epochs, std::map<std::string, torch::Tensor>& losses, float running_total_time) {
                if (total_lbfgs_epochs % args.print_interval == 0 || epoch + 1 == args.lbfgs_args.lbfgs_max_epochs) {
                    stats_logger.printStats(losses, epoch, running_total_time, lambda_res, lambda_bnd, lambda_ic, current_lr,
                                            solvingpde, verbose, args.print_model_trn_info);
                }
            },
            []() {},
            [](const torch::Tensor& loss) { return torch::isnan(loss).any().item<bool>(); });
    }

    std::map<std::string, torch::Tensor> getPDELosses(torch::optim::Optimizer& optimizer) {
        (void)optimizer;
        std::map<std::string, torch::Tensor> losses;

        torch::Tensor pde_loss = getPDELoss();
        losses["pde_loss"] = pde_loss;
        losses["bnd_loss"] = getBndLoss();

        if (pde_domain->initial_cond) {
            torch::Tensor ic_loss = getICLoss();
            losses["ic_loss"] = ic_loss;
        }

        torch::Tensor reg_loss = getRegularizationLoss();
        losses["reg"] = reg_loss;

        losses["total_loss"] = accumulateTotalLoss(losses, torch::zeros_like(pde_loss));

        forward_count++;
        return losses;
    }

    void lstsqUpdates() {
        pde_domain->clearBatch(); // ensure full-data LSTSQ
        /*
         *  A = [ [ A_bnd_ff, A_bnd_nn ]
                  [ A_res_ff, A_res_nn ] ]
         */
        PDEInputs PDE_DATA = pde_domain->getPDEData();
        BndInputs BND_DATA = (pde_domain->bnd_typ == "periodic")
                               ? pde_domain->getPeriodicBndDataPair()
                               : pde_domain->getBndData();

        runLstsqUpdateLoop(lstsq_upd_count, args, [&]() {
            ICInputs IC_DATA;
            ICInputs* IC_PTR = nullptr;
            torch::Tensor b;
            const std::string pde_name = pde->getPDEProblem();
            const bool safeguarded = args.lstsq_safeguard && pde_name == "allencahn";
            auto merit = [&]() {
                auto loss = getPDELoss();
                loss = loss + getBndLoss();
                return loss.template item<double>();
            };
            const double original_loss = safeguarded ? merit() : 0.0;
            if (safeguarded && !std::isfinite(original_loss)) {
                std::cout << "  LS safeguard: skipped; starting nonlinear loss is non-finite. No pruning.\n";
                return;
            }
            auto is_time_dependent = [&](const std::string& name) {
                return name=="onewaywave_inhomo" || name=="nonss_allencahn";
            };
            if (pde_domain->initial_cond && is_time_dependent(pde_name)) {
                IC_DATA = pde_domain->getICData();
                IC_PTR = &IC_DATA;
            }

            torch::Tensor ftrue;
            if (!pde_domain->homogeneous_pde && PDE_DATA.f != nullptr) {
                ftrue = *PDE_DATA.f;
            } else {
                TORCH_CHECK(PDE_DATA.data.X != nullptr, "PDE_DATA requires X.");
                ftrue = torch::zeros({PDE_DATA.data.X->size(0), 1}, PDE_DATA.data.X->options());
            }

            torch::Tensor bnd_block;
            if (pde_domain->bnd_typ == "periodic") {
                TORCH_CHECK(BND_DATA.data.X != nullptr, "Periodic BND_DATA requires X.");
                int nbnd = static_cast<int>(BND_DATA.data.X->size(0));
                bnd_block = torch::zeros({2 * nbnd, 1}, BND_DATA.data.X->options());
            } else {
                TORCH_CHECK(BND_DATA.u != nullptr, "BND_DATA must contain Soln for LSTSQ.");
                bnd_block = *BND_DATA.u;
            }

            torch::Tensor pde_block;
            // Freeze the reaction term at the current prediction for steady Allen-Cahn.
            if (pde_name == "allencahn") {
                torch::Tensor u_pred = model_->forward(PDE_DATA.data);
                torch::Tensor s = u_pred * (u_pred * u_pred - 1.0);
                pde_block = ftrue - s;
            } else {
                pde_block = ftrue;
            }

            if (IC_PTR != nullptr && IC_PTR->u != nullptr) {
                torch::Tensor ic_block = *IC_PTR->u;
                b = torch::cat({bnd_block, ic_block, pde_block}, 0);
            } else {
                b = torch::cat({bnd_block, pde_block}, 0);
            }

            torch::Tensor A = model_->getLSTSQ_A(PDE_DATA, BND_DATA, IC_PTR, pde);
            if (safeguarded && (!torch::isfinite(A).all().template item<bool>() ||
                                !torch::isfinite(b).all().template item<bool>())) {
                std::cout << "  LS safeguard: rejected non-finite system. No pruning.\n";
                return;
            }
            std::cout << "  System: A=" << A.sizes() << ", b=" << b.sizes()
                      << " | input: " << A.device() << " " << A.dtype() << "\n";
            std::cout << "  Solver: CPU float64, SVD (gelsd), rcond: "
                      << leastSquaresRankCutoff(A, args.lstsq_rcond) << ", column scaling: disabled\n";

            auto lstsq_result = solveLeastSquaresOnCpu(A, b, pde_domain->getDevice(), pde_domain->getDtype(),
                                                     args.lstsq_rcond);
            torch::Tensor soln = lstsq_result.first;
            int64_t rank = lstsq_result.second;

            std::cout << "  Rank: " << rank << "/" << A.size(1)
                      << " | returned coefficients: " << soln.device() << " " << soln.dtype() << "\n";

            double accepted_loss = original_loss;
            if (safeguarded) {
                if (!torch::isfinite(soln).all().template item<bool>()) {
                    std::cout << "  LS safeguard: rejected non-finite coefficients. No pruning.\n";
                    return;
                }
                auto parameters = model_->parameters();
                TORCH_CHECK(!parameters.empty(), "Safeguarded LS requires registered model parameters.");
                auto snapshot = [&]() {
                    std::vector<torch::Tensor> values;
                    for (const auto& parameter : parameters) values.push_back(parameter.detach().clone());
                    return values;
                };
                const auto old = snapshot();
                std::vector<torch::Tensor> proposed;
                {
                    torch::NoGradGuard guard;
                    try {
                        model_->lstsqUpdates(soln);
                        proposed = snapshot();
                    } catch (...) {
                        for (size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(old[i]);
                        throw;
                    }
                    for (size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(old[i]);
                }
                auto apply = [&](double step) {
                    torch::NoGradGuard guard;
                    for (size_t i = 0; i < parameters.size(); ++i) {
                        if (step == 0.0) parameters[i].copy_(old[i]);
                        else parameters[i].copy_(old[i] + step * (proposed[i] - old[i]));
                    }
                };
                const auto result = backtrackLstsqUpdate(original_loss, apply, merit);
                std::cout << "  LS safeguard: " << (result.accepted ? "accepted" : "rejected")
                          << " | step: " << result.step << " | nonlinear loss: "
                          << original_loss << " -> " << result.loss << "\n";
                if (!result.accepted) {
                    std::cout << "  Pruning: skipped for rejected LS update.\n";
                    return;
                }
                accepted_loss = result.loss;
            } else {
                soln.set_requires_grad(true);
                model_->lstsqUpdates(soln);
            }

            std::vector<int64_t> before;
            for (const auto& group : model_->getBasisCoefficientGroups()) {
                before.push_back(group.mask.defined() ? group.mask.sum().template item<int64_t>() : 0);
            }

            if (model_->getTruncateModel()) {
                std::cout<<"Truncating after lstsq updates.\n";
                if (safeguarded) {
                    // Pruning may undo the line search's improvement; roll it back independently.
                    auto parameters = model_->parameters();
                    std::vector<torch::Tensor> saved, masks;
                    for (const auto& parameter : parameters) saved.push_back(parameter.detach().clone());
                    for (const auto& group : model_->getBasisCoefficientGroups()) {
                        masks.push_back(group.mask.defined() ? group.mask.clone() : torch::Tensor());
                    }
                    model_->truncateCoeffs();
                    const double pruned_loss = merit();
                    if (!std::isfinite(pruned_loss) || pruned_loss > accepted_loss) {
                        torch::NoGradGuard guard;
                        for (size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(saved[i]);
                        auto groups = model_->getBasisCoefficientGroups();
                        for (size_t i = 0; i < groups.size(); ++i) {
                            if (masks[i].defined()) groups[i].mask.copy_(masks[i]);
                        }
                        std::cout << "  LS safeguard: pruning rolled back (nonlinear loss increased).\n";
                    }
                } else model_->truncateCoeffs();
            } else {
                std::cout << "  Pruning: disabled\n";
            }
            printBasisSummary(std::cout, model_->getBasisCoefficientGroups(), before,
                              args.print_model_trn_info);
        });
    }


    torch::Tensor getRegularizationLoss() {
        torch::Tensor reg_loss;
        PDEInputs PDE_DATA = pde_domain->getPDEData();
        // Coefficient penalties are independent of the number of data blocks.
        reg_loss = model_->getRegularizationLoss(PDE_DATA.data);

        return reg_loss;
    }


    torch::Tensor getPDELoss() {
        torch::Tensor pde_loss;
        PDEInputs PDE_DATA = pde_domain->getPDEData();
        torch::Tensor f_pred = model_->forward_pde(PDE_DATA, pde);
        torch::Tensor diff;
        if (pde_domain->homogeneous_pde) diff = f_pred;
        else {
            TORCH_CHECK(PDE_DATA.f != nullptr, "PDE_DATA must contain Soln for non-homogeneous PDEs.");
            torch::Tensor f_true = *PDE_DATA.f;
            diff = f_pred - f_true;
        }

        pde_loss = torch::mean(diff.pow(2));
        pde_loss = (this->lambda_res*pde_loss);
        return pde_loss;
    }

    torch::Tensor getBndLoss() {
        torch::Tensor u_pred, bnd_loss;
        if (pde_domain->bnd_typ=="dirichlet") {
            BndInputs BND_DATA = pde_domain->getBndData();
            u_pred = model_->forward(BND_DATA.data);
            TORCH_CHECK(BND_DATA.u != nullptr, "BND_DATA must contain Soln.");
            torch::Tensor u_true = *BND_DATA.u;
            bnd_loss = torch::mean((u_pred - u_true).pow(2));

        } else if (pde_domain->bnd_typ=="periodic") {
            BndInputs BND_DATA_A = pde_domain->getBndData_A();
            BndInputs BND_DATA_B = pde_domain->getBndData_B();
            bnd_loss = model_->getPeriodicBndLoss(BND_DATA_A, BND_DATA_B);
        }

        bnd_loss = (this->lambda_bnd*bnd_loss);
        return bnd_loss;
    }

    torch::Tensor getICLoss() {
        torch::Tensor ic_loss;
        ICInputs IC_DATA = pde_domain->getICData();

        torch::Tensor uic_pred = model_->forward(IC_DATA.data);
        TORCH_CHECK(IC_DATA.u != nullptr, "IC_DATA must contain Soln.");
        torch::Tensor uic_true = *IC_DATA.u;

        ic_loss = torch::mean((uic_pred - uic_true).pow(2));
        ic_loss = (this->lambda_ic*ic_loss);
        return ic_loss;
    }


    void printLearningRateSchedule(double min_lr, double max_lr, int max_epochs) {
        TrainingLearningRateScheduler::printSchedule(args, min_lr, max_lr, max_epochs);
    }

    void updateLearningRate(torch::optim::Optimizer& optimizer, double min_lr, double max_lr, int epoch, int max_epochs) {
        this->current_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
            optimizer, args, min_lr, max_lr, epoch, max_epochs, this->current_lr);
    }

};


#include "Trainer.tpp"
