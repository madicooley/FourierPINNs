#pragma once
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <functional>
#include "fourierpinns/models/CustomNet.h"

template<typename ModelType>
class StatsLogger {
public:
    StatsLogger(const TrainingConfig& args,
                LoadedPDEDomain* domain,
                ModelType* model)
        : args_(args), domain_(domain), model_(model) {}

    using TensorFn = std::function<torch::Tensor()>;

    void printStats(std::map<std::string, torch::Tensor>& losses_dict,
                    int epoch,
                    float& total_time,
                    float lambda_res,
                    float lambda_bnd,
                    float lambda_ic,
                    double current_lr,
                    bool solvingpde,
                    bool verbose,
                    bool print_model_trn_info) {
        if (solvingpde) {
            printPDEStats(losses_dict, epoch, total_time, lambda_res, lambda_bnd, lambda_ic,
                          current_lr, verbose, print_model_trn_info);
        }
    }


    int saveStats(std::string filename_root, std::string datadir, bool regressionStats=false) {
        std::string filename = (std::filesystem::path(datadir) / "stats" / ("stats_" + filename_root + ".txt")).string();
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());

        // Open the file for writing
        std::ofstream outFile(filename);

        // Check if the file was successfully opened
        if (!outFile.is_open()) {
            std::cout << "Error: Could not open the file for writing. Filename-> "<<filename<< std::endl;
            return 1; // Return an error code
        }

        outFile << "TEST_ERRORS" << std::endl;
        writeVector(outFile, test_errors);

        outFile << "\nTRAIN_ERRORS" << std::endl;
        writeVector(outFile, train_errors);

        outFile << "\nTEST_LOSSES" << std::endl;
        writeVector(outFile, test_losses);

        outFile << "\nTRAIN_LOSSES" << std::endl;
        writeVector(outFile, train_losses);

        outFile << "\nTRAIN_TIMES" << std::endl;
        writeVector(outFile, train_times);

        outFile << "\nTRAIN_ITERS" << std::endl;
        writeVector(outFile, iters);

        if (regressionStats==false) {
            outFile << "\nPDE_LOSSES" << std::endl;
            writeVector(outFile, train_pde_losses);

            outFile << "\nBND_LOSSES" << std::endl;
            writeVector(outFile, train_bnd_losses);

            outFile << "\nIC_LOSSES" << std::endl;
            writeVector(outFile, train_ic_losses);

            outFile << "\nLAMBDA_RES" << std::endl;
            writeVector(outFile, train_lambda_res);

            outFile << "\nLAMBDA_BND" << std::endl;
            writeVector(outFile, train_lambda_bnds);

            outFile << "\nLAMBDA_IC" << std::endl;
            writeVector(outFile, train_lambda_ics);


        }

        outFile.close();
        std::cout << "Data has been written to " << filename << std::endl;

        return 0;
    }


    void addTestPrediction(std::string epochname) {
        std::cout<<"Adding to test_predictions. epochname="<<epochname<<"\n";
        torch::Tensor u_pred_test = getTestPred().detach(); // Full test prediction

        test_predictions[epochname]=u_pred_test;
    }

    int savePDEPredictions(std::string pde_problem, std::string filename_root, std::string datadir) {
        (void)pde_problem;
        std::string filename = (std::filesystem::path(datadir) / "predictions" / ("predictions_u_" + filename_root + ".pt")).string();
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());

        addTestPrediction("finalprediction"); // final prediction
        saveTensorMap(test_predictions, filename);

        std::cout<<"PDE Solutions have been written to "<<filename<<"\n";
        return 0;
    }

    int saveRgrPredictions(std::string filename_root, std::string datadir) {
        std::string filename = (std::filesystem::path(datadir) / "predictions" / ("predictions_u_" + filename_root + ".pt")).string();
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        torch::Tensor u_pred_test  = getTestPred().detach();
        torch::save(u_pred_test, filename);
        std::cout<<"Regression Solutions have been written to "<<filename<<"\n";
        return 0;
    }

    double getFinalError()     { return test_errors.back();  }
    double getFinalLoss()      { return train_losses.back(); }
    double getFinalTrainTime() { return train_times.back();  }
    int getFinalIter()         { return static_cast<int>(iters.back()); }

private:
    const TrainingConfig& args_;
    LoadedPDEDomain* domain_;
    ModelType* model_;

    std::vector<double> test_errors,  test_losses,  test_mses,  test_maes,  test_r2s;
    std::vector<double> train_errors, train_losses, train_mses, train_maes, train_r2s;
    std::vector<double> iters, train_times, condition_nums;
    std::vector<float> train_pde_losses, train_bnd_losses, train_ic_losses;
    std::vector<float> train_lambda_bnds, train_lambda_res, train_lambda_ics;
    std::map<std::string, torch::Tensor> test_predictions;

    void printPDEStats(std::map<std::string, torch::Tensor>& losses_dict,
                       int epoch,
                       float& total_time,
                       float lambda_res,
                       float lambda_bnd,
                       float lambda_ic,
                       double current_lr,
                       bool verbose,
                       bool print_model_trn_info) {
        const double tst_l2err = getTestRelativeError();

        torch::Tensor train_loss = losses_dict["total_loss"];

        if (verbose) {
            const std::string GREEN = "\033[32m";
            const std::string RESET = "\033[0m";

            std::cout << GREEN;
            std::cout << "epoch: " << epoch;
            std::cout << " | test_l2error: " << tst_l2err;

            for (const auto& entry : losses_dict) {
                std::cout << "  |  " + entry.first << ": " << entry.second.item<float>();
                if      (entry.first=="pde_loss") std::cout << ", lambda_res:" << lambda_res;
                else if (entry.first=="bnd_loss") std::cout << ", lambda_bnd:" << lambda_bnd;
                else if (entry.first=="ic_loss")  std::cout << ", lambda_ic: "<< lambda_ic;
            }

            std::cout << " | lr: " << current_lr;
            std::cout << " | time: " << total_time << " (ms)/" << total_time/1000
                           << "(s)/" << (total_time/1000)/60 << "(min)\n";

            if (print_model_trn_info) model_->printTrainingInfo();
            std::cout << RESET;
            std::cout << std::flush;
        }

        for (const auto& entry : losses_dict) {
            if (entry.first == "pde_loss") {
                train_pde_losses.push_back(entry.second.item<float>());
                if (args_.reglr_args.loss_reweight_scheme != "none")
                    train_lambda_res.push_back(lambda_res);
            } else if (entry.first == "bnd_loss") {
                train_bnd_losses.push_back(entry.second.item<float>());
                if (args_.reglr_args.loss_reweight_scheme != "none")
                    train_lambda_bnds.push_back(lambda_bnd);
            } else if (entry.first == "ic_loss") {
                train_ic_losses.push_back(entry.second.item<float>());
                if (args_.reglr_args.loss_reweight_scheme != "none")
                    train_lambda_ics.push_back(lambda_ic);
            }
        }

        train_losses.push_back(train_loss.item<float>());
        test_errors.push_back(tst_l2err);
        train_times.push_back(total_time);
        iters.push_back(epoch);
    }

    struct EvaluationResult {
        torch::Tensor predictions;
        double relative_error;
    };

    EvaluationResult evaluateTest2D(bool save_predictions) {
        TORCH_CHECK(args_.test_batch_size > 0, "test_batch_size must be positive");
        struct ModeGuard {
            ModelType* model;
            bool was_training;
            explicit ModeGuard(ModelType* m) : model(m), was_training(m->is_training()) { model->eval(); }
            ~ModeGuard() { model->train(was_training); }
        } mode(model_);
        torch::NoGradGuard no_grad;
        auto test = domain_->getTestData();
        TORCH_CHECK(test.data.X && test.u, "Test coordinates and targets are required");
        const int64_t n = test.data.X->size(0);
        TORCH_CHECK(n > 0 && test.u->size(0) == n, "Test coordinates and targets must have matching nonempty rows");
        auto squared_error = torch::zeros({}, test.u->options().dtype(torch::kDouble));
        auto squared_target = torch::zeros_like(squared_error);
        torch::Tensor predictions;
        for (int64_t start = 0; start < n; start += args_.test_batch_size) {
            const int64_t count = std::min<int64_t>(args_.test_batch_size, n-start);
            auto X = test.data.X->narrow(0, start, count);
            auto basis = model_->buildEvaluationBasis(X);
            DataInputs inputs;
            inputs.X = &X;
            if (basis.defined()) {
                inputs.P.basis = &basis;
            }
            auto predicted = model_->forward(inputs);
            auto target = test.u->narrow(0, start, count);
            TORCH_CHECK(predicted.sizes() == target.sizes(), "Test prediction/target shapes must match");
            squared_error += (predicted.to(torch::kDouble) - target.to(torch::kDouble)).square().sum();
            squared_target += target.to(torch::kDouble).square().sum();
            if (save_predictions) {
                if (!predictions.defined()) {
                    auto shape = predicted.sizes().vec();
                    shape[0] = n;
                    predictions = torch::empty(shape, predicted.options());
                }
                predictions.narrow(0, start, count).copy_(predicted);
            }
        }
        return {predictions, (squared_error/squared_target).sqrt().template item<double>()};
    }

public:
    double getTestRelativeError() {
        if (domain_->dim == 2) return evaluateTest2D(false).relative_error;
        auto target = domain_->getTestingSolution();
        return ((getTestPred()-target).norm()/target.norm()).template item<float>();
    }

    torch::Tensor getTestPred() {
        if (domain_->dim == 2) return evaluateTest2D(true).predictions;
        model_->eval();
        TestInputs TEST_DATA = domain_->getTestData();
        torch::Tensor upred = model_->forward(TEST_DATA.data);
        model_->train();
        return upred;
    }

private:
    template<typename T>
    static void writeVector(std::ostream& out, const std::vector<T>& data) {
        for (const auto& v : data) out << v << std::endl;
    }

    template<typename T>
    static bool saveVector(const std::vector<T>& data, const std::string& filename, const std::string& header="") {
        std::ofstream out(filename);
        if (!out.is_open()) {
            // log(Level::Error, "Could not open file for writing: ", filename);
            std::cout<<"ERROR: Could not open file for writing: "<<filename<<"\n";
            return false;
        }
        if (!header.empty()) out << header << std::endl;
        writeVector(out, data);
        return true;
    }
};
