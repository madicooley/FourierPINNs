#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>

#include "fourierpinns/config/Args.h"
#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/training/utils_trainer.h"
#include "fourierpinns/training/LossLambdaUpdater.h"

TEST_CASE("LS does not amplify float32 near-dependence into huge coefficients", "[Trainer][Algorithm1]") {
    auto x = torch::linspace(0.0, 2.0 * M_PI, 128, torch::kFloat32);
    auto a = torch::sin(x);
    auto A = torch::stack({a, 10000.0 * torch::sin(2.0 * x),
                           a + 1e-7 * torch::cos(3.0 * x)}, 1);
    auto b = (a + 10000.0 * torch::sin(2.0 * x) + 0.001 * torch::cos(3.0 * x)).unsqueeze(1);
    auto result = solveLeastSquaresOnCpu(A, b, torch::kCPU, torch::kFloat32);
    REQUIRE(result.second == 2);
    REQUIRE(result.first.abs().max().item<float>() < 10.0f);
    REQUIRE((A.matmul(result.first).unsqueeze(1) - b).abs().max().item<float>() < 0.01f);
}

TEST_CASE("LS always uses unscaled columns", "[Trainer][Algorithm1]") {
    auto A = torch::diag(torch::tensor({1e-5f, 10000.0f}));
    auto b = torch::tensor({2e-5f, 30000.0f}).unsqueeze(1);
    auto unscaled = solveLeastSquaresOnCpu(A, b, torch::kCPU, torch::kFloat32);
    REQUIRE(unscaled.second == 1);
    REQUIRE(unscaled.first[0].item<float>() == Catch::Approx(0.0f));
    REQUIRE(unscaled.first[1].item<float>() == Catch::Approx(3.0f));
}

TEST_CASE("Unscaled LS retains directions above the 1e-5 relative cutoff", "[Trainer][Algorithm1]") {
    auto A = torch::zeros({10400, 2}, torch::kFloat32);
    A.index_put_({0, 0}, 1.0f);
    A.index_put_({1, 1}, 10000.0f);
    auto b = A.matmul(torch::tensor({2.0f, 3.0f})).unsqueeze(1);
    auto automatic = solveLeastSquaresOnCpu(A, b, torch::kCPU, torch::kFloat32);
    REQUIRE(automatic.second == 1);
    REQUIRE(leastSquaresRankCutoff(A, -1.0) == Catch::Approx(0.001239776611328125));
    auto result = solveLeastSquaresOnCpu(A, b, torch::kCPU, torch::kFloat32, 1e-5);
    REQUIRE(result.second == 2);
    REQUIRE(result.first[0].item<float>() == Catch::Approx(2.0f));
    REQUIRE(result.first[1].item<float>() == Catch::Approx(3.0f));
}

TEST_CASE("LS batches respect the global budget", "[Trainer][Algorithm1]") {
    TrainingConfig cfg;
    cfg.n_lstsq_upds = 7;
    cfg.n_lstsq_upds_per_epoch = 5;
    int count = 0;
    int calls = 0;
    runLstsqUpdateLoop(count, cfg, [&] { ++calls; });
    REQUIRE(calls == 5);
    runLstsqUpdateLoop(count, cfg, [&] { ++calls; });
    REQUIRE(calls == 7);
    runLstsqUpdateLoop(count, cfg, [&] { ++calls; });
    REQUIRE(calls == 7);
}

TEST_CASE("LS backtracking accepts only finite non-increasing nonlinear merit", "[Trainer][LSTSQ][AllenCahn]") {
    double value = 0.0;
    SECTION("Full step accepted") {
        auto result = backtrackLstsqUpdate(1.0, [&](double step) { value = step; },
                                           [&]() { return (value-1)*(value-1); });
        REQUIRE(result.accepted);
        REQUIRE(result.step == 1.0);
        REQUIRE(value == 1.0);
    }
    SECTION("Overshoot is damped") {
        auto result = backtrackLstsqUpdate(1.0, [&](double step) { value = 8*step; },
                                           [&]() { return std::pow(value-1, 4); });
        REQUIRE(result.accepted);
        REQUIRE(result.step < 1.0);
        REQUIRE(result.loss <= 1.0);
    }
    SECTION("Wrong direction rejected and original state restored") {
        auto result = backtrackLstsqUpdate(1.0, [&](double step) { value = -step; },
                                           [&]() { return (value-1)*(value-1); });
        REQUIRE_FALSE(result.accepted);
        REQUIRE(value == 0.0);
    }
    SECTION("Non-finite losses rejected") {
        auto result = backtrackLstsqUpdate(1.0, [&](double step) { value = step; },
                                           []() { return std::numeric_limits<double>::infinity(); });
        REQUIRE_FALSE(result.accepted);
        REQUIRE(value == 0.0);
    }
}

TEST_CASE("Adam LS cutoff is inclusive and does not affect the Adam budget", "[Trainer][Algorithm1]") {
    TrainingConfig cfg;
    cfg.lstsq_updates = true;
    cfg.lstsq_upd_interval = 1000;
    cfg.lstsq_stop_epoch = 20000;
    cfg.adam_args.adam_max_epochs = 40000;
    int blocks = 0;
    for (int step = 0; step <= cfg.adam_args.adam_max_epochs; ++step) {
        if (shouldRunLstsqAfterAdam(cfg, step)) ++blocks;
    }
    REQUIRE(blocks == 20);
    REQUIRE(shouldRunLstsqAfterAdam(cfg, 20000));
    REQUIRE_FALSE(shouldRunLstsqAfterAdam(cfg, 21000));
    cfg.lstsq_stop_epoch = 19999;
    REQUIRE_FALSE(shouldRunLstsqAfterAdam(cfg, 20000));
    cfg.lstsq_stop_epoch = 0;
    REQUIRE_FALSE(shouldRunLstsqAfterAdam(cfg, 1000));
    cfg.lstsq_stop_epoch = -1;
    REQUIRE(shouldRunLstsqAfterAdam(cfg, 40000));
    cfg.lstsq_updates = false;
    REQUIRE_FALSE(shouldRunLstsqAfterAdam(cfg, 1000));
}

TEST_CASE("Adam alternates after exact completed step blocks", "[Trainer][Algorithm1]") {
    TrainingConfig cfg;
    cfg.adam_args.adam_max_epochs = 5;
    cfg.print_interval = 5;
    auto weight = torch::tensor(1.0, torch::TensorOptions().requires_grad(true));
    torch::optim::SGD optimizer({weight}, torch::optim::SGDOptions(0.01));
    int completed = 0;
    float time = 0;
    std::vector<int> lr_epochs, ls_steps, prune_steps;
    runAdamTrainingStage(cfg, optimizer, completed, time,
        [&](int epoch, int) { lr_epochs.push_back(epoch); },
        [&](int step) { if (step % 2 == 0) ls_steps.push_back(completed); },
        [&](int step) { if (step % 2 == 0) prune_steps.push_back(completed); },
        [] {}, [] {},
        [&](auto&) { return std::map<std::string, torch::Tensor>{{"total_loss", weight.square()}}; },
        [](int, auto&, auto&) {},
        [](int, auto&, float) {},
        [](const auto&) { return false; });
    REQUIRE(completed == 5);
    REQUIRE(lr_epochs == std::vector<int>{0, 1, 2, 3, 4});
    REQUIRE(ls_steps == std::vector<int>{2, 4});
    REQUIRE(prune_steps == ls_steps);
    REQUIRE(weight.item<double>() < 1.0);
}

TEST_CASE("LBFGS line search keeps scheduling outside closure evaluations", "[Trainer][Algorithm1]") {
    struct RepeatedClosureOptimizer {
        torch::optim::SGD optimizer;
        explicit RepeatedClosureOptimizer(torch::Tensor p)
            : optimizer({p}, torch::optim::SGDOptions(0.01)) {}
        void zero_grad() { optimizer.zero_grad(); }
        void step(const std::function<torch::Tensor()>& closure) {
            closure();
            closure();
        }
    };
    TrainingConfig cfg;
    cfg.lbfgs_args.lbfgs_max_epochs = 3;
    auto p = torch::tensor(1.0, torch::TensorOptions().requires_grad(true));
    RepeatedClosureOptimizer optimizer(p);
    int completed = 0, schedules = 0, reweights = 0, truncations = 0, evaluations = 0;
    float time = 0;
    runLBFGSTrainingStage(cfg, optimizer, completed, time,
        [&](auto&) { return std::map<std::string, torch::Tensor>{{"total_loss", p.square()}}; },
        [&](int, int, auto&) { ++schedules; },
        [&](int, auto&, auto&) { ++reweights; },
        [&](int) { ++truncations; },
        [&](int, auto&, float) {
            ++evaluations;
            REQUIRE(schedules == completed + 1);
            REQUIRE(reweights == schedules);
            REQUIRE(truncations == schedules);
        },
        [](int, int, auto&, float) {}, [] {},
        [](const auto&) { return false; });
    REQUIRE(completed == 3);
    REQUIRE(evaluations == 6);
    REQUIRE(schedules == 3);
}

TEST_CASE("TrainingLearningRateScheduler updates warmup and decay learning rates", "[Trainer][Utils][Scheduler]") {
    TrainingConfig cfg;
    cfg.lr_decaytype = "cosine";
    cfg.lr_warmup_steps = 2;
    cfg.lr_decay_interval = 2;
    cfg.lr_decay_steps = 8;

    auto param = torch::randn({2, 1}, torch::TensorOptions().requires_grad(true));
    torch::optim::SGD optimizer({param}, torch::optim::SGDOptions(0.5));

    double current_lr = 0.5;
    const double min_lr = 0.1;
    const double max_lr = 0.5;
    const int max_epochs = 10;

    current_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
        optimizer, cfg, min_lr, max_lr, /*epoch=*/1, max_epochs, current_lr);
    REQUIRE(current_lr == Catch::Approx(0.3));
    auto lr_warmup = static_cast<torch::optim::SGDOptions&>(optimizer.param_groups()[0].options()).lr();
    REQUIRE(lr_warmup == Catch::Approx(0.3));

    current_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
        optimizer, cfg, min_lr, max_lr, /*epoch=*/4, max_epochs, current_lr);
    const double expected_cosine = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + std::cos(M_PI * 4.0 / max_epochs));
    REQUIRE(current_lr == Catch::Approx(expected_cosine));
    auto lr_decay = static_cast<torch::optim::SGDOptions&>(optimizer.param_groups()[0].options()).lr();
    REQUIRE(lr_decay == Catch::Approx(expected_cosine));
}

TEST_CASE("TrainingLearningRateScheduler keeps LR constant outside update windows", "[Trainer][Utils][Scheduler]") {
    TrainingConfig cfg;
    cfg.lr_decaytype = "exp";
    cfg.lr_warmup_steps = 1;
    cfg.lr_decay_interval = 2;
    cfg.lr_decay_steps = 6;
    cfg.lr_decay_rate = 0.5f;

    auto param = torch::randn({2, 1}, torch::TensorOptions().requires_grad(true));
    torch::optim::SGD optimizer({param}, torch::optim::SGDOptions(0.4));

    double current_lr = 0.4;
    const double min_lr = 0.1;
    const double max_lr = 0.4;
    const int max_epochs = 10;

    current_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
        optimizer, cfg, min_lr, max_lr, /*epoch=*/2, max_epochs, current_lr);
    const double expected_updated = max_lr * std::pow(cfg.lr_decay_rate, (2 - cfg.lr_warmup_steps) / cfg.lr_decay_interval);
    REQUIRE(current_lr == Catch::Approx(expected_updated));

    current_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
        optimizer, cfg, min_lr, max_lr, /*epoch=*/3, max_epochs, current_lr);
    REQUIRE(current_lr == Catch::Approx(expected_updated));
}

TEST_CASE("TrainingLearningRateScheduler handles zero warmup and decay interval safely", "[Trainer][Utils][Scheduler]") {
    TrainingConfig cfg;
    cfg.lr_decaytype = "exp";
    cfg.lr_warmup_steps = 0;
    cfg.lr_decay_interval = 0;
    cfg.lr_decay_steps = 10;
    cfg.lr_decay_rate = 0.5f;

    auto param = torch::randn({2, 1}, torch::TensorOptions().requires_grad(true));
    torch::optim::SGD optimizer({param}, torch::optim::SGDOptions(0.2));

    const double min_lr = 0.05;
    const double max_lr = 0.2;
    const int max_epochs = 10;
    const double initial_lr = 0.2;

    const double updated_lr = TrainingLearningRateScheduler::updateOptimizerLearningRate(
        optimizer, cfg, min_lr, max_lr, /*epoch=*/4, max_epochs, initial_lr);

    REQUIRE(updated_lr == Catch::Approx(initial_lr));
    auto optimizer_lr = static_cast<torch::optim::SGDOptions&>(optimizer.param_groups()[0].options()).lr();
    REQUIRE(optimizer_lr == Catch::Approx(initial_lr));
}

TEST_CASE("accumulateTotalLoss preserves historical aggregation formula", "[Trainer][Utils][Loss]") {
    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor pde = torch::tensor(1.0, opts);
    torch::Tensor bnd = torch::tensor(2.0, opts);
    torch::Tensor reg = torch::tensor(3.0, opts);

    std::map<std::string, torch::Tensor> losses;
    losses["pde_loss"] = pde;
    losses["bnd_loss"] = bnd;
    losses["reg"] = reg;

    torch::Tensor total = accumulateTotalLoss(losses, pde);
    REQUIRE(total.item<double>() == Catch::Approx(7.0));
}

TEST_CASE("LossLambdaUpdater grad scheme remains finite for near-zero gradients", "[Trainer][Utils][Loss]") {
    TrainingConfig cfg;
    cfg.reglr_args.loss_reweight_scheme = "grad";

    auto param = torch::randn({2, 1}, torch::TensorOptions().requires_grad(true));
    torch::optim::SGD optimizer({param}, torch::optim::SGDOptions(0.1));

    auto zero_loss = (param * 0.0).sum();
    std::map<std::string, torch::Tensor> losses;
    losses["pde_loss"] = zero_loss;
    losses["bnd_loss"] = zero_loss;
    losses["ic_loss"] = zero_loss;

    float lambda_bnd = 1.0f;
    float lambda_ic = 1.0f;

    LossLambdaUpdater updater(cfg);
    REQUIRE_NOTHROW(updater.updateLossLambdas(losses, optimizer, lambda_bnd, lambda_ic));

    REQUIRE(std::isfinite(lambda_bnd));
    REQUIRE(std::isfinite(lambda_ic));
    REQUIRE(lambda_bnd == Catch::Approx(0.9f));
    REQUIRE(lambda_ic == Catch::Approx(0.9f));
}
