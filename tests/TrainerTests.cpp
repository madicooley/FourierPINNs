#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <torch/torch.h>

#include "fourierpinns/config/Args.h"
#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/training/Trainer.h"
#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/models/NeuralNet.h"

namespace {
struct MockPeriodicModel final : public CustomNetwork {
    bool periodic_called = false;
    bool forward_called = false;
    torch::Tensor forward(const torch::Tensor& inputs) override {
        forward_called = true;
        return torch::zeros({inputs.size(0), 1}, inputs.options());
    }
    torch::Tensor forward(const DataInputs& inputs) override {
        TORCH_CHECK(inputs.X != nullptr, "DataInputs requires X.");
        return forward(*inputs.X);
    }
    torch::Tensor forward_pde(const PDEInputs& inputs, PDE* pde) override {
        TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X.");
        auto u = forward(*inputs.data.X);
        return pde->forwardPDE(u, *inputs.data.X);
    }
    void lstsqUpdates(torch::Tensor& /*soln*/) override {}
    torch::Tensor getLSTSQ_A(const PDEInputs& pdeData,
                             const BndInputs& bndData,
                             const ICInputs* /*icData*/,
                             PDE* /*pde*/) override {
        const int64_t nbnd = bndData.data.X->size(0);
        const int64_t ncol = pdeData.data.X->size(0);
        return torch::zeros({nbnd + ncol, 1}, bndData.data.X->options());
    }
    void printTrainingInfo() override {}
    void truncateCoeffs() override {}
    torch::Tensor getRegularizationLoss(const DataInputs& /*inputs*/) override {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kDouble));
    }
    torch::Tensor getPeriodicBndLoss(const BndInputs& /*inputs_a*/,
                                     const BndInputs& /*inputs_b*/) override {
        periodic_called = true;
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kDouble));
    }
};

struct MockRegModel final : public CustomNetwork {
    torch::Tensor forward(const torch::Tensor& inputs) override {
        return torch::zeros({inputs.size(0), 1}, inputs.options());
    }
    torch::Tensor forward(const DataInputs& inputs) override {
        TORCH_CHECK(inputs.X != nullptr, "DataInputs requires X.");
        return forward(*inputs.X);
    }
    torch::Tensor forward_pde(const PDEInputs& inputs, PDE* /*pde*/) override {
        TORCH_CHECK(inputs.data.X != nullptr, "PDEInputs requires X.");
        return torch::zeros({inputs.data.X->size(0), 1}, inputs.data.X->options());
    }
    void lstsqUpdates(torch::Tensor& /*soln*/) override {}
    torch::Tensor getLSTSQ_A(const PDEInputs&,
                             const BndInputs&,
                             const ICInputs* /*icData*/,
                             PDE* /*pde*/) override {
        return torch::zeros({1, 1});
    }
    void printTrainingInfo() override {}
    void truncateCoeffs() override {}
    torch::Tensor getRegularizationLoss(const DataInputs& /*inputs*/) override {
        return torch::ones({}, torch::TensorOptions().dtype(torch::kDouble));
    }
    torch::Tensor getPeriodicBndLoss(const BndInputs&,
                                     const BndInputs&) override {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kDouble));
    }
};

LoadedPDEDomain makeDomain(const std::string& pde, const std::string& bnd_typ, bool homogeneous, bool initial_cond) {
    DomainConfig dcfg;
    dcfg.dim = (pde == "onewaywave_inhomo" || pde == "nonss_allencahn") ? 2 : 1;
    dcfg.pde_problem = pde;
    dcfg.bnd_typ = bnd_typ;
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = (dcfg.dim == 2) ? "test7" : "test1_k1";
    dcfg.homogeneous_pde = homogeneous;
    dcfg.initial_cond = initial_cond;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = initial_cond ? 3 : 0;
    if (dcfg.dim == 1) {
        dcfg.domain_bounds = {{0.0, 1.0}};
    } else {
        dcfg.domain_bounds = {{0.0, 1.0}, {-1.0, 1.0}};
    }

    LoadedPDEDomain domain(dcfg.dim,
                           dcfg.Ncol_pts,
                           dcfg.Nbnd_pts,
                           dcfg.Ntest_pts,
                           dcfg.Nic_pts,
                           dcfg.test_func,
                           dcfg.pde_problem,
                           dcfg.bnd_typ,
                           dcfg.domain_typ,
                           dcfg.point_sampling,
                           dcfg.homogeneous_pde,
                           dcfg.initial_cond,
                           dcfg.domain_bounds,
                           torch::kCPU,
                           torch::kDouble);
    domain.generateDomainData();
    return domain;
}
} // namespace

// Verifies periodic boundary path is used when bnd_typ=periodic.
TEST_CASE("Trainer uses periodic boundary loss for periodic bnd_typ", "[Trainer]") {
    auto domain = makeDomain("onewaywave_inhomo", "periodic", /*homogeneous=*/true, /*initial_cond=*/false);

    TrainingConfig tcfg;
    tcfg.reglr_args.lambda_bnd = 1.0;
    tcfg.reglr_args.lambda_res = 1.0;
    tcfg.dim = domain.dim;
    tcfg.seed = 0;

    PDE pde("onewaywave_inhomo");
    MockPeriodicModel model;

    ModelTrainer<MockPeriodicModel> trainer(tcfg, &model, &domain, &pde);
    (void)trainer.getBndLossForTesting();

    REQUIRE(model.periodic_called);
    REQUIRE_FALSE(model.forward_called);
}

// Verifies homogeneous PDE loss uses f_pred directly.
TEST_CASE("Trainer homogeneous PDE loss uses prediction only", "[Trainer]") {
    auto domain = makeDomain("poisson", "dirichlet", /*homogeneous=*/true, /*initial_cond=*/false);

    TrainingConfig tcfg;
    tcfg.reglr_args.lambda_res = 1.0;
    tcfg.dim = domain.dim;
    tcfg.seed = 0;

    PDE pde("poisson");
    MockRegModel model;

    ModelTrainer<MockRegModel> trainer(tcfg, &model, &domain, &pde);
    auto loss = trainer.getPDELossForTesting();
    REQUIRE(loss.item<double>() == Catch::Approx(0.0));
}

// Verifies non-homogeneous PDE loss subtracts f_true.
TEST_CASE("Trainer non-homogeneous PDE loss subtracts f_true", "[Trainer]") {
    auto domain = makeDomain("poisson", "dirichlet", /*homogeneous=*/false, /*initial_cond=*/false);

    TrainingConfig tcfg;
    tcfg.reglr_args.lambda_res = 1.0;
    tcfg.dim = domain.dim;
    tcfg.seed = 0;

    PDE pde("poisson");
    MockRegModel model;

    ModelTrainer<MockRegModel> trainer(tcfg, &model, &domain, &pde);
    auto loss = trainer.getPDELossForTesting();
    REQUIRE(loss.item<double>() > 0.0);
}

// The coefficient penalty is included once regardless of data blocks.
TEST_CASE("Trainer regularization is counted once", "[Trainer]") {
    auto domain = makeDomain("poisson", "dirichlet", /*homogeneous=*/true, /*initial_cond=*/true);

    TrainingConfig tcfg;
    tcfg.reglr_args.lambda_res = 1.0;
    tcfg.dim = domain.dim;
    tcfg.seed = 0;

    PDE pde("poisson");
    MockRegModel model;

    ModelTrainer<MockRegModel> trainer(tcfg, &model, &domain, &pde);
    auto reg = trainer.getRegularizationLossForTesting();
    REQUIRE(reg.item<double>() == Catch::Approx(1.0));
}
