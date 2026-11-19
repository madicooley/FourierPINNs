#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>
#include "fourierpinns/utils/utils.h"
#include "fourierpinns/config/Args.h"
#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/training/Trainer.h"
#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/models/CustomNet.h"

namespace {
struct MockModel : public CustomNetwork {
    torch::Tensor last_soln;
    std::vector<torch::Tensor> solutions;
    bool advance_prediction_on_ls = false;
    double u_const = 0.5;
    torch::Device device = torch::kCPU;
    torch::Dtype dtype = torch::kDouble;

    torch::Tensor forward(const torch::Tensor& inputs) override {
        return torch::full({inputs.size(0), 1}, u_const, torch::TensorOptions().device(inputs.device()).dtype(inputs.dtype()));
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

    void lstsqUpdates(torch::Tensor& soln) override {
        last_soln = soln.detach().clone();
        solutions.push_back(last_soln);
        if (advance_prediction_on_ls) u_const += 0.25;
    }

    torch::Tensor getLSTSQ_A(const PDEInputs& pdeData,
                             const BndInputs& bndData,
                             const ICInputs* icData,
                             PDE* /*pde*/) override {
        TORCH_CHECK(bndData.data.X != nullptr, "BndInputs requires X.");
        TORCH_CHECK(pdeData.data.X != nullptr, "PDEInputs requires X.");
        int nbnd = static_cast<int>(bndData.data.X->size(0));
        int ncol = static_cast<int>(pdeData.data.X->size(0));
        int nbnd_rows = (bndData.data_b.X != nullptr) ? (2 * nbnd) : nbnd;
        int nic = (icData != nullptr && icData->data.X != nullptr) ? static_cast<int>(icData->data.X->size(0)) : 0;
        int n = nbnd_rows + nic + ncol;
        return torch::eye(n, torch::TensorOptions().dtype(dtype).device(device));
    }

    void printTrainingInfo() override {}
    void truncateCoeffs() override {}
    torch::Tensor getRegularizationLoss(const DataInputs& /*inputs*/) override {
        return torch::zeros({}, torch::TensorOptions().dtype(dtype).device(device));
    }
    torch::Tensor getPeriodicBndLoss(const BndInputs& /*inputs_a*/,
                                     const BndInputs& /*inputs_b*/) override {
        return torch::zeros({}, torch::TensorOptions().dtype(dtype).device(device));
    }
};

struct SafeguardModel final : public MockModel {
    using MockModel::forward;
    torch::Tensor amplitude, mask;
    double proposal = 1e100;
    int pruning_calls = 0;

    SafeguardModel() {
        amplitude = register_parameter("amplitude", torch::tensor({0.1}, torch::kDouble));
        mask = register_buffer("mask", torch::ones({1}, torch::kDouble));
        truncate_ = true;
    }
    torch::Tensor forward(const torch::Tensor& X) override { return amplitude * mask * torch::sin(X); }
    void lstsqUpdates(torch::Tensor&) override { amplitude.data().fill_(proposal); }
    void truncateCoeffs() override {
        ++pruning_calls;
        amplitude.data().zero_();
        mask.zero_();
    }
    std::vector<BasisCoefficientGroup> getBasisCoefficientGroups() override {
        return {{"test coefficient", amplitude, mask, {}}};
    }
};
} // namespace

// Verifies LS RHS for Allen-Cahn uses f - u(u^2-1) with current u_pred.
TEST_CASE("Trainer lstsqUpdates builds Allen-Cahn RHS with nonlinear term", "[Trainer][LSTSQ]") {
    torch::manual_seed(0);

    DomainConfig dcfg;
    dcfg.dim = 1;
    dcfg.pde_problem = "allencahn";
    dcfg.bnd_typ = "dirichlet";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = "test1_k1";
    dcfg.homogeneous_pde = false;
    dcfg.initial_cond = false;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = 0;
    dcfg.domain_bounds = {{0.0, 1.0}};

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

    TrainingConfig tcfg;
    tcfg.lstsq_safeguard = false; // This mock inspects the raw algebra, not a differentiable model.
    tcfg.lstsq_updates = true;
    tcfg.lstsq_upd_interval = 1;
    tcfg.n_lstsq_upds = 1;
    tcfg.n_lstsq_upds_per_epoch = 1;
    tcfg.reglr_args.lambda_bnd = 1.0;
    tcfg.reglr_args.lambda_res = 1.0;

    PDE pde("allencahn");
    MockModel model;

    ModelTrainer<MockModel> trainer(tcfg, &model, &domain, &pde);
    trainer.runLstsqUpdatesForTesting();

    auto bnd = domain.getBndData();
    auto pde_data = domain.getPDEData();

    auto u_bnd = *bnd.u;
    auto ftrue = *pde_data.f;

    auto u_pred = model.forward(pde_data.data);
    auto s = u_pred * (u_pred * u_pred - 1.0);
    auto f_rhs = ftrue - s;

    auto expected_b = torch::cat({u_bnd, f_rhs}, 0).squeeze(1);

    REQUIRE(model.last_soln.defined());
    REQUIRE(model.last_soln.sizes() == expected_b.sizes());
    REQUIRE(torch::allclose(model.last_soln, expected_b, 1e-8, 1e-8));
}

TEST_CASE("Allen-Cahn refreshes nonlinear RHS after every LS coefficient update", "[Trainer][LSTSQ][AllenCahn]") {
    LoadedPDEDomain domain(1, 9, 4, 17, 0, "test1_k1", "allencahn", "dirichlet",
        "box", "equispaced", false, false, {{0.0, 1.0}}, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    TrainingConfig cfg;
    cfg.lstsq_safeguard = false;
    cfg.lstsq_updates = true;
    cfg.n_lstsq_upds = 3;
    cfg.n_lstsq_upds_per_epoch = 3;
    MockModel model;
    model.advance_prediction_on_ls = true;
    PDE pde("allencahn");
    ModelTrainer<MockModel> trainer(cfg, &model, &domain, &pde);
    trainer.runLstsqUpdatesForTesting();
    REQUIRE(model.solutions.size() == 3);
    auto bnd = domain.getBndData();
    auto data = domain.getPDEData();
    for (int pass = 0; pass < 3; ++pass) {
        const double u = 0.5 + pass * 0.25;
        auto expected = torch::cat({*bnd.u, *data.f - u*(u*u-1)}, 0).squeeze(1);
        REQUIRE(torch::allclose(model.solutions[pass], expected, 1e-10, 1e-10));
    }
    REQUIRE_FALSE(torch::allclose(model.solutions[0], model.solutions[1]));
    REQUIRE_FALSE(torch::allclose(model.solutions[1], model.solutions[2]));
}

TEST_CASE("Allen-Cahn safeguard restores rejected updates and rolls back harmful pruning", "[Trainer][LSTSQ][AllenCahn]") {
    LoadedPDEDomain domain(1, 17, 4, 33, 0, "test1_k1", "allencahn", "dirichlet",
        "box", "equispaced", false, false, {{0.0, 1.0}}, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    TrainingConfig cfg;
    cfg.lstsq_updates = true;
    cfg.n_lstsq_upds = 1;
    cfg.n_lstsq_upds_per_epoch = 1;
    PDE pde("allencahn");
    SafeguardModel model;
    ModelTrainer<SafeguardModel> trainer(cfg, &model, &domain, &pde);
    const auto before = (trainer.getPDELossForTesting() + trainer.getBndLossForTesting()).item<double>();

    SECTION("Non-finite losses from a huge proposal cannot change the model or prune") {
        auto old = model.amplitude.detach().clone();
        trainer.runLstsqUpdatesForTesting();
        REQUIRE(torch::equal(model.amplitude, old));
        REQUIRE(model.mask.item<double>() == 1.0);
        REQUIRE(model.pruning_calls == 0);
    }
    SECTION("Non-finite proposed parameters are restored") {
        model.proposal = std::numeric_limits<double>::quiet_NaN();
        auto old = model.amplitude.detach().clone();
        trainer.runLstsqUpdatesForTesting();
        REQUIRE(torch::equal(model.amplitude, old));
        REQUIRE(model.pruning_calls == 0);
    }
    SECTION("Accepted updates cannot be undone by destructive pruning") {
        model.proposal = 1.0;
        trainer.runLstsqUpdatesForTesting();
        const auto after = (trainer.getPDELossForTesting() + trainer.getBndLossForTesting()).item<double>();
        REQUIRE(std::isfinite(after));
        REQUIRE(after < before);
        REQUIRE(model.pruning_calls == 1);
        REQUIRE(model.mask.item<double>() == 1.0);
        REQUIRE(model.amplitude.abs().item<double>() > 0.0);
    }
    SECTION("A nonlinear overshoot is damped before acceptance") {
        model.proposal = 10.0;
        trainer.runLstsqUpdatesForTesting();
        const auto after = (trainer.getPDELossForTesting() + trainer.getBndLossForTesting()).item<double>();
        REQUIRE(after < before);
        REQUIRE(model.amplitude.item<double>() < 10.0);
        REQUIRE(model.amplitude.item<double>() > 0.1);
    }
}

// Verifies Poisson LSTSQ uses getLSTSQ_b() (boundary + ftrue).
TEST_CASE("Trainer lstsqUpdates uses getLSTSQ_b for Poisson", "[Trainer][LSTSQ]") {
    torch::manual_seed(0);

    DomainConfig dcfg;
    dcfg.dim = 1;
    dcfg.pde_problem = "poisson";
    dcfg.bnd_typ = "dirichlet";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = "test1_k1";
    dcfg.homogeneous_pde = false;
    dcfg.initial_cond = false;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = 0;
    dcfg.domain_bounds = {{0.0, 1.0}};

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

    TrainingConfig tcfg;
    tcfg.lstsq_updates = true;
    tcfg.lstsq_upd_interval = 1;
    tcfg.n_lstsq_upds = 1;
    tcfg.n_lstsq_upds_per_epoch = 1;
    tcfg.reglr_args.lambda_bnd = 1.0;
    tcfg.reglr_args.lambda_res = 1.0;

    PDE pde("poisson");
    MockModel model;

    ModelTrainer<MockModel> trainer(tcfg, &model, &domain, &pde);
    trainer.runLstsqUpdatesForTesting();

    auto bnd = domain.getBndData();
    auto pde_data = domain.getPDEData();
    auto u_bnd = *bnd.u;
    auto ftrue = *pde_data.f;

    auto expected_b = torch::cat({u_bnd, ftrue}, 0).squeeze(1);

    REQUIRE(model.last_soln.defined());
    REQUIRE(model.last_soln.sizes() == expected_b.sizes());
    REQUIRE(torch::allclose(model.last_soln, expected_b, 1e-8, 1e-8));
}

// Verifies periodic LSTSQ uses zero boundary constraints and ftrue for Poisson.
TEST_CASE("Trainer lstsqUpdates builds periodic RHS for Poisson", "[Trainer][LSTSQ]") {
    torch::manual_seed(0);

    DomainConfig dcfg;
    dcfg.dim = 1;
    dcfg.pde_problem = "poisson";
    dcfg.bnd_typ = "periodic";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = "test1_k1";
    dcfg.homogeneous_pde = false;
    dcfg.initial_cond = false;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = 0;
    dcfg.domain_bounds = {{0.0, 1.0}};

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

    TrainingConfig tcfg;
    tcfg.lstsq_updates = true;
    tcfg.lstsq_upd_interval = 1;
    tcfg.n_lstsq_upds = 1;
    tcfg.n_lstsq_upds_per_epoch = 1;
    tcfg.reglr_args.lambda_bnd = 1.0;
    tcfg.reglr_args.lambda_res = 1.0;

    PDE pde("poisson");
    MockModel model;

    ModelTrainer<MockModel> trainer(tcfg, &model, &domain, &pde);
    trainer.runLstsqUpdatesForTesting();

    auto bnd = domain.getPeriodicBndDataPair();
    auto pde_data = domain.getPDEData();
    auto ic_data = domain.getICData();
    auto ftrue = *pde_data.f;
    auto uic = *ic_data.u;

    int nbnd = static_cast<int>(bnd.data.X->size(0));
    auto zeros_bnd = torch::zeros({2 * nbnd, 1}, ftrue.options());
    auto expected_b = torch::cat({zeros_bnd, uic, ftrue}, 0).squeeze(1);

    REQUIRE(model.last_soln.defined());
    REQUIRE(model.last_soln.sizes() == expected_b.sizes());
    REQUIRE(torch::allclose(model.last_soln, expected_b, 1e-8, 1e-8));
}

// Verifies periodic LSTSQ uses zero boundary constraints and ftrue for one-way wave.
TEST_CASE("Trainer lstsqUpdates builds periodic RHS for onewaywave_inhomo", "[Trainer][LSTSQ]") {
    torch::manual_seed(0);

    DomainConfig dcfg;
    dcfg.dim = 2;
    dcfg.pde_problem = "onewaywave_inhomo";
    dcfg.bnd_typ = "periodic";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = "test7";
    dcfg.homogeneous_pde = false;
    dcfg.initial_cond = true;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = 4;
    dcfg.domain_bounds = {{0.0, 1.0}, {-1.0, 1.0}};

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

    TrainingConfig tcfg;
    tcfg.lstsq_updates = true;
    tcfg.lstsq_upd_interval = 1;
    tcfg.n_lstsq_upds = 1;
    tcfg.n_lstsq_upds_per_epoch = 1;
    tcfg.reglr_args.lambda_bnd = 1.0;
    tcfg.reglr_args.lambda_res = 1.0;

    PDE pde("onewaywave_inhomo");
    MockModel model;

    ModelTrainer<MockModel> trainer(tcfg, &model, &domain, &pde);
    trainer.runLstsqUpdatesForTesting();

    auto bnd = domain.getPeriodicBndDataPair();
    auto pde_data = domain.getPDEData();
    auto ic_data = domain.getICData();
    auto ftrue = *pde_data.f;
    auto uic = *ic_data.u;

    int nbnd = static_cast<int>(bnd.data.X->size(0));
    auto zeros_bnd = torch::zeros({2 * nbnd, 1}, ftrue.options());
    auto expected_b = torch::cat({zeros_bnd, uic, ftrue}, 0).squeeze(1);

    REQUIRE(model.last_soln.defined());
    REQUIRE(model.last_soln.sizes() == expected_b.sizes());
    REQUIRE(torch::allclose(model.last_soln, expected_b, 1e-8, 1e-8));
}

// nonss_allencahn manufactured data generation is currently unsupported in generateDomainData.
TEST_CASE("nonss_allencahn manufactured data path is currently unsupported", "[Trainer][LSTSQ]") {
    torch::manual_seed(0);

    DomainConfig dcfg;
    dcfg.dim = 2;
    dcfg.pde_problem = "nonss_allencahn";
    dcfg.bnd_typ = "periodic";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.test_func = "test7";
    dcfg.homogeneous_pde = false;
    dcfg.initial_cond = true;
    dcfg.Ncol_pts = 5;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 5;
    dcfg.Nic_pts = 4;
    dcfg.domain_bounds = {{0.0, 1.0}, {-1.0, 1.0}};

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
    REQUIRE_THROWS(domain.generateDomainData());
}
