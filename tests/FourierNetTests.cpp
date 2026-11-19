#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <torch/torch.h>
#include <map>

#include "fourierpinns/models/NeuralNet.h"
#include "fourierpinns/models/FourierNet.h"
#include "fourierpinns/config/Args.h"
#include "fourierpinns/config/ArgParser.h"
#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/basis/FourierBasisUtils.h"
#include "fourierpinns/training/Trainer.h"

TEST_CASE("Algorithm 1 LS recovers a Fourier solution without changing hidden weights", "[FourierNet][Algorithm1]") {
    torch::manual_seed(1234);
    ModelConfig cfg;
    cfg.model_type = "fouriernn";
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0.0, 2.0 * M_PI}};
    FourierNetArgs arch;
    arch.neuralnet_args.layers = {1, 4, 1};
    arch.neuralnet_args.width = 4;
    arch.neuralnet_args.nn_truncate = true;
    arch.neuralnet_args.nn_trunc_thresh = 1e-4;
    arch.fourierlayer_args.ell = 4;
    arch.fourierlayer_args.scale_to_period = true;
    arch.fourierlayer_args.include_zero_mode = false;
    arch.fourierlayer_args.pn_truncate = true;
    arch.fourierlayer_args.pn_trunc_thresh = 1e-4;
    arch.fourierlayer_args.pn_l2_reg_coeff = 1e-4;
    cfg.arch = arch;
    VanillaFourierNeuralNet model(cfg, "poisson", torch::kCPU, torch::kDouble);
    LoadedPDEDomain domain(1, 64, 8, 128, 0, "test1_k2", "poisson", "dirichlet",
        "box", "equispaced", false, false, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto fl_cfg = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(cfg, arch.fourierlayer_args);
    VanillaFourierLayer layer(fl_cfg, "poisson", torch::kCPU, torch::kDouble);
    domain.setPolyBasis(buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble), false);
    auto pde_data = domain.getPDEData();
    auto bnd_data = domain.getBndData();
    auto test_data = domain.getTestData();
    DataInputs raw = test_data.data;
    raw.P = {};
    REQUIRE(torch::allclose(model->forward(raw), model->forward(test_data.data), 1e-10, 1e-10));
    std::map<std::string, torch::Tensor> hidden;
    for (const auto& named : model->named_parameters()) {
        if (named.key().find("linear_") != std::string::npos) hidden[named.key()] = named.value().detach().clone();
    }
    TrainingConfig training;
    training.n_lstsq_upds = 5;
    training.n_lstsq_upds_per_epoch = 5;
    training.lstsq_updates = true;
    PDE pde("poisson");
    ModelTrainer<VanillaFourierNeuralNetImpl> trainer(training, model.get(), &domain, &pde);
    REQUIRE(torch::isfinite(trainer.getRegularizationLossForTesting()).item<bool>());
    trainer.runLstsqUpdatesForTesting();
    auto predicted = model->forward(test_data.data);
    REQUIRE(torch::allclose(predicted, *test_data.u, 1e-6, 1e-6));
    for (const auto& named : model->named_parameters()) {
        if (hidden.count(named.key())) REQUIRE(torch::equal(hidden.at(named.key()), named.value()));
    }
    REQUIRE(model->getPolyBasisSelection().sum().item<int>() < 8);
    auto mask = model->getPolyBasisSelection().clone();
    model->truncateCoeffs();
    REQUIRE(torch::equal(mask, model->getPolyBasisSelection()));
    auto A = model->getLSTSQ_A(pde_data, bnd_data, nullptr, &pde);
    auto coeffs = torch::cat({model->getPolyBasisCoeffs().flatten(),
                             model->getNNetBias().flatten(), model->getNNetBasisCoeffs().flatten()});
    auto selection = torch::cat({model->getPolyBasisSelection().flatten(),
                                 model->neuralnet->getBiasSelection().flatten(),
                                 model->neuralnet->getBasisSelection().flatten()}).to(torch::kBool);
    auto Aw = A.matmul(coeffs.index({selection}).unsqueeze(1));
    auto actual = torch::cat({model->forward(bnd_data.data), model->forward_pde(pde_data, &pde)}, 0);
    REQUIRE(torch::allclose(Aw, actual, 1e-7, 1e-7));
}

TEST_CASE("Steady Allen-Cahn hybrid converges from zero output through Adam LS and LBFGS", "[FourierNet][AllenCahn][Algorithm1]") {
    torch::manual_seed(1234);
    ModelConfig cfg;
    cfg.model_type = "fouriernn";
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0.0, 2*M_PI}};
    FourierNetArgs arch;
    arch.neuralnet_args.layers = {1, 4, 1};
    arch.neuralnet_args.width = 4;
    arch.fourierlayer_args.ell = 12;
    arch.fourierlayer_args.scale_to_period = true;
    arch.fourierlayer_args.include_zero_mode = false;
    cfg.arch = arch;
    VanillaFourierNeuralNet model(cfg, "allencahn", torch::kCPU, torch::kDouble);
    // A controlled basin-of-attraction check, not a random-start convergence claim.
    {
        torch::NoGradGuard guard;
        for (auto& parameter : model->parameters()) parameter.zero_();
    }
    LoadedPDEDomain domain(1, 96, 8, 257, 0, "test1_k4", "allencahn", "dirichlet",
        "box", "equispaced", false, false, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto fl_cfg = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(cfg, arch.fourierlayer_args);
    VanillaFourierLayer layer(fl_cfg, "allencahn", torch::kCPU, torch::kDouble);
    domain.setPolyBasis(buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble), false);
    PDE pde("allencahn");
    TrainingConfig training;
    training.training_order = "adamfirst";
    training.adam_args.adam_max_epochs = 8;
    training.adam_args.adam_max_lr = 1e-4;
    training.lstsq_updates = true;
    training.lstsq_upd_interval = 1;
    training.lstsq_stop_epoch = 6;
    training.n_lstsq_upds = 30;
    training.n_lstsq_upds_per_epoch = 5;
    training.lstsq_rcond = 1e-10;
    training.lbfgs_args.lbfgs_max_epochs = 2;
    training.lbfgs_args.lbfgs_max_iter = 5;
    training.print_interval = 8;
    ModelTrainer<VanillaFourierNeuralNetImpl> trainer(training, model.get(), &domain, &pde);
    auto test = domain.getTestData();
    const auto initial = trainer.getPDELossForTesting().item<double>();
    trainer.trainModel();
    auto predicted = model->forward(test.data);
    auto error = (predicted - *test.u).norm() / test.u->norm();
    REQUIRE(torch::isfinite(predicted).all().item<bool>());
    REQUIRE(error.item<double>() < 1e-4);
    const auto final_loss = trainer.getPDELossForTesting().item<double>();
    REQUIRE(std::isfinite(final_loss));
    REQUIRE(final_loss < 1e-5);
    REQUIRE(final_loss < initial * 1e-5);
    REQUIRE(trainer.getBndLossForTesting().item<double>() < 1e-8);
}

namespace {
ModelConfig make_fouriernet_config(int dim, int width, int depth, int ell) {
    ModelConfig cfg;
    cfg.model_type = "fouriernn";
    cfg.input_dim = dim;
    cfg.output_dim = 1;
    cfg.verbose = false;
    cfg.domain_bounds.clear();
    for (int d = 0; d < dim; ++d) {
        cfg.domain_bounds.push_back({0.0, 1.0});
    }

    NeuralNetArgs nn;
    nn.width = width;
    nn.depth = depth;
    nn.net_act = "tanh";
    nn.nn_truncate = false;
    nn.nn_trunc_thresh = 1e-6f;
    nn.nn_l1_reg_coeff = 0.0f;
    nn.nn_l2_reg_coeff = 0.0f;
    nn.layers = getLayerList(cfg.input_dim, cfg.output_dim, nn.width, nn.depth);

    FourierLayerArgs fl;
    fl.ell = ell;
    fl.include_zero_mode = true;
    fl.scale_to_period = false;
    fl.periodic_deriv_dim = -1;
    fl.pn_truncate = false;
    fl.pn_trunc_thresh = 1e-6f;
    fl.pn_l1_reg_coeff = 0.0f;
    fl.pn_l2_reg_coeff = 0.0f;

    FourierNetArgs fargs;
    fargs.neuralnet_args = nn;
    fargs.fourierlayer_args = fl;
    cfg.arch = fargs;
    return cfg;
}
} // namespace

// Checks that FourierNet forward(DataInputs) returns output with correct shape and combines NN + Fourier parts.
TEST_CASE("FourierNet forward(DataInputs) produces expected shape", "[FourierNet]") {
    auto cfg = make_fouriernet_config(/*dim=*/1, /*width=*/4, /*depth=*/1, /*ell=*/1);
    auto net = VanillaFourierNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X = torch::linspace(0.0, 1.0, 5, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);
    DataInputs inputs;
    inputs.X = &X;

    auto out = net->forward(inputs);
    REQUIRE(out.sizes()[0] == 5);
    REQUIRE(out.sizes()[1] == 1);
}

// Verifies LS design matrix shape matches expected rows/cols after selection masks.
TEST_CASE("FourierNet getLSTSQ_A returns expected shape", "[FourierNet]") {
    auto cfg = make_fouriernet_config(/*dim=*/1, /*width=*/3, /*depth=*/1, /*ell=*/1);
    auto net = VanillaFourierNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto Xbnd = torch::tensor({{0.0}, {1.0}}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(true));
    auto Xcol = torch::linspace(0.0, 1.0, 3, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1).requires_grad_(true);

    BndInputs bnd;
    PDEInputs col;
    bnd.data.X = &Xbnd;
    col.data.X = &Xcol;

    PDE pde("poisson");
    auto A = net->getLSTSQ_A(col, bnd, nullptr, &pde);

    const int64_t rows = Xbnd.size(0) + Xcol.size(0);
    const int64_t cols = net->fouriernet->getNRemainingBases() + net->neuralnet->getNRemainingBases();
    REQUIRE(A.sizes()[0] == rows);
    REQUIRE(A.sizes()[1] == cols);
    REQUIRE(torch::isfinite(A).all().item<bool>());
}

// Verifies periodic LSTSQ A includes value/derivative rows across NN + Fourier blocks.
TEST_CASE("FourierNet periodic getLSTSQ_A has value/derivative rows", "[FourierNet]") {
    auto cfg = make_fouriernet_config(/*dim=*/2, /*width=*/3, /*depth=*/1, /*ell=*/1);
    auto net = VanillaFourierNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X_a = torch::tensor({{0.0, -1.0}, {0.5, -1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_b = torch::tensor({{0.0,  1.0}, {0.5,  1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_col = torch::tensor({{0.25, 0.0}, {0.75, 0.5}, {0.9, -0.2}}, torch::TensorOptions().dtype(torch::kDouble));
    X_a.requires_grad_(true);
    X_b.requires_grad_(true);
    X_col.requires_grad_(true);

    BndInputs bnd;
    bnd.data.X = &X_a;
    bnd.data_b.X = &X_b;
    auto P_a_x = net->fouriernet->buildBasisDxFromData(X_a, /*deriv_dim=*/1);
    auto P_b_x = net->fouriernet->buildBasisDxFromData(X_b, /*deriv_dim=*/1);
    bnd.P_x = &P_a_x;
    bnd.P_x_b = &P_b_x;

    PDEInputs col;
    col.data.X = &X_col;

    PDE pde("poisson");
    auto A = net->getLSTSQ_A(col, bnd, nullptr, &pde);

    const int64_t nbnd = X_a.size(0);
    const int64_t ncol = X_col.size(0);
    const int64_t expected_rows = 2 * nbnd + ncol;
    const int64_t expected_cols = net->fouriernet->getNRemainingBases() + net->neuralnet->getNRemainingBases();

    REQUIRE(A.sizes()[0] == expected_rows);
    REQUIRE(A.sizes()[1] == expected_cols);
    REQUIRE(torch::isfinite(A).all().item<bool>());
}

// Ensures lstsqUpdates splits the solution vector without errors.
TEST_CASE("FourierNet lstsqUpdates accepts joint solution vector", "[FourierNet]") {
    auto cfg = make_fouriernet_config(/*dim=*/1, /*width=*/3, /*depth=*/1, /*ell=*/1);
    auto net = VanillaFourierNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    int n_ff = net->fouriernet->getNRemainingBases();
    int n_nn = net->neuralnet->getNRemainingBases();
    auto soln = torch::zeros({n_ff + n_nn}, torch::TensorOptions().dtype(torch::kDouble));

    REQUIRE_NOTHROW(net->lstsqUpdates(soln));
}

TEST_CASE("FourierNet forward throws when both branches are inactive", "[FourierNet]") {
    auto cfg = make_fouriernet_config(/*dim=*/1, /*width=*/2, /*depth=*/1, /*ell=*/1);
    auto net = VanillaFourierNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    net->getNNBasisSelection().zero_();
    if (net->getNNBiasSelection().defined()) {
        net->getNNBiasSelection().zero_();
    }
    net->getPolyBasisSelection().zero_();
    if (net->getPolyBiasSelection().defined()) {
        net->getPolyBiasSelection().zero_();
    }

    auto X = torch::linspace(0.0, 1.0, 4, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);
    DataInputs inputs;
    inputs.X = &X;

    REQUIRE_THROWS_WITH(net->forward(inputs),
                        Catch::Matchers::ContainsSubstring("both branches inactive"));
}
