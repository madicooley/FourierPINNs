#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <torch/torch.h>
#include <map>
#include <cmath>

#include "fourierpinns/models/NeuralNet.h"
#include "fourierpinns/models/FourierNet.h"
#include "fourierpinns/config/Args.h"

namespace {
constexpr double kTol = 1e-9;

ModelConfig make_model_config(int input_dim, int output_dim, int width, int depth) {
    ModelConfig cfg;
    cfg.input_dim = input_dim;
    cfg.output_dim = output_dim;

    NeuralNetArgs nn;
    nn.width = width;
    nn.depth = depth;
    nn.net_act = "tanh";
    nn.nn_truncate = false;
    nn.nn_trunc_thresh = 1e-5f;
    nn.nn_l1_reg_coeff = 0.0f;
    nn.nn_l2_reg_coeff = 0.0f;

    cfg.layers.clear();
    cfg.layers.push_back(input_dim);
    for (int i = 0; i < depth; ++i) cfg.layers.push_back(width);
    cfg.layers.push_back(output_dim);
    cfg.arch = nn;
    return cfg;
}

} // namespace

TEST_CASE("Steady Allen-Cahn network uses physical coordinate derivatives", "[NeuralNet][AllenCahn]") {
    auto cfg = make_model_config(1, 1, 1, 1);
    auto net = VanillaNeuralNet(cfg, "allencahn", torch::kCPU, torch::kDouble);
    {
        torch::NoGradGuard guard;
        for (auto& parameter : net->named_parameters()) {
            if (parameter.key().find("linear_0.weight") != std::string::npos) parameter.value().fill_(1.0);
            if (parameter.key().find("linear_0.bias") != std::string::npos) parameter.value().zero_();
        }
    }
    auto coeffs = torch::tensor({0.0, 1.0}, torch::kDouble);
    net->lstsqUpdates(coeffs);
    auto x = torch::linspace(0.05, 0.95, 17, torch::kDouble).unsqueeze(1).requires_grad_(true);
    auto u = net->forward(x);
    auto exact = torch::tanh(x);
    REQUIRE(torch::allclose(u, exact, 1e-12, 1e-12));
    auto d2 = -2 * exact * (1-exact.square());
    auto expected = d2 + exact * (exact.square()-1);
    PDE pde("allencahn");
    REQUIRE(torch::allclose(pde.forwardPDE(u, x), expected, 1e-10, 1e-10));
    auto basis = net->forwardBasis(x);
    REQUIRE(torch::allclose(pde.forwardBasisPDE(basis, x), d2, 1e-10, 1e-10));
}

// Verifies basis and output shapes for a basic forward pass.
TEST_CASE("NeuralNet forward and basis shapes", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/3, /*output_dim=*/1, /*width=*/5, /*depth=*/2);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X = torch::randn({7, 3}, torch::TensorOptions().dtype(torch::kDouble));
    auto basis = net->forwardBasis(X);
    auto out = net->forward(X);

    REQUIRE(basis.sizes()[0] == 7);
    REQUIRE(basis.sizes()[1] == 5);
    REQUIRE(out.sizes()[0] == 7);
    REQUIRE(out.sizes()[1] == 1);
}

// Ensures forward(DataInputs) reads X and produces correct output shape.
TEST_CASE("NeuralNet forward(DataInputs) uses X", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/4, /*depth=*/1);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X = torch::zeros({3, 2}, torch::TensorOptions().dtype(torch::kDouble));
    DataInputs inputs;
    inputs.X = &X;

    auto out = net->forward(inputs);
    REQUIRE(out.sizes()[0] == 3);
    REQUIRE(out.sizes()[1] == 1);
}

// Checks that zero input yields the final-layer bias when weights are zeroed.
TEST_CASE("NeuralNet forward equals bias for zero input with zero hidden biases", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/3, /*depth=*/2);
    NeuralNetArgs nn = std::get<NeuralNetArgs>(cfg.arch);
    nn.nn_l1_reg_coeff = 0.0f;
    nn.nn_l2_reg_coeff = 0.0f;
    cfg.arch = nn;

    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    // Set final layer bias=2, weights=0
    int n_weights = 3; // width
    auto soln = torch::zeros({1 + n_weights}, torch::TensorOptions().dtype(torch::kDouble));
    soln.index_put_({0}, 2.0);
    net->lstsqUpdates(soln);

    auto X = torch::zeros({4, 2}, torch::TensorOptions().dtype(torch::kDouble));
    auto out = net->forward(X);
    REQUIRE(out.mean().item<double>() == Catch::Approx(2.0).margin(kTol));
}

// Validates L1/L2 regularization computations against known values.
TEST_CASE("NeuralNet regularization losses", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/3, /*depth=*/1);
    NeuralNetArgs nn = std::get<NeuralNetArgs>(cfg.arch);
    nn.nn_l1_reg_coeff = 0.5f;
    nn.nn_l2_reg_coeff = 0.25f;
    cfg.arch = nn;

    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto soln = torch::tensor({1.0, 1.0, -2.0, 3.0}, torch::TensorOptions().dtype(torch::kDouble));
    net->lstsqUpdates(soln);

    auto l1 = net->getL1RegLoss().item<double>();
    auto l2 = net->getL2RegLoss().item<double>();

    // weights = [1, -2, 3], bias=1
    // mean over width with bias broadcast across width
    double l1_expected = 0.5 * ((std::abs(1.0) + std::abs(-2.0) + std::abs(3.0)) + 3.0 * std::abs(1.0)) / 3.0;
    double l2_expected = 0.5 * 0.25 * (1.0 + 4.0 + 9.0 + 1.0);

    REQUIRE(l1 == Catch::Approx(l1_expected).margin(kTol));
    REQUIRE(l2 == Catch::Approx(l2_expected).margin(kTol));
}

// Ensures truncation masks coefficients below the threshold.
TEST_CASE("NeuralNet truncation removes all coefficients below threshold", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/1, /*output_dim=*/1, /*width=*/3, /*depth=*/1);
    NeuralNetArgs nn = std::get<NeuralNetArgs>(cfg.arch);
    nn.nn_truncate = true;
    nn.nn_trunc_thresh = 0.1f;
    cfg.arch = nn;

    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);
    auto soln = torch::tensor({0.01, 0.01, 0.01, 0.01}, torch::TensorOptions().dtype(torch::kDouble));
    net->lstsqUpdates(soln);
    net->truncateCoeffs();

    REQUIRE(net->getNRemainingBases() == 0);
}

// Confirms full basis selection mask includes bias + weights and has correct shape.
TEST_CASE("NeuralNet getFullBasisSelection shape", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/4, /*depth=*/1);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);
    auto sel = net->getFullBasisSelection();
    REQUIRE(sel.sizes().size() == 1);
    REQUIRE(sel.size(0) == 5); // bias + width
}

// Verifies LS design matrix dimensions and finite values.
TEST_CASE("NeuralNet getLSTSQ_A returns expected shape", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/1, /*output_dim=*/1, /*width=*/4, /*depth=*/1);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    PDE pde("poisson");

    auto X_bnd = torch::linspace(0.0, 1.0, 2, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);
    X_bnd.requires_grad_(true);
    auto X_col = torch::linspace(0.0, 1.0, 3, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);
    X_col.requires_grad_(true);

    BndInputs bnd;
    PDEInputs col;
    bnd.data.X = &X_bnd;
    col.data.X = &X_col;

    auto A = net->getLSTSQ_A(col, bnd, nullptr, &pde);
    REQUIRE(A.sizes()[0] == 5);
    REQUIRE(A.sizes()[1] == 5); // bias + width
    REQUIRE(torch::isfinite(A).all().item<bool>());
}

// Verifies periodic LSTSQ A includes value and derivative constraint rows.
TEST_CASE("NeuralNet periodic getLSTSQ_A has value/derivative rows", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/4, /*depth=*/1);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    PDE pde("poisson");

    auto X_a = torch::tensor({{0.0, -1.0}, {0.5, -1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_b = torch::tensor({{0.0,  1.0}, {0.5,  1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_col = torch::tensor({{0.25, 0.0}, {0.75, 0.5}, {0.9, -0.2}}, torch::TensorOptions().dtype(torch::kDouble));
    X_a.requires_grad_(true);
    X_b.requires_grad_(true);
    X_col.requires_grad_(true);

    BndInputs bnd;
    bnd.data.X = &X_a;
    bnd.data_b.X = &X_b;

    PDEInputs col;
    col.data.X = &X_col;

    auto A = net->getLSTSQ_A(col, bnd, nullptr, &pde);
    const int64_t nbnd = X_a.size(0);
    const int64_t ncol = X_col.size(0);
    const int64_t expected_rows = 2 * nbnd + ncol;
    const int64_t expected_cols = 1 + cfg.layers[1]; // bias + width

    REQUIRE(A.sizes()[0] == expected_rows);
    REQUIRE(A.sizes()[1] == expected_cols);
    REQUIRE(torch::isfinite(A).all().item<bool>());
}

// Ensures periodic boundary loss is zero when both boundary sets are identical.
TEST_CASE("NeuralNet periodic boundary loss is zero for identical inputs", "[NeuralNet]") {
    auto cfg = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/3, /*depth=*/1);
    auto net = VanillaNeuralNet(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X = torch::tensor({{0.0, -1.0}, {1.0, 1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    X.requires_grad_(true);
    BndInputs A;
    BndInputs B;
    A.data.X = &X;
    B.data.X = &X;

    auto loss = net->getPeriodicBndLoss(A, B);
    REQUIRE(loss.item<double>() == Catch::Approx(0.0).margin(1e-7));
}

// Checks FourierNet config builders preserve relevant base settings.
TEST_CASE("FourierNet config builders preserve base config settings", "[FourierNet]") {
    ModelConfig base = make_model_config(/*input_dim=*/2, /*output_dim=*/1, /*width=*/4, /*depth=*/2);
    base.domain_bounds = {{-1.0, 1.0}, {0.0, 2.0}};

    FourierNetArgs fargs;
    fargs.neuralnet_args = std::get<NeuralNetArgs>(base.arch);
    fargs.fourierlayer_args = FourierLayerArgs{};

    ModelConfig nn_cfg = VanillaFourierNeuralNetImpl::makeNeuralNetConfig(base, fargs.neuralnet_args);
    ModelConfig fl_cfg = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(base, fargs.fourierlayer_args);

    REQUIRE(nn_cfg.domain_bounds.size() == 2);

    REQUIRE(fl_cfg.domain_bounds.size() == 2);
    REQUIRE(fl_cfg.layers.empty());
    REQUIRE(fl_cfg.model_type == "fourierlayer");
}
