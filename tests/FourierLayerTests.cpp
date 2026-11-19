#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <torch/torch.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "fourierpinns/models/FourierLayer.h"
#include "fourierpinns/basis/FourierBasisUtils.h"
#include "fourierpinns/data/LoadedDomain.h"

TEST_CASE("Raw Fourier features preserve constants and match cached evaluation", "[FourierLayer][RawBases]") {
    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0, 1}};
    FourierLayerArgs fl;
    fl.ell = 2;
    fl.include_zero_mode = true;
    fl.fourier_periods = {2};
    cfg.arch = fl;
    VanillaFourierLayer layer(cfg, "poisson", torch::kCPU, torch::kDouble);
    LoadedPDEDomain domain(1, 17, 4, 23, 0, "test1_pi_k1", "poisson", "dirichlet",
        "box", "random", false, false, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    domain.setPolyBasis(bases);
    auto col = domain.getPDEData();
    REQUIRE(torch::equal(*col.data.P.basis, layer->buildBasisFromData(*col.data.X)));
    REQUIRE(torch::equal(col.data.P.basis->select(1, 0), torch::ones({17}, torch::kDouble)));
    {
        torch::NoGradGuard guard;
        for (auto& parameter : layer->parameters()) parameter.zero_();
        layer->named_parameters()["trunc_neuralnet.weight"].flatten().index_put_({0}, 3.0);
    }
    auto cached = layer->forward(col.data);
    REQUIRE(torch::equal(cached, layer->forward(*col.data.X)));
    REQUIRE(torch::equal(cached, torch::full({17, 1}, 3.0, torch::kDouble)));
    PDE pde("poisson");
    REQUIRE(layer->forward_pde(col, &pde).abs().max().item<double>() == 0);
}

namespace {

constexpr double kTol = 1e-10;

std::vector<double> build_phi_1d(double x,
                                 int ell,
                                 bool include_zero_mode,
                                 bool scale_to_period,
                                 double a,
                                 double b) {
    double x_scaled = x;
    if (scale_to_period) {
        x_scaled = (x - a) / (b - a);
    }
    std::vector<double> phi;
    if (include_zero_mode) {
        phi.push_back(1.0);
    }
    for (int w = 1; w <= ell; ++w) {
        double phase = 2.0 * M_PI * w * x_scaled;
        phi.push_back(std::cos(phase));
        phi.push_back(std::sin(phase));
    }
    return phi;
}

std::vector<double> build_phi_1d_dx(double x,
                                    int ell,
                                    bool include_zero_mode,
                                    bool scale_to_period,
                                    double a,
                                    double b) {
    double x_scaled = x;
    double period = 1.0;
    if (scale_to_period) {
        x_scaled = (x - a) / (b - a);
        period = (b - a);
    }
    std::vector<double> phi;
    if (include_zero_mode) {
        phi.push_back(0.0);
    }
    for (int w = 1; w <= ell; ++w) {
        double phase = 2.0 * M_PI * w * x_scaled;
        double scale = 2.0 * M_PI * w / period;
        phi.push_back(-std::sin(phase) * scale);
        phi.push_back( std::cos(phase) * scale);
    }
    return phi;
}

std::vector<double> build_phi_1d_laplacian(double x,
                                           int ell,
                                           bool include_zero_mode,
                                           bool scale_to_period,
                                           double a,
                                           double b) {
    double x_scaled = x;
    double period = 1.0;
    if (scale_to_period) {
        x_scaled = (x - a) / (b - a);
        period = (b - a);
    }
    std::vector<double> phi;
    if (include_zero_mode) {
        phi.push_back(0.0);
    }
    for (int w = 1; w <= ell; ++w) {
        double phase = 2.0 * M_PI * w * x_scaled;
        double scale = -(2.0 * M_PI * w / period) * (2.0 * M_PI * w / period);
        phi.push_back(std::cos(phase) * scale);
        phi.push_back(std::sin(phase) * scale);
    }
    return phi;
}

torch::Tensor make_one_hot(int n, int idx, torch::Device device, torch::Dtype dtype) {
    auto t = torch::zeros({n}, torch::TensorOptions().device(device).dtype(dtype));
    t.index_put_({idx}, 1.0);
    return t;
}

VanillaFourierLayer make_layer_1d(int ell,
                                  bool include_zero_mode,
                                  bool scale_to_period,
                                  double a,
                                  double b,
                                  double trunc_thresh = 1e-5) {
    FourierLayerArgs args;
    args.ell = ell;
    args.include_zero_mode = include_zero_mode;
    args.scale_to_period = scale_to_period;
    args.pn_trunc_thresh = static_cast<float>(trunc_thresh);

    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{a, b}};
    cfg.arch = args;

    return VanillaFourierLayer(cfg,
                               /*pde_problem=*/"poisson",
                               torch::kCPU,
                               torch::kDouble);
}

ModelConfig make_layer_config(int input_dim, int output_dim, int ell) {
    FourierLayerArgs args;
    args.ell = ell;
    args.include_zero_mode = true;
    args.scale_to_period = false;

    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = input_dim;
    cfg.output_dim = output_dim;
    cfg.domain_bounds.clear();
    for (int d = 0; d < input_dim; ++d) {
        cfg.domain_bounds.push_back({0.0, 1.0});
    }
    cfg.arch = args;
    return cfg;
}

} // namespace

// Ensures FourierLayer basis size matches tensor-product count for the config.
TEST_CASE("FourierLayer basis size matches tensor product", "[FourierLayer]") {
    FourierLayerArgs args;
    args.ell = 2;
    args.include_zero_mode = true;
    args.scale_to_period = false;
    std::vector<std::vector<double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};

    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.domain_bounds = bounds;
    cfg.arch = args;
    auto layer = VanillaFourierLayer(cfg, "poisson", torch::kCPU, torch::kDouble);

    int per_dim = 1 + 2 * args.ell;
    int expected = per_dim * per_dim;
    REQUIRE(layer->getNRemainingBases() == expected);
}

// Checks analytical basis values at x=0 for 1D modes.
TEST_CASE("FourierLayer 1D basis values at x=0", "[FourierLayer]") {
    int ell = 3;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);

    auto expected = build_phi_1d(0.0, ell, true, false, 0.0, 1.0);
    REQUIRE(static_cast<int>(expected.size()) == layer->getNRemainingBases());

    auto X = torch::tensor({{0.0}}, torch::TensorOptions().dtype(torch::kDouble));
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        auto soln = make_one_hot(layer->getNRemainingBases(), i, torch::kCPU, torch::kDouble);
        layer->lstsqUpdates(soln);
        auto y = layer->forward(X).item<double>();
        REQUIRE(y == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(kTol));
    }
}

// Confirms include_zero_mode changes the number of basis functions.
TEST_CASE("FourierLayer zero-mode toggle changes basis size", "[FourierLayer]") {
    int ell = 2;
    auto layer_zero = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);
    auto layer_nozero = make_layer_1d(ell, /*include_zero_mode=*/false, /*scale_to_period=*/false, 0.0, 1.0);

    REQUIRE(layer_zero->getNRemainingBases() == 1 + 2 * ell);
    REQUIRE(layer_nozero->getNRemainingBases() == 2 * ell);
}

// Validates that period scaling aligns the left boundary with zero phase.
TEST_CASE("FourierLayer period scaling maps x=a to zero phase", "[FourierLayer]") {
    int ell = 2;
    double a = -1.0;
    double b = 1.0;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/true, a, b);

    auto expected = build_phi_1d(a, ell, true, true, a, b);
    auto X = torch::tensor({{a}}, torch::TensorOptions().dtype(torch::kDouble));

    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        auto soln = make_one_hot(layer->getNRemainingBases(), i, torch::kCPU, torch::kDouble);
        layer->lstsqUpdates(soln);
        auto y = layer->forward(X).item<double>();
        REQUIRE(y == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(kTol));
    }
}

// Verifies tensor-product ordering in 2D basis construction is consistent.
TEST_CASE("FourierLayer tensor product ordering (2D)", "[FourierLayer]") {
    FourierLayerArgs args;
    args.ell = 1;
    args.include_zero_mode = true;
    args.scale_to_period = false;
    std::vector<std::vector<double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};

    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.domain_bounds = bounds;
    cfg.arch = args;
    auto layer = VanillaFourierLayer(cfg, "poisson", torch::kCPU, torch::kDouble);

    double x = 0.0;
    double y = 0.25;
    auto phi_x = build_phi_1d(x, 1, true, false, 0.0, 1.0);
    auto phi_y = build_phi_1d(y, 1, true, false, 0.0, 1.0);

    int B = static_cast<int>(phi_y.size());
    auto X = torch::tensor({{x, y}}, torch::TensorOptions().dtype(torch::kDouble));

    for (int a = 0; a < static_cast<int>(phi_x.size()); ++a) {
        for (int b = 0; b < static_cast<int>(phi_y.size()); ++b) {
            int idx = a * B + b;
            auto soln = make_one_hot(layer->getNRemainingBases(), idx, torch::kCPU, torch::kDouble);
            layer->lstsqUpdates(soln);
            auto yhat = layer->forward(X).item<double>();
            double expected = phi_x[static_cast<std::size_t>(a)] * phi_y[static_cast<std::size_t>(b)];
            REQUIRE(yhat == Catch::Approx(expected).margin(kTol));
        }
    }
}

// Checks Laplacian basis scaling against analytic second derivatives.
TEST_CASE("FourierLayer Laplacian scaling matches analytic", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);
    auto X = torch::tensor({{0.3}}, torch::TensorOptions().dtype(torch::kDouble));

    auto expected = build_phi_1d_laplacian(0.3, ell, true, false, 0.0, 1.0);
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        auto soln = make_one_hot(layer->getNRemainingBases(), i, torch::kCPU, torch::kDouble);
        layer->lstsqUpdates(soln);
        auto y = layer->forwardPDEFromData(X).item<double>();
        REQUIRE(y == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
    }
}

// Compares derivative basis values to analytic derivatives for correctness.
TEST_CASE("FourierLayer derivative basis matches analytic", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);
    auto X = torch::tensor({{0.4}}, torch::TensorOptions().dtype(torch::kDouble));

    auto expected = build_phi_1d_dx(0.4, ell, true, false, 0.0, 1.0);
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        auto soln = make_one_hot(layer->getNRemainingBases(), i, torch::kCPU, torch::kDouble);
        layer->lstsqUpdates(soln);
        auto y = layer->forwardDxFromData(X, 0).item<double>();
        REQUIRE(y == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
    }
}

// Validates second-derivative basis in FourierLayer against analytic values in 1D.
TEST_CASE("FourierLayer second derivative basis matches analytic", "[FourierLayer]") {
    auto cfg = make_layer_config(/*input_dim=*/1, /*output_dim=*/1, /*ell=*/1);
    auto layer = VanillaFourierLayer(cfg, "poisson", torch::kCPU, torch::kDouble);

    auto X = torch::tensor({{0.25}}, torch::kDouble);
    auto P_dxx = layer->buildBasisDxxFromData(X, /*deriv_dim=*/0);

    double phase = 2.0 * M_PI * 0.25;
    double d2cos = -(2.0 * M_PI) * (2.0 * M_PI) * std::cos(phase);
    double d2sin = -(2.0 * M_PI) * (2.0 * M_PI) * std::sin(phase);

    REQUIRE(P_dxx.sizes()[0] == 1);
    REQUIRE(P_dxx.sizes()[1] == 3);
    double v0 = P_dxx.index({0, 0}).item<double>();
    double v1 = P_dxx.index({0, 1}).item<double>();
    double v2 = P_dxx.index({0, 2}).item<double>();
    REQUIRE(v0 == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(v1 == Catch::Approx(d2cos).margin(1e-8));
    REQUIRE(v2 == Catch::Approx(d2sin).margin(1e-8));
}

// Ensures buildFourierBases includes required derivative bases for time-dependent PDEs.
TEST_CASE("FourierBasisUtils builds time-derivative bases for nonss_allencahn and onewaywave", "[FourierLayer]") {
    DomainConfig dcfg;
    dcfg.dim = 2;
    dcfg.pde_problem = "nonss_allencahn";
    dcfg.bnd_typ = "periodic";
    dcfg.domain_typ = "box";
    dcfg.point_sampling = "equispaced";
    dcfg.Ncol_pts = 4;
    dcfg.Nbnd_pts = 4;
    dcfg.Ntest_pts = 4;
    dcfg.Nic_pts = 4;
    dcfg.initial_cond = true;
    dcfg.homogeneous_pde = true;
    dcfg.test_func = "test7";
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

    auto cfg = make_layer_config(/*input_dim=*/2, /*output_dim=*/1, /*ell=*/1);
    auto layer = VanillaFourierLayer(cfg, "nonss_allencahn", torch::kCPU, torch::kDouble);

    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    REQUIRE(bases.count("P_col_dx") == 1);
    REQUIRE(bases.count("P_col_dy") == 1);
    REQUIRE(bases.count("P_col_dyy") == 1);
    REQUIRE(bases.count("P_bnd_min_dy") == 1);
    REQUIRE(bases.count("P_bnd_max_dy") == 1);
    REQUIRE(bases.count("P_bnd_min_dyy") == 1);
    REQUIRE(bases.count("P_bnd_max_dyy") == 1);

    dcfg.pde_problem = "onewaywave_inhomo";
    LoadedPDEDomain domain2(dcfg.dim,
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
    domain2.generateDomainData();

    auto layer2 = VanillaFourierLayer(cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    auto bases2 = buildFourierBases(&domain2, layer2, torch::kCPU, torch::kDouble);
    REQUIRE(bases2.count("P_col_dx") == 1);
    REQUIRE(bases2.count("P_col_dy") == 1);
    REQUIRE(bases2.count("P_col_dyy") == 0);
}

// Ensures precomputed basis and on-the-fly basis produce identical outputs.
TEST_CASE("FourierLayer precomputed basis yields same output as on-the-fly", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);

    auto X = torch::tensor({{0.25}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = layer->buildBasisFromData(X);

    auto soln = make_one_hot(layer->getNRemainingBases(), 3, torch::kCPU, torch::kDouble);
    layer->lstsqUpdates(soln);

    auto y_direct = layer->forward(X).item<double>();
    auto y_basis  = layer->forwardBasis(P).item<double>();
    REQUIRE(y_direct == Catch::Approx(y_basis).margin(kTol));
}

// Confirms forward(DataInputs) consumes provided Basis instead of recomputing.
TEST_CASE("FourierLayer forward(DataInputs) uses provided precomputed basis", "[FourierLayer]") {
    int ell = 1;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);

    auto X = torch::tensor({{0.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = layer->buildBasisFromData(X);

    auto soln = make_one_hot(layer->getNRemainingBases(), 2, torch::kCPU, torch::kDouble);
    layer->lstsqUpdates(soln);

    DataInputs inputs;
    inputs.P.basis = &P;
    auto y = layer->forward(inputs).item<double>();
    auto expected = P.index({0, 2}).item<double>();
    REQUIRE(y == Catch::Approx(expected).margin(kTol));
}

// Verifies truncation zeros coefficients below the configured threshold.
TEST_CASE("FourierLayer truncation removes small weights", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0, /*trunc_thresh=*/0.1);

    int n = layer->getNRemainingBases();
    auto soln = torch::full({n}, 0.01, torch::TensorOptions().dtype(torch::kDouble));
    layer->lstsqUpdates(soln);
    layer->truncateCoeffs();

    REQUIRE(layer->getNRemainingBases() == 0);
}

// Checks that outputs are differentiable w.r.t. inputs (autograd works).
TEST_CASE("FourierLayer forward is differentiable", "[FourierLayer]") {
    auto layer = make_layer_1d(2, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);
    auto X = torch::tensor({{0.2}}, torch::TensorOptions().dtype(torch::kDouble).requires_grad(true));
    auto y = layer->forward(X);
    y.sum().backward();
    REQUIRE(X.grad().defined());
    REQUIRE(torch::isfinite(X.grad()).all().item<bool>());
}

// Ensures LS matrix uses provided precomputed basis matrices when available.
TEST_CASE("FourierLayer getLSTSQ_A uses provided precomputed bases", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);

    auto X_bnd = torch::linspace(0.0, 1.0, 2, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);
    auto X_col = torch::linspace(0.0, 1.0, 3, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);

    auto P_bnd = layer->buildBasisFromData(X_bnd);
    auto P_pde = layer->buildBasisPDEFromData(X_col);

    BndInputs bnd;
    PDEInputs col;
    bnd.data.X = &X_bnd;
    bnd.data.P.basis = &P_bnd;
    col.data.X = &X_col;
    col.P_pde = &P_pde;

    auto A = layer->getLSTSQ_A(col, bnd, nullptr, nullptr);
    REQUIRE(A.sizes()[0] == 5);
    REQUIRE(A.sizes()[1] == layer->getNRemainingBases());
    REQUIRE(torch::isfinite(A).all().item<bool>());
}

// Verifies periodic LSTSQ A uses value + derivative difference rows.
TEST_CASE("FourierLayer periodic getLSTSQ_A builds value/derivative constraints", "[FourierLayer]") {
    int ell = 2;
    auto layer = make_layer_1d(ell, /*include_zero_mode=*/true, /*scale_to_period=*/false, 0.0, 1.0);

    auto X_a = torch::tensor({{0.0}, {0.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_b = torch::tensor({{1.0}, {1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto X_col = torch::linspace(0.0, 1.0, 3, torch::TensorOptions().dtype(torch::kDouble)).unsqueeze(1);

    auto P_a = layer->buildBasisFromData(X_a);
    auto P_b = layer->buildBasisFromData(X_b);
    auto P_a_x = layer->buildBasisDxFromData(X_a, 0);
    auto P_b_x = layer->buildBasisDxFromData(X_b, 0);
    auto P_pde = layer->buildBasisPDEFromData(X_col);

    BndInputs bnd;
    bnd.data.X = &X_a;
    bnd.data.P.basis = &P_a;
    bnd.data_b.X = &X_b;
    bnd.data_b.P.basis = &P_b;
    bnd.P_x = &P_a_x;
    bnd.P_x_b = &P_b_x;

    PDEInputs pde;
    pde.data.X = &X_col;
    pde.P_pde = &P_pde;

    auto A = layer->getLSTSQ_A(pde, bnd, nullptr, nullptr);

    const int64_t nbnd = X_a.size(0);
    const int64_t ncol = X_col.size(0);

    auto zeros_bnd = torch::zeros({nbnd, 1}, P_a.options());
    auto A_val = torch::cat({zeros_bnd, P_a - P_b}, 1);
    auto A_dx = torch::cat({zeros_bnd, P_a_x - P_b_x}, 1);
    auto A_bnd = torch::cat({A_val, A_dx}, 0);

    auto zeros_col = torch::zeros({ncol, 1}, P_pde.options());
    auto A_col = torch::cat({zeros_col, P_pde}, 1);

    auto expected = torch::cat({A_bnd, A_col}, 0);
    auto selection = layer->getFullBasisSelection();
    expected = expected.index({torch::indexing::Slice(), selection});

    REQUIRE(A.sizes() == expected.sizes());
    REQUIRE(torch::allclose(A, expected, 1e-8, 1e-8));
}
