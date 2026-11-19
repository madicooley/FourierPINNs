#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <limits>

#include "fourierpinns/basis/FourierBasisBuilder.h"

TEST_CASE("Explicit Fourier periods set phases and all analytic derivatives", "[FourierPeriods]") {
    auto X = torch::tensor({{2.13, -0.71}, {2.64, 0.37}, {2.97, 0.81}}, torch::kDouble);
    FourierBasisBuilder basis(2, 2, BasisGenType::TensorProduct, true, false,
        -1, {{2, 3}, {-1, 1}}, torch::kCPU, torch::kDouble, {3, 5});
    auto a = (2*M_PI/3)*(X.index({torch::indexing::Slice(), 0})-2);
    auto b = (2*M_PI/5)*(X.index({torch::indexing::Slice(), 1})+1);
    auto u = torch::sin(a)*torch::sin(b);
    constexpr int column = 2*5+2;
    REQUIRE(torch::allclose(basis.buildBasis(X).select(1, column), u, 1e-12, 1e-12));
    REQUIRE(torch::allclose(basis.buildBasisDx(X, 0).select(1, column), (2*M_PI/3)*torch::cos(a)*torch::sin(b), 1e-12, 1e-12));
    REQUIRE(torch::allclose(basis.buildBasisDx(X, 1).select(1, column), (2*M_PI/5)*torch::sin(a)*torch::cos(b), 1e-12, 1e-12));
    REQUIRE(torch::allclose(basis.buildBasisDxx(X, 0).select(1, column), -std::pow(2*M_PI/3, 2)*u, 1e-12, 1e-12));
    REQUIRE(torch::allclose(basis.buildBasisPDE(X).select(1, column),
        -(std::pow(2*M_PI/3, 2)+std::pow(2*M_PI/5, 2))*u, 1e-12, 1e-12));
    FourierBasisBuilder enabled(2, 2, BasisGenType::TensorProduct, true, true,
        -1, {{2, 3}, {-1, 1}}, torch::kCPU, torch::kDouble, {3, 5});
    REQUIRE(torch::equal(enabled.buildBasis(X), basis.buildBasis(X)));
}

TEST_CASE("Default Fourier periods retain legacy scaling and explicit periods are validated", "[FourierPeriods]") {
    auto X = torch::tensor({{2.13}, {2.64}}, torch::kDouble);
    for (bool scale : {false, true}) {
        FourierBasisBuilder old(1, 2, BasisGenType::TensorProduct, true, scale,
            -1, {{2, 5}}, torch::kCPU, torch::kDouble);
        FourierBasisBuilder explicit_period(1, 2, BasisGenType::TensorProduct, true, false,
            -1, scale ? std::vector<std::vector<double>>{{2, 5}} : std::vector<std::vector<double>>{},
            torch::kCPU, torch::kDouble, {scale ? 3.0 : 1.0});
        REQUIRE(torch::equal(old.buildBasis(X), explicit_period.buildBasis(X)));
        REQUIRE(torch::equal(old.buildBasisDx(X, 0), explicit_period.buildBasisDx(X, 0)));
        REQUIRE(torch::equal(old.buildBasisPDE(X), explicit_period.buildBasisPDE(X)));
    }
    for (const std::vector<double> periods : {std::vector<double>{0}, {-1},
            {std::numeric_limits<double>::infinity()}, {std::numeric_limits<double>::quiet_NaN()}, {1, 2}}) {
        REQUIRE_THROWS(FourierBasisBuilder(1, 2, BasisGenType::TensorProduct, true, false,
            -1, {{0, 1}}, torch::kCPU, torch::kDouble, periods));
    }
    FourierBasisBuilder half_wave(1, 2, BasisGenType::TensorProduct, true, false,
        -1, {{0, 1}}, torch::kCPU, torch::kDouble, {2});
    REQUIRE(torch::allclose(half_wave.buildBasis(X).select(1, 2), torch::sin(M_PI*X.squeeze(1)), 1e-12, 1e-12));
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

} // namespace

// Verifies tensor-product mode count matches expected basis size for given dim/ell and zero-mode setting.
TEST_CASE("FourierBasisBuilder basis size matches tensor product", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/2,
                                /*ell=*/2,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/false,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{0.0, 1.0}, {0.0, 1.0}},
                                torch::kCPU,
                                torch::kDouble);
    int per_dim = 1 + 2 * 2;
    REQUIRE(builder.getPerDimSize() == per_dim);
    REQUIRE(builder.getNPoly() == per_dim * per_dim);
}

// Confirms builder getters (dim, ell, include-zero, scaling) return the configured values.
TEST_CASE("FourierBasisBuilder getters reflect configuration", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/4,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/false,
                                /*scale_to_period=*/true,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{-2.0, 2.0}},
                                torch::kCPU,
                                torch::kDouble);
    REQUIRE(builder.getEll() == 4);
    REQUIRE(builder.getIncludeZeroMode() == false);
    REQUIRE(builder.getScaleToPeriod() == true);
    REQUIRE(builder.getBasisGenType() == BasisGenType::TensorProduct);
}

// Ensures basis construction respects the requested tensor dtype.
TEST_CASE("FourierBasisBuilder output dtype matches builder dtype", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/2,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/false,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{0.0, 1.0}},
                                torch::kCPU,
                                torch::kFloat);
    auto X = torch::tensor({{0.25}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = builder.buildBasis(X);
    REQUIRE(P.dtype() == torch::kFloat);
}

// Checks analytic values at x=0 for sin/cos modes to validate basis evaluation.
TEST_CASE("FourierBasisBuilder 1D basis values at x=0", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/3,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/false,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{0.0, 1.0}},
                                torch::kCPU,
                                torch::kDouble);

    auto expected = build_phi_1d(0.0, 3, true, false, 0.0, 1.0);
    auto X = torch::tensor({{0.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = builder.buildBasis(X);

    REQUIRE(P.size(1) == static_cast<long>(expected.size()));
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        REQUIRE(P.index({0, i}).item<double>() == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(kTol));
    }
}

// Validates period scaling so the left boundary maps to zero phase in the basis.
TEST_CASE("FourierBasisBuilder period scaling maps x=a to zero phase", "[FourierBasisBuilder]") {
    double a = -1.0;
    double b = 1.0;
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/2,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/true,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{a, b}},
                                torch::kCPU,
                                torch::kDouble);

    auto expected = build_phi_1d(a, 2, true, true, a, b);
    auto X = torch::tensor({{a}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = builder.buildBasis(X);
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        REQUIRE(P.index({0, i}).item<double>() == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(kTol));
    }
}

// Compares computed derivative bases against analytic derivatives for correctness.
TEST_CASE("FourierBasisBuilder derivative basis matches analytic", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/2,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/false,
                                /*periodic_deriv_dim=*/0,
                                /*domain_bounds=*/{{0.0, 1.0}},
                                torch::kCPU,
                                torch::kDouble);

    auto expected = build_phi_1d_dx(0.4, 2, true, false, 0.0, 1.0);
    auto X = torch::tensor({{0.4}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = builder.buildBasisDx(X, 0);
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        REQUIRE(P.index({0, i}).item<double>() == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
    }
}

// Verifies Laplacian basis computation against analytic second derivatives.
TEST_CASE("FourierBasisBuilder Laplacian basis matches analytic", "[FourierBasisBuilder]") {
    FourierBasisBuilder builder(/*input_dim=*/1,
                                /*ell=*/2,
                                BasisGenType::TensorProduct,
                                /*include_zero_mode=*/true,
                                /*scale_to_period=*/false,
                                /*periodic_deriv_dim=*/-1,
                                /*domain_bounds=*/{{0.0, 1.0}},
                                torch::kCPU,
                                torch::kDouble);

    auto expected = build_phi_1d_laplacian(0.3, 2, true, false, 0.0, 1.0);
    auto X = torch::tensor({{0.3}}, torch::TensorOptions().dtype(torch::kDouble));
    auto P = builder.buildBasisPDE(X);
    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        REQUIRE(P.index({0, i}).item<double>() == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
    }
}

// Checks second-derivative basis against analytic values in 1D.
TEST_CASE("FourierBasisBuilder second derivative basis matches analytic", "[FourierBasisBuilder]") {
    FourierLayerArgs args;
    args.ell = 1;
    args.include_zero_mode = true;
    args.scale_to_period = false;

    FourierBasisBuilder builder(1, args.ell, args.basis_gen_typ,
                                args.include_zero_mode, args.scale_to_period,
                                /*periodic_deriv_dim=*/-1,
                                {{0.0, 1.0}},
                                torch::kCPU, torch::kDouble);

    auto x = torch::tensor({{0.25}}, torch::kDouble);
    auto P_dxx = builder.buildBasisDxx(x, /*deriv_dim=*/0);

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
