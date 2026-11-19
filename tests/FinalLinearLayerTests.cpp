#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <torch/torch.h>
#include <vector>

#include "fourierpinns/models/FinalLinearLayer.h"
#include "fourierpinns/training/BasisLogging.h"

TEST_CASE("Basis summary separates groups and reports post-pruning counts", "[Logging][Bases]") {
    auto layer = FinalLinearLayer(3, 1, 0.1, true, "constant", torch::kCPU);
    auto soln = torch::tensor({0.01f, 0.01f, 0.2f, -0.4f});
    layer->reassignLayerWeights(soln);
    layer->truncateCoeffs();
    std::vector<BasisCoefficientGroup> groups = {
        {"NN bases", layer->getBasisCoeffs(), layer->getBasisSelection(), {}},
        {"NN bias", layer->getBasisBias(), layer->getBiasSelection(), {"bias"}}};
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2);
    const auto flags = stream.flags();
    printBasisSummary(stream, groups, {3, 1});
    const auto text = stream.str();
    REQUIRE(text.find("NN bases") != std::string::npos);
    REQUIRE(text.find("NN bias") != std::string::npos);
    REQUIRE(text.find("2/3") != std::string::npos);
    REQUIRE(text.find("0/1") != std::string::npos);
    REQUIRE(text.find("Pruned") != std::string::npos);
    REQUIRE(text.find("basis[2]=-4.000000e-01") != std::string::npos);
    REQUIRE(text.find("basis[0]=") == std::string::npos);
    REQUIRE(stream.flags() == flags);
    REQUIRE(stream.precision() == 2);
}

TEST_CASE("Basis summary limits coefficient output unless details are requested", "[Logging][Bases]") {
    auto values = torch::arange(1, 8, torch::kDouble);
    auto mask = torch::ones_like(values);
    BasisCoefficientGroup group{"Fourier", values, mask,
        {"constant", "cos(k=1)", "sin(k=1)", "cos(k=2)", "sin(k=2)", "cos(k=3)", "sin(k=3)"}};
    std::ostringstream compact, full;
    printBasisSummary(compact, {group}, {7});
    printBasisSummary(full, {group}, {7}, true);
    REQUIRE(compact.str().find("7/7") != std::string::npos);
    REQUIRE(compact.str().find("showing 5 of 7") != std::string::npos);
    REQUIRE(compact.str().find("sin(k=3)=7.000000e+00") != std::string::npos);
    REQUIRE(compact.str().find("constant=") == std::string::npos);
    REQUIRE(full.str().find("constant=1.000000e+00") != std::string::npos);
}

namespace {
constexpr double kTol = 1e-10;
}

// Verifies forward output equals manual linear transform with bias and mask.
TEST_CASE("FinalLinearLayer forward matches manual computation", "[FinalLinearLayer]") {
    auto layer = FinalLinearLayer(/*in_features=*/2, /*out_features=*/1, /*threshold=*/0.0,
                                  /*include_bias=*/true, /*init_typ=*/"constant", torch::kCPU);
    layer->to(torch::kCPU, torch::kDouble);

    // soln = [bias, w00, w01]
    auto soln = torch::tensor({2.0, 3.0, -1.0}, torch::TensorOptions().dtype(torch::kDouble));
    layer->reassignLayerWeights(soln);

    auto X = torch::tensor({{4.0, 5.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto y = layer->forward(X).item<double>();
    double expected = 2.0 + 3.0 * 4.0 + (-1.0) * 5.0;
    REQUIRE(y == Catch::Approx(expected).margin(kTol));
}

// Checks truncation masks weights/bias below threshold and zeroes them out.
TEST_CASE("FinalLinearLayer truncation zeros small weights and bias", "[FinalLinearLayer]") {
    auto layer = FinalLinearLayer(/*in_features=*/2, /*out_features=*/1, /*threshold=*/0.1,
                                  /*include_bias=*/true, /*init_typ=*/"constant", torch::kCPU);
    layer->to(torch::kCPU, torch::kDouble);

    auto soln = torch::tensor({0.01, 0.05, -0.02}, torch::TensorOptions().dtype(torch::kDouble));
    layer->reassignLayerWeights(soln);
    layer->truncateCoeffs();

    REQUIRE(layer->getNRemainingBases() == 0);
    auto X = torch::tensor({{1.0, 2.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto y = layer->forward(X).item<double>();
    REQUIRE(y == Catch::Approx(0.0).margin(kTol));
}

// Ensures LS weight reassignment respects the active mask and shape.
TEST_CASE("FinalLinearLayer reassignLayerWeights respects mask", "[FinalLinearLayer]") {
    auto layer = FinalLinearLayer(/*in_features=*/2, /*out_features=*/1, /*threshold=*/0.0,
                                  /*include_bias=*/true, /*init_typ=*/"constant", torch::kCPU);
    layer->to(torch::kCPU, torch::kDouble);

    // First set to small values and truncate to mask one weight
    auto soln_small = torch::tensor({1.0, 0.0, 0.0}, torch::TensorOptions().dtype(torch::kDouble));
    layer->reassignLayerWeights(soln_small);
    layer->truncateCoeffs(); // threshold=0.0, so weights stay; now mask all ones

    // Now reassign
    auto soln = torch::tensor({1.0, 2.0, 3.0}, torch::TensorOptions().dtype(torch::kDouble));
    layer->reassignLayerWeights(soln);

    auto X = torch::tensor({{1.0, 1.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto y = layer->forward(X).item<double>();
    double expected = 1.0 + 2.0 + 3.0;
    REQUIRE(y == Catch::Approx(expected).margin(kTol));
}

// Confirms gradients for truncated coefficients are zeroed.
TEST_CASE("FinalLinearLayer gradients are masked after truncation", "[FinalLinearLayer]") {
    auto layer = FinalLinearLayer(/*in_features=*/2, /*out_features=*/1, /*threshold=*/0.1,
                                  /*include_bias=*/true, /*init_typ=*/"constant", torch::kCPU);
    layer->to(torch::kCPU, torch::kDouble);

    auto soln = torch::tensor({0.01, 0.01, 0.01}, torch::TensorOptions().dtype(torch::kDouble));
    layer->reassignLayerWeights(soln);
    layer->truncateCoeffs();

    auto X = torch::tensor({{1.0, 2.0}}, torch::TensorOptions().dtype(torch::kDouble));
    auto y = layer->forward(X);
    y.sum().backward();

    auto grads = layer->getGrads();
    auto bias_grads = layer->getBiasGrad();
    REQUIRE(torch::all(grads == 0).item<bool>());
    REQUIRE(torch::all(bias_grads == 0).item<bool>());
}
