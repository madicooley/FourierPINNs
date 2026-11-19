#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>
#include <cmath>

#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/basis/FourierBasisBuilder.h"

namespace {
constexpr double kTol = 1e-6;

std::pair<torch::Tensor, torch::Tensor> allenCahnExact(const torch::Tensor& x, const std::string& test) {
    torch::Tensor u, d2;
    if (test == "test1_k100") {
        u = torch::sin(100 * x);
        d2 = -10000 * u;
    } else if (test == "test2") {
        u = torch::sin(x) + 0.1 * torch::sin(20*x) + 0.05 * torch::cos(100*x);
        d2 = -torch::sin(x) - 40 * torch::sin(20*x) - 500 * torch::cos(100*x);
    } else {
        // Independent sum-of-sines identity avoids duplicating the generator's product rule.
        u = 0.5 * (torch::sin(106*x) - torch::sin(94*x));
        d2 = -0.5 * 106*106 * torch::sin(106*x) + 0.5 * 94*94 * torch::sin(94*x);
    }
    return {u, d2};
}
}

TEST_CASE("Steady Allen-Cahn manufactured residual and linear basis match analytic solutions", "[PDE][AllenCahn]") {
    for (const std::string test : {"test1_k100", "test2", "test3"}) {
        INFO(test);
        auto x = torch::linspace(0.013, 6.2, 129, torch::kDouble).unsqueeze(1).requires_grad_(true);
        auto exact = allenCahnExact(x, test);
        auto u = exact.first;
        auto expected = exact.second + u * (u.square() - 1);
        PDE pde("allencahn");
        auto actual = pde.forwardPDE(u, x);
        REQUIRE(torch::allclose(actual, expected, 1e-10, 1e-9));
        auto basis = torch::cat({u, torch::sin(3*x)}, 1);
        auto linear = pde.forwardBasisPDE(basis, x);
        REQUIRE(torch::allclose(linear.index({torch::indexing::Slice(), 0}).unsqueeze(1),
                               exact.second, 1e-10, 1e-9));
        REQUIRE(torch::allclose(linear.index({torch::indexing::Slice(), 1}).unsqueeze(1),
                               -9*torch::sin(3*x), 1e-10, 1e-9));
    }
}

TEST_CASE("Steady Allen-Cahn generated forcing and boundaries stay in physical coordinates", "[PDE][AllenCahn][LoadedDomain]") {
    for (const std::string test : {"test1_k100", "test2", "test3"}) {
        LoadedPDEDomain domain(1, 129, 8, 257, 0, test, "allencahn", "dirichlet",
            "box", "equispaced", false, false, {{0.0, 2*M_PI}}, torch::kCPU, torch::kDouble);
        domain.generateDomainData();
        auto data = domain.getPDEData();
        auto x = *data.data.X;
        REQUIRE(x.min().item<double>() == 0.0);
        REQUIRE(std::abs(x.max().item<double>() - 2*M_PI) < 1e-12);
        auto exact = allenCahnExact(x, test);
        auto expected = exact.second + exact.first * (exact.first.square() - 1);
        REQUIRE(torch::allclose(*data.f, expected, 1e-10, 1e-8));
        auto bnd = domain.getBndData();
        REQUIRE(torch::allclose(*bnd.u, allenCahnExact(*bnd.data.X, test).first, 1e-10, 1e-10));
        auto tst = domain.getTestData();
        REQUIRE(torch::allclose(*tst.u, allenCahnExact(*tst.data.X, test).first, 1e-10, 1e-10));
    }
}

TEST_CASE("Section 6.4.1 Poisson generated forcing and boundary targets match analytic solutions", "[PDE][Section641]") {
    torch::manual_seed(1234);
    for (const std::string test : {"test4", "test5"}) {
        INFO(test);
        // Eight samples per edge avoid placing every test5 boundary point at a node.
        LoadedPDEDomain domain(2, 37, 32, 43, 0, test, "poisson", "dirichlet",
            "box", "random", false, false, {{0.0, 2*M_PI}, {0.0, 2*M_PI}}, torch::kCPU, torch::kDouble);
        domain.generateDomainData();
        auto exact = [&](const torch::Tensor& X, bool forcing) -> torch::Tensor {
            auto x = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
            auto y = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);
            if (test == "test4") {
                auto u = torch::sin(100*x)*torch::sin(100*y);
                return forcing ? -20000*u : u;
            }
            auto hi = torch::sin(26*x) + torch::sin(26*y);
            auto lo = torch::sin(14*x) + torch::sin(14*y);
            return forcing ? 0.5*(-676*hi + 196*lo) : 0.5*(hi-lo);
        };
        auto data = domain.getPDEData();
        REQUIRE(data.data.X->size(0) == 37);
        REQUIRE(torch::allclose(*data.f, exact(*data.data.X, true), 1e-10, 1e-8));
        auto tst = domain.getTestData();
        REQUIRE(tst.data.X->size(0) == 43);
        REQUIRE(torch::allclose(*tst.u, exact(*tst.data.X, false), 1e-10, 1e-10));
        auto bnd = domain.getBndData();
        REQUIRE(bnd.data.X->size(0) == 32);
        REQUIRE(torch::allclose(*bnd.u, exact(*bnd.data.X, false), 1e-10, 1e-10));
        if (test == "test5") REQUIRE(bnd.u->abs().max().item<double>() > 0.1);

        auto x = data.data.X->index({torch::indexing::Slice(), 0}).unsqueeze(1);
        auto y = data.data.X->index({torch::indexing::Slice(), 1}).unsqueeze(1);
        auto u = test == "test4" ? torch::sin(100*x)*torch::sin(100*y)
                                : torch::sin(6*x)*torch::cos(20*x) + torch::sin(6*y)*torch::cos(20*y);
        PDE pde("poisson");
        REQUIRE(torch::allclose(pde.forwardPDE(u, x, y), *data.f, 1e-10, 1e-8));
    }
}

TEST_CASE("Section 6.4.2 steady Allen-Cahn forcing and nonzero boundaries match Equation 41", "[PDE][AllenCahn][Section642]") {
    torch::manual_seed(1234);
    LoadedPDEDomain domain(2, 37, 32, 43, 0, "test6", "allencahn", "dirichlet",
        "box", "random", false, false, {{0, 2*M_PI}, {0, 2*M_PI}}, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto g = [](const torch::Tensor& s) {
        return torch::sin(s) + 0.1*torch::sin(20*s) + torch::cos(100*s);
    };
    auto exact = [&](const torch::Tensor& X) {
        return g(X.index({torch::indexing::Slice(), 0})).unsqueeze(1)
             * g(X.index({torch::indexing::Slice(), 1})).unsqueeze(1);
    };
    auto data = domain.getPDEData();
    auto X = data.data.X->detach().clone().requires_grad_(true);
    auto u = exact(X);
    // Differentiate the manufactured solution independently of the forcing generator.
    auto du = torch::autograd::grad({u}, {X}, {torch::ones_like(u)}, true, true)[0];
    auto laplacian = torch::zeros_like(u);
    for (int d = 0; d < 2; ++d) {
        auto first = du.index({torch::indexing::Slice(), d});
        auto second = torch::autograd::grad({first}, {X}, {torch::ones_like(first)}, true, true)[0];
        laplacian += second.index({torch::indexing::Slice(), d}).unsqueeze(1);
    }
    auto expected = laplacian + u*(u.square()-1);
    REQUIRE(torch::allclose(*data.f, expected, 1e-10, 1e-8));
    PDE pde("allencahn");
    auto x = X.index({torch::indexing::Slice(), 0}).unsqueeze(1).detach().requires_grad_(true);
    auto y = X.index({torch::indexing::Slice(), 1}).unsqueeze(1).detach().requires_grad_(true);
    auto separated_u = g(x)*g(y);
    REQUIRE(torch::allclose(pde.forwardPDE(separated_u, x, y), expected, 1e-10, 1e-8));
    auto test = domain.getTestData();
    REQUIRE(torch::allclose(*test.u, exact(*test.data.X), 1e-12, 1e-12));
    auto bnd = domain.getBndData();
    REQUIRE(torch::allclose(*bnd.u, exact(*bnd.data.X), 1e-12, 1e-12));
    REQUIRE(bnd.u->abs().max().item<double>() > 0.5);
    auto corners = torch::tensor({{0., 0.}, {0., 2*M_PI}, {2*M_PI, 0.}, {2*M_PI, 2*M_PI}}, torch::kDouble);
    REQUIRE(torch::allclose(exact(corners), torch::ones({4, 1}, torch::kDouble), 1e-12, 1e-12));
}

TEST_CASE("Section 6.4.1 tensor product includes target and axis-only modes", "[FourierBasisBuilder][Section641]") {
    auto X = torch::tensor({{0.013, 0.071}, {1.17, 2.31}, {3.03, 4.19}}, torch::kDouble);
    auto x = X.index({torch::indexing::Slice(), 0});
    auto y = X.index({torch::indexing::Slice(), 1});
    for (const int k : {26, 100}) {
        FourierBasisBuilder builder(2, k, BasisGenType::TensorProduct, true, true, -1,
            {{0.0, 2*M_PI}, {0.0, 2*M_PI}}, torch::kCPU, torch::kDouble);
        auto basis = builder.buildBasis(X);
        auto laplacian = builder.buildBasisPDE(X);
        const int n = 2*k+1;
        REQUIRE(builder.getNPoly() == n*n);
        if (k == 100) {
            const auto column = 200*n+200;
            auto expected = torch::sin(100*x)*torch::sin(100*y);
            REQUIRE(torch::allclose(basis.index({torch::indexing::Slice(), column}), expected, 1e-10, 1e-10));
            REQUIRE(torch::allclose(laplacian.index({torch::indexing::Slice(), column}), -20000*expected, 1e-10, 1e-8));
        } else {
            auto w = torch::zeros({n*n}, torch::kDouble);
            w.index_put_({52*n}, 0.5);
            w.index_put_({28*n}, -0.5);
            w.index_put_({52}, 0.5);
            w.index_put_({28}, -0.5);
            auto expected = torch::sin(6*x)*torch::cos(20*x) + torch::sin(6*y)*torch::cos(20*y);
            auto f = -338*(torch::sin(26*x)+torch::sin(26*y))
                     +98*(torch::sin(14*x)+torch::sin(14*y));
            REQUIRE(torch::allclose(basis.matmul(w), expected, 1e-10, 1e-10));
            REQUIRE(torch::allclose(laplacian.matmul(w), f, 1e-10, 1e-8));
        }
    }
}

// Verifies forwardBasisPDE for poisson in 1D computes second derivative of each basis column.
TEST_CASE("PDE forwardBasisPDE poisson 1D matches analytic second derivative", "[PDE][Basis]") {
    PDE pde("poisson");

    auto X = torch::tensor({{0.0}, {0.25}, {0.5}}, torch::TensorOptions().dtype(torch::kDouble));
    X.requires_grad_(true);
    auto u = torch::sin(2.0 * M_PI * X); // column basis

    auto A = pde.forwardBasisPDE(u, X);
    auto expected = -(2.0 * M_PI) * (2.0 * M_PI) * torch::sin(2.0 * M_PI * X);

    REQUIRE(A.sizes() == expected.sizes());
    REQUIRE(torch::allclose(A, expected, kTol, kTol));
}

// Verifies forwardBasisPDE for onewaywave_inhomo in 2D computes u_t + 10 u_x.
TEST_CASE("PDE forwardBasisPDE onewaywave_inhomo matches analytic", "[PDE][Basis]") {
    PDE pde("onewaywave_inhomo");

    auto X = torch::tensor({{0.0, 0.0}, {0.25, 0.25}, {0.5, 0.5}},
                           torch::TensorOptions().dtype(torch::kDouble));
    X.requires_grad_(true);
    auto t = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
    auto x = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);

    auto u = torch::sin(2.0 * M_PI * t) + torch::sin(2.0 * M_PI * x);
    auto A = pde.forwardBasisPDE(u, X);

    auto expected = 2.0 * M_PI * torch::cos(2.0 * M_PI * t) + 10.0 * (2.0 * M_PI * torch::cos(2.0 * M_PI * x));
    REQUIRE(A.sizes() == expected.sizes());
    REQUIRE(torch::allclose(A, expected, kTol, kTol));
}

// Verifies forwardBasisPDE for nonss_allencahn linear part u_t - 0.0001 u_xx.
TEST_CASE("PDE forwardBasisPDE nonss_allencahn linear part matches analytic", "[PDE][Basis]") {
    PDE pde("nonss_allencahn");

    auto X = torch::tensor({{0.0, 0.0}, {0.25, 0.25}, {0.5, 0.5}},
                           torch::TensorOptions().dtype(torch::kDouble));
    X.requires_grad_(true);
    auto t = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
    auto x = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);

    auto u = torch::sin(2.0 * M_PI * t) + torch::sin(2.0 * M_PI * x);
    auto A = pde.forwardBasisPDE(u, X);

    auto u_t = 2.0 * M_PI * torch::cos(2.0 * M_PI * t);
    auto u_xx = -(2.0 * M_PI) * (2.0 * M_PI) * torch::sin(2.0 * M_PI * x);
    auto expected = u_t - 0.0001 * u_xx;

    REQUIRE(A.sizes() == expected.sizes());
    REQUIRE(torch::allclose(A, expected, kTol, kTol));
}
