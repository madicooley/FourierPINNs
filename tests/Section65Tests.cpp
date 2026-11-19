#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>
#include "fourierpinns/models/NeuralNet.h"
#include "fourierpinns/models/FourierNet.h"
#include "fourierpinns/basis/FourierBasisUtils.h"
#include "fourierpinns/training/utils_trainer.h"
#include "fourierpinns/training/StatsLogger.h"

namespace {
torch::Tensor waveSolution(const torch::Tensor& X) {
    auto t = X.index({torch::indexing::Slice(), 0}).unsqueeze(1);
    auto x = X.index({torch::indexing::Slice(), 1}).unsqueeze(1);
    return torch::sin(M_PI*x)*torch::cos(10*M_PI*t)
         + torch::sin(2*M_PI*x)*torch::cos(20*M_PI*t);
}
}

TEST_CASE("Section 6.5 generated wave data use time first and spatial boundaries only", "[Section65][PDE]") {
    torch::manual_seed(1234);
    LoadedPDEDomain domain(2, 37, 18, 43, 19, "test7", "onewaywave_inhomo", "dirichlet",
        "box", "random", false, true, {{0, 1}, {0, 1}}, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto col = domain.getPDEData();
    auto X = col.data.X->detach().clone().requires_grad_(true);
    auto u = waveSolution(X);
    auto grad = torch::autograd::grad({u}, {X}, {torch::ones_like(u)}, true, true)[0];
    auto forcing = (grad.index({torch::indexing::Slice(), 0})
                  + 10*grad.index({torch::indexing::Slice(), 1})).unsqueeze(1);
    REQUIRE(torch::allclose(*col.f, forcing, 1e-10, 1e-10));
    auto t = X.index({torch::indexing::Slice(), 0}).unsqueeze(1).detach().requires_grad_(true);
    auto x = X.index({torch::indexing::Slice(), 1}).unsqueeze(1).detach().requires_grad_(true);
    auto separated_u = torch::sin(M_PI*x)*torch::cos(10*M_PI*t)
                     + torch::sin(2*M_PI*x)*torch::cos(20*M_PI*t);
    PDE pde("onewaywave_inhomo");
    REQUIRE(torch::allclose(pde.forwardPDE(separated_u, t, x), forcing, 1e-10, 1e-10));
    auto bnd = domain.getBndData();
    auto bx = bnd.data.X->index({torch::indexing::Slice(), 1});
    REQUIRE(torch::logical_or(bx == 0, bx == 1).all().item<bool>());
    REQUIRE(bnd.data.X->size(0) == 18);
    REQUIRE(bnd.u->abs().max().item<double>() < 1e-12);
    auto ic = domain.getICData();
    REQUIRE(ic.data.X->size(0) == 19);
    REQUIRE(ic.data.X->index({torch::indexing::Slice(), 0}).abs().max().item<double>() == 0);
    auto ix = ic.data.X->index({torch::indexing::Slice(), 1}).unsqueeze(1);
    REQUIRE(torch::allclose(*ic.u, torch::sin(M_PI*ix)+torch::sin(2*M_PI*ix), 1e-12, 1e-12));
    auto test = domain.getTestData();
    REQUIRE(torch::allclose(*test.u, waveSolution(*test.data.X), 1e-12, 1e-12));
}

TEST_CASE("Section 6.5 hybrid LS matrix matches boundary initial and transport predictions", "[Section65][LSTSQ]") {
    torch::manual_seed(1234);
    ModelConfig cfg;
    cfg.model_type = "fouriernn";
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0, 1}, {0, 1}};
    FourierNetArgs arch;
    arch.neuralnet_args.width = 4;
    arch.neuralnet_args.layers = {2, 4, 1};
    arch.fourierlayer_args.ell = 2;
    arch.fourierlayer_args.include_zero_mode = true;
    arch.fourierlayer_args.scale_to_period = true;
    arch.fourierlayer_args.fourier_periods = {1, 2};
    cfg.arch = arch;
    VanillaFourierNeuralNet model(cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    LoadedPDEDomain domain(2, 37, 18, 43, 19, "test7", "onewaywave_inhomo", "dirichlet",
        "box", "random", false, true, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto layer_cfg = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(cfg, arch.fourierlayer_args);
    VanillaFourierLayer layer(layer_cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    domain.setPolyBasis(buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble), false);
    auto col = domain.getPDEData();
    auto bnd = domain.getBndData();
    auto ic = domain.getICData();
    PDE pde("onewaywave_inhomo");
    for (bool prune : {false, true}) {
        if (prune) {
            model->fouriernet->getBasisSelection().flatten().index_put_({1}, 0);
            model->neuralnet->getBasisSelection().flatten().index_put_({1}, 0);
        }
        for (bool cached_derivatives : {true, false}) {
            auto inputs = col;
            if (!cached_derivatives) {
                inputs.P_t = nullptr;
                inputs.P_x = nullptr;
            }
            auto A = model->getLSTSQ_A(inputs, bnd, &ic, &pde);
            auto coefficients = torch::cat({model->getPolyBasisCoeffs().flatten(),
                model->getNNetBias().flatten(), model->getNNetBasisCoeffs().flatten()});
            auto mask = torch::cat({model->getPolyBasisSelection().flatten(),
                model->neuralnet->getBiasSelection().flatten(),
                model->neuralnet->getBasisSelection().flatten()}).to(torch::kBool);
            auto predicted = torch::cat({model->forward(bnd.data), model->forward(ic.data),
                                         model->forward_pde(inputs, &pde)}, 0);
            REQUIRE(A.size(0) == 18+19+37);
            REQUIRE(torch::allclose(A.matmul(coefficients.index({mask}).unsqueeze(1)),
                                   predicted, 1e-9, 1e-9));
        }
    }
}

TEST_CASE("Explicit Fourier periods exactly include the wave solution and its derivatives", "[Section65][FourierPeriods]") {
    ModelConfig cfg;
    cfg.model_type = "fouriernn";
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0, 1}, {0, 1}};
    FourierNetArgs arch;
    arch.neuralnet_args.width = 4;
    arch.neuralnet_args.layers = {2, 4, 1};
    arch.fourierlayer_args.ell = 10;
    arch.fourierlayer_args.include_zero_mode = true;
    arch.fourierlayer_args.fourier_periods = {1, 2};
    cfg.arch = arch;
    VanillaFourierNeuralNet model(cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    {
        torch::NoGradGuard guard;
        for (auto& parameter : model->parameters()) parameter.zero_();
        auto weights = model->fouriernet->named_parameters()["trunc_neuralnet.weight"].flatten();
        // Per-axis order: constant, cos(1), sin(1), ..., cos(10), sin(10).
        weights.index_put_({9*21+2}, 1);  // cos(10*pi*t) sin(pi*x)
        weights.index_put_({19*21+4}, 1); // cos(20*pi*t) sin(2*pi*x)
    }
    LoadedPDEDomain domain(2, 37, 18, 43, 19, "test7", "onewaywave_inhomo", "dirichlet",
        "box", "random", false, true, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto layer_cfg = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(cfg, arch.fourierlayer_args);
    VanillaFourierLayer layer(layer_cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    domain.setPolyBasis(buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble), false);
    auto col = domain.getPDEData();
    PDE pde("onewaywave_inhomo");
    REQUIRE(torch::allclose(model->forward_pde(col, &pde), *col.f, 1e-10, 1e-10));
    TrainingConfig training;
    training.test_batch_size = 7;
    StatsLogger<VanillaFourierNeuralNetImpl> logger(training, &domain, model.get());
    REQUIRE(torch::allclose(logger.getTestPred(), domain.getTestingSolution(), 1e-12, 1e-12));
    REQUIRE(logger.getTestRelativeError() < 1e-12);
}
