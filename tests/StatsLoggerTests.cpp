#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>
#include "fourierpinns/models/NeuralNet.h"
#include "fourierpinns/models/FourierNet.h"
#include "fourierpinns/basis/FourierBasisUtils.h"
#include "fourierpinns/training/utils_trainer.h"
#include "fourierpinns/training/StatsLogger.h"

namespace {
template<class Model>
void checkEvaluation(Model& model, LoadedPDEDomain& domain, torch::Tensor& basis) {
    auto test = domain.getTestData();
    auto inputs = test.data;
    inputs.P.basis = &basis;
    torch::Tensor reference;
    {
        torch::NoGradGuard guard;
        reference = model->forward(inputs);
    }
    const double error = ((reference-*test.u).norm()/test.u->norm()).template item<double>();
    TrainingConfig training;
    StatsLogger<typename Model::ContainedType> logger(training, &domain, model.get());
    for (int batch : {1, 5, 100}) {
        training.test_batch_size = batch;
        StatsLogger<typename Model::ContainedType> chunked(training, &domain, model.get());
        model->train();
        auto predicted = chunked.getTestPred();
        REQUIRE(model->is_training());
        REQUIRE_FALSE(predicted.requires_grad());
        REQUIRE(torch::allclose(predicted, reference, 1e-10, 1e-10));
        REQUIRE(std::abs(chunked.getTestRelativeError()-error) < 1e-10);
        model->eval();
        chunked.getTestRelativeError();
        REQUIRE_FALSE(model->is_training());
    }
    const auto directory = (std::filesystem::path(__FILE__).parent_path().parent_path()
                            / "tmp/chunked_test_eval").string()+"/";
    REQUIRE(logger.saveRgrPredictions("ordered", directory) == 0);
    torch::Tensor saved;
    torch::load(saved, directory+"predictions/predictions_u_ordered.pt");
    REQUIRE(torch::allclose(saved, reference, 1e-10, 1e-10));
}

struct BatchProbe : VanillaNeuralNetImpl {
    using VanillaNeuralNetImpl::VanillaNeuralNetImpl;
    int64_t largest_batch = 0;
    bool fail = false;
    torch::Tensor forward(const DataInputs& inputs) override {
        REQUIRE_FALSE(torch::GradMode::is_enabled());
        REQUIRE_FALSE(is_training());
        largest_batch = std::max(largest_batch, inputs.X->size(0));
        if (fail) throw std::runtime_error("evaluation failure");
        return VanillaNeuralNetImpl::forward(inputs);
    }
};
}

TEST_CASE("2D chunked evaluation matches full predictions before and after pruning", "[StatsLogger][ChunkedTest]") {
    torch::manual_seed(1234);
    ModelConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0, 2*M_PI}, {0, 2*M_PI}};
    NeuralNetArgs nn;
    nn.layers = {2, 4, 1};
    nn.width = 4;
    nn.nn_truncate = true;
    nn.nn_trunc_thresh = 0.1;
    FourierLayerArgs fl;
    fl.ell = 2;
    fl.include_zero_mode = true;
    fl.scale_to_period = true;
    fl.fourier_periods = {3, 5};
    fl.pn_truncate = true;
    fl.pn_trunc_thresh = 0.1;
    LoadedPDEDomain domain(2, 37, 16, 13, 0, "test5", "poisson", "dirichlet",
        "box", "random", false, false, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    cfg.model_type = "fourierlayer";
    cfg.arch = fl;
    VanillaFourierLayer layer(cfg, "poisson", torch::kCPU, torch::kDouble);
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    REQUIRE(bases.count("P_test") == 0);
    domain.setPolyBasis(bases, false);
    REQUIRE(domain.getTestData().data.P.basis == nullptr);
    auto basis = layer->buildBasisFromData(domain.getTestingNodes());
    cfg.model_type = "fouriernn";
    FourierNetArgs hybrid;
    hybrid.neuralnet_args = nn;
    hybrid.fourierlayer_args = fl;
    cfg.arch = hybrid;
    VanillaFourierNeuralNet net(cfg, "poisson", torch::kCPU, torch::kDouble);
    cfg.model_type = "neuralnet";
    cfg.arch = nn;
    cfg.layers = nn.layers;
    VanillaNeuralNet neural(cfg, "poisson", torch::kCPU, torch::kDouble);
    checkEvaluation(layer, domain, basis);
    checkEvaluation(net, domain, basis);
    checkEvaluation(neural, domain, basis);
    {
        torch::NoGradGuard guard;
        auto weights = layer->named_parameters()["trunc_neuralnet.weight"];
        weights.fill_(0.5);
        weights.flatten().index_put_({0}, 0.01);
        net->fouriernet->named_parameters()["trunc_neuralnet.weight"].copy_(weights);
    }
    layer->truncateCoeffs();
    net->truncateCoeffs();
    REQUIRE(layer->getNRemainingBases() < basis.size(1));
    checkEvaluation(layer, domain, basis);
    checkEvaluation(net, domain, basis);
}

TEST_CASE("2D evaluation bounds batches and restores mode on failure", "[StatsLogger][ChunkedTest]") {
    ModelConfig cfg;
    cfg.model_type = "neuralnet";
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    NeuralNetArgs nn;
    nn.layers = {2, 4, 1};
    nn.width = 4;
    cfg.arch = nn;
    cfg.layers = nn.layers;
    BatchProbe model(cfg, "poisson", torch::kCPU, torch::kDouble);
    LoadedPDEDomain domain(2, 17, 16, 13, 0, "test5", "poisson", "dirichlet",
        "box", "random", false, false, {{0, 2*M_PI}, {0, 2*M_PI}}, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    TrainingConfig training;
    training.test_batch_size = 5;
    StatsLogger<BatchProbe> logger(training, &domain, &model);
    REQUIRE(logger.getTestPred().size(0) == 13);
    REQUIRE(std::isfinite(logger.getTestRelativeError()));
    REQUIRE(model.largest_batch == 5);
    model.fail = true;
    REQUIRE_THROWS(logger.getTestPred());
    REQUIRE(model.is_training());
}

TEST_CASE("1D Fourier test basis remains cached", "[StatsLogger][ChunkedTest]") {
    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    cfg.domain_bounds = {{0, 2*M_PI}};
    FourierLayerArgs fl;
    fl.ell = 2;
    cfg.arch = fl;
    VanillaFourierLayer layer(cfg, "poisson", torch::kCPU, torch::kDouble);
    LoadedPDEDomain domain(1, 17, 4, 13, 0, "test1_k2", "poisson", "dirichlet",
        "box", "equispaced", false, false, cfg.domain_bounds, torch::kCPU, torch::kDouble);
    domain.generateDomainData();
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    REQUIRE(bases.count("P_test") == 1);
    domain.setPolyBasis(bases, false);
    REQUIRE(domain.getTestData().data.P.basis != nullptr);
    TrainingConfig training;
    training.test_batch_size = 1;
    StatsLogger<VanillaFourierLayerImpl> logger(training, &domain, layer.get());
    REQUIRE(torch::allclose(logger.getTestPred(), layer->forward(domain.getTestData().data)));
}
