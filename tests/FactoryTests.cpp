#include <catch2/catch_test_macros.hpp>

#include "fourierpinns/solvers/SolverFactory.h"

using fourierpinns::ISolverRunner;
using fourierpinns::ModelFactory;
using fourierpinns::ModelKind;
using fourierpinns::SolverFactory;

TEST_CASE("ModelFactory maps supported model names", "[Factory]") {
    REQUIRE(ModelFactory::fromModelName("neuralnet") == ModelKind::NeuralNet);
    REQUIRE(ModelFactory::fromModelName("fouriernn") == ModelKind::FourierNN);
    REQUIRE(ModelFactory::fromModelName("fourierlayer") == ModelKind::FourierLayer);
}

TEST_CASE("ModelFactory rejects unsupported model names", "[Factory]") {
    REQUIRE_THROWS(ModelFactory::fromModelName("unknown_model"));
}

TEST_CASE("SolverFactory creates runners for each model kind", "[Factory]") {
    std::unique_ptr<ISolverRunner> standard = SolverFactory::create(ModelKind::NeuralNet);
    std::unique_ptr<ISolverRunner> fouriernn = SolverFactory::create(ModelKind::FourierNN);
    std::unique_ptr<ISolverRunner> fourierlayer = SolverFactory::create(ModelKind::FourierLayer);

    REQUIRE(standard != nullptr);
    REQUIRE(fouriernn != nullptr);
    REQUIRE(fourierlayer != nullptr);
}
