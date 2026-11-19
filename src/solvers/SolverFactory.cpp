#include "fourierpinns/solvers/SolverFactory.h"

#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/pdes/PDEs.h"
#include "fourierpinns/models/CustomNet.h"
#include "fourierpinns/training/Trainer.h"
#include "fourierpinns/models/FourierLayer.h"
#include "fourierpinns/models/NeuralNet.h"
#include "fourierpinns/models/FourierNet.h"
#include "fourierpinns/config/Args.h"
#include "fourierpinns/config/ArgParser.h"
#include "fourierpinns/solvers/Solver.h"
#include "fourierpinns/solvers/StandardPINNSolver.h"
#include "fourierpinns/solvers/FourierPINNSolver.h"
#include "fourierpinns/solvers/FourierPILayerSolver.h"

namespace fourierpinns {

ModelKind ModelFactory::fromModelName(const std::string& model_name) {
    if (model_name == "neuralnet") return ModelKind::NeuralNet;
    if (model_name == "fouriernn") return ModelKind::FourierNN;
    if (model_name == "fourierlayer") return ModelKind::FourierLayer;
    throw std::invalid_argument("Unsupported --model for fourierpinns: " + model_name);
}

namespace {

bool prepareDataset(PDESolver& solver, const SolverRunContext& ctx) {
    const auto& data_mode = ctx.data_mode;
    const auto& datadir = ctx.datadir;
    if (data_mode == "generate") {
        solver.generateDomainData();
        if (ctx.command == "gen-data") {
            solver.saveDomainData(datadir);
            return false;
        }
    } else if (data_mode == "load_all") {
        solver.loadDomainsFromFile(datadir);
    } else if (data_mode == "load_test_only") {
        solver.loadDomainsFromFile(datadir, /*test_only=*/true);
        solver.generateDomainData(/*gen_test=*/false, /*gen_bnd=*/true, /*gen_ic=*/true, /*gen_col=*/true);
    } else {
        throw std::invalid_argument("Unknown data_mode: " + data_mode);
    }
    return true;
}

class StandardSolverRunner final : public ISolverRunner {
public:
    void run(const Args& args,
             torch::Device device,
             torch::Dtype dtype,
             ArgParser& parser,
             const SolverRunContext& ctx) override {
        StandardPINNSolver solver(args, device, dtype);
        if (!prepareDataset(solver, ctx)) return;
        if (ctx.command == "train" && ctx.load_trained) solver.loadModel(ctx.modelfname);
        if (ctx.command == "train" && ctx.solve_pde) {
            solver.solvePDE(args.training_args, parser, /*saveStats=*/true, ctx.filename_root, ctx.datadir);
        }
    }
};

class FourierNNSolverRunner final : public ISolverRunner {
public:
    void run(const Args& args,
             torch::Device device,
             torch::Dtype dtype,
             ArgParser& parser,
             const SolverRunContext& ctx) override {
        FourierPINNSolver solver(args, device, dtype);
        if (!prepareDataset(solver, ctx)) return;
        solver.setPolyBasis(args);
        if (ctx.command == "train" && ctx.load_trained) solver.loadModel(ctx.modelfname);
        if (ctx.command == "train" && ctx.solve_pde) {
            solver.solvePDE(args.training_args, parser, /*saveStats=*/true, ctx.filename_root, ctx.datadir);
        }
    }
};

class FourierLayerSolverRunner final : public ISolverRunner {
public:
    void run(const Args& args,
             torch::Device device,
             torch::Dtype dtype,
             ArgParser& parser,
             const SolverRunContext& ctx) override {
        FourierPILayerSolver solver(args, device, dtype);
        if (!prepareDataset(solver, ctx)) return;
        solver.setPolyBasis(args);
        if (ctx.command == "train" && ctx.load_trained) solver.loadModel(ctx.modelfname);
        if (ctx.command == "train" && ctx.solve_pde) {
            solver.solvePDE(args.training_args, parser, /*saveStats=*/true, ctx.filename_root, ctx.datadir);
        }
    }
};

} // namespace

std::unique_ptr<ISolverRunner> SolverFactory::create(ModelKind kind) {
    if (kind == ModelKind::NeuralNet) return std::make_unique<StandardSolverRunner>();
    if (kind == ModelKind::FourierNN) return std::make_unique<FourierNNSolverRunner>();
    if (kind == ModelKind::FourierLayer) return std::make_unique<FourierLayerSolverRunner>();
    throw std::invalid_argument("Unsupported model kind for solver factory.");
}

} // namespace fourierpinns
