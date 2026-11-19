#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

struct Args;
class ArgParser;

namespace fourierpinns {

enum class ModelKind {
    NeuralNet,
    FourierNN,
    FourierLayer
};

class ModelFactory {
public:
    static ModelKind fromModelName(const std::string& model_name);
};

struct SolverRunContext {
    std::string command;
    std::string datadir;
    std::string data_mode;
    std::string filename_root;
    std::string modelfname;
    bool load_trained = false;
    bool solve_pde = false;
};

class ISolverRunner {
public:
    virtual ~ISolverRunner() = default;
    virtual void run(const Args& args,
                     torch::Device device,
                     torch::Dtype dtype,
                     ArgParser& parser,
                     const SolverRunContext& ctx) = 0;
};

class SolverFactory {
public:
    static std::unique_ptr<ISolverRunner> create(ModelKind kind);
};

} // namespace fourierpinns
