#include "CLI.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "fourierpinns/config/ArgParser.h"
#include "fourierpinns/solvers/SolverFactory.h"

namespace {

class ScopedCoutRedirect {
public:
    bool enable(const std::string& path) {
        out_file_.open(path);
        if (!out_file_.is_open()) return false;
        old_buf_ = std::cout.rdbuf(out_file_.rdbuf());
        active_ = true;
        return true;
    }

    void disable() {
        if (!active_) return;
        std::cout.rdbuf(old_buf_);
        out_file_.close();
        active_ = false;
        old_buf_ = nullptr;
    }

    bool isActive() const {
        return active_;
    }

    ~ScopedCoutRedirect() {
        disable();
    }

private:
    std::ofstream out_file_;
    std::streambuf* old_buf_ = nullptr;
    bool active_ = false;
};

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  fourierpinns [train|gen-data|validate-config|help] --key=value ...\n\n"
        << "Examples:\n"
        << "  fourierpinns train --model=neuralnet --datadir=runs/tmp/ --expid=run1\n"
        << "  fourierpinns gen-data --model=fouriernn --datadir=runs/tmp/ --expid=run2 --data_mode=generate\n"
        << "  fourierpinns validate-config --config=/path/to/config.json --datadir=runs/tmp/ --expid=dryrun\n";
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string parseCommand(int argc, char* argv[], int& first_arg_idx) {
    first_arg_idx = 1;
    if (argc < 2) return "train";

    const std::string first = argv[1];
    if (first.rfind("--", 0) == 0 || first.find('=') != std::string::npos) {
        return "train";
    }

    first_arg_idx = 2;
    return toLower(first);
}

void validateCommandAndArgs(const std::string& command, const std::string& model) {
    const bool valid_cmd = (command == "train" || command == "gen-data" || command == "validate-config");
    if (!valid_cmd) {
        throw std::invalid_argument("Unknown command: " + command + ". Use help for usage.");
    }

    (void)fourierpinns::ModelFactory::fromModelName(model);
}

void validateCrossFieldCombinations(const std::string& command,
                                    const std::string& data_mode,
                                    bool solve_pde,
                                    bool load_trained) {
    if (command == "gen-data" && solve_pde) {
        throw std::invalid_argument(
            "Invalid option combination: command 'gen-data' cannot be used with --solve_pde=true. "
            "Use 'train' to solve the PDE or set --solve_pde=false.");
    }
    if (command == "gen-data" && load_trained) {
        throw std::invalid_argument(
            "Invalid option combination: command 'gen-data' cannot be used with --load_pretrn_model=true.");
    }
    if (command == "gen-data" && data_mode != "generate") {
        throw std::invalid_argument(
            "Invalid option combination: command 'gen-data' requires --data_mode=generate.");
    }
}

std::string getConfigPathFromArgv(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto pos = arg.find('=');
        if (pos == std::string::npos) continue;
        const std::string name = arg.substr(0, pos);
        if (name == "--config") return arg.substr(pos + 1);
    }
    return "";
}

void validateStrictConfigSchema(const std::string& config_path) {
    std::ifstream in(config_path);
    if (!in) {
        throw std::invalid_argument("Failed to open config file: " + config_path);
    }

    nlohmann::json j;
    in >> j;
    if (!j.is_object()) {
        throw std::invalid_argument("Config file must contain a JSON object at root.");
    }

    static const std::unordered_set<std::string> allowed_keys = {
        "lstsq_rcond", "lstsq_stop_epoch", "lstsq_safeguard",
        "model", "datadir", "expid", "gpu_device", "solve_pde", "write_logs", "load_pretrn_model",
        "data_mode", "print_all_args", "print_relevant_args", "include_zero_mode", "scale_to_period",
        "fourier_periods", "periodic_deriv_dim", "dtype", "lr_decaytype", "lbfgs_min_lr", "lbfgs_max_lr", "lbfgs_max_epochs",
        "lbfgs_history_size", "lbfgs_max_iter", "lbfgs_tolerance_grad", "lbfgs_tolerance_change", "lbfgs_line_search_fn",
        "adam_min_lr", "adam_max_lr", "adam_max_epochs", "adam_eps", "seed", "dim",
        "lstsq_updates", "lstsq_upd_interval", "n_lstsq_upds", "n_lstsq_upds_per_epoch", "col_batch_size",
        "col_batch_shuffle", "training_order", "pde_problem", "Ncol_pts", "Nbnd_pts", "Ntest_pts", "Nic_pts",
        "bnd_typ", "point_sampling", "test_func", "dim1_min", "dim1_max", "dim2_min", "dim2_max",
        "initial_cond", "homogeneous_pde", "depth", "width", "net_act",
        "nn_l1_reg_coeff", "nn_l2_reg_coeff", "nn_trunc_thresh", "nn_truncate", "truncate_interval", "ell",
        "pn_l1_reg_coeff", "pn_l2_reg_coeff", "pn_trunc_thresh", "pn_truncate", "verbose",
        "lambda_upd_interval", "lambda_bnd", "lambda_ic", "lambda_res",
        "loss_reweight_scheme", "lr_decay_interval", "lr_warmup_steps", "lr_decay_steps", "lr_decay_rate",
        "pred_save_interval", "print_model_trn_info", "print_lr_sched", "print_interval", "test_batch_size"
    };

    static const std::unordered_map<std::string, std::unordered_set<std::string>> enum_rules = {
        {"model", {"neuralnet", "fourierlayer", "fouriernn"}},
        {"data_mode", {"generate", "load_all", "load_test_only"}},
        {"dtype", {"float32", "float64"}},
        {"lr_decaytype", {"cosine", "exp", "none"}},
        {"training_order", {"adamfirst", "lbfgsfirst", "adamonly", "lbfgsonly"}},
        {"pde_problem", {"poisson", "allencahn", "nonss_allencahn", "onewaywave_inhomo"}},
        {"bnd_typ", {"dirichlet", "periodic"}},
        {"point_sampling", {"equispaced", "random"}},
        {"net_act", {"tanh", "relu", "sigmoid", "elu", "repu", "softplus", "gelu", "selu", "swish"}},
        {"loss_reweight_scheme", {"none", "grad"}},
        {"lbfgs_line_search_fn", {"strong_wolfe"}}
    };

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = it.key();
        if (allowed_keys.find(key) == allowed_keys.end()) {
            throw std::invalid_argument("Unknown config key in strict schema: " + key);
        }

        const auto enum_it = enum_rules.find(key);
        if (enum_it == enum_rules.end()) continue;

        if (!it.value().is_string()) {
            throw std::invalid_argument("Config key '" + key + "' must be a string enum value.");
        }
        const std::string val = it.value().get<std::string>();
        if (enum_it->second.find(val) == enum_it->second.end()) {
            throw std::invalid_argument("Invalid enum value for config key '" + key + "': " + val);
        }
    }
}

std::string makeFilenameRoot(const std::string& expid, const std::string& model, const std::string& problem) {
    return expid + "_" + model + "_" + problem;
}

std::string writeResolvedSnapshot(const ArgParser& parser,
                                  const std::string& command,
                                  const std::string& datadir,
                                  const std::string& filename_root) {
    const std::filesystem::path args_dir = std::filesystem::path(datadir) / "args";
    std::filesystem::create_directories(args_dir);

    const std::filesystem::path snapshot_path = args_dir / ("resolved_config_" + filename_root + ".json");

    std::map<std::string, std::string> resolved_sorted;
    for (const auto& pair : parser.getResolvedArguments()) {
        std::string key = pair.first;
        if (key.rfind("--", 0) == 0) key = key.substr(2);
        resolved_sorted[key] = pair.second;
    }

    nlohmann::json snapshot;
    snapshot["metadata"] = {
        {"command", command},
        {"filename_root", filename_root}
    };
    snapshot["resolved_args"] = resolved_sorted;

    std::ofstream out(snapshot_path);
    if (!out.is_open()) {
        throw std::invalid_argument("Failed to write resolved config snapshot: " + snapshot_path.string());
    }
    out << snapshot.dump(2) << "\n";
    return snapshot_path.string();
}

} // namespace

int runFourierPINNsCLI(int argc, char* argv[]) {
    try {
        const std::string config_path = getConfigPathFromArgv(argc, argv);
        if (!config_path.empty()) {
            validateStrictConfigSchema(config_path);
        }

        int first_arg_idx = 1;
        const std::string command = parseCommand(argc, argv, first_arg_idx);
        if (command == "help") {
            printUsage();
            return 0;
        }

        std::vector<std::string> forwarded;
        forwarded.reserve(static_cast<size_t>(argc - first_arg_idx + 1));
        forwarded.emplace_back(argv[0]);
        for (int i = first_arg_idx; i < argc; ++i) {
            forwarded.emplace_back(argv[i]);
        }
        std::vector<char*> forwarded_argv;
        forwarded_argv.reserve(forwarded.size());
        for (std::string& s : forwarded) {
            forwarded_argv.push_back(const_cast<char*>(s.c_str()));
        }

        ArgParser parser;
        generateMyExperimentArgs(parser, command != "gen-data");
        parser.parse(static_cast<int>(forwarded_argv.size()), forwarded_argv.data());
        const Args& args = parser.getArgs();

        const std::string datadir   = parser.getArgument("--datadir");
        std::string expid           = parser.getArgument("--expid");
        const std::string model     = parser.getArgument("--model");
        const std::string problem   = "forwardpde";
        const std::string arg_dtype = parser.getArgument("--dtype");
        const std::string data_mode = parser.getArgument("--data_mode");
        const bool solve_pde       = parser.getBoolArgument("--solve_pde");
        const bool load_trained    = parser.getBoolArgument("--load_pretrn_model");
        const bool write_logs      = parser.getBoolArgument("--write_logs");
        const bool print_all_args  = parser.getBoolArgument("--print_all_args");
        const bool print_rel_args  = parser.getBoolArgument("--print_relevant_args");

        validateCommandAndArgs(command, model);
        validateCrossFieldCombinations(command, data_mode, solve_pde, load_trained);
        if (command == "gen-data") {
            const auto data_dir = std::filesystem::path(datadir) / "data";
            if (std::filesystem::exists(data_dir)) {
                throw std::invalid_argument("Refusing to overwrite existing dataset directory: " + data_dir.string());
            }
        }

        const std::string filename_root = makeFilenameRoot(expid, model, problem);
        const std::string resolved_snapshot_path = writeResolvedSnapshot(parser, command, datadir, filename_root);
        if (command == "validate-config") {
            std::cout << "Config validation successful.\n";
            std::cout << "  command=" << command
                      << " model=" << model
                      << " problem=" << problem
                      << " datadir=" << datadir
                      << " expid=" << expid
                      << " dtype=" << arg_dtype
                      << " data_mode=" << data_mode
                      << "\n";
            std::cout << "Resolved config snapshot: " << resolved_snapshot_path << "\n";
            return 0;
        }

        std::string train_filename_root = filename_root;
        std::string modelfname;
        if (load_trained) {
            modelfname = (std::filesystem::path(datadir) / "models" / ("models_" + train_filename_root + ".pt")).string();
            expid += "_retrn";
            train_filename_root = makeFilenameRoot(expid, model, problem);
        }

        const std::string log_fname = (std::filesystem::path(datadir) / "logs" / ("logs_" + train_filename_root + ".txt")).string();
        ScopedCoutRedirect log_redirect;
        if (write_logs) {
            std::filesystem::create_directories(std::filesystem::path(log_fname).parent_path());
            if (!log_redirect.enable(log_fname)) {
                std::cerr << "Failed to open file for writing: " << log_fname << "\n";
                return 1;
            }
        }

        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            const int gpu_device = parser.getIntArgument("--gpu_device");
            std::cout << "CUDA is available! Using GPU " << gpu_device << "\n";
            device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(gpu_device));
        }

        torch::Dtype dtype = (arg_dtype == "float64") ? torch::kFloat64 : torch::kFloat32;
        std::cout << "DEVICE: " << device << " DTYPE: " << dtype << "\n";
        std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << "\n";

        const std::string args_fname = (std::filesystem::path(datadir) / "args" / ("args_" + train_filename_root + ".txt")).string();
        if (print_all_args) parser.printAllArgs(/*saveargs=*/true, args_fname);
        if (print_rel_args) parser.printRelArgs();

        const fourierpinns::ModelKind model_kind = fourierpinns::ModelFactory::fromModelName(model);
        std::unique_ptr<fourierpinns::ISolverRunner> solver_runner = fourierpinns::SolverFactory::create(model_kind);
        fourierpinns::SolverRunContext run_ctx;
        run_ctx.command = command;
        run_ctx.datadir = datadir;
        run_ctx.data_mode = data_mode;
        run_ctx.filename_root = train_filename_root;
        run_ctx.modelfname = modelfname;
        run_ctx.load_trained = load_trained;
        run_ctx.solve_pde = solve_pde;
        solver_runner->run(args, device, dtype, parser, run_ctx);

        if (log_redirect.isActive()) {
            log_redirect.disable();
            std::cout << "Logs written to: " << log_fname << "\n";
        }
        std::cout << "Resolved config snapshot: " << resolved_snapshot_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fourierpinns error: " << ex.what() << "\n";
        std::cerr << "Run `fourierpinns help` for usage.\n";
        return 1;
    }
}
