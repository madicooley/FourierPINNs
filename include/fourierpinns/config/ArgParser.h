#pragma once
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <functional>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>  // For setprecision, std::setw
#include <map>
#include <ctime>
#include <chrono>
#include <math.h> // for M_PI
#include <cmath>

#include <nlohmann/json.hpp>

#include "fourierpinns/config/Args.h"

inline std::vector<int> getLayerList(int input_dim, int output_dim, int net_width, int net_depth) {
    std::vector<int> layers = {};
    layers.push_back(input_dim);

    for (int i=0; i<net_depth; i++) {
        layers.push_back(net_width);
    }
    layers.push_back(output_dim);  // output dimension
    return layers;
}


inline bool isValidDouble(const std::string& val) {
    try {
        std::size_t pos;
        (void)std::stod(val, &pos);
        return pos == val.length();
    } catch (const std::invalid_argument& e) {
        return false;
    } catch (const std::out_of_range& e) {
        return false;
    }
}

inline bool isValidInt(const std::string& val) {
    try {
        std::size_t pos;
        (void)std::stoi(val, &pos);
        return pos == val.length(); // Ensure entire string was processed
    } catch (const std::invalid_argument& e) {
        return false; // Not a valid integer
    } catch (const std::out_of_range& e) {
        return false; // Value out of int range
    }
}

inline bool isValidFloat(const std::string& val) {
    try {
        std::size_t pos;
        (void)std::stof(val, &pos);
        return pos == val.length(); // Ensure the entire string was processed
    } catch (const std::invalid_argument& e) {
        return false; // Not a valid float
    } catch (const std::out_of_range& e) {
        return false; // Value out of float range
    }
}

inline std::vector<double> parseFourierPeriods(const std::string& value) {
    std::vector<double> result;
    if (!value.empty() && value.front() == '[') {
        auto array = nlohmann::json::parse(value);
        if (!array.is_array()) throw std::invalid_argument("fourier_periods must be an array");
        for (const auto& entry : array) {
            if (!entry.is_number()) throw std::invalid_argument("fourier_periods requires numbers");
            result.push_back(entry.get<double>());
        }
    } else {
        if (value.empty() || value.back() == ',') throw std::invalid_argument("Empty Fourier period");
        std::istringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ',')) {
            const auto first = token.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) throw std::invalid_argument("Empty Fourier period");
            token = token.substr(first, token.find_last_not_of(" \t\r\n")-first+1);
            size_t used = 0;
            const double period = std::stod(token, &used);
            if (used != token.size()) throw std::invalid_argument("Invalid Fourier period");
            result.push_back(period);
        }
    }
    if (result.empty()) throw std::invalid_argument("fourier_periods must not be empty");
    for (double period : result) {
        if (!std::isfinite(period) || period <= 0) throw std::invalid_argument("Fourier periods must be finite and positive");
    }
    return result;
}

inline bool isValidBool(const std::string& val) {
    // Convert string to lower case for case-insensitive comparison
    std::string lowerVal = val;
    std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    return lowerVal == "true" || lowerVal == "false" ||
           lowerVal == "1" || lowerVal == "0";
}


class ArgParser {
public:
    ArgParser() : args_initialized(false) { }

    // Function type for validation
    using ValidatorFunc = std::function<bool(const std::string&)>;

    // Add an argument with itexp_argss validator
    void addArgument(const std::string& argName,
                     ValidatorFunc validator,
                     const std::string& description,
                     bool isRequired = false) {
        validators[argName] = validator;
        descriptions[argName] = description;
        if (isRequired) {
            requiredArgs.insert(argName);
        }
    }

    void addArgument(const std::string& argName,
                     ValidatorFunc validator,
                     bool isRequired = false) {
        validators[argName] = validator;
        if (isRequired) {
            requiredArgs.insert(argName);
        }
    }

    void addInitArgument(const std::string& argName, ValidatorFunc validator, bool value ) {
        validators[argName]=validator;
        arguments[argName]=std::to_string(value);
    }

    // Parse and validate arguments
    void parse(int argc, char* argv[]) {
        std::unordered_set<std::string> suppliedArgs;

        // Precedence: defaults < config file < explicit CLI args.
        // Load config first, then apply CLI args in the second pass.
        for (int i = 1; i < argc; ++i) {
            std::string argStr(argv[i]);
            auto pos = argStr.find('=');
            if (pos == std::string::npos) continue;
            std::string name = argStr.substr(0, pos);
            std::string value = argStr.substr(pos + 1);
            if (name == "--config") {
                auto cfg = loadConfigFile(value);
                applyConfigMap(cfg, suppliedArgs);
                arguments[name] = value;
            }
        }

        for (int i = 1; i < argc; ++i) {
            std::string argStr(argv[i]);
            full_input_arg_str+=argStr+" ";

            auto pos = argStr.find('=');
            if (pos != std::string::npos) {
                std::string name = argStr.substr(0, pos);
                std::string value = argStr.substr(pos + 1);

                if (validators.find(name) == validators.end()) {
                    throw std::invalid_argument("Unknown argument: " + name);
                }

                if (!validators[name](value)) {
                    throw std::invalid_argument("Invalid value for argument: " + name);
                }

                arguments[name] = value;
                suppliedArgs.insert(name);
            } else {
                throw std::invalid_argument("Invalid argument format");
            }
        }

        std::vector<std::string> missingArgs;
        for (const auto& reqArg : requiredArgs) {
            if (arguments.find(reqArg) == arguments.end()) {
                missingArgs.push_back(reqArg);
            }
        }

        if (!missingArgs.empty()) {
            std::string errorMsg = "Missing required arguments:\n";
            for (const auto& arg : missingArgs) {
                errorMsg += arg + ": " + descriptions[arg] + "\n";
            }

            errorMsg += "\nValid required options are:\n";
            for (const auto& arg : missingArgs) {
                errorMsg += arg + ": " + descriptions[arg] + "\n";
            }

            throw std::invalid_argument(errorMsg);
        }

        setArgumentStruct();
    }

    void parseConfigMap(const std::unordered_map<std::string, std::string>& config_map) {
        std::unordered_set<std::string> suppliedArgs;

        std::cout<<"parseConfigMap.\n";
        for (const auto& pair : config_map) {
            std::string name  = "--"+pair.first;
            std::string value = pair.second;

            std::cout<<"name="<<name<<", value="<<value<<"\n";
            if (validators.find(name) == validators.end()) {
                throw std::invalid_argument("Unknown argument: " + name);
            }

            if (!validators[name](value)) {
                throw std::invalid_argument("Invalid value for argument: " + name);
            }

            arguments[name] = value;
            suppliedArgs.insert(name);
        }

        std::vector<std::string> missingArgs;
        for (const auto& reqArg : requiredArgs) {
            if (suppliedArgs.find(reqArg) == suppliedArgs.end()) {
                missingArgs.push_back(reqArg);
            }
        }

        if (!missingArgs.empty()) {
            std::string errorMsg = "Missing required arguments:\n";
            for (const auto& arg : missingArgs) {
                errorMsg += arg + ": " + descriptions[arg] + "\n";
            }

            errorMsg += "\nValid required options are:\n";
            for (const auto& arg : missingArgs) {
                errorMsg += arg + ": " + descriptions[arg] + "\n";
            }

            throw std::invalid_argument(errorMsg);
        }

        setArgumentStruct();
    }

    // Get parsed argument value
    std::string getArgument(const std::string& argName) {
        if (arguments.find(argName) != arguments.end()) {
            return arguments[argName];
        }
        throw std::invalid_argument("Argument not found: " + argName);
    }

    const Args& getArgs() const {
        if (!args_initialized) std::cerr<<"Args not initialized.\n";
        return this->args;
    }

    const std::unordered_map<std::string, std::string>& getResolvedArguments() const {
        return arguments;
    }

    std::string getCurrentDateTime() {
        // Function to get the current date and time as a formatted string
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time);

        std::ostringstream oss;
        oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S"); // Format: "YYYY-MM-DD HH:MM:SS"
        return oss.str();
    }

    int saveArgCSVLine(double final_error, double final_loss, double final_traintime, int final_iter, std::string filename_root, std::string datadir) {
        std::cout<<"Saving Arg CSV line: final_error="<<final_error<<", final_loss="<<final_loss<<", final_traintime="<<final_traintime<<"\n";

        std::string filename = (std::filesystem::path(datadir) / "csvs" / ("csvs_" + filename_root + ".txt")).string();
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing. Filename-> " << filename << std::endl;
            return 1;
        }
        std::string timestamp = getCurrentDateTime();

        for (const auto& pair : arguments) {
            outFile<<pair.first<<", ";
        }
        outFile<<"final_error, final_loss, final_traintime, final_iter,Timestamp"<<std::endl;
        for (const auto& pair : arguments) {
            outFile<<pair.second<<", ";
        }
        outFile<<final_error<<", "<<final_loss<<", "<<final_traintime<<", "<<final_iter<<", "<<timestamp<<std::endl;

        outFile.close();
        std::cout << "CSV Args have been written to " << filename << std::endl;
        return 0;
    }

    void printAllArgs(bool saveargs, const std::string& filename) {
        // Transfer elements to a std::map to sort them by key
        std::map<std::string, std::string> orderedArguments(arguments.begin(), arguments.end());

        // Find the length of the longest key to align the output
        size_t maxKeyLength = 0;
        for (const auto& pair : orderedArguments) {
            maxKeyLength = std::max(maxKeyLength, pair.first.length());
        }

        // Now iterate through the std::map to print elements in alphabetical order by key with aligned values
        for (const auto& pair : orderedArguments) {
            std::cout << std::left << std::setw(static_cast<int>(maxKeyLength + 2)) // +2 for some spacing between key and value
                    << pair.first << "Value: " << pair.second << "\n";
        }

        if (saveargs) {
            std::ofstream outFile(filename);
            // for (const auto& pair : orderedArguments) {
            //     outFile << pair.first << "=" << pair.second << " ";
            // }
            outFile<<full_input_arg_str;
            outFile.close();
            std::cout << "Args has been written to " << filename << std::endl;
        }
    }

    void printRelArgs() {
        args.print();
    }

    bool getBoolArgument(const std::string& argName) {
        bool value;
        try {
            std::string val = getArgument(argName);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "true" || val == "1") {
                value=true;
            } else if (val == "false" || val == "0") {
                value=false;
            } else {
                throw std::invalid_argument("Invalid boolean value");
            }
        } catch (const std::invalid_argument&) {
            value=false;
        }
        return value;
    }

    int getIntArgument(const std::string& argName) {
        int value;
        try {
            value=std::stoi(getArgument(argName));
        } catch (const std::invalid_argument&) {
            value=0;
        }
        return value;
    }

private:
    std::unordered_map<std::string, ValidatorFunc> validators;
    std::unordered_map<std::string, std::string> arguments;
    std::unordered_map<std::string, std::string> descriptions;
    std::unordered_set<std::string> requiredArgs;

    std::string full_input_arg_str;
    bool args_initialized;
    Args args;

    std::unordered_map<std::string, std::string> loadConfigFile(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::invalid_argument("Failed to open config file: " + path);
        }

        nlohmann::json j;
        in >> j;

        if (!j.is_object()) {
            throw std::invalid_argument("Config file must contain a JSON object at root.");
        }

        std::unordered_map<std::string, std::string> config_map;
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string key = it.key();
            const auto& val = it.value();
            if (val.is_boolean()) {
                config_map[key] = val.get<bool>() ? "true" : "false";
            } else if (val.is_number_integer()) {
                config_map[key] = std::to_string(val.get<long long>());
            } else if (val.is_number_float()) {
                config_map[key] = std::to_string(val.get<double>());
            } else if (val.is_string()) {
                config_map[key] = val.get<std::string>();
            } else if (val.is_array()) {
                if (key == "fourier_periods") {
                    config_map[key] = val.dump();
                } else if (val.size() == 2 && val[0].is_number() && val[1].is_number()) {
                    config_map[key] = "(" + std::to_string(val[0].get<double>()) + "," +
                                      std::to_string(val[1].get<double>()) + ")";
                } else {
                    throw std::invalid_argument("Unsupported array in config for key: " + key);
                }
            } else {
                throw std::invalid_argument("Unsupported config value type for key: " + key);
            }
        }

        return config_map;
    }

    void applyConfigMap(const std::unordered_map<std::string, std::string>& config_map,
                        std::unordered_set<std::string>& suppliedArgs) {
        for (const auto& pair : config_map) {
            std::string name  = "--"+pair.first;
            std::string value = pair.second;

            if (validators.find(name) == validators.end()) {
                throw std::invalid_argument("Unknown argument in config: " + name);
            }

            if (!validators[name](value)) {
                throw std::invalid_argument("Invalid value for argument: " + name);
            }

            // Config overrides defaults; CLI (parsed later) may override config.
            arguments[name] = value;
            suppliedArgs.insert(name);
        }
    }

    void setArgumentStruct() {
        // args = setArgs();
        setGeneralArgs();
        setDomainArgs();
        setTrainingArgs();
        setModelArgs();
        validateExperimentArgs();

        args_initialized=true;
    }

    void validateExperimentArgs() const {
        auto require = [&](bool cond, const std::string& msg) {
            if (!cond) throw std::invalid_argument(msg);
        };

        require(args.training_args.print_interval > 0, "print_interval must be > 0");
        require(args.training_args.reglr_args.lambda_upd_interval > 0, "lambda_upd_interval must be > 0");

        auto tf_it = arguments.find("--test_func");
        if (tf_it == arguments.end()) return;
        const std::string& tf = tf_it->second;

        const int dim = args.domain_args.dim;
        const auto& domain = args.domain_args;
        const auto& model = args.model_args;
        const bool generates_training_data = domain.data_mode != "load_all";
        if (domain.pde_problem == "nonss_allencahn") {
            require(dim == 2, "nonss_allencahn requires dim=2");
            require(!generates_training_data, "nonss_allencahn requires data_mode=load_all; manufactured data is not supported");
            require(!args.training_args.lstsq_updates, "nonss_allencahn does not support least-squares updates");
        }
        if (domain.pde_problem == "onewaywave_inhomo") {
            require(dim == 2, "onewaywave_inhomo requires dim=2");
        }
        if (generates_training_data && domain.initial_cond) {
            require(tf == "test7", "Generated initial conditions are only supported for test7");
        }
        if (domain.bnd_typ == "periodic") {
            require(dim == 2 || model.model_type == "fourierlayer",
                    "1D periodic boundaries require model=fourierlayer");
            if (generates_training_data) {
                require(domain.Nbnd_pts >= 2 && domain.Nbnd_pts % 2 == 0,
                        "Generated periodic boundaries require an even Nbnd_pts >= 2");
            }
            const auto* layer = std::get_if<FourierLayerArgs>(&model.arch);
            const auto* hybrid = std::get_if<FourierNetArgs>(&model.arch);
            if (hybrid) layer = &hybrid->fourierlayer_args;
            if (layer) {
                const int axis = layer->periodic_deriv_dim;
                require(axis == -1 || axis == dim - 1,
                        "periodic_deriv_dim must match the last coordinate (or be -1)");
            }
        }
        if (args.training_args.lstsq_updates) {
            require(args.training_args.lstsq_upd_interval > 0, "lstsq_updates requires lstsq_upd_interval>0");
            require(args.training_args.n_lstsq_upds > 0, "lstsq_updates requires n_lstsq_upds>0");
            require(args.training_args.n_lstsq_upds_per_epoch > 0, "lstsq_updates requires n_lstsq_upds_per_epoch>0");
        }
        if (tf.rfind("test1_k", 0) == 0 || tf.rfind("test1_pi_k", 0) == 0 || tf == "test2" || tf == "test3") {
            require(dim == 1, "test1/test2/test3 require dim=1");
        }
        if (tf == "test4" || tf == "test5" || tf == "test6" || tf == "test7") {
            require(dim == 2, "test4/test5/test6/test7 require dim=2");
        }
        if (tf == "test4" || tf == "test5") {
            require(args.domain_args.pde_problem == "poisson", "test4/test5 require pde_problem=poisson");
        }
        if (tf == "test6") {
            require(args.domain_args.pde_problem == "allencahn", "test6 requires pde_problem=allencahn");
        }
        if (tf == "test7") {
            require(args.domain_args.pde_problem == "onewaywave_inhomo", "test7 requires pde_problem=onewaywave_inhomo");
            require(args.domain_args.initial_cond, "test7 requires initial_cond=true");
        }
        if (tf == "test8") {
            require(dim == 2, "test8 requires dim=2");
            require(args.domain_args.pde_problem == "nonss_allencahn", "test8 requires pde_problem=nonss_allencahn");
            require(args.domain_args.initial_cond, "test8 requires initial_cond=true");
        }
        require(args.domain_args.Ncol_pts > 0, "Ncol_pts must be > 0");
        require(args.domain_args.Nbnd_pts > 0, "Nbnd_pts must be > 0");
        require(args.domain_args.Ntest_pts > 0, "Ntest_pts must be > 0");
        if (args.domain_args.initial_cond) {
            require(args.domain_args.Nic_pts > 0, "Nic_pts must be > 0 when initial_cond=true");
        }
    }

    double getDoubleArgument(const std::string& argName, double defaultValue) {
        double value;
        try {
            value=std::stod(getArgument(argName));
        } catch (const std::invalid_argument&) {
            value=defaultValue;
        }
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        arguments[argName]=ss.str();
        return value;
    }

    int getIntArgument(const std::string& argName, int defaultValue) {
        int value;
        try {
            value=std::stoi(getArgument(argName));
        } catch (const std::invalid_argument&) {
            value=defaultValue;
        }
        arguments[argName]=std::to_string(value);
        return value;
    }

    std::string getStringArgument(const std::string& argName, const std::string& defaultValue) {
        std::string value;
        try {
            value = getArgument(argName);
        } catch (const std::invalid_argument&) {
            value = defaultValue;
        }
        arguments[argName]=value;
        return value;
    }

    float getFloatArgument(const std::string& argName, float defaultValue) {
        float value;
        try {
            value=std::stof(getArgument(argName));
        } catch (const std::invalid_argument&) {
            value=defaultValue;
        }
        std::ostringstream ss;
        ss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
        arguments[argName] = ss.str();
        return value;
    }

    bool getBoolArgument(const std::string& argName, bool defaultValue) {
        bool value;
        try {
            std::string val = getArgument(argName);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "true" || val == "1") {
                value=true;
            } else if (val == "false" || val == "0") {
                value=false;
            } else {
                throw std::invalid_argument("Invalid boolean value");
            }
        } catch (const std::invalid_argument&) {
            value=defaultValue;
        }

        arguments[argName]=std::to_string(value);
        return value;
    }

    void setGeneralArgs() {
        this->args.dim = getIntArgument("--dim", this->args.dim);
        if (args.dim != 1 && args.dim != 2) throw std::invalid_argument("dim must be 1 or 2");
        this->args.seed =  getIntArgument("--seed", this->args.seed);
    }

    void setDomainArgs() {
        this->args.domain_args.initial_cond    = getBoolArgument("--initial_cond", this->args.domain_args.initial_cond);
        this->args.domain_args.homogeneous_pde = getBoolArgument("--homogeneous_pde", this->args.domain_args.homogeneous_pde);
        this->args.domain_args.pde_problem     = getStringArgument("--pde_problem", this->args.domain_args.pde_problem);
        this->args.domain_args.bnd_typ         = getStringArgument("--bnd_typ", this->args.domain_args.bnd_typ);
        this->args.domain_args.point_sampling  = getStringArgument("--point_sampling", this->args.domain_args.point_sampling);
        this->args.domain_args.Ncol_pts        = getIntArgument("--Ncol_pts", this->args.domain_args.Ncol_pts);
        this->args.domain_args.Nbnd_pts        = getIntArgument("--Nbnd_pts", this->args.domain_args.Nbnd_pts);
        this->args.domain_args.Ntest_pts       = getIntArgument("--Ntest_pts", this->args.domain_args.Ntest_pts);
        this->args.domain_args.Nic_pts         = getIntArgument("--Nic_pts", this->args.domain_args.Nic_pts);
        this->args.domain_args.test_func       = getStringArgument("--test_func", this->args.domain_args.test_func);
        this->args.domain_args.dim             = this->args.dim;

        this->args.domain_args.data_mode = getStringArgument("--data_mode", this->args.domain_args.data_mode);

        if (this->args.domain_args.pde_problem=="nonss_allencahn" || this->args.domain_args.pde_problem=="onewaywave_inhomo") {
            std::cout<<"Note: setting args.initial_cond to True (if not already set), for pde_problem: "<<this->args.domain_args.pde_problem<<"\n";
            this->args.domain_args.initial_cond=true;
        }

    }

    void setTrainingArgs() {
        args.training_args.lstsq_safeguard = getBoolArgument("--lstsq_safeguard", true);
        args.training_args.lstsq_stop_epoch = getIntArgument("--lstsq_stop_epoch", -1);
        if (args.training_args.lstsq_stop_epoch < -1) {
            throw std::invalid_argument("lstsq_stop_epoch must be -1 or nonnegative");
        }
        args.training_args.lstsq_rcond = getDoubleArgument("--lstsq_rcond", -1.0);
        if (!(args.training_args.lstsq_rcond == -1.0 ||
              (args.training_args.lstsq_rcond >= 0.0 && args.training_args.lstsq_rcond < 1.0))) {
            throw std::invalid_argument("lstsq_rcond must be -1 (automatic) or in [0, 1)");
        }

        // General training options
        this->args.training_args.training_order   = getStringArgument("--training_order", this->args.training_args.training_order);
        this->args.training_args.dtype            = getStringArgument("--dtype", this->args.training_args.dtype);

        // Scheduler
        this->args.training_args.lr_decaytype     = getStringArgument("--lr_decaytype", this->args.training_args.lr_decaytype);
        this->args.training_args.lr_decay_interval= getIntArgument("--lr_decay_interval", this->args.training_args.lr_decay_interval);
        this->args.training_args.lr_warmup_steps  = getIntArgument("--lr_warmup_steps", this->args.training_args.lr_warmup_steps);
        this->args.training_args.lr_decay_steps   = getIntArgument("--lr_decay_steps", this->args.training_args.lr_decay_steps);
        this->args.training_args.lr_decay_rate    = getFloatArgument("--lr_decay_rate", this->args.training_args.lr_decay_rate);


        this->args.training_args.lstsq_updates          = getBoolArgument("--lstsq_updates", this->args.training_args.lstsq_updates);
        // at what training interval freq to call updated lstsq_updates
        this->args.training_args.lstsq_upd_interval     = getIntArgument("--lstsq_upd_interval", this->args.training_args.lstsq_upd_interval);
        // total number of lstsq updates during training
        this->args.training_args.n_lstsq_upds           = getIntArgument("--n_lstsq_upds", this->args.training_args.n_lstsq_upds);
        // number of lstsq updates to do concurrently (without grad descent updates in between)
        this->args.training_args.n_lstsq_upds_per_epoch = getIntArgument("--n_lstsq_upds_per_epoch", this->args.training_args.n_lstsq_upds_per_epoch);

        this->args.training_args.col_batch_size    = getIntArgument("--col_batch_size", this->args.training_args.col_batch_size);
        this->args.training_args.col_batch_shuffle = getBoolArgument("--col_batch_shuffle", this->args.training_args.col_batch_shuffle);

        // Logging and output
        this->args.training_args.print_model_trn_info = getBoolArgument("--print_model_trn_info", this->args.training_args.print_model_trn_info);
        this->args.training_args.print_lr_sched       = getBoolArgument("--print_lr_sched", this->args.training_args.print_lr_sched);
        this->args.training_args.print_interval       = getIntArgument("--print_interval", this->args.training_args.print_interval);
        args.training_args.test_batch_size = getIntArgument("--test_batch_size", 256);
        if (args.training_args.test_batch_size <= 0) {
            throw std::invalid_argument("test_batch_size must be positive");
        }
        this->args.training_args.pred_save_interval   = getIntArgument("--pred_save_interval", this->args.training_args.pred_save_interval);

        this->args.training_args.truncate_interval = getIntArgument("--truncate_interval", this->args.training_args.truncate_interval);

        this->args.training_args.dim = this->args.dim;
        this->args.training_args.seed = this->args.seed;

        // -----------------------
        // LBFGS optimizer config
        // -----------------------
        if (this->args.training_args.training_order!="adamonly") {
            auto& lbfgs = this->args.training_args.lbfgs_args;
            lbfgs.lbfgs_min_lr         = getDoubleArgument("--lbfgs_min_lr", lbfgs.lbfgs_min_lr);
            lbfgs.lbfgs_max_lr         = getDoubleArgument("--lbfgs_max_lr", lbfgs.lbfgs_max_lr);
            lbfgs.lbfgs_max_epochs     = getIntArgument("--lbfgs_max_epochs", lbfgs.lbfgs_max_epochs);
            lbfgs.lbfgs_history_size   = getIntArgument("--lbfgs_history_size", lbfgs.lbfgs_history_size);
            lbfgs.lbfgs_max_iter       = getIntArgument("--lbfgs_max_iter", lbfgs.lbfgs_max_iter);
            lbfgs.lbfgs_tolerance_grad = getDoubleArgument("--lbfgs_tolerance_grad", lbfgs.lbfgs_tolerance_grad);
            lbfgs.lbfgs_tolerance_change = getDoubleArgument("--lbfgs_tolerance_change", lbfgs.lbfgs_tolerance_change);
            lbfgs.lbfgs_line_search_fn = getStringArgument("--lbfgs_line_search_fn", lbfgs.lbfgs_line_search_fn);
        }

        // -----------------------
        // Adam optimizer config
        // -----------------------
        if (this->args.training_args.training_order!="lbfgsonly") {
            auto& adam = this->args.training_args.adam_args;
            adam.adam_min_lr     = getDoubleArgument("--adam_min_lr", adam.adam_min_lr);
            adam.adam_max_lr     = getDoubleArgument("--adam_max_lr", adam.adam_max_lr);
            adam.adam_max_epochs = getIntArgument("--adam_max_epochs", adam.adam_max_epochs);
            adam.adam_eps        = getDoubleArgument("--adam_eps", adam.adam_eps);
        }

        // -----------------------
        // Regularization config
        // -----------------------
        auto& reg = this->args.training_args.reglr_args;
        reg.lambda_bnd           = getFloatArgument("--lambda_bnd", reg.lambda_bnd);
        reg.lambda_ic            = getFloatArgument("--lambda_ic", reg.lambda_ic);
        reg.lambda_res           = getFloatArgument("--lambda_res", reg.lambda_res);
        reg.lambda_upd_interval  = getIntArgument("--lambda_upd_interval", reg.lambda_upd_interval);
        reg.loss_reweight_scheme = getStringArgument("--loss_reweight_scheme", reg.loss_reweight_scheme);
    }

    void setModelArgs() {
        std::cout << "Setting Model Args\n";

        // Basic model type and dimensions
        this->args.model_args.model_type          = getStringArgument("--model", this->args.model_args.model_type);
        this->args.model_args.input_dim           = this->args.dim;
        this->args.model_args.output_dim          = getIntArgument("--output_dim", this->args.model_args.output_dim);
        this->args.model_args.verbose             = getBoolArgument("--verbose", this->args.model_args.verbose);
        std::vector<double> fourier_periods;
        if (arguments.count("--fourier_periods")) {
            fourier_periods = parseFourierPeriods(arguments.at("--fourier_periods"));
            if (fourier_periods.size() != static_cast<size_t>(this->args.dim))
                throw std::invalid_argument("fourier_periods requires one period per input dimension");
            if (this->args.model_args.model_type != "fouriernn" && this->args.model_args.model_type != "fourierlayer")
                throw std::invalid_argument("fourier_periods requires a Fourier model");
        }

        double dim1_min = getFloatArgument("--dim1_min", -1.0);
        double dim1_max = getFloatArgument("--dim1_max", 1.0);

        double dim2_min = getFloatArgument("--dim2_min", -1.0);
        double dim2_max = getFloatArgument("--dim2_max", 1.0);


        for (int i=0; i<this->args.dim; i++) {
            std::vector<double> dimi_bnds = {};
            if (i==0) {
                dimi_bnds.push_back(dim1_min);
                dimi_bnds.push_back(dim1_max);
            } else if (i==1) {
                dimi_bnds.push_back(dim2_min);
                dimi_bnds.push_back(dim2_max);
            }

            this->args.model_args.domain_bounds.push_back(dimi_bnds);
            this->args.domain_args.domain_bounds.push_back(dimi_bnds);
        }


        // Now set the appropriate architecture variant based on model_type
        const std::string& model_type = this->args.model_args.model_type;
        if (model_type == "neuralnet") {
            NeuralNetArgs net;
            net.depth     = getIntArgument("--depth", net.depth);
            net.width     = getIntArgument("--width", net.width);
            net.net_act   = getStringArgument("--net_act", net.net_act);
            net.nn_truncate       = getBoolArgument("--nn_truncate", net.nn_truncate);
            net.nn_trunc_thresh   = getFloatArgument("--nn_trunc_thresh", net.nn_trunc_thresh);
            net.nn_l1_reg_coeff   = getFloatArgument("--nn_l1_reg_coeff", net.nn_l1_reg_coeff);
            net.nn_l2_reg_coeff   = getFloatArgument("--nn_l2_reg_coeff", net.nn_l2_reg_coeff);
            this->args.model_args.arch = net;
            this->args.model_args.layers = getLayerList(this->args.model_args.input_dim,
                                                        this->args.model_args.output_dim,
                                                        net.width,
                                                        net.depth);

        } else if (model_type=="fourierlayer") {
            FourierLayerArgs poly;
            poly.ell              = getIntArgument("--ell", poly.ell);
            // Hard-default to tensorproduct; ignore user input.
            poly.basis_gen_typ    = BasisGenType::TensorProduct;
            poly.include_zero_mode = getBoolArgument("--include_zero_mode", poly.include_zero_mode);
            poly.scale_to_period   = getBoolArgument("--scale_to_period", poly.scale_to_period);
            poly.fourier_periods = fourier_periods;
            poly.periodic_deriv_dim = getIntArgument("--periodic_deriv_dim", poly.periodic_deriv_dim);

            poly.pn_l1_reg_coeff = getFloatArgument("--pn_l1_reg_coeff", poly.pn_l1_reg_coeff);
            poly.pn_l2_reg_coeff = getFloatArgument("--pn_l2_reg_coeff", poly.pn_l2_reg_coeff);
            poly.pn_trunc_thresh = getFloatArgument("--pn_trunc_thresh", poly.pn_trunc_thresh);
            poly.pn_truncate     = getBoolArgument("--pn_truncate", poly.pn_truncate);

            this->args.model_args.arch = poly;

        } else if (model_type=="fouriernn") {
            FourierNetArgs polynet;

            // Neural Net Arguments
            polynet.neuralnet_args.depth     = getIntArgument("--depth", polynet.neuralnet_args.depth);
            polynet.neuralnet_args.width     = getIntArgument("--width", polynet.neuralnet_args.width);
            polynet.neuralnet_args.net_act   = getStringArgument("--net_act", polynet.neuralnet_args.net_act);
            polynet.neuralnet_args.nn_truncate       = getBoolArgument("--nn_truncate", polynet.neuralnet_args.nn_truncate);
            polynet.neuralnet_args.nn_trunc_thresh   = getFloatArgument("--nn_trunc_thresh", polynet.neuralnet_args.nn_trunc_thresh);
            polynet.neuralnet_args.nn_l1_reg_coeff   = getFloatArgument("--nn_l1_reg_coeff", polynet.neuralnet_args.nn_l1_reg_coeff);
            polynet.neuralnet_args.nn_l2_reg_coeff   = getFloatArgument("--nn_l2_reg_coeff", polynet.neuralnet_args.nn_l2_reg_coeff);

            polynet.neuralnet_args.layers = getLayerList(this->args.model_args.input_dim,
                                                         this->args.model_args.output_dim,
                                                         polynet.neuralnet_args.width,
                                                         polynet.neuralnet_args.depth);

            // FourierLayer Arguments
            polynet.fourierlayer_args.ell                = getIntArgument("--ell", polynet.fourierlayer_args.ell);
            // Hard-default to tensorproduct; ignore user input.
            polynet.fourierlayer_args.basis_gen_typ      = BasisGenType::TensorProduct;
            polynet.fourierlayer_args.include_zero_mode  = getBoolArgument("--include_zero_mode", polynet.fourierlayer_args.include_zero_mode);
            polynet.fourierlayer_args.scale_to_period    = getBoolArgument("--scale_to_period", polynet.fourierlayer_args.scale_to_period);
            polynet.fourierlayer_args.fourier_periods = fourier_periods;
            polynet.fourierlayer_args.periodic_deriv_dim = getIntArgument("--periodic_deriv_dim", polynet.fourierlayer_args.periodic_deriv_dim);

            polynet.fourierlayer_args.pn_l1_reg_coeff = getFloatArgument("--pn_l1_reg_coeff", polynet.fourierlayer_args.pn_l1_reg_coeff);
            polynet.fourierlayer_args.pn_l2_reg_coeff = getFloatArgument("--pn_l2_reg_coeff", polynet.fourierlayer_args.pn_l2_reg_coeff);
            polynet.fourierlayer_args.pn_trunc_thresh = getFloatArgument("--pn_trunc_thresh", polynet.fourierlayer_args.pn_trunc_thresh);
            polynet.fourierlayer_args.pn_truncate     = getBoolArgument("--pn_truncate", polynet.fourierlayer_args.pn_truncate);

            this->args.model_args.arch = polynet;

        } else {
            std::cerr << "[ERROR] Unsupported model type: " << model_type << "\n";
            throw std::invalid_argument("Invalid --model argument: " + model_type);
        }
    }
};


inline void generateMyExperimentArgs(ArgParser& parser, bool solve_by_default = false) {
    parser.addArgument("--lstsq_rcond", isValidDouble);
    parser.addArgument("--lstsq_stop_epoch", isValidInt);
    parser.addArgument("--lstsq_safeguard", isValidBool);
    parser.addArgument("--config", [](const std::string& val) -> bool { return !val.empty(); } ); // JSON config path.

    parser.addArgument("--model", [](const std::string& val) -> bool {
                        return val=="neuralnet" || val=="fourierlayer" || val=="fouriernn"; },
                       /*isRequired=*/false); // Model type.

    parser.addArgument("--datadir", [](const std::string&) -> bool { return true; }, /*isRequired=*/true); // Data directory root.
    parser.addArgument("--expid", [](const std::string&) -> bool { return true; }, /*isRequired=*/true);    // Experiment id (name).
    parser.addArgument("--gpu_device", isValidInt);                                                             // CUDA device index.

    parser.addInitArgument("--solve_pde", isValidBool, solve_by_default);          // Train/solve PDE.
    parser.addInitArgument("--write_logs", isValidBool, false);         // Redirect stdout to log file.
    parser.addInitArgument("--load_pretrn_model", isValidBool, false);  // Load pretrained model.
    parser.addArgument("--data_mode", [](const std::string& val) -> bool {
                        return val=="generate" || val=="load_all" || val=="load_test_only"; } );              // Domain data mode.
    parser.addInitArgument("--print_all_args", isValidBool, false);                                            // Print all args.
    parser.addInitArgument("--print_relevant_args", isValidBool, false);                                       // Print relevant args.

    // Fourier basis options
    parser.addArgument("--include_zero_mode", isValidBool); // Include zero-frequency mode.
    parser.addArgument("--scale_to_period", isValidBool);   // Scale inputs by period lengths.
    parser.addArgument("--fourier_periods", [](const std::string& value) {
        try { parseFourierPeriods(value); return true; } catch (const std::exception&) { return false; }
    });
    parser.addArgument("--periodic_deriv_dim", isValidInt); // Periodic derivative dimension.

    parser.addArgument("--dtype", [](const std::string& val) -> bool { return val=="float32" || val=="float64"; } ); // Dtype.

    parser.addArgument("--lr_decaytype", [](const std::string& val) -> bool {
                        return val=="cosine" || val=="exp" || val=="none"; } ); // LR schedule type.

    // lbfgs argsparse
    parser.addArgument("--lbfgs_min_lr", isValidDouble);                                              // LBFGS min lr.
    parser.addArgument("--lbfgs_max_lr", isValidDouble);                                              // LBFGS max lr.
    parser.addArgument("--lbfgs_max_epochs", isValidInt);                                             // LBFGS epochs.
    parser.addArgument("--lbfgs_history_size", isValidInt);                                           // LBFGS history size.
    parser.addArgument("--lbfgs_max_iter", isValidInt);                                               // LBFGS max iter.
    parser.addArgument("--lbfgs_tolerance_grad", isValidDouble);                                      // LBFGS grad tolerance.
    parser.addArgument("--lbfgs_tolerance_change", isValidDouble);                                    // LBFGS change tolerance.
    parser.addArgument("--lbfgs_line_search_fn", [](const std::string& val) -> bool { return val=="strong_wolfe"; } ); // Line search.

    // adam args
    parser.addArgument("--adam_min_lr", isValidDouble);     // Adam min lr.
    parser.addArgument("--adam_max_lr", isValidDouble);     // Adam max lr.
    parser.addArgument("--adam_max_epochs", isValidInt);    // Adam epochs.
    parser.addArgument("--adam_eps", isValidDouble);        // Adam epsilon.

    // experiment args
    parser.addArgument("--seed", isValidInt);                             // RNG seed.
    parser.addArgument("--dim", isValidInt);                              // Input dimension.
    parser.addInitArgument("--lstsq_updates", isValidBool, false);        // Enable least-squares updates.
    parser.addArgument("--lstsq_upd_interval", isValidInt);               // LSTSQ update interval.
    parser.addArgument("--n_lstsq_upds", isValidInt);                     // Number of LSTSQ updates.
    parser.addArgument("--n_lstsq_upds_per_epoch", isValidInt);           // LSTSQ updates per epoch.
    parser.addArgument("--col_batch_size", isValidInt);                   // Collocation batch size (0 => full).
    parser.addArgument("--col_batch_shuffle", isValidBool);               // Shuffle collocation batch.
    parser.addArgument("--training_order", [](const std::string& val) -> bool {
                        return val=="adamfirst" || val=="lbfgsfirst" || val=="adamonly" || val=="lbfgsonly"; } ); // Training order.

    // NOTE loadedDomain args
    // NOTE allencahn is assumed to be steady-state, nonss-allencahn is the non-steady state
    parser.addArgument("--pde_problem", [](const std::string& val) -> bool { return val=="poisson" || val=="allencahn" || val=="nonss_allencahn" || val=="onewaywave_inhomo"; } ); // PDE type.
    parser.addArgument("--Ncol_pts", isValidInt); // Collocation points.
    parser.addArgument("--Nbnd_pts", isValidInt); // Boundary points.
    parser.addArgument("--Ntest_pts", isValidInt); // Test points.
    parser.addArgument("--Nic_pts", isValidInt); // Initial-condition points.
    parser.addArgument("--bnd_typ", [](const std::string& val) -> bool { return val=="dirichlet" || val=="periodic"; } ); // Boundary type.
    parser.addArgument("--point_sampling", [](const std::string& val) -> bool { return val=="equispaced" || val=="random"; } ); // Sampling type.
    parser.addArgument("--test_func", [](const std::string&) -> bool { return true; }); // Manufactured solution id.

    parser.addArgument("--dim1_min", isValidFloat); // Min bound for dim1.
    parser.addArgument("--dim1_max", isValidFloat); // Max bound for dim1.

    parser.addArgument("--dim2_min", isValidFloat); // Min bound for dim2.
    parser.addArgument("--dim2_max", isValidFloat); // Max bound for dim2.

    parser.addArgument("--initial_cond", isValidBool);    // Use initial conditions.
    parser.addArgument("--homogeneous_pde", isValidBool); // PDE forcing f=0.

    // neural net args
    parser.addArgument("--depth", isValidInt); // Network depth.
    parser.addArgument("--width", isValidInt); // Network width.
    parser.addArgument("--net_act", [](const std::string& val) -> bool {
                        return val=="tanh" || val=="relu" || val=="sigmoid" || val=="elu" || val=="repu" || val=="softplus" || val=="gelu" || val=="selu" || val=="swish"; } ); // Activation.
    parser.addArgument("--nn_l1_reg_coeff", isValidFloat);  // NN L1 regularization.
    parser.addArgument("--nn_l2_reg_coeff", isValidFloat);  // NN L2 regularization.
    parser.addArgument("--nn_trunc_thresh", isValidFloat);  // NN truncation threshold.
    parser.addArgument("--nn_truncate", isValidBool);       // Enable NN truncation.
    parser.addArgument("--truncate_interval", isValidInt);  // Truncation interval.

    // fourier layer args
    parser.addArgument("--ell", isValidInt);               // Max Fourier frequency per dim.
    parser.addArgument("--pn_l1_reg_coeff", isValidFloat); // Fourier L1 regularization.
    parser.addArgument("--pn_l2_reg_coeff", isValidFloat); // Fourier L2 regularization.
    parser.addArgument("--pn_trunc_thresh", isValidFloat); // Fourier truncation threshold.
    parser.addArgument("--pn_truncate", isValidBool); // Enable Fourier truncation.

    // misc args
    parser.addArgument("--verbose", isValidBool);             // Verbose logging.

    // lagrange multipliers
    parser.addArgument("--lambda_upd_interval", isValidInt); // Lambda update interval.
    parser.addArgument("--lambda_bnd", isValidFloat);        // Boundary loss weight.
    parser.addArgument("--lambda_ic", isValidFloat);         // Initial condition loss weight.
    parser.addArgument("--lambda_res", isValidFloat);        // Residual loss weight.

    parser.addArgument("--loss_reweight_scheme", [](const std::string& val) -> bool {
                        return val=="none" || val=="grad"; } ); // Loss reweighting scheme.

    parser.addArgument("--lr_decay_interval", isValidInt); // LR decay interval.
    parser.addArgument("--lr_warmup_steps", isValidInt);   // LR warmup steps.
    parser.addArgument("--lr_decay_steps", isValidInt);    // LR decay steps.
    parser.addArgument("--lr_decay_rate", isValidFloat);   // LR decay rate.

    parser.addArgument("--pred_save_interval", isValidInt); // Prediction save interval.

    parser.addArgument("--print_model_trn_info", isValidBool); // Print training info.
    parser.addArgument("--print_lr_sched", isValidBool);       // Print LR schedule.
    parser.addArgument("--print_interval", isValidInt);        // Print interval.
    parser.addArgument("--test_batch_size", isValidInt);

}
