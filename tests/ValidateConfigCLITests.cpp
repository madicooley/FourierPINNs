#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "CLI.h"

namespace {
std::string fourierpinnsTmpDir() {
    const std::filesystem::path repo_root =
        std::filesystem::path(__FILE__).parent_path().parent_path();
    return (repo_root / "tmp").string() + "/";
}

struct CliRunResult {
    int rc = 1;
    std::string out;
    std::string err;
};

CliRunResult runCliWithArgs(const std::vector<std::string>& args) {
    std::vector<std::string> argv_storage = args;
    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (std::string& arg : argv_storage) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }

    std::ostringstream out_capture;
    std::ostringstream err_capture;
    std::streambuf* old_out = std::cout.rdbuf(out_capture.rdbuf());
    std::streambuf* old_err = std::cerr.rdbuf(err_capture.rdbuf());

    CliRunResult result;
    result.rc = runFourierPINNsCLI(static_cast<int>(argv.size()), argv.data());

    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    result.out = out_capture.str();
    result.err = err_capture.str();
    return result;
}

std::string writeJsonFile(const std::string& filename, const std::string& json_content) {
    const std::filesystem::path cfg_dir = std::filesystem::path(fourierpinnsTmpDir()) / "configs";
    std::filesystem::create_directories(cfg_dir);
    const std::string path = (cfg_dir / filename).string();
    std::ofstream out(path);
    out << json_content;
    out.close();
    return path;
}

} // namespace

TEST_CASE("Fourier periods accept CLI lists and JSON arrays with dimension validation", "[CLI][FourierPeriods]") {
    for (const std::string value : {"1,2", "[1,2]", "1,2.5"}) {
        auto result = runCliWithArgs({"fourierpinns", "validate-config", "--model=fouriernn", "--dim=2",
            "--test_func=test4", "--pde_problem=poisson",
            "--datadir="+fourierpinnsTmpDir(), "--expid=periods_cli",
            "--fourier_periods="+value});
        INFO(result.err);
        REQUIRE(result.rc == 0);
    }
    for (const std::string value : {"", "[]", "1", "1,2,3", "0,2", "-1,2", "nan,2", "inf,2", "1,", "1,,2", "[true,2]"}) {
        auto result = runCliWithArgs({"fourierpinns", "validate-config", "--model=fouriernn", "--dim=2",
            "--test_func=test4", "--pde_problem=poisson",
            "--datadir="+fourierpinnsTmpDir(), "--expid=periods_invalid",
            "--fourier_periods="+value});
        REQUIRE(result.rc != 0);
    }
    for (const std::string array : {"[2]", "[2.0]"}) {
        auto path = writeJsonFile("periods_1d.json", "{\"model\":\"fourierlayer\",\"dim\":1,\"fourier_periods\":"+array+"}");
        auto result = runCliWithArgs({"fourierpinns", "validate-config", "--config="+path, "--dim=1",
            "--datadir="+fourierpinnsTmpDir(), "--expid=periods_json"});
        REQUIRE(result.rc == 0);
    }
    auto path = writeJsonFile("periods_2d.json", "{\"model\":\"fouriernn\",\"dim\":2,\"fourier_periods\":[1,2]}");
    auto result = runCliWithArgs({"fourierpinns", "validate-config", "--config="+path, "--dim=2",
        "--test_func=test4", "--pde_problem=poisson",
        "--fourier_periods=2,3", "--datadir="+fourierpinnsTmpDir(), "--expid=periods_override"});
    REQUIRE(result.rc == 0);
    std::ifstream snapshot(fourierpinnsTmpDir()+"args/resolved_config_periods_override_fouriernn_forwardpde.json");
    std::string contents((std::istreambuf_iterator<char>(snapshot)), std::istreambuf_iterator<char>());
    REQUIRE(contents.find("2,3") != std::string::npos);
}

TEST_CASE("Public CLI rejects removed model feature options", "[CLI][PublicCleanup]") {
    const std::vector<std::pair<std::string, std::string>> removed = {
        {"fourier_feat_embedd", "true"}, {"periodic_feat_embedd", "true"},
        {"fourier_feat_scale", "1"},
        {"strong_bnd_exp_init", "1"}, {"homogeneous_bnds", "true"},
        {"strbnd_alpha_lr_mult", "10"}, {"strbnd_alpha_only_warmup_epochs", "10"},
        {"strbnd_alpha_freeze_epoch", "100"},
        {"input_norm_type", "standardize"}, {"normalize_fourier_basis", "false"},
        {"norm_forward", "true"}, {"problem", "forwardpde"}, {"domain_typ", "box"},
        {"basis_typ", "fourier"}, {"save_poly_bases", "true"}, {"dim3_min", "0"}, {"dim3_max", "1"},
        {"lambda_skip", "0"}, {"lbfgs_lr_decay_rate", "100"}, {"adam_betas", "(0.9,0.999)"}
    };
    for (const auto& option : removed) {
        INFO(option.first);
        auto result = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=fouriernn", "--dim=1",
            "--datadir="+fourierpinnsTmpDir(), "--expid=removed_option",
            "--"+option.first+"="+option.second});
        REQUIRE(result.rc != 0);
        REQUIRE(result.err.find("Unknown argument") != std::string::npos);
        const auto path = writeJsonFile("removed_option.json",
            "{\"model\":\"fouriernn\",\""+option.first+"\":\""+option.second+"\"}");
        auto config_result = runCliWithArgs({"fourierpinns", "validate-config",
            "--config="+path, "--dim=1",
            "--datadir="+fourierpinnsTmpDir(), "--expid=removed_option_json"});
        REQUIRE(config_result.rc != 0);
        REQUIRE(config_result.err.find("Unknown config key") != std::string::npos);
    }
}

TEST_CASE("validate-config succeeds for valid minimal config", "[CLI][validate-config]") {
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--model=neuralnet",
        "--datadir=" + fourierpinnsTmpDir(),
        "--expid=validate_ok",
        "--dtype=float32",
        "--data_mode=generate"
    });

    REQUIRE(res.rc == 0);
}

TEST_CASE("validate-config requires positive test chunk sizes", "[CLI][ChunkedTest]") {
    for (const std::string value : {"1", "256", "0", "-1"}) {
        const auto res = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=neuralnet",
            "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_test_chunk",
            "--test_batch_size=" + value});
        REQUIRE((res.rc == 0) == (value == "1" || value == "256"));
    }
}

TEST_CASE("validate-config requires positive training intervals", "[CLI][validate-config]") {
    for (const std::string key : {"print_interval", "lambda_upd_interval"}) {
        for (const std::string value : {"0", "-1", "1"}) {
            INFO(key << "=" << value);
            const auto path = writeJsonFile("interval_" + key + ".json",
                "{\"" + key + "\":" + value + "}");
            for (const auto& option : {"--" + key + "=" + value, "--config=" + path}) {
                INFO(option);
                const auto res = runCliWithArgs({"fourierpinns", "validate-config",
                    "--model=neuralnet",
                    "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_interval",
                    "--loss_reweight_scheme=none", option});
                REQUIRE((res.rc == 0) == (value == "1"));
                if (value != "1") REQUIRE(res.err.find(key + " must be > 0") != std::string::npos);
            }
        }
    }
}

TEST_CASE("validate-config accepts only supported loss reweighting schemes", "[CLI][validate-config]") {
    for (const std::string scheme : {"none", "grad", "ntk"}) {
        const auto path = writeJsonFile("loss_reweight_" + scheme + ".json",
            "{\"loss_reweight_scheme\":\"" + scheme + "\"}");
        for (const auto& option : {"--loss_reweight_scheme=" + scheme, "--config=" + path}) {
            INFO(option);
            const auto res = runCliWithArgs({"fourierpinns", "validate-config",
                "--model=neuralnet",
                "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_reweight", option});
            REQUIRE((res.rc == 0) == (scheme != "ntk"));
            if (scheme == "ntk") REQUIRE(res.err.find("loss_reweight_scheme") != std::string::npos);
        }
    }
}

TEST_CASE("validate-config accepts automatic and explicit LS rank cutoffs", "[CLI][validate-config]") {
    for (const auto& value : {"-1", "0", "1e-5"}) {
        const auto res = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=neuralnet",
            "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_rcond",
            std::string("--lstsq_rcond=") + value});
        REQUIRE(res.rc == 0);
    }
    for (const auto& value : {"-2", "1", "nan", "inf"}) {
        const auto res = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=neuralnet",
            "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_rcond_invalid",
            std::string("--lstsq_rcond=") + value});
        REQUIRE(res.rc != 0);
    }
}

TEST_CASE("validate-config checks LS Adam step cutoff", "[CLI][validate-config]") {
    for (const auto& value : {"-1", "0", "20000", "-2"}) {
        const auto res = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=neuralnet",
            "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_ls_stop",
            std::string("--lstsq_stop_epoch=") + value});
        if (std::string(value) == "-2") REQUIRE(res.rc != 0);
        else REQUIRE(res.rc == 0);
    }
}

TEST_CASE("Removed LS initialization and scaling switches are rejected", "[CLI][validate-config]") {
    for (const std::string key : {"lstsq_initial", "lstsq_column_scaling"}) {
        const auto cli = runCliWithArgs({"fourierpinns", "validate-config",
            "--model=neuralnet", "--" + key + "=false"});
        REQUIRE(cli.rc != 0);
        const auto path = writeJsonFile("removed_" + key + ".json",
            "{\"model\":\"neuralnet\",\"problem\":\"forwardpde\",\"" + key + "\":false}");
        const auto config = runCliWithArgs({"fourierpinns", "validate-config", "--config=" + path});
        REQUIRE(config.rc != 0);
    }
}

TEST_CASE("validate-config fails for unknown command", "[CLI][validate-config]") {
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-unknown",
        "--model=neuralnet",
        "--datadir=" + fourierpinnsTmpDir(),
        "--expid=validate_bad_cmd",
        "--dtype=float32",
        "--data_mode=generate"
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("validate-config fails when required args are missing", "[CLI][validate-config]") {
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--model=neuralnet",
        "--dtype=float32",
        "--data_mode=generate"
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("validate-config succeeds with valid --config json", "[CLI][validate-config]") {
    const std::string cfg = writeJsonFile(
        "fourierpinns_validate_ok.json",
        R"({
  "model": "neuralnet",
  "datadir": "runs/tmp/",
  "expid": "from_config",
  "dtype": "float32",
  "data_mode": "generate"
})");

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--config=" + cfg
    });

    REQUIRE(res.rc == 0);
}

TEST_CASE("validate-config uses the shipped 2D dimension unless explicitly overridden", "[CLI][validate-config]") {
    const auto config = std::filesystem::path(__FILE__).parent_path().parent_path() /
                        "experiments/section_6_4_1/configs/single_scale.json";
    std::vector<std::string> args = {"fourierpinns", "validate-config",
        "--config=" + config.string(), "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_2d"};
    const auto from_config = runCliWithArgs(args);
    INFO(from_config.err);
    REQUIRE(from_config.rc == 0);
    args.push_back("--dim=1");
    const auto overridden = runCliWithArgs(args);
    REQUIRE(overridden.rc != 0);
    REQUIRE(overridden.err.find("require dim=2") != std::string::npos);
}

TEST_CASE("validate-config fails on unknown config key", "[CLI][validate-config]") {
    const std::string cfg = writeJsonFile(
        "fourierpinns_validate_unknown_key.json",
        R"({
  "model": "neuralnet",
  "datadir": "runs/tmp/",
  "expid": "bad_key",
  "dtype": "float32",
  "data_mode": "generate",
  "unknown_option": "oops"
})");

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--config=" + cfg
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("validate-config fails on invalid enum value in config", "[CLI][validate-config]") {
    const std::string cfg = writeJsonFile(
        "fourierpinns_validate_bad_enum.json",
        R"({
  "model": "neuralnet",
  "datadir": "runs/tmp/",
  "expid": "bad_enum",
  "dtype": "float16",
  "data_mode": "generate"
})");

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--config=" + cfg
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("validate-config uses CLI values over config values", "[CLI][validate-config][precedence]") {
    const std::string cfg = writeJsonFile(
        "fourierpinns_validate_cli_precedence.json",
        R"({
  "model": "neuralnet",
  "datadir": "runs/tmp/from_config/",
  "expid": "config_expid",
  "dtype": "float32",
  "data_mode": "generate"
})");

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--config=" + cfg,
        "--model=fouriernn",
        "--datadir=" + fourierpinnsTmpDir() + "from_cli/",
        "--expid=cli_expid",
        "--dtype=float64",
        "--data_mode=load_all"
    });

    REQUIRE(res.rc == 0);
    REQUIRE(res.out.find("model=fouriernn") != std::string::npos);
    REQUIRE(res.out.find("datadir=" + fourierpinnsTmpDir() + "from_cli/") != std::string::npos);
    REQUIRE(res.out.find("expid=cli_expid") != std::string::npos);
    REQUIRE(res.out.find("dtype=float64") != std::string::npos);
    REQUIRE(res.out.find("data_mode=load_all") != std::string::npos);
}

TEST_CASE("validate-config allows CLI to supply required keys missing in config", "[CLI][validate-config][precedence]") {
    const std::string cfg = writeJsonFile(
        "fourierpinns_validate_cli_supplies_required.json",
        R"({
  "model": "neuralnet",
  "dtype": "float32",
  "data_mode": "generate"
})");

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--config=" + cfg,
        "--datadir=" + fourierpinnsTmpDir() + "from_cli_required/",
        "--expid=cli_required"
    });

    REQUIRE(res.rc == 0);
    REQUIRE(res.out.find("datadir=" + fourierpinnsTmpDir() + "from_cli_required/") != std::string::npos);
    REQUIRE(res.out.find("expid=cli_required") != std::string::npos);
}

TEST_CASE("gen-data rejects solve_pde true with actionable diagnostic", "[CLI][diagnostics]") {
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "gen-data",
        "--model=neuralnet",
        "--datadir=" + fourierpinnsTmpDir(),
        "--expid=bad_combo_solve",
        "--dtype=float32",
        "--data_mode=generate",
        "--solve_pde=true"
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("gen-data rejects non-generate data_mode", "[CLI][diagnostics]") {
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "gen-data",
        "--model=neuralnet",
        "--datadir=" + fourierpinnsTmpDir(),
        "--expid=bad_combo_mode",
        "--dtype=float32",
        "--data_mode=load_all"
    });

    REQUIRE(res.rc == 1);
}

TEST_CASE("resolved config preserves small float settings", "[CLI][snapshot]") {
    const auto datadir = fourierpinnsTmpDir();
    const auto res = runCliWithArgs({"fourierpinns", "validate-config", "--model=fouriernn",
        "--datadir=" + datadir, "--expid=float_precision", "--pn_trunc_thresh=1e-8"});
    INFO(res.err);
    REQUIRE(res.rc == 0);
    std::ifstream in(datadir + "args/resolved_config_float_precision_fouriernn_forwardpde.json");
    REQUIRE(in.is_open());
    nlohmann::json snapshot;
    in >> snapshot;
    REQUIRE(std::stof(snapshot.at("resolved_args").at("pn_trunc_thresh").get<std::string>()) == 1e-8f);
}

TEST_CASE("validate-config writes resolved config snapshot", "[CLI][snapshot]") {
    const std::string expid = "snapshot_validate_case";
    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "validate-config",
        "--model=neuralnet",
        "--datadir=" + fourierpinnsTmpDir(),
        "--expid=" + expid,
        "--dtype=float32",
        "--data_mode=generate"
    });

    REQUIRE(res.rc == 0);

    const std::string snapshot_path = fourierpinnsTmpDir() + "args/resolved_config_" + expid + "_neuralnet_forwardpde.json";
    REQUIRE(std::filesystem::exists(snapshot_path));

    std::ifstream in(snapshot_path);
    REQUIRE(in.is_open());
    nlohmann::json j;
    in >> j;

    REQUIRE(j["metadata"]["command"] == "validate-config");
    REQUIRE(j["resolved_args"]["expid"] == expid);
    REQUIRE(j["resolved_args"]["model"] == "neuralnet");
    REQUIRE(j["resolved_args"]["data_mode"] == "generate");
    REQUIRE(j["resolved_args"]["dim"] == "1");
    REQUIRE(j["resolved_args"]["test_func"] == "test1_k1");
}

TEST_CASE("validate-config rejects unsupported dimensions and PDE boundary combinations", "[CLI][diagnostics]") {
    const auto validate = [](const std::string& model, std::vector<std::string> options) {
        std::vector<std::string> args{"fourierpinns", "validate-config", "--model=" + model,
            "--datadir=" + fourierpinnsTmpDir(), "--expid=validate_combinations"};
        args.insert(args.end(), options.begin(), options.end());
        return runCliWithArgs(args);
    };
    const auto reject = [&](const std::string& model, std::vector<std::string> options, const std::string& error) {
        const auto result = validate(model, options);
        REQUIRE(result.rc != 0);
        REQUIRE(result.err.find(error) != std::string::npos);
    };
    reject("fouriernn", {"--dim=3"}, "dim must be 1 or 2");
    reject("fouriernn", {"--dim=2", "--pde_problem=nonss_allencahn", "--test_func=test8"},
           "requires data_mode=load_all");
    reject("neuralnet", {"--bnd_typ=periodic"}, "require model=fourierlayer");
    reject("fouriernn", {"--bnd_typ=periodic"}, "require model=fourierlayer");
    reject("fourierlayer", {"--bnd_typ=periodic", "--Nbnd_pts=3"}, "even Nbnd_pts");
    reject("fouriernn", {"--dim=2", "--test_func=test4", "--bnd_typ=periodic", "--periodic_deriv_dim=0"},
           "periodic_deriv_dim must match");
    REQUIRE(validate("fourierlayer", {"--bnd_typ=periodic"}).rc == 0);
    REQUIRE(validate("fouriernn", {"--dim=2", "--test_func=test4", "--bnd_typ=periodic",
                                    "--periodic_deriv_dim=1"}).rc == 0);
    REQUIRE(validate("fouriernn", {"--dim=2", "--pde_problem=nonss_allencahn", "--test_func=test8",
                                    "--data_mode=load_all"}).rc == 0);
}
