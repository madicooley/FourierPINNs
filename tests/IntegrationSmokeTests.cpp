#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "CLI.h"
#include "fourierpinns/data/LoadedDomain.h"

namespace {
std::size_t occurrences(const std::string& text, const std::string& needle) {
    std::size_t result = 0, pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++result;
        pos += needle.size();
    }
    return result;
}
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

int runCliNoCapture(const std::vector<std::string>& args) {
    std::vector<std::string> argv_storage = args;
    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (std::string& arg : argv_storage) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    return runFourierPINNsCLI(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST_CASE("train smoke run succeeds with deterministic small config", "[integration][smoke]") {
    const std::string datadir = fourierpinnsTmpDir() + "fourierpinns_smoke_train/";
    const std::string expid = "smoke_train";
    const std::string filename_root = expid + "_neuralnet_forwardpde";
    const std::filesystem::path snapshot_path =
        std::filesystem::path(datadir) / "args" / ("resolved_config_" + filename_root + ".json");

    std::filesystem::remove(snapshot_path);

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "train",
        "--model=neuralnet",
        "--datadir=" + datadir,
        "--expid=" + expid,
        "--dtype=float32",
        "--data_mode=generate",
        "--solve_pde=false",
        "--seed=123",
        "--dim=1",
        "--pde_problem=poisson",
        "--bnd_typ=dirichlet",
        "--point_sampling=equispaced",
        "--dim1_min=0.0",
        "--dim1_max=1.0",
        "--Ncol_pts=8",
        "--Nbnd_pts=4",
        "--Ntest_pts=8",
        "--test_func=test1_k1",
        "--depth=1",
        "--width=8",
        "--net_act=tanh"
    });

    REQUIRE(res.rc == 0);
    REQUIRE(std::filesystem::exists(snapshot_path));

    std::ifstream in(snapshot_path);
    REQUIRE(in.is_open());
    nlohmann::json j;
    in >> j;
    REQUIRE(j["metadata"]["command"] == "train");
    REQUIRE(j["resolved_args"]["seed"] == "123");
    REQUIRE(j["resolved_args"]["solve_pde"] == "false");
}

TEST_CASE("gen-data writes a reloadable dataset without overwriting it", "[integration][smoke]") {
    const std::string datadir = fourierpinnsTmpDir() + "fourierpinns_smoke_gendata/";
    const std::string expid = "smoke_gendata";
    const std::string filename_root = expid + "_neuralnet_forwardpde";
    const std::filesystem::path snapshot_path =
        std::filesystem::path(datadir) / "args" / ("resolved_config_" + filename_root + ".json");

    std::filesystem::remove_all(datadir);

    std::vector<std::string> args = {
        "fourierpinns",
        "gen-data",
        "--model=neuralnet",
        "--datadir=" + datadir,
        "--expid=" + expid,
        "--dtype=float32",
        "--data_mode=generate",
        "--seed=456",
        "--dim=1",
        "--pde_problem=poisson",
        "--bnd_typ=dirichlet",
        "--point_sampling=equispaced",
        "--dim1_min=0.0",
        "--dim1_max=1.0",
        "--Ncol_pts=8",
        "--Nbnd_pts=4",
        "--Ntest_pts=8",
        "--test_func=test1_k1",
        "--depth=1",
        "--width=8",
        "--net_act=tanh"
    };
    const CliRunResult res = runCliWithArgs(args);

    REQUIRE(res.rc == 0);
    REQUIRE(std::filesystem::exists(snapshot_path));

    std::ifstream in(snapshot_path);
    REQUIRE(in.is_open());
    nlohmann::json j;
    in >> j;
    REQUIRE(j["metadata"]["command"] == "gen-data");
    REQUIRE(j["resolved_args"]["seed"] == "456");
    REQUIRE(j["resolved_args"]["data_mode"] == "generate");

    const auto dataset = std::filesystem::path(datadir) / "data";
    const auto generate_domain = [] {
        return LoadedPDEDomain(1, 8, 4, 8, 0, "test1_k1", "poisson", "dirichlet",
            "box", "equispaced", false, false, {{0, 1}}, torch::kCPU, torch::kFloat32);
    };
    auto expected = generate_domain();
    expected.generateDomainData();
    auto loaded = generate_domain();
    loaded.loadDomainsFromFile(datadir);
    REQUIRE(torch::equal(*loaded.getPDEData().data.X, *expected.getPDEData().data.X));
    REQUIRE(torch::allclose(*loaded.getPDEData().f, *expected.getPDEData().f));
    REQUIRE(torch::equal(*loaded.getBndData().data.X, *expected.getBndData().data.X));
    REQUIRE(torch::allclose(*loaded.getBndData().u, *expected.getBndData().u));
    REQUIRE(torch::equal(*loaded.getTestData().data.X, *expected.getTestData().data.X));
    REQUIRE(torch::allclose(*loaded.getTestData().u, *expected.getTestData().u));
    REQUIRE(loaded.getPDEData().data.X->requires_grad());
    REQUIRE_FALSE(loaded.getPDEData().f->requires_grad());

    const auto refused = runCliWithArgs(args);
    REQUIRE(refused.rc != 0);
    REQUIRE(refused.err.find("Refusing to overwrite") != std::string::npos);

    const auto duplicate = dataset / "other_XcolTrain.pt";
    std::ofstream(duplicate).close();
    auto test_only = generate_domain();
    test_only.loadDomainsFromFile(datadir, true);
    REQUIRE(torch::equal(*test_only.getTestData().data.X, *expected.getTestData().data.X));
    REQUIRE(torch::equal(*test_only.getTestData().u, *loaded.getTestData().u));
    REQUIRE_THROWS_WITH(loaded.loadDomainsFromFile(datadir),
        Catch::Matchers::ContainsSubstring("Duplicate dataset tensor XcolTrain"));
    std::filesystem::remove(duplicate);

    const auto test_target = (dataset / "uTest.pt").string();
    torch::save(torch::zeros({7, 1}), test_target);
    REQUIRE_THROWS_WITH(loaded.loadDomainsFromFile(datadir, true),
        Catch::Matchers::ContainsSubstring("must have matching row counts"));
    torch::save(torch::zeros({8, 2}), test_target);
    REQUIRE_THROWS_WITH(loaded.loadDomainsFromFile(datadir, true),
        Catch::Matchers::ContainsSubstring("Dataset tensor uTest must have shape"));
    std::filesystem::rename(test_target, dataset / "uTest.bin");
    REQUIRE_THROWS_WITH(loaded.loadDomainsFromFile(datadir, true),
        Catch::Matchers::ContainsSubstring("Missing required domain tensor: uTest"));
    REQUIRE_THROWS_WITH(loaded.loadDomainsFromFile((dataset / "absent").string()),
        Catch::Matchers::ContainsSubstring("Dataset directory not found"));
}

TEST_CASE("Hybrid CLI executes Algorithm 1 and creates logs on a fresh run", "[integration][Algorithm1]") {
    const std::string datadir = fourierpinnsTmpDir() + "algorithm1_hybrid/";
    std::filesystem::remove_all(datadir);
    auto result = runCliWithArgs({"fourierpinns", "train", "--model=fouriernn",
        "--solve_pde=true", "--data_mode=generate",
        "--datadir=" + datadir, "--expid=algorithm1", "--seed=1234", "--dim=1",
        "--dtype=float64", "--pde_problem=poisson", "--bnd_typ=dirichlet",
        "--dim1_min=0", "--dim1_max=6.283185307179586",
        "--test_func=test1_k2", "--Ncol_pts=32", "--Nbnd_pts=4", "--Ntest_pts=64",
        "--width=4", "--depth=1", "--ell=4", "--scale_to_period=true", "--include_zero_mode=false", "--lstsq_updates=true", "--lstsq_upd_interval=2",
        "--n_lstsq_upds=10", "--n_lstsq_upds_per_epoch=5",
        "--nn_truncate=true", "--pn_truncate=true", "--training_order=adamfirst",
        "--adam_max_epochs=4", "--lbfgs_max_epochs=1", "--lbfgs_max_iter=2",
        "--print_interval=1", "--write_logs=true"});
    INFO(result.err);
    REQUIRE(result.rc == 0);
    std::ifstream log(datadir + "logs/logs_algorithm1_fouriernn_forwardpde.txt");
    REQUIRE(log.good());
    const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
    REQUIRE(occurrences(text, "[LS ") == 10);
    REQUIRE(occurrences(text, "Truncating after lstsq updates.") == 10);
    REQUIRE(occurrences(text, "LSTSQ Updates epoch=2") == 1);
    REQUIRE(occurrences(text, "LSTSQ Updates epoch=4") == 1);
    REQUIRE(text.find("Starting Adam training") < text.find("[LS 1/10"));
    REQUIRE(text.find("column scaling: enabled") == std::string::npos);
    const auto lbfgs = text.find("Starting LBFGS training");
    REQUIRE(lbfgs != std::string::npos);
    REQUIRE(text.find("[LS ", lbfgs) == std::string::npos);
    REQUIRE(text.find("Group") != std::string::npos);
    REQUIRE(text.find("Fourier") != std::string::npos);
    REQUIRE(text.find("NN bases") != std::string::npos);
    REQUIRE(text.find("NN bias") != std::string::npos);
    REQUIRE(text.find("Truncating", lbfgs) == std::string::npos);
    const auto final_counts = text.find("Final Active Basis Counts (active/total)");
    REQUIRE(final_counts != std::string::npos);
    REQUIRE(final_counts > lbfgs);
    REQUIRE(final_counts < text.find("Finished training"));
    REQUIRE(occurrences(text, "Final Active Basis Counts") == 1);
    const auto summary = text.substr(final_counts, text.find("Finished training") - final_counts);
    REQUIRE(summary.find("Fourier: 1/8") != std::string::npos);
    bool valid_nn_count = false;
    for (int active = 0; active <= 4; ++active) {
        valid_nn_count |= summary.find("NN bases: " + std::to_string(active) + "/4\n") != std::string::npos;
    }
    REQUIRE(valid_nn_count);
    REQUIRE(summary.find("NN bias: 0/1") != std::string::npos);
}

TEST_CASE("LS cutoff preserves remaining Adam and LBFGS training", "[integration][Algorithm1]") {
    const auto datadir = fourierpinnsTmpDir() + "ls_cutoff_hybrid/";
    auto result = runCliWithArgs({"fourierpinns", "train", "--model=fouriernn",
        "--solve_pde=true", "--data_mode=generate",
        "--datadir=" + datadir, "--expid=cutoff", "--dim=1", "--dtype=float64",
        "--pde_problem=poisson", "--test_func=test1_k2", "--dim1_min=0",
        "--dim1_max=6.283185307179586", "--Ncol_pts=32", "--Nbnd_pts=4",
        "--Ntest_pts=64", "--width=4", "--depth=1", "--ell=4",
        "--lstsq_updates=true", "--lstsq_upd_interval=2",
        "--lstsq_stop_epoch=2", "--n_lstsq_upds=20", "--n_lstsq_upds_per_epoch=5",
        "--nn_truncate=true", "--pn_truncate=true", "--training_order=adamfirst",
        "--adam_max_epochs=4", "--lbfgs_max_epochs=1", "--lbfgs_max_iter=2",
        "--print_interval=1", "--write_logs=true"});
    INFO(result.err);
    REQUIRE(result.rc == 0);
    std::ifstream log(datadir + "logs/logs_cutoff_fouriernn_forwardpde.txt");
    REQUIRE(log.good());
    const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
    REQUIRE(occurrences(text, "[LS ") == 5);
    REQUIRE(text.find("LSTSQ Updates epoch=2") != std::string::npos);
    REQUIRE(text.find("LSTSQ Updates epoch=4") == std::string::npos);
    REQUIRE(text.find("epoch: 4 |") != std::string::npos);
    const auto lbfgs = text.find("Starting LBFGS training");
    REQUIRE(lbfgs != std::string::npos);
    REQUIRE(text.find("[LS ", lbfgs) == std::string::npos);
    REQUIRE(text.find("Truncating", text.find("epoch: 3 |")) == std::string::npos);
    REQUIRE(text.find("Finished training") != std::string::npos);
}

TEST_CASE("write_logs error path restores cout stream state", "[integration][smoke]") {
    const std::string datadir = fourierpinnsTmpDir() + "fourierpinns_smoke_logs_restore/";
    std::filesystem::remove_all(datadir);
    // A directory at the intended log filename must make file opening fail.
    std::filesystem::create_directories(datadir + "logs/logs_logs_restore_neuralnet_forwardpde.txt");

    const int rc_fail = runCliNoCapture({
        "fourierpinns",
        "train",
        "--model=neuralnet",
        "--datadir=" + datadir,
        "--expid=logs_restore",
        "--dtype=float32",
        "--data_mode=generate",
        "--write_logs=true",
        "--solve_pde=true",
        "--seed=123",
        "--dim=1",
        "--pde_problem=poisson",
        "--bnd_typ=dirichlet",
        "--point_sampling=equispaced",
        "--dim1_min=0.0",
        "--dim1_max=1.0",
        "--Ncol_pts=8",
        "--Nbnd_pts=4",
        "--Ntest_pts=8",
        "--test_func=test1_k1",
        "--training_order=adamonly",
        "--adam_max_epochs=1",
        "--adam_min_lr=0.001",
        "--adam_max_lr=0.001",
        "--depth=1",
        "--width=8",
        "--net_act=tanh"
    });
    REQUIRE(rc_fail == 1);

    // If cout redirection was not restored on failure, this help call is prone to UB/crash.
    const int rc_help = runCliNoCapture({"fourierpinns", "help"});
    REQUIRE(rc_help == 0);
}

TEST_CASE("train defaults to solving and joins output paths without a trailing slash", "[integration][smoke]") {
    const std::string datadir = fourierpinnsTmpDir() + "fourierpinns_smoke_output_dirs";
    const std::string expid = "smoke_output_dirs";
    const std::string filename_root = expid + "_neuralnet_forwardpde";

    const std::filesystem::path model_path =
        std::filesystem::path(datadir) / "models" / ("models_" + filename_root + ".pt");
    const std::filesystem::path stats_path =
        std::filesystem::path(datadir) / "stats" / ("stats_" + filename_root + ".txt");
    const std::filesystem::path predictions_path =
        std::filesystem::path(datadir) / "predictions" / ("predictions_u_" + filename_root + ".pt");
    const std::filesystem::path csv_path =
        std::filesystem::path(datadir) / "csvs" / ("csvs_" + filename_root + ".txt");

    std::filesystem::remove_all(datadir);

    const CliRunResult res = runCliWithArgs({
        "fourierpinns",
        "train",
        "--model=neuralnet",
        "--datadir=" + datadir,
        "--expid=" + expid,
        "--dtype=float32",
        "--data_mode=generate",
        "--write_logs=true",
        "--print_all_args=true",
        "--seed=321",
        "--dim=1",
        "--pde_problem=poisson",
        "--bnd_typ=dirichlet",
        "--point_sampling=equispaced",
        "--dim1_min=0.0",
        "--dim1_max=1.0",
        "--Ncol_pts=8",
        "--Nbnd_pts=4",
        "--Ntest_pts=8",
        "--test_func=test1_k1",
        "--training_order=adamonly",
        "--adam_max_epochs=1",
        "--adam_min_lr=0.001",
        "--adam_max_lr=0.001",
        "--depth=1",
        "--width=8",
        "--net_act=tanh"
    });

    REQUIRE(res.rc == 0);
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(stats_path));
    REQUIRE(std::filesystem::exists(predictions_path));
    REQUIRE(std::filesystem::exists(csv_path));
    REQUIRE(std::filesystem::exists(std::filesystem::path(datadir) / "logs" / ("logs_" + filename_root + ".txt")));
    REQUIRE(std::filesystem::exists(std::filesystem::path(datadir) / "args" / ("args_" + filename_root + ".txt")));
}
