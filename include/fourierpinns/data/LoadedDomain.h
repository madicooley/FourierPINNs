#pragma once
#include <algorithm>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "fourierpinns/data/DataBundle.h"

void printXinfo(torch::Tensor& X, std::string name, bool printminmax);

/**
 * @class LoadedPDEDomain
 * @brief Utility for loading PDE datasets and managing tensor batches.
 *
 * This class reads domain tensors from disk and stores them in keyed maps.
 * It exposes helpers for generating batches of collocation, boundary and
 * initial condition points and retrieving
 * test data. All tensors are kept on the configured device and dtype.
 */

class LoadedPDEDomain {
private:
    torch::Tensor batch_indices, XcolTrain_batch, f_train_batch;

    std::map<std::string, torch::Tensor> data;
    std::map<std::string, torch::Tensor> basis;
    std::map<std::string, torch::Tensor> basis_batch;

    torch::Device device_;
    torch::Dtype dtype_;

    std::string datadir, domain_typ, test_func, point_sampling;
    int Ncol;
    int Nbnd;
    int Ntest;
    int Nic;
    bool abatchhasbeenset, polybasishasbeenset;

    torch::Tensor* lookupBasis(const std::string& key) {
        if (abatchhasbeenset && basis_batch.count(key)) {
            return &basis_batch[key];
        }
        if (basis.count(key)) {
            return &basis[key];
        }
        return nullptr;
    }


    void fillDataInputs(DataInputs& out, torch::Tensor* X, const std::string* basis_key) {
        out.X = X;
        if (polybasishasbeenset && basis_key != nullptr) {
            torch::Tensor* basis_ptr = lookupBasis(*basis_key);
            TORCH_CHECK(basis_ptr != nullptr, "Missing basis: ", *basis_key);
            out.P.basis = basis_ptr;
        }
    }

    void fillPDEBasisInputs(PDEInputs& out) {
        if (!polybasishasbeenset) {
            return;
        }
        if (pde_problem=="poisson" || pde_problem=="allencahn") {
            TORCH_CHECK(lookupBasis("P_pde") != nullptr, "Missing basis P_pde for PDE problem.");
            out.P_pde = lookupBasis("P_pde");
        } else if (pde_problem=="onewaywave_inhomo") {
            TORCH_CHECK(lookupBasis("P_col_dx") != nullptr, "Missing basis P_col_dx for PDE problem.");
            TORCH_CHECK(lookupBasis("P_col_dy") != nullptr, "Missing basis P_col_dy for PDE problem.");
            out.P_t = lookupBasis("P_col_dx");
            out.P_x = lookupBasis("P_col_dy");
        } else if (pde_problem=="nonss_allencahn") {
            TORCH_CHECK(lookupBasis("P_col_dx") != nullptr, "Missing basis P_col_dx for PDE problem.");
            TORCH_CHECK(lookupBasis("P_col_dyy") != nullptr, "Missing basis P_col_dyy for PDE problem.");
            out.P_t  = lookupBasis("P_col_dx");
            out.P_xx = lookupBasis("P_col_dyy");
        }
    }

public:
    LoadedPDEDomain(int dim_value,
                    int ncol_value,
                    int nbnd_value,
                    int ntest_value,
                    int nic_value,
                    std::string test_func_value,
                    std::string pde_problem_value,
                    std::string bnd_typ_value,
                    std::string domain_typ_value,
                    std::string point_sampling_value,
                    bool homogeneous_pde_value,
                    bool initial_cond_value,
                    const std::vector<std::vector<double>>& domain_bounds,
                    torch::Device device,
                    torch::Dtype dtype)
        : device_(device),
          dtype_(dtype),
          domain_typ(domain_typ_value),
          test_func(test_func_value),
          point_sampling(point_sampling_value),
          Ncol(ncol_value),
          Nbnd(nbnd_value),
          Ntest(ntest_value),
          Nic(nic_value),
          abatchhasbeenset(false),
          polybasishasbeenset(false),
          dim(dim_value),
          pde_problem(pde_problem_value),
          bnd_typ(bnd_typ_value),
          homogeneous_pde(homogeneous_pde_value),
          initial_cond(initial_cond_value),
          domain_bounds_(domain_bounds) {}

    void generateDomainData(bool gen_test=true, bool gen_bnd=true, bool gen_ic=true, bool gen_col=true) {
        TORCH_CHECK(dim >= 1, "generateDomainData requires dim >= 1");
        TORCH_CHECK(domain_typ == "box", "generateDomainData only supports domain_typ='box' for now.");
        TORCH_CHECK(static_cast<int>(domain_bounds_.size()) == dim, "domain_bounds size must match dim.");

        for (int d = 0; d < dim; ++d) {
            const auto idx = static_cast<std::size_t>(d);
            TORCH_CHECK(domain_bounds_[idx].size() == 2, "Each domain_bounds entry must have size 2.");
            TORCH_CHECK(domain_bounds_[idx][1] > domain_bounds_[idx][0], "domain_bounds max must exceed min.");
        }

        auto opts = torch::TensorOptions().device(device_).dtype(dtype_);
        auto sample_1d = [&](int n, double a, double b) {
            if (point_sampling == "random") {
                return (b - a) * torch::rand({n, 1}, opts) + a;
            }
            // default: equispaced
            return torch::linspace(a, b, n, opts).unsqueeze(1);
        };

        auto sample_2d = [&](int n, double ax, double bx, double ay, double by) {
            if (point_sampling == "random") {
                auto xr = (bx - ax) * torch::rand({n, 1}, opts) + ax;
                auto yr = (by - ay) * torch::rand({n, 1}, opts) + ay;
                return torch::cat({xr, yr}, 1);
            }
            int n1 = static_cast<int>(std::round(std::sqrt(static_cast<double>(n))));
            if (n1 < 2) n1 = 2;
            int n2 = static_cast<int>(std::ceil(static_cast<double>(n) / n1));
            auto xs = torch::linspace(ax, bx, n1, opts);
            auto ys = torch::linspace(ay, by, n2, opts);
            auto grid_x = xs.repeat({n2}).reshape({n2, n1});
            auto grid_y = ys.unsqueeze(1).repeat({1, n1});
            auto X = torch::cat({grid_x.reshape({-1, 1}), grid_y.reshape({-1, 1})}, 1);
            if (X.size(0) > n) {
                X = X.index({torch::indexing::Slice(0, n), torch::indexing::Slice()});
            }
            return X;
        };

        auto parse_test1_k = [&]() -> double {
            const std::string prefix = "test1_k";
            auto pos = test_func.find(prefix);
            TORCH_CHECK(pos != std::string::npos, "Expected test_func like 'test1_k10', got: ", test_func);
            std::string kstr = test_func.substr(pos + prefix.size());
            TORCH_CHECK(!kstr.empty(), "Missing k in test_func: ", test_func);
            return std::stod(kstr);
        };

        auto parse_test1_pi_k = [&]() -> double {
            const std::string prefix = "test1_pi_k";
            auto pos = test_func.find(prefix);
            TORCH_CHECK(pos != std::string::npos, "Expected test_func like 'test1_pi_k1', got: ", test_func);
            std::string kstr = test_func.substr(pos + prefix.size());
            TORCH_CHECK(!kstr.empty(), "Missing k in test_func: ", test_func);
            return std::stod(kstr);
        };

        auto eval_u = [&](const torch::Tensor& X) {
            if (dim == 1) {
                auto x = X.index({torch::indexing::Slice(), 0});
                if (test_func.rfind("test1_k", 0) == 0) {
                    // test1_kK (single-frequency manufactured solution): u(x) = sin(kx)
                    double k = parse_test1_k();
                    return torch::sin(k * x).unsqueeze(1);
                }
                if (test_func.rfind("test1_pi_k", 0) == 0) {
                    // test1_pi_kK: u(x) = sin(pi*k*x) on [0,1]
                    double k = parse_test1_pi_k();
                    return torch::sin(M_PI * k * x).unsqueeze(1);
                }
                if (test_func == "test2") {
                    // test2 (Sec 6.3): u(x) = sin(x) + 0.1 sin(20x) + 0.05 cos(100x)
                    auto u = torch::sin(x)
                           + 0.1 * torch::sin(20.0 * x)
                           + 0.05 * torch::cos(100.0 * x);
                    return u.unsqueeze(1);
                }
                if (test_func == "test3") {
                    // test3 (Sec 6.3): u(x) = sin(6x) cos(100x)
                    auto u = torch::sin(6.0 * x) * torch::cos(100.0 * x);
                    return u.unsqueeze(1);
                }
                TORCH_CHECK(false, "Unsupported test_func in generateDomainData: ", test_func);
            }
            if (dim == 2) {
                auto x = X.index({torch::indexing::Slice(), 0});
                auto y = X.index({torch::indexing::Slice(), 1});
                if (test_func == "test4") {
                    // test4 (Sec 6.4.1): u(x,y) = sin(100x) sin(100y)
                    auto u = torch::sin(100.0 * x) * torch::sin(100.0 * y);
                    return u.unsqueeze(1);
                }
                if (test_func == "test5") {
                    // test5 (Sec 6.4.1): u(x,y) = sin(6x)cos(20x) + sin(6y)cos(20y)
                    auto u = torch::sin(6.0 * x) * torch::cos(20.0 * x)
                           + torch::sin(6.0 * y) * torch::cos(20.0 * y);
                    return u.unsqueeze(1);
                }
                if (test_func == "test6") {
                    // test6 (Sec 6.4.2): u(x,y) = (sin x + 0.1 sin 20x + cos 100x)
                    //                          * (sin y + 0.1 sin 20y + cos 100y)
                    auto ux = torch::sin(x) + 0.1 * torch::sin(20.0 * x) + torch::cos(100.0 * x);
                    auto uy = torch::sin(y) + 0.1 * torch::sin(20.0 * y) + torch::cos(100.0 * y);
                    auto u = ux * uy;
                    return u.unsqueeze(1);
                }
                if (test_func == "test7") {
                    // test7 (Sec 6.5): 1D one-way wave, u(t,x) = sin(pi x) cos(10 pi t)
                    //                                  + sin(2 pi x) cos(20 pi t)
                    // Note: t is first column, x is second.
                    auto t = x;
                    auto xx = y;
                    auto u = torch::sin(M_PI * xx) * torch::cos(10.0 * M_PI * t)
                           + torch::sin(2.0 * M_PI * xx) * torch::cos(20.0 * M_PI * t);
                    return u.unsqueeze(1);
                }
                TORCH_CHECK(false, "Unsupported test_func in generateDomainData: ", test_func);
            }
            TORCH_CHECK(false, "generateDomainData only implemented for dim=1 or dim=2");
            return torch::zeros({X.size(0), 1}, opts);
        };

        auto eval_f = [&](const torch::Tensor& X) {
            if (dim == 1) {
                auto x = X.index({torch::indexing::Slice(), 0});
                torch::Tensor u;
                torch::Tensor u_xx;
                if (test_func.rfind("test1_k", 0) == 0) {
                    // test1_kK (single-frequency manufactured solution): u(x) = sin(kx)
                    double k = parse_test1_k();
                    u = torch::sin(k * x);
                    u_xx = -k * k * u;

                } else if (test_func.rfind("test1_pi_k", 0) == 0) {
                    // test1_pi_kK: u(x) = sin(pi*k*x) on [0,1]
                    double k = parse_test1_pi_k();
                    u = torch::sin(M_PI * k * x);
                    u_xx = -(M_PI * M_PI * k * k) * u;

                } else if (test_func == "test2") {
                    // test2 (Sec 6.3): u(x) = sin(x) + 0.1 sin(20x) + 0.05 cos(100x)
                    u = torch::sin(x)
                      + 0.1 * torch::sin(20.0 * x)
                      + 0.05 * torch::cos(100.0 * x);
                    u_xx = -torch::sin(x)
                         - 0.1 * (20.0 * 20.0) * torch::sin(20.0 * x)
                         - 0.05 * (100.0 * 100.0) * torch::cos(100.0 * x);

                } else if (test_func == "test3") {
                    // test3 (Sec 6.3): u(x) = sin(6x) cos(100x)
                    auto sin6 = torch::sin(6.0 * x);
                    auto cos6 = torch::cos(6.0 * x);
                    auto sin100 = torch::sin(100.0 * x);
                    auto cos100 = torch::cos(100.0 * x);
                    u = sin6 * cos100;
                    u_xx = -(10036.0) * sin6 * cos100 - 1200.0 * cos6 * sin100;

                } else {
                    TORCH_CHECK(false, "Unsupported test_func in generateDomainData: ", test_func);
                }

                if (pde_problem == "poisson") {
                    // PoissonPDE returns Laplacian(u)
                    return u_xx.unsqueeze(1);
                }

                if (pde_problem == "allencahn") {
                    // u_xx + u(u^2 - 1) = f(x)
                    auto reaction = u * (u * u - 1.0);
                    return (u_xx + reaction).unsqueeze(1);
                }
                TORCH_CHECK(false, "Unsupported pde_problem in generateDomainData: ", pde_problem);
            }

            if (dim == 2) {
                auto x = X.index({torch::indexing::Slice(), 0});
                auto y = X.index({torch::indexing::Slice(), 1});
                torch::Tensor u;
                torch::Tensor u_xx;
                torch::Tensor u_yy;
                if (test_func == "test4") {
                    // test4 (Sec 6.4.1): u(x,y) = sin(100x) sin(100y)
                    u = torch::sin(100.0 * x) * torch::sin(100.0 * y);
                    u_xx = -(100.0 * 100.0) * u;
                    u_yy = -(100.0 * 100.0) * u;

                } else if (test_func == "test5") {
                    // test5 (Sec 6.4.1): u(x,y) = sin(6x)cos(20x) + sin(6y)cos(20y)
                    auto sin6x = torch::sin(6.0 * x);
                    auto cos6x = torch::cos(6.0 * x);
                    auto sin20x = torch::sin(20.0 * x);
                    auto cos20x = torch::cos(20.0 * x);
                    auto sin6y = torch::sin(6.0 * y);
                    auto cos6y = torch::cos(6.0 * y);
                    auto sin20y = torch::sin(20.0 * y);
                    auto cos20y = torch::cos(20.0 * y);
                    auto u_x = sin6x * cos20x;
                    auto u_y = sin6y * cos20y;
                    u = u_x + u_y;
                    u_xx = -436.0 * sin6x * cos20x - 240.0 * cos6x * sin20x;
                    u_yy = -436.0 * sin6y * cos20y - 240.0 * cos6y * sin20y;

                } else if (test_func == "test6") {
                    // test6 (Sec 6.4.2): 2D steady Allen-Cahn manufactured solution
                    auto sinx = torch::sin(x);
                    auto cosx = torch::cos(x);
                    auto sin20x = torch::sin(20.0 * x);
                    auto cos20x = torch::cos(20.0 * x);
                    auto sin100x = torch::sin(100.0 * x);
                    auto cos100x = torch::cos(100.0 * x);

                    auto siny = torch::sin(y);
                    auto cosy = torch::cos(y);
                    auto sin20y = torch::sin(20.0 * y);
                    auto cos20y = torch::cos(20.0 * y);
                    auto sin100y = torch::sin(100.0 * y);
                    auto cos100y = torch::cos(100.0 * y);

                    auto ux = sinx + 0.1 * sin20x + cos100x;
                    auto uy = siny + 0.1 * sin20y + cos100y;
                    u = ux * uy;

                    auto uxx = -sinx - 0.1 * 400.0 * sin20x - 10000.0 * cos100x;
                    auto uyy = -siny - 0.1 * 400.0 * sin20y - 10000.0 * cos100y;
                    u_xx = uxx * uy;
                    u_yy = ux * uyy;

                } else if (test_func == "test7") {
                    // test7 (Sec 6.5): 1D one-way wave, return forcing v = u_t + 10 u_x
                    // Note: t is first column, x is second.
                    auto t = x;
                    auto xx = y;
                    auto sin1 = torch::sin(M_PI * xx);
                    auto sin2 = torch::sin(2.0 * M_PI * xx);
                    auto cos1 = torch::cos(M_PI * xx);
                    auto cos2 = torch::cos(2.0 * M_PI * xx);
                    auto cost1 = torch::cos(10.0 * M_PI * t);
                    auto cost2 = torch::cos(20.0 * M_PI * t);
                    auto sint1 = torch::sin(10.0 * M_PI * t);
                    auto sint2 = torch::sin(20.0 * M_PI * t);

                    u = sin1 * cost1 + sin2 * cost2;
                    auto u_t = -10.0 * M_PI * sin1 * sint1 - 20.0 * M_PI * sin2 * sint2;
                    auto u_x = M_PI * cos1 * cost1 + 2.0 * M_PI * cos2 * cost2;
                    if (pde_problem == "onewaywave_inhomo") {
                        return (u_t + 10.0 * u_x).unsqueeze(1);
                    }
                    TORCH_CHECK(false, "Unsupported pde_problem in generateDomainData: ", pde_problem);
                } else {
                    TORCH_CHECK(false, "Unsupported test_func in generateDomainData: ", test_func);
                }

                if (test_func == "test7") {
                    TORCH_CHECK(false, "Unexpected test7 path in generateDomainData");
                }

                if (pde_problem == "poisson") {
                    return (u_xx + u_yy).unsqueeze(1);
                }

                if (pde_problem == "allencahn") {
                    auto reaction = u * (u * u - 1.0);
                    return (u_xx + u_yy + reaction).unsqueeze(1);
                }
                TORCH_CHECK(false, "Unsupported pde_problem in generateDomainData: ", pde_problem);
            }
            TORCH_CHECK(false, "generateDomainData only implemented for dim=1 or dim=2");
            return torch::zeros({X.size(0), 1}, opts);
        };

        double ax = domain_bounds_[0][0];
        double bx = domain_bounds_[0][1];
        double ay = (dim > 1) ? domain_bounds_[1][0] : 0.0;
        double by = (dim > 1) ? domain_bounds_[1][1] : 0.0;

        if (gen_col) {
            torch::Tensor Xcol = (dim == 1) ? sample_1d(Ncol, ax, bx) : sample_2d(Ncol, ax, bx, ay, by);
            Xcol.set_requires_grad(true);
            data["XcolTrain"] = Xcol;
        }

        if (gen_test) {
            torch::Tensor Xtest = (dim == 1) ? sample_1d(Ntest, ax, bx) : sample_2d(Ntest, ax, bx, ay, by);
            Xtest.set_requires_grad(false);
            data["XTest"] = Xtest;
            data["uTest"] = eval_u(Xtest).to(opts).detach();
        }

        if (gen_bnd && (bnd_typ == "dirichlet" || bnd_typ == "none")) {
            if (dim == 1) {
                int nmin = Nbnd / 2;
                int nmax = Nbnd - nmin;
                auto Xmin = torch::full({nmin, 1}, ax, opts);
                auto Xmax = torch::full({nmax, 1}, bx, opts);
                auto Xbnd = torch::cat({Xmin, Xmax}, 0);
                Xbnd.set_requires_grad(true);
                data["XbndTrain"] = Xbnd;
                data["uTrain"]    = eval_u(Xbnd).to(opts).detach();
            } else if (dim == 2) {
                if (test_func == "test7") {
                    // Only spatial boundaries: x=0 and x=1 across t in [0,1]
                    int nmin = Nbnd / 2;
                    int nmax = Nbnd - nmin;
                    auto tmin = torch::linspace(ax, bx, nmin, opts).unsqueeze(1);
                    auto tmax = torch::linspace(ax, bx, nmax, opts).unsqueeze(1);
                    auto xmin = torch::full({nmin, 1}, ay, opts);
                    auto xmax = torch::full({nmax, 1}, by, opts);
                    auto Xmin = torch::cat({tmin, xmin}, 1);
                    auto Xmax = torch::cat({tmax, xmax}, 1);
                    auto Xbnd = torch::cat({Xmin, Xmax}, 0);
                    Xbnd.set_requires_grad(true);
                    data["XbndTrain"] = Xbnd;
                    data["uTrain"]    = eval_u(Xbnd).to(opts).detach();
                } else {
                    int per_edge = Nbnd / 4;
                    int rem = Nbnd - 4 * per_edge;
                    auto xs = torch::linspace(ax, bx, per_edge + (rem > 0 ? 1 : 0), opts).unsqueeze(1);
                    auto ys = torch::linspace(ay, by, per_edge + (rem > 1 ? 1 : 0), opts).unsqueeze(1);
                    auto xb = torch::linspace(ax, bx, per_edge + (rem > 2 ? 1 : 0), opts).unsqueeze(1);
                    auto yb = torch::linspace(ay, by, per_edge + (rem > 3 ? 1 : 0), opts).unsqueeze(1);

                    auto bottom = torch::cat({xs, torch::full_like(xs, ay)}, 1);
                    auto top    = torch::cat({xs, torch::full_like(xs, by)}, 1);
                    auto left   = torch::cat({torch::full_like(ys, ax), ys}, 1);
                    auto right  = torch::cat({torch::full_like(yb, bx), yb}, 1);
                    auto Xbnd = torch::cat({bottom, top, left, right}, 0);
                    if (Xbnd.size(0) > Nbnd) {
                        Xbnd = Xbnd.index({torch::indexing::Slice(0, Nbnd), torch::indexing::Slice()});
                    }
                    Xbnd.set_requires_grad(true);
                    data["XbndTrain"] = Xbnd;
                    data["uTrain"]    = eval_u(Xbnd).to(opts).detach();
                }
            }
        } else if (gen_bnd && bnd_typ == "periodic") {
            if (dim == 1) {
                int nmin = Nbnd / 2;
                int nmax = Nbnd - nmin;
                auto Xmin = torch::full({nmin, 1}, ax, opts);
                auto Xmax = torch::full({nmax, 1}, bx, opts);
                Xmin.set_requires_grad(true);
                Xmax.set_requires_grad(true);
                data["XbndTrainMin"] = Xmin;
                data["XbndTrainMax"] = Xmax;
                auto Xbnd = torch::cat({Xmin, Xmax}, 0);
                data["XbndTrain"] = Xbnd;
                data["uTrain"] = eval_u(Xbnd).to(opts).detach();
            } else {
                // periodic in x (second column), t in first column
                int nmin = Nbnd / 2;
                int nmax = Nbnd - nmin;
                auto tmin = torch::linspace(ax, bx, nmin, opts).unsqueeze(1);
                auto tmax = torch::linspace(ax, bx, nmax, opts).unsqueeze(1);
                auto xmin = torch::full({nmin, 1}, ay, opts);
                auto xmax = torch::full({nmax, 1}, by, opts);
                auto Xmin = torch::cat({tmin, xmin}, 1);
                auto Xmax = torch::cat({tmax, xmax}, 1);
                Xmin.set_requires_grad(true);
                Xmax.set_requires_grad(true);
                data["XbndTrainMin"] = Xmin;
                data["XbndTrainMax"] = Xmax;
                auto Xbnd = torch::cat({Xmin, Xmax}, 0);
                data["XbndTrain"] = Xbnd;
                data["uTrain"] = eval_u(Xbnd).to(opts).detach();
            }
        } else {
            TORCH_CHECK(false, "Unsupported bnd_typ in generateDomainData: ", bnd_typ);
        }

        if (gen_col && !homogeneous_pde) {
            data["fTrain"] = eval_f(data["XcolTrain"]).to(opts).detach();
        }

        if (gen_ic && initial_cond && dim == 2 && test_func == "test7") {
            // initial condition at t=0, x in [0,1]
            auto xx = torch::linspace(ay, by, Nic, opts).unsqueeze(1);
            auto tt = torch::zeros({Nic, 1}, opts);
            auto Xic = torch::cat({tt, xx}, 1);
            Xic.set_requires_grad(true);
            data["XicTrain"] = Xic;
            data["uicTrain"] = eval_u(Xic).to(opts).detach();
        }

        printDomainInfo();
        std::cout << "Done generating datasets." << "\n";
        std::cout << "_____________________________________________________________________________________________\n";
    }

    void saveDomainData(const std::string& dir) const {
        validateDataset(data, false);
        const auto data_dir = std::filesystem::path(dir) / "data";
        std::filesystem::create_directories(data_dir.parent_path());
        if (!std::filesystem::create_directory(data_dir)) {
            throw std::runtime_error("Refusing to overwrite existing dataset directory: " + data_dir.string());
        }
        for (const auto& [key, tensor] : data) {
            torch::save(tensor.detach().cpu(), (data_dir / (key + ".pt")).string());
        }
        std::cout << "Saved dataset to: " << data_dir << "\n";
    }

    void loadDomainsFromFile(const std::string& dir, bool test_only = false) {
        const auto data_dir = std::filesystem::path(dir) / "data";
        if (!std::filesystem::is_directory(data_dir)) {
            throw std::runtime_error("Dataset directory not found: " + data_dir.string());
        }

        const auto required = requiredDatasetKeys(test_only);
        std::map<std::string, std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            const auto& path = entry.path();
            if (!entry.is_regular_file() || path.extension() != ".pt") continue;
            const auto stem = path.stem().string();
            const auto separator = stem.find_last_of('_');
            const auto key = separator == std::string::npos ? stem : stem.substr(separator + 1);
            if (std::find(required.begin(), required.end(), key) == required.end()) continue;
            const auto [it, inserted] = files.emplace(key, path);
            if (!inserted) {
                throw std::runtime_error("Duplicate dataset tensor " + key + ": " +
                                         it->second.string() + " and " + path.string());
            }
        }

        requireDatasetKeys(files, required);
        std::map<std::string, torch::Tensor> loaded;
        for (const auto& [key, path] : files) {
            torch::Tensor tensor;
            torch::load(tensor, path.string(), torch::kCPU);
            loaded.emplace(key, tensor.detach().to(device_, dtype_));
        }
        validateDataset(loaded, test_only);
        if (!test_only && bnd_typ == "periodic") {
            loaded["XbndTrain"] = torch::cat({loaded.at("XbndTrainMin"), loaded.at("XbndTrainMax")}, 0);
        }
        for (auto& [key, tensor] : loaded) {
            tensor.set_requires_grad(key.front() == 'X' && key != "XTest");
            data[key] = std::move(tensor);
        }
        datadir = data_dir.string();
        printDomainInfo();
        std::cout << "Loaded dataset from: " << data_dir << "\n";
    }

private:
    std::vector<std::string> requiredDatasetKeys(bool test_only) const {
        std::vector<std::string> keys{"XTest", "uTest"};
        if (test_only) return keys;
        keys.push_back("XcolTrain");
        if (!homogeneous_pde) keys.push_back("fTrain");
        if (initial_cond) {
            keys.push_back("XicTrain");
            keys.push_back("uicTrain");
        }
        if (bnd_typ == "periodic") {
            keys.push_back("XbndTrainMin");
            keys.push_back("XbndTrainMax");
        } else {
            keys.push_back("XbndTrain");
            keys.push_back("uTrain");
        }
        return keys;
    }

    template <typename DatasetMap>
    static void requireDatasetKeys(const DatasetMap& tensors, const std::vector<std::string>& required) {
        for (const auto& key : required) {
            if (!tensors.count(key)) throw std::runtime_error("Missing required domain tensor: " + key);
        }
    }

    void validateDataset(const std::map<std::string, torch::Tensor>& tensors, bool test_only) const {
        const auto required = requiredDatasetKeys(test_only);
        requireDatasetKeys(tensors, required);
        for (const auto& key : required) {
            const auto& tensor = tensors.at(key);
            const int columns = key.front() == 'X' ? dim : 1;
            TORCH_CHECK(tensor.defined() && tensor.dim() == 2 && tensor.size(0) > 0 &&
                        tensor.size(1) == columns, "Dataset tensor ", key,
                        " must have shape [N, ", columns, "] with N > 0");
        }
        auto checkRows = [&](const std::string& a, const std::string& b) {
            TORCH_CHECK(tensors.at(a).size(0) == tensors.at(b).size(0),
                        "Dataset tensors ", a, " and ", b, " must have matching row counts");
        };
        checkRows("XTest", "uTest");
        if (test_only) return;
        if (!homogeneous_pde) checkRows("XcolTrain", "fTrain");
        if (initial_cond) checkRows("XicTrain", "uicTrain");
        if (bnd_typ == "periodic") checkRows("XbndTrainMin", "XbndTrainMax");
        else checkRows("XbndTrain", "uTrain");
    }

public:


    void printDomainInfo() {
        std::cout << "\n**************************** Printing Domain Info ****************************"<< std::endl;
        for (const auto& entry : data) {
            std::string name = entry.first;
            torch::Tensor P  = entry.second;
            printXinfo(P, name, /*printminmax=*/false);
            std::cout<<name<<" min="<<torch::min(P).item<float>()<<", max="<<torch::max(P).item<float>()<<"\n";
        }
        std::cout << "\n******************************************************************************"<< std::endl;
    }

    std::string getDataDir() { return this->datadir; }
    int getNcol()            { return static_cast<int>(data["XcolTrain"].sizes()[0]); }

    torch::Tensor getTestingSolutionMean() { return torch::mean(data["uTest"]); }


    torch::Device getDevice() { return device_; }
    torch::Dtype getDtype()   { return dtype_;  } 

    
    void clearBatch() {
        abatchhasbeenset = false;
        batch_indices = torch::Tensor();
        XcolTrain_batch = torch::Tensor();
        f_train_batch = torch::Tensor();
        basis_batch.clear();
    }

    void setBatchIndices(const torch::Tensor& idx) {
        TORCH_CHECK(idx.defined(), "Batch indices must be defined.");
        batch_indices = idx;
        abatchhasbeenset = true;

        XcolTrain_batch = data["XcolTrain"].index({idx}).contiguous();
        if (!homogeneous_pde) {
            f_train_batch = data["fTrain"].index({idx}).contiguous();
        }

        if (polybasishasbeenset) {
            auto slice = [&](const std::string& key) {
                if (basis.count(key)) {
                    basis_batch[key] = basis[key].index({idx}).contiguous();
                }
            };
            slice("P_col");
            slice("P_col_dx");
            slice("P_col_dy");
            slice("P_col_dyy");
            slice("P_col_dyyy");
            slice("P_col_dyyyy");
            slice("P_pde");
        }
    }

    void setRandomCollocationBatch(int batch_size, bool shuffle=true) {
        const auto n = data["XcolTrain"].size(0);
        const auto batch_size_i64 = static_cast<int64_t>(batch_size);
        if (batch_size <= 0 || batch_size_i64 >= n) {
            clearBatch();
            return;
        }
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(data["XcolTrain"].device());
        auto idx = torch::arange(n, opts);
        if (shuffle) {
            idx = idx.index({torch::randperm(n, opts)});
        }
        idx = idx.index({torch::indexing::Slice(0, batch_size_i64)});
        setBatchIndices(idx);
    }

    void setPolyBasis(std::map<std::string, torch::Tensor> basis_map, bool requires_grad = false) {
        (void)requires_grad;
        this->basis=std::move(basis_map);

        std::cout<<"\nsetting bases.\n";
        TORCH_CHECK(this->basis.count("P_col"), "P_col basis required.");
        for (auto& entry : this->basis) {
            auto& P = entry.second;
            P.set_requires_grad(false);
        }

        for (const auto& entry : this->basis) {
            std::string name = entry.first;
            torch::Tensor P  = entry.second;
            printXinfo(P, name, /*printminmax=*/false);
            std::cout<<name<<" min="<<torch::min(P).item<float>()<<", max="<<torch::max(P).item<float>()<<"\n";
        }

        polybasishasbeenset=true;
    }


    BndInputs getBndData() {
        BndInputs inputs;
        const std::string key = "P_bnd";
        fillDataInputs(inputs.data, &data["XbndTrain"], &key);
        inputs.u = &data["uTrain"];
        return inputs;
    }

    BndInputs getBndData_A() {
        BndInputs inputs;
        const std::string key = "P_bnd_min";
        fillDataInputs(inputs.data, &data["XbndTrainMin"], &key);

        if (polybasishasbeenset && bnd_typ=="periodic") {
            inputs.P_x = &basis["P_bnd_min_dy"];
            if (basis.count("P_bnd_min_dyy"))   inputs.P_xx = &basis["P_bnd_min_dyy"];
            if (basis.count("P_bnd_min_dyyy"))  inputs.P_xxx = &basis["P_bnd_min_dyyy"];
            if (basis.count("P_bnd_min_dyyyy")) inputs.P_xxxx = &basis["P_bnd_min_dyyyy"];
        }
        return inputs;
    }

    BndInputs getBndData_B() {
        BndInputs inputs;
        const std::string key = "P_bnd_max";
        fillDataInputs(inputs.data, &data["XbndTrainMax"], &key);

        if (polybasishasbeenset && bnd_typ=="periodic") {
            inputs.P_x = &basis["P_bnd_max_dy"];
            if (basis.count("P_bnd_max_dyy"))   inputs.P_xx = &basis["P_bnd_max_dyy"];
            if (basis.count("P_bnd_max_dyyy"))  inputs.P_xxx = &basis["P_bnd_max_dyyy"];
            if (basis.count("P_bnd_max_dyyyy")) inputs.P_xxxx = &basis["P_bnd_max_dyyyy"];
        }
        return inputs;
    }

    BndInputs getPeriodicBndDataPair() {
        BndInputs inputs;
        const std::string key_min = "P_bnd_min";
        const std::string key_max = "P_bnd_max";
        fillDataInputs(inputs.data, &data["XbndTrainMin"], &key_min);
        fillDataInputs(inputs.data_b, &data["XbndTrainMax"], &key_max);

        if (polybasishasbeenset && bnd_typ=="periodic") {
            inputs.P_x = &basis["P_bnd_min_dy"];
            inputs.P_x_b = &basis["P_bnd_max_dy"];
            if (basis.count("P_bnd_min_dyy") && basis.count("P_bnd_max_dyy")) {
                inputs.P_xx = &basis["P_bnd_min_dyy"];
                inputs.P_xx_b = &basis["P_bnd_max_dyy"];
            }
            if (basis.count("P_bnd_min_dyyy") && basis.count("P_bnd_max_dyyy")) {
                inputs.P_xxx = &basis["P_bnd_min_dyyy"];
                inputs.P_xxx_b = &basis["P_bnd_max_dyyy"];
            }
            if (basis.count("P_bnd_min_dyyyy") && basis.count("P_bnd_max_dyyyy")) {
                inputs.P_xxxx = &basis["P_bnd_min_dyyyy"];
                inputs.P_xxxx_b = &basis["P_bnd_max_dyyyy"];
            }
        }
        return inputs;
    }

    void updateICData(torch::Tensor XicTrain) {
        XicTrain.set_requires_grad(true);
        data["XicTrain"]=XicTrain;
    }

    void updateICSoln(torch::Tensor uic_pred) {
        uic_pred.set_requires_grad(true);
        data["uicTrain"]=uic_pred;
    }

    ICInputs getICData() {
        ICInputs inputs;
        const std::string key = "P_ic";
        fillDataInputs(inputs.data, &data["XicTrain"], &key);
        inputs.u = &data["uicTrain"];
        return inputs;
    }

    PDEInputs getPDEData() {
        PDEInputs inputs;
        const std::string key = "P_col";
        fillDataInputs(inputs.data, &getColNodes(), &key);

        if (!homogeneous_pde) inputs.f = &getPDESolution();

        fillPDEBasisInputs(inputs);

        return inputs;
    }

    TestInputs getTestData() {
        TestInputs inputs;
        if (dim == 2) {
            // Only coordinates are needed for chunked test evaluation.
            inputs.data.X = &data["XTest"];
        } else {
            const std::string key = "P_test";
            fillDataInputs(inputs.data, &data["XTest"], &key);
        }
        inputs.u = &data["uTest"];
        return inputs;
    }

    torch::Tensor& getColNodes() {
        if (abatchhasbeenset) return XcolTrain_batch;
        return data["XcolTrain"];
    }

    torch::Tensor& getPDESolution() {
        if (abatchhasbeenset) return f_train_batch;
        return data["fTrain"];
    }

    torch::Tensor& getBndNodes()             { return data["XbndTrain"]; }
    torch::Tensor& getBndNodesA()            { return data["XbndTrainMin"]; }
    torch::Tensor& getBndNodesB()            { return data["XbndTrainMax"]; }
    torch::Tensor& getICNodes()              { return data["XicTrain"];  }
    torch::Tensor& getTestingNodes()         { return data["XTest"];     }

    torch::Tensor& getBndSolution()          { return data["uTrain"]; }
    torch::Tensor& getTestingSolution()      { return data["uTest"]; }

    int dim;
    std::string pde_problem, bnd_typ;
    bool homogeneous_pde, initial_cond;
    std::vector<std::vector<double>> domain_bounds_;

};
