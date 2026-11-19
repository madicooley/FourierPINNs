#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>

#include "fourierpinns/config/Args.h"
#include "fourierpinns/data/LoadedDomain.h"
#include "fourierpinns/basis/FourierBasisUtils.h"
#include "fourierpinns/models/FourierLayer.h"

namespace {
constexpr double kTol = 1e-8;

ModelConfig make_fourierlayer_config(int dim, int ell, const std::vector<std::vector<double>>& bounds) {
    ModelConfig cfg;
    cfg.model_type = "fourierlayer";
    cfg.input_dim = dim;
    cfg.output_dim = 1;
    cfg.verbose = false;
    cfg.domain_bounds = bounds;

    FourierLayerArgs fl;
    fl.ell = ell;
    fl.include_zero_mode = true;
    fl.scale_to_period = false;
    fl.periodic_deriv_dim = -1;
    fl.pn_truncate = false;
    fl.pn_trunc_thresh = 1e-6f;
    fl.pn_l1_reg_coeff = 0.0f;
    fl.pn_l2_reg_coeff = 0.0f;
    cfg.arch = fl;
    return cfg;
}
} // namespace

// Checks basic dataset generation, shapes, and gradient flags.
TEST_CASE("LoadedDomain generateDomainData shapes and grads", "[LoadedDomain]") {
    LoadedPDEDomain domain(/*dim=*/1,
                           /*Ncol=*/5,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/0,
                           /*test_func=*/"test1_k2",
                           /*pde_problem=*/"poisson",
                           /*bnd_typ=*/"dirichlet",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/false,
                           /*initial_cond=*/false,
                           /*domain_bounds=*/{{0.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    REQUIRE(domain.getColNodes().sizes() == torch::IntArrayRef({5, 1}));
    REQUIRE(domain.getBndNodes().sizes() == torch::IntArrayRef({4, 1}));
    REQUIRE(domain.getTestingNodes().sizes() == torch::IntArrayRef({6, 1}));

    REQUIRE(domain.getColNodes().requires_grad());
    REQUIRE(domain.getBndNodes().requires_grad());
    REQUIRE_FALSE(domain.getTestingNodes().requires_grad());
}

// Ensures periodic min/max boundary nodes are created at domain bounds.
TEST_CASE("LoadedDomain periodic boundary min/max sets", "[LoadedDomain]") {
    LoadedPDEDomain domain(/*dim=*/1,
                           /*Ncol=*/5,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/0,
                           /*test_func=*/"test1_k1",
                           /*pde_problem=*/"poisson",
                           /*bnd_typ=*/"periodic",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/true,
                           /*initial_cond=*/false,
                           /*domain_bounds=*/{{-1.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    auto Xmin = domain.getBndNodesA();
    auto Xmax = domain.getBndNodesB();
    auto min_expected = torch::full_like(Xmin, -1.0);
    auto max_expected = torch::full_like(Xmax, 1.0);

    REQUIRE(torch::allclose(Xmin, min_expected, kTol, kTol));
    REQUIRE(torch::allclose(Xmax, max_expected, kTol, kTol));
}

// Verifies PDE basis wiring for one-way wave problem.
TEST_CASE("LoadedDomain PDE basis wiring for onewaywave_inhomo", "[LoadedDomain]") {
    LoadedPDEDomain domain(/*dim=*/2,
                           /*Ncol=*/5,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/4,
                           /*test_func=*/"test7",
                           /*pde_problem=*/"onewaywave_inhomo",
                           /*bnd_typ=*/"dirichlet",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/true,
                           /*initial_cond=*/true,
                           /*domain_bounds=*/{{0.0, 1.0}, {-1.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    auto cfg = make_fourierlayer_config(/*dim=*/2, /*ell=*/1, {{0.0, 1.0}, {-1.0, 1.0}});
    auto layer = VanillaFourierLayer(cfg, "onewaywave_inhomo", torch::kCPU, torch::kDouble);
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    domain.setPolyBasis(bases, /*requires_grad=*/false);

    auto pde = domain.getPDEData();
    REQUIRE(pde.data.P.basis != nullptr);
    REQUIRE(pde.P_t != nullptr);
    REQUIRE(pde.P_x != nullptr);
}

// Verifies PDE basis wiring for non-steady-state Allen-Cahn.
TEST_CASE("LoadedDomain PDE basis wiring for nonss_allencahn", "[LoadedDomain]") {
    LoadedPDEDomain domain(/*dim=*/2,
                           /*Ncol=*/5,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/4,
                           /*test_func=*/"test7",
                           /*pde_problem=*/"nonss_allencahn",
                           /*bnd_typ=*/"periodic",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/true,
                           /*initial_cond=*/true,
                           /*domain_bounds=*/{{0.0, 1.0}, {-1.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    auto cfg = make_fourierlayer_config(/*dim=*/2, /*ell=*/1, {{0.0, 1.0}, {-1.0, 1.0}});
    auto layer = VanillaFourierLayer(cfg, "nonss_allencahn", torch::kCPU, torch::kDouble);
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    domain.setPolyBasis(bases, /*requires_grad=*/false);

    auto pde = domain.getPDEData();
    REQUIRE(pde.P_t != nullptr);
    REQUIRE(pde.P_xx != nullptr);
}

// Homogeneous PDEs should omit forcing tensor in PDE inputs.
TEST_CASE("LoadedDomain homogeneous PDE omits forcing data", "[LoadedDomain]") {
    LoadedPDEDomain domain(/*dim=*/1,
                           /*Ncol=*/4,
                           /*Nbnd=*/2,
                           /*Ntest=*/3,
                           /*Nic=*/0,
                           /*test_func=*/"test1_k1",
                           /*pde_problem=*/"poisson",
                           /*bnd_typ=*/"dirichlet",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/true,
                           /*initial_cond=*/false,
                           /*domain_bounds=*/{{0.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();
    auto pde = domain.getPDEData();
    REQUIRE(pde.f == nullptr);
}

// Collocation batching should return subset tensors and restore full data after clear.
TEST_CASE("LoadedDomain collocation batching selects subset", "[LoadedDomain][Batch]") {
    LoadedPDEDomain domain(/*dim=*/1,
                           /*Ncol=*/10,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/0,
                           /*test_func=*/"test1_k2",
                           /*pde_problem=*/"poisson",
                           /*bnd_typ=*/"dirichlet",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/false,
                           /*initial_cond=*/false,
                           /*domain_bounds=*/{{0.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    domain.setRandomCollocationBatch(/*batch_size=*/4, /*shuffle=*/false);
    REQUIRE(domain.getColNodes().sizes() == torch::IntArrayRef({4, 1}));
    REQUIRE(domain.getPDESolution().sizes() == torch::IntArrayRef({4, 1}));

    domain.clearBatch();
    REQUIRE(domain.getColNodes().sizes() == torch::IntArrayRef({10, 1}));
    REQUIRE(domain.getPDESolution().sizes() == torch::IntArrayRef({10, 1}));
}

// Batched PDE inputs should use sliced Fourier bases.
TEST_CASE("LoadedDomain batching uses sliced bases for nonss_allencahn", "[LoadedDomain][Batch]") {
    LoadedPDEDomain domain(/*dim=*/2,
                           /*Ncol=*/8,
                           /*Nbnd=*/4,
                           /*Ntest=*/6,
                           /*Nic=*/4,
                           /*test_func=*/"test7",
                           /*pde_problem=*/"nonss_allencahn",
                           /*bnd_typ=*/"periodic",
                           /*domain_typ=*/"box",
                           /*point_sampling=*/"equispaced",
                           /*homogeneous_pde=*/true,
                           /*initial_cond=*/true,
                           /*domain_bounds=*/{{0.0, 1.0}, {-1.0, 1.0}},
                           torch::kCPU,
                           torch::kDouble);

    domain.generateDomainData();

    auto cfg = make_fourierlayer_config(/*dim=*/2, /*ell=*/1, {{0.0, 1.0}, {-1.0, 1.0}});
    auto layer = VanillaFourierLayer(cfg, "nonss_allencahn", torch::kCPU, torch::kDouble);
    auto bases = buildFourierBases(&domain, layer, torch::kCPU, torch::kDouble);
    domain.setPolyBasis(bases, /*requires_grad=*/false);

    domain.setRandomCollocationBatch(/*batch_size=*/3, /*shuffle=*/false);
    auto pde = domain.getPDEData();

    REQUIRE(pde.data.P.basis != nullptr);
    REQUIRE(pde.P_t != nullptr);
    REQUIRE(pde.P_xx != nullptr);

    auto batch_P_col = *pde.data.P.basis;
    auto batch_P_t   = *pde.P_t;
    auto batch_P_xx  = *pde.P_xx;

    REQUIRE(batch_P_col.size(0) == 3);
    REQUIRE(batch_P_t.size(0) == 3);
    REQUIRE(batch_P_xx.size(0) == 3);

    auto full_P_col = bases["P_col"].index({torch::indexing::Slice(0, 3)});
    auto full_P_t   = bases["P_col_dx"].index({torch::indexing::Slice(0, 3)});
    auto full_P_xx  = bases["P_col_dyy"].index({torch::indexing::Slice(0, 3)});

    REQUIRE(torch::allclose(batch_P_col, full_P_col, kTol, kTol));
    REQUIRE(torch::allclose(batch_P_t, full_P_t, kTol, kTol));
    REQUIRE(torch::allclose(batch_P_xx, full_P_xx, kTol, kTol));
}
