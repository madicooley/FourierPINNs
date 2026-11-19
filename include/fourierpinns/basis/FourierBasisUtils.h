#pragma once

#include <map>
#include <string>
#include <torch/torch.h>

#include "fourierpinns/models/FourierLayer.h"
#include "fourierpinns/data/LoadedDomain.h"

inline std::map<std::string, torch::Tensor> buildFourierBases(LoadedPDEDomain* domain,
                                                              VanillaFourierLayer& layer,
                                                              torch::Device device,
                                                              torch::Dtype dtype) {
    std::map<std::string, torch::Tensor> bases;

    torch::Tensor X_col = domain->getColNodes();
    bases["P_col"] = layer->buildBasisFromData(X_col).detach().to(device, dtype);

    if (domain->pde_problem == "poisson" || domain->pde_problem == "allencahn") {
        bases["P_pde"] = layer->buildBasisPDEFromData(X_col).detach().to(device, dtype);
    }
    if (domain->pde_problem == "nonss_allencahn" || domain->pde_problem == "onewaywave_inhomo") {
        if (X_col.size(1) >= 1) {
            bases["P_col_dx"] = layer->buildBasisDxFromData(X_col, 0).detach().to(device, dtype); // dt
        }
        if (X_col.size(1) >= 2) {
            bases["P_col_dy"]  = layer->buildBasisDxFromData(X_col, 1).detach().to(device, dtype); // dx
        }
        if (domain->pde_problem == "nonss_allencahn" && X_col.size(1) >= 2) {
            bases["P_col_dyy"] = layer->buildBasisDxxFromData(X_col, 1).detach().to(device, dtype); // dxx
        }
    }

    torch::Tensor X_bnd = domain->getBndNodes();
    bases["P_bnd"] = layer->buildBasisFromData(X_bnd).detach().to(device, dtype);

    if (domain->initial_cond) {
        torch::Tensor X_ic = domain->getICNodes();
        bases["P_ic"] = layer->buildBasisFromData(X_ic).detach().to(device, dtype);
    }

    if (domain->dim != 2) {
        torch::Tensor X_test = domain->getTestingNodes();
        bases["P_test"] = layer->buildBasisFromData(X_test).detach().to(device, dtype);
    }

    if (domain->bnd_typ == "periodic") {
        int deriv_dim = layer->getPeriodicDerivDimValue();
        torch::Tensor X_min = domain->getBndNodesA();
        torch::Tensor X_max = domain->getBndNodesB();
        bases["P_bnd_min"] = layer->buildBasisFromData(X_min).detach().to(device, dtype);
        bases["P_bnd_max"] = layer->buildBasisFromData(X_max).detach().to(device, dtype);
        bases["P_bnd_min_dy"] = layer->buildBasisDxFromData(X_min, deriv_dim).detach().to(device, dtype);
        bases["P_bnd_max_dy"] = layer->buildBasisDxFromData(X_max, deriv_dim).detach().to(device, dtype);
        if (domain->pde_problem == "nonss_allencahn") {
            bases["P_bnd_min_dyy"] = layer->buildBasisDxxFromData(X_min, deriv_dim).detach().to(device, dtype);
            bases["P_bnd_max_dyy"] = layer->buildBasisDxxFromData(X_max, deriv_dim).detach().to(device, dtype);
        }
    }

    return bases;
}
