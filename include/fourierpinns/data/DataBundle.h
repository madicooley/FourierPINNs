#ifndef FOURIERPINNS_DATABUNDLE_H
#define FOURIERPINNS_DATABUNDLE_H

#include <torch/torch.h>


// Optional precomputed basis features.
struct BasisData {
    torch::Tensor* basis = nullptr;
};

// Generic input data with optional basis.
struct DataInputs {
    torch::Tensor* X = nullptr;
    BasisData P{};
};

// Boundary condition inputs.
struct BndInputs {
    DataInputs data{};
    torch::Tensor* u = nullptr;
    torch::Tensor* P_x = nullptr;
    torch::Tensor* P_xx = nullptr;
    torch::Tensor* P_xxx = nullptr;
    torch::Tensor* P_xxxx = nullptr;
    DataInputs data_b{};
    torch::Tensor* P_x_b = nullptr;
    torch::Tensor* P_xx_b = nullptr;
    torch::Tensor* P_xxx_b = nullptr;
    torch::Tensor* P_xxxx_b = nullptr;
};

// Initial condition inputs.
struct ICInputs {
    DataInputs data{};
    torch::Tensor* u = nullptr;
};

// PDE residual inputs (collocation + operator bases).
struct PDEInputs {
    DataInputs data{};
    torch::Tensor* f = nullptr;     // optional; nullptr for homogeneous
    torch::Tensor* P_pde = nullptr; // laplacian basis for poisson/allencahn
    torch::Tensor* P_t = nullptr;
    torch::Tensor* P_x = nullptr;
    torch::Tensor* P_xx = nullptr;
    torch::Tensor* P_xxx = nullptr;
    torch::Tensor* P_xxxx = nullptr;
};

// Test/validation inputs.
struct TestInputs {
    DataInputs data{};
    torch::Tensor* u = nullptr;
};

#endif
