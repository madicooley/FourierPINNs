#pragma once

#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <string>

#include "fourierpinns/config/Args.h"

class FourierBasisBuilder {
public:
    FourierBasisBuilder(int input_dim,
                        int ell,
                        BasisGenType basis_gen,
                        bool include_zero_mode,
                        bool scale_to_period,
                        int periodic_deriv_dim,
                        std::vector<std::vector<double>> domain_bounds,
                        torch::Device device,
                        torch::Dtype dtype,
                        std::vector<double> fourier_periods = {})
    : inputDim_(input_dim),
      ell_(ell),
      basisGenTyp_(basis_gen),
      includeZeroMode_(include_zero_mode),
      scaleToPeriod_(scale_to_period),
      periodicDerivDim_(periodic_deriv_dim),
      domainBounds_(std::move(domain_bounds)),
      fourierPeriods_(std::move(fourier_periods)),
      device_(device),
      dtype_(dtype) {
        init();
    }

    int64_t getNPoly() const { return nPoly_; }
    int getPerDimSize() const { return perDimSize_; }
    int getEll() const { return ell_; }
    bool getIncludeZeroMode() const { return includeZeroMode_; }
    bool getScaleToPeriod() const { return scaleToPeriod_; }
    const torch::Tensor& getPeriods() const { return period_; }
    BasisGenType getBasisGenType() const { return basisGenTyp_; }

    int getPeriodicDerivDim() const {
        if (periodicDerivDim_ >= 0 && periodicDerivDim_ < inputDim_) {
            return periodicDerivDim_;
        }
        return inputDim_ - 1;
    }

    torch::Tensor buildBasis(const torch::Tensor& X) const {
        TORCH_CHECK(X.sizes().size() == 2, "FourierBasisBuilder expects 2D input [N, dim]");
        TORCH_CHECK(X.sizes()[1] == inputDim_, "Input dim mismatch: expected ", inputDim_, ", got ", X.sizes()[1]);
        auto X_in = X.to(device_, dtype_);

        std::vector<torch::Tensor> phis;
        phis.reserve(static_cast<std::size_t>(inputDim_));
        for (int d = 0; d < inputDim_; ++d) {
            auto x = X_in.index({torch::indexing::Slice(), d}).unsqueeze(1);
            phis.push_back(computePhiForDim(x, d));
        }
        return tensorProduct(phis);
    }

    torch::Tensor buildBasisPDE(const torch::Tensor& X) const {
        torch::Tensor P = buildBasis(X);
        return P * laplacianScale_;
    }

    torch::Tensor buildBasisDx(const torch::Tensor& X, int deriv_dim) const {
        TORCH_CHECK(X.sizes().size() == 2, "FourierBasisBuilder expects 2D input [N, dim]");
        TORCH_CHECK(X.sizes()[1] == inputDim_, "Input dim mismatch: expected ", inputDim_, ", got ", X.sizes()[1]);
        int d = deriv_dim;
        if (d < 0 || d >= inputDim_) {
            d = inputDim_ - 1;
        }
        auto X_in = X.to(device_, dtype_);

        std::vector<torch::Tensor> phis;
        phis.reserve(static_cast<std::size_t>(inputDim_));
        for (int i = 0; i < inputDim_; ++i) {
            auto x = X_in.index({torch::indexing::Slice(), i}).unsqueeze(1);
            if (i == d) {
                phis.push_back(computePhiDxForDim(x, i));
            } else {
                phis.push_back(computePhiForDim(x, i));
            }
        }
        return tensorProduct(phis);
    }

    torch::Tensor buildBasisDxx(const torch::Tensor& X, int deriv_dim) const {
        TORCH_CHECK(X.sizes().size() == 2, "FourierBasisBuilder expects 2D input [N, dim]");
        TORCH_CHECK(X.sizes()[1] == inputDim_, "Input dim mismatch: expected ", inputDim_, ", got ", X.sizes()[1]);
        int d = deriv_dim;
        if (d < 0 || d >= inputDim_) {
            d = inputDim_ - 1;
        }
        auto X_in = X.to(device_, dtype_);

        std::vector<torch::Tensor> phis;
        phis.reserve(static_cast<std::size_t>(inputDim_));
        for (int i = 0; i < inputDim_; ++i) {
            auto x = X_in.index({torch::indexing::Slice(), i}).unsqueeze(1);
            if (i == d) {
                phis.push_back(computePhiDxxForDim(x, i));
            } else {
                phis.push_back(computePhiForDim(x, i));
            }
        }
        return tensorProduct(phis);
    }

private:
    void init() {
        if (basisGenTyp_ != BasisGenType::TensorProduct) {
            std::cerr << "WARNING: FourierBasisBuilder only supports basis_gen_typ='tensorproduct'. Found '"
                      << toString(basisGenTyp_) << "'.\n";
        }
        TORCH_CHECK(inputDim_ > 0, "FourierBasisBuilder requires input_dim > 0.");
        if (ell_ <= 0 && !includeZeroMode_) {
            throw std::runtime_error("FourierBasisBuilder requires ell > 0 when include_zero_mode=false.");
        }

        omegaBase_ = buildOmegaBase();
        perDimFreqs_ = buildPerDimFreqs();
        perDimSize_ = static_cast<int>(perDimFreqs_.sizes()[0]);
        nPoly_ = computeTensorProductSize(inputDim_, perDimSize_);

        initPeriods();
        laplacianScale_ = buildLaplacianScale();
    }

    torch::Tensor buildOmegaBase() const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        if (ell_ <= 0) {
            return torch::zeros({0}, opts);
        }
        return torch::arange(1, ell_ + 1, opts);
    }

    torch::Tensor buildPerDimFreqs() const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        if (ell_ <= 0) {
            return torch::zeros({1}, opts);
        }
        auto omega_rep = omegaBase_.repeat_interleave(2);
        if (includeZeroMode_) {
            auto zeros = torch::zeros({1}, opts);
            return torch::cat({zeros, omega_rep}, 0);
        }
        return omega_rep;
    }

    void initPeriods() {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        if (!fourierPeriods_.empty()) {
            TORCH_CHECK(fourierPeriods_.size() == static_cast<size_t>(inputDim_),
                        "fourier_periods requires one period per input dimension");
            std::vector<double> mins(inputDim_, 0.0);
            for (int d = 0; d < inputDim_; ++d) {
                TORCH_CHECK(std::isfinite(fourierPeriods_[d]) && fourierPeriods_[d] > 0,
                            "Fourier periods must be finite and positive");
                if (static_cast<size_t>(d) < domainBounds_.size() && domainBounds_[d].size() == 2)
                    mins[d] = domainBounds_[d][0];
            }
            domainMin_ = torch::tensor(mins, opts);
            period_ = torch::tensor(fourierPeriods_, opts);
            return;
        }
        auto ones = torch::ones({inputDim_}, opts);
        if (!scaleToPeriod_ || domainBounds_.empty()) {
            period_ = ones;
            domainMin_ = torch::zeros({inputDim_}, opts);
            return;
        }
        std::vector<double> mins;
        std::vector<double> periods;
        mins.reserve(static_cast<std::size_t>(inputDim_));
        periods.reserve(static_cast<std::size_t>(inputDim_));
        for (int d = 0; d < inputDim_; ++d) {
            const auto idx = static_cast<std::size_t>(d);
            if (idx >= domainBounds_.size() || domainBounds_[idx].size() != 2) {
                mins.push_back(0.0);
                periods.push_back(1.0);
                continue;
            }
            double a = domainBounds_[idx][0];
            double b = domainBounds_[idx][1];
            mins.push_back(a);
            periods.push_back(b - a);
        }
        domainMin_ = torch::tensor(mins, opts);
        period_    = torch::tensor(periods, opts);
    }

    torch::Tensor buildLaplacianScale() const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        auto freq = perDimFreqs_.to(opts);
        if (freq.sizes()[0] == 1 && freq.item<double>() == 0.0) {
            return torch::zeros({1}, opts);
        }
        auto two_pi = 2.0 * M_PI;
        torch::Tensor scale;
        for (int d = 0; d < inputDim_; ++d) {
            auto denom = period_[d];
            auto term = (two_pi * freq / denom).pow(2);
            if (d == 0) {
                scale = term;
            } else {
                scale = scale.unsqueeze(1) + term.unsqueeze(0);
                scale = scale.reshape({-1});
            }
        }
        return -scale;
    }

    torch::Tensor computePhiForDim(const torch::Tensor& x, int dim_index) const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        auto x_scaled = x;
        if (scaleToPeriod_ || !fourierPeriods_.empty()) {
            x_scaled = (x - domainMin_[dim_index]) / period_[dim_index];
        }
        if (ell_ <= 0) {
            return torch::ones({x.size(0), 1}, opts);
        }
        bool has_zero = includeZeroMode_;
        auto omega = omegaBase_.to(x_scaled.dtype()).unsqueeze(0); // [1, K]
        auto phase = 2.0 * M_PI * x_scaled * omega; // [N, K]
        auto cosv = torch::cos(phase);
        auto sinv = torch::sin(phase);
        auto interleaved = torch::stack({cosv, sinv}, -1).reshape({x.size(0), -1});

        if (has_zero) {
            auto ones = torch::ones({x.size(0), 1}, opts);
            return torch::cat({ones, interleaved}, 1);
        }
        return interleaved;
    }

    torch::Tensor computePhiDxForDim(const torch::Tensor& x, int dim_index) const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        auto x_scaled = x;
        if (scaleToPeriod_ || !fourierPeriods_.empty()) {
            x_scaled = (x - domainMin_[dim_index]) / period_[dim_index];
        }
        if (ell_ <= 0) {
            return torch::zeros({x.size(0), 1}, opts);
        }
        bool has_zero = includeZeroMode_;
        auto omega = omegaBase_.to(x_scaled.dtype());
        auto phase = 2.0 * M_PI * x_scaled * omega.unsqueeze(0);
        auto cosv = torch::cos(phase);
        auto sinv = torch::sin(phase);

        auto scale = 2.0 * M_PI * omega / period_[dim_index];
        scale = scale.to(x_scaled.dtype());

        auto dcos = -sinv * scale;
        auto dsin =  cosv * scale;
        auto interleaved = torch::stack({dcos, dsin}, -1).reshape({x.size(0), -1});

        if (has_zero) {
            auto zeros = torch::zeros({x.size(0), 1}, opts);
            return torch::cat({zeros, interleaved}, 1);
        }
        return interleaved;
    }

    torch::Tensor computePhiDxxForDim(const torch::Tensor& x, int dim_index) const {
        auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
        auto x_scaled = x;
        if (scaleToPeriod_ || !fourierPeriods_.empty()) {
            x_scaled = (x - domainMin_[dim_index]) / period_[dim_index];
        }
        if (ell_ <= 0) {
            return torch::zeros({x.size(0), 1}, opts);
        }
        bool has_zero = includeZeroMode_;
        auto omega = omegaBase_.to(x_scaled.dtype());
        auto phase = 2.0 * M_PI * x_scaled * omega.unsqueeze(0);
        auto cosv = torch::cos(phase);
        auto sinv = torch::sin(phase);

        auto scale = 2.0 * M_PI * omega / period_[dim_index];
        scale = scale.to(x_scaled.dtype());
        auto scale2 = scale * scale;

        auto d2cos = -cosv * scale2;
        auto d2sin = -sinv * scale2;
        auto interleaved = torch::stack({d2cos, d2sin}, -1).reshape({x.size(0), -1});

        if (has_zero) {
            auto zeros = torch::zeros({x.size(0), 1}, opts);
            return torch::cat({zeros, interleaved}, 1);
        }
        return interleaved;
    }

    torch::Tensor tensorProduct(const std::vector<torch::Tensor>& phis) const {
        TORCH_CHECK(!phis.empty(), "tensorProduct requires at least one basis vector");
        torch::Tensor out = phis[0];
        for (size_t i = 1; i < phis.size(); ++i) {
            out = (out.unsqueeze(2) * phis[i].unsqueeze(1)).reshape({out.size(0), -1});
        }
        return out;
    }

    int64_t computeTensorProductSize(int dim, int per_dim) const {
        int64_t size = 1;
        for (int i = 0; i < dim; ++i) {
            size *= per_dim;
        }
        return size;
    }

    int inputDim_;
    int ell_;
    BasisGenType basisGenTyp_;
    bool includeZeroMode_;
    bool scaleToPeriod_;
    int periodicDerivDim_;
    std::vector<std::vector<double>> domainBounds_;
    std::vector<double> fourierPeriods_;
    torch::Device device_;
    torch::Dtype dtype_;

    int64_t nPoly_ = 0;
    int perDimSize_ = 0;
    torch::Tensor omegaBase_;
    torch::Tensor perDimFreqs_;
    torch::Tensor domainMin_;
    torch::Tensor period_;
    torch::Tensor laplacianScale_;
};
