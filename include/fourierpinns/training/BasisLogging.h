#pragma once

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <torch/torch.h>

struct BasisCoefficientGroup {
    std::string name;
    torch::Tensor coefficients;
    torch::Tensor mask;
    std::vector<std::string> labels;
};

inline void printBasisSummary(std::ostream& output,
                              const std::vector<BasisCoefficientGroup>& groups,
                              const std::vector<int64_t>& before,
                              bool all_coefficients = false) {
    // Use a local stream so logging does not change later epoch formatting.
    std::ostringstream text;
    text << "  " << std::left << std::setw(15) << "Group"
         << std::right << std::setw(15) << "Active/total" << std::setw(10) << "Pruned"
         << std::setw(14) << "min |coef|" << std::setw(14) << "max |coef|"
         << std::setw(14) << "L2 norm" << "\n";
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const auto& group = groups[g];
        if (!group.mask.defined() || !group.coefficients.defined()) continue;
        auto mask = group.mask.detach().flatten().to(torch::kCPU).to(torch::kBool);
        auto values = group.coefficients.detach().flatten().to(torch::kCPU).to(torch::kDouble);
        auto indices = torch::nonzero(mask).flatten();
        const int64_t active = indices.numel();
        auto selected = values.index({mask});
        const int64_t removed = g < before.size() ? before[g] - active : 0;
        const std::string count = std::to_string(active) + "/" + std::to_string(mask.numel());
        text << "  " << std::left << std::setw(15) << group.name << std::right
             << std::setw(15) << count << std::setw(10) << removed
             << std::scientific << std::setprecision(3)
             << std::setw(14) << (active ? selected.abs().min().item<double>() : 0.0)
             << std::setw(14) << (active ? selected.abs().max().item<double>() : 0.0)
             << std::setw(14) << (active ? selected.norm().item<double>() : 0.0) << "\n";
        if (!active) continue;
        const int64_t shown = all_coefficients ? active : std::min<int64_t>(5, active);
        auto order = std::get<1>(selected.abs().sort(/*dim=*/0, /*descending=*/true));
        text << "    " << (all_coefficients ? "Active coefficients: " : "Largest coefficients: ");
        for (int64_t i = 0; i < shown; ++i) {
            const int64_t index = indices[order[i].item<int64_t>()].item<int64_t>();
            if (i) text << ", ";
            if (static_cast<std::size_t>(index) < group.labels.size()) text << group.labels[index];
            else text << "basis[" << index << "]";
            text << "=" << std::setprecision(6) << values[index].item<double>();
        }
        if (shown < active) text << " (showing " << shown << " of " << active << ")";
        text << "\n";
    }
    output << text.str();
}
