#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <string>

void printXinfo(torch::Tensor& X, std::string name, bool printminmax=true) {
        std::cout<<"\n"<<name<<"\nsize: "<<X.sizes()<<", device: "<<X.device()<<", requires grad: "<<X.requires_grad()<<", dtype: "<<X.dtype()<<"\n";

        if (printminmax) {
            const int64_t ncols = X.sizes()[1];
            for (int64_t i = 0; i < ncols; ++i) {
                auto column = X.select(1, i);
                auto min_value = column.min();
                auto max_value = column.max();
                std::cout << "    [ Min: " << min_value.item<double>();
                std::cout << ",\tMax: " << max_value.item<double>()<<"\t]"<<std::endl;
            }
        }
}
