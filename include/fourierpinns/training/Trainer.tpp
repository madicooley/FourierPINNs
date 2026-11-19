#ifndef FOURIERPINNS_TRAINER_TPP
#define FOURIERPINNS_TRAINER_TPP

#include "fourierpinns/training/StatsLogger.h"

template <typename ModelType>
ModelTrainer<ModelType>::ModelTrainer(const TrainingConfig& training_args,
                                      ModelType* model,
                                      LoadedPDEDomain* domain_ptr,
                                      PDE* pde_ptr)
    : stats_logger(training_args, domain_ptr, model),
      args(training_args),
      pde_domain(domain_ptr),
      pde(pde_ptr),
      model_(model),
      lambda_updater(training_args),
      solvingpde(true),
      verbose(true),
      forward_count(-1),
      curr_epoch(0),
      total_epochs(0),
      lstsq_upd_count(0),
      current_lr(0.0) {
    if (training_args.reglr_args.loss_reweight_scheme!="none") {
        lambda_bnd=1.0;
        lambda_ic=1.0;
    } else {
        lambda_bnd=training_args.reglr_args.lambda_bnd;
        lambda_ic=training_args.reglr_args.lambda_ic;
    }
    lambda_res=training_args.reglr_args.lambda_res;

    if (args.training_order == "adamfirst") {
        this->total_epochs=args.adam_args.adam_max_epochs+(args.lbfgs_args.lbfgs_max_epochs*args.lbfgs_args.lbfgs_max_iter);
    } else if (args.training_order == "lbfgsfirst") {
        this->total_epochs=args.adam_args.adam_max_epochs+(args.lbfgs_args.lbfgs_max_epochs*args.lbfgs_args.lbfgs_max_iter);
    } else if (args.training_order == "adamonly") {
        this->total_epochs=args.adam_args.adam_max_epochs;
    } else if (args.training_order == "lbfgsonly") {
        this->total_epochs=(args.lbfgs_args.lbfgs_max_epochs*args.lbfgs_args.lbfgs_max_iter);
    }
}

template <typename ModelType>
void ModelTrainer<ModelType>::printModelInfo() {
    std::cout<<"\n---------------------------------------- ModelTrainer().\n"
                <<"lambda_res: "<<lambda_res<<" | lambda_bnd: "<<lambda_bnd<<" | lambda_ic: "<<lambda_ic<<"\n";

    int totalparams=0;
    int i=0;
    std::vector<torch::Tensor> parameters = model_->parameters();

    for (const auto& param : parameters) {
        std::cout << "i: "<<i<<", Parameter: " << param.sizes() << std::endl;
        totalparams+=static_cast<int>(param.view(-1).sizes()[0]);
        i++;
    }
    std::cout<<"total number of parameters: "<<totalparams<<"\n";

    i=0;
    totalparams=0;
    std::cout<<"\nnumber of parameter: "<<model_->parameters().size()<<"\n";
    for (const auto& pair : model_->named_parameters()) {
        const std::string& name = pair.key();
        const torch::Tensor& param = pair.value();
        std::cout<<"i: "<<i<<" Name: "<<name<<", Parameter: "<<param.sizes()<<", requires_grad: "<<param.requires_grad()<<", device: "<<param.device()<<std::endl;
        totalparams+=static_cast<int>(param.view(-1).sizes()[0]);
        i++;
    }
    std::cout<<"total number of parameters: "<<totalparams<<"\n";

    if (args.print_lr_sched) printLearningRateSchedule(args.adam_args.adam_min_lr, args.adam_args.adam_max_lr, args.adam_args.adam_max_epochs);

    std::cout<<"total_epochs: "<<this->total_epochs<<"\n";
    std::cout<<"\n----------------------------------------\n";
}

template <typename ModelType>
void ModelTrainer<ModelType>::trainModel() {
    torch::manual_seed(static_cast<uint64_t>(args.seed));
    float total_time=0.0;
    torch::optim::Adam adamoptimizer(
        model_->parameters(),
        torch::optim::AdamOptions(args.adam_args.adam_max_lr)
            .betas(args.adam_args.adam_betas)
            .eps(args.adam_args.adam_eps));

    torch::optim::LBFGS lbfgsoptimizer(model_->parameters(),
                                       torch::optim::LBFGSOptions(args.lbfgs_args.lbfgs_max_lr)
                                        .max_iter(args.lbfgs_args.lbfgs_max_iter)
                                        .tolerance_grad(args.lbfgs_args.lbfgs_tolerance_grad)
                                        .tolerance_change(args.lbfgs_args.lbfgs_tolerance_change)
                                        .history_size(args.lbfgs_args.lbfgs_history_size)
                                        .line_search_fn(args.lbfgs_args.lbfgs_line_search_fn));

    printModelInfo();

    // Decide the training order
    model_->train();
    if (args.training_order == "adamfirst") {
        // total_epochs=args.adam_max_epochs+(args.lbfgs_max_epochs*args.lbfgs_max_iter);
        trainAdam(adamoptimizer, total_time);
        trainLBFGS(lbfgsoptimizer, total_time);
    } else if (args.training_order == "lbfgsfirst") {
        // total_epochs=args.adam_max_epochs+(args.lbfgs_max_epochs*args.lbfgs_max_iter);;
        trainLBFGS(lbfgsoptimizer, total_time);
        trainAdam(adamoptimizer, total_time);
    } else if (args.training_order == "adamonly") {
        // total_epochs=args.adam_max_epochs;
        trainAdam(adamoptimizer, total_time);
    } else if (args.training_order == "lbfgsonly") {
        // total_epochs=(args.lbfgs_max_epochs*args.lbfgs_max_iter);;
        trainLBFGS(lbfgsoptimizer, total_time);
    } else {
        throw std::invalid_argument("Invalid training order option.");
    }

    torch::NoGradGuard no_grad;
    const auto groups = model_->getBasisCoefficientGroups();
    if (!groups.empty()) {
        std::cout << "\nFinal Active Basis Counts (active/total)\n";
        for (const auto& group : groups) {
            if (!group.mask.defined()) continue;
            std::cout << "  " << group.name << ": "
                      << group.mask.count_nonzero().template item<int64_t>()
                      << "/" << group.mask.numel() << "\n";
        }
        std::cout << "\n";
    }
}

#endif // FOURIERPINNS_TRAINER_TPP
