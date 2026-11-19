/**
 * @class FourierPILayerSolver
 * @brief Automates problem-solving for PDE problems using a Fourier layer within a neural network,
 *        using autograd for all gradient computations.
 *
 */

#include "fourierpinns/basis/FourierBasisUtils.h"

class FourierPILayerSolver : public PDESolver {
private:
    VanillaFourierLayer net = nullptr;

public:
    /**
    * @param args Struct or class containing the necessary configuration parameters. Must have 'seed' and 'pde_problem' members.
    */
    FourierPILayerSolver(const Args& args,
                          torch::Device device,
                          torch::Dtype dtype)
    : PDESolver(args.domain_args,
                args.seed,
                device,
                dtype,
                args.model_args.verbose) {
    }

    void setPolyBasis(const Args& args) {
        const FourierLayerArgs* polyargs = std::get_if<FourierLayerArgs>(&args.model_args.arch);
        if (!polyargs) {
            throw std::runtime_error("Expected FourierLayerArgs in ModelConfig.arch, but found another type.");
        }

        // Precompute Fourier bases in torch for differentiability and reuse.
        net = VanillaFourierLayer(args.model_args, domain->pde_problem, device_, dtype_);

        std::map<std::string, torch::Tensor> Bases = buildFourierBases(domain.get(), net, device_, dtype_);
        domain->setPolyBasis(Bases, true);
    }

    void loadModel(std::string modelfname) {
        std::cout<<"\nloading model: "<<modelfname<<"\n";
        torch::load(net, modelfname);
        std::cout<<"done loading\n";
    }

    template <typename TrainingConfig, typename ArgParser>
    void solvePDE(const TrainingConfig& args, ArgParser& parser, bool saveStats, std::string filename_root, std::string datadir) {
        ModelTrainer trainer(args, net.get(), domain.get(), pde.get());
        std::cout << "Started training\n";
        trainer.trainModel();
        std::cout << "Finished training\n";

        if (saveStats) {
            const int status1 = trainer.stats_logger.saveStats(filename_root, datadir);
            const int status2 = trainer.stats_logger.savePDEPredictions(domain->pde_problem, filename_root, datadir);
            const int status3 = parser.saveArgCSVLine(trainer.stats_logger.getFinalError(),
                                                      trainer.stats_logger.getFinalLoss(),
                                                      trainer.stats_logger.getFinalTrainTime(),
                                                      trainer.stats_logger.getFinalIter(),
                                                      filename_root, datadir);
            const int status4 = saveModel(net, filename_root, datadir);
            if (status1 != 0 || status2 != 0 || status3 != 0 || status4 != 0) {
                throw std::runtime_error("Failed to persist one or more training outputs.");
            }
        }
    }
};
