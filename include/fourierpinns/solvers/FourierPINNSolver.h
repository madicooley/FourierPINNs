/**
 * @class FourierPINNSolver
 * @brief Solves PDEs using a physics-informed fourier neural network with automatic differentiation.
 *
 * The FourierPINNSolver class encapsulates the setup and execution of a physics-informed neural network (PINN) that integrates
 * a fourier basis for solving partial differential equations (PDEs). It is designed to leverage automatic differentiation,
 * facilitating the gradient computation process and making it highly suitable for complex PDE problems where traditional
 * numerical methods may be challenging to apply. The solver includes features like truncation of the fourier basis.
 *
 */

#include "fourierpinns/basis/FourierBasisUtils.h"
class FourierPINNSolver : public PDESolver {
private:
    VanillaFourierNeuralNet net = nullptr;

public:
    /**
    * Constructs a new FourierPINNSolver instance.
    *
    * @param args Struct or class containing the necessary configuration parameters. Must have 'seed' and 'pde_problem' members.
    * @param device The torch::Device on which the computations will be performed (e.g., CPU or CUDA device).
    * @param dtype The torch::Dtype defining the data type for computations (e.g., float or double).
    */
    FourierPINNSolver(const Args& args,
                       torch::Device device,
                       torch::Dtype dtype)
    : PDESolver(args.domain_args,
                args.seed,
                device,
                dtype,
                args.model_args.verbose) { }

    void setPolyBasis(const Args& args) {
        const FourierNetArgs* netargs = std::get_if<FourierNetArgs>(&args.model_args.arch);
        if (!netargs) {
            throw std::runtime_error("Expected FourierNetArgs in ModelConfig.arch, but found another type.");
        }

        // Precompute Fourier bases in torch for differentiability and reuse.
        auto basis_config = VanillaFourierNeuralNetImpl::makeFourierLayerConfig(
            args.model_args, netargs->fourierlayer_args);
        VanillaFourierLayer basis_layer(basis_config, domain->pde_problem, device_, dtype_);

        std::map<std::string, torch::Tensor> Bases = buildFourierBases(domain.get(), basis_layer, device_, dtype_);
        domain->setPolyBasis(Bases, true);

        net = VanillaFourierNeuralNet(args.model_args, domain->pde_problem, device_, dtype_);
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
