/**
 * @class StandardPINNSolver
 * @brief Implements a solver for Physics-Informed Neural Networks (PINNs) using a standard approach.
 *
 * The StandardPINNSolver class provides the functionality to solve PDEs using PINNs with a standard training
 * and solution strategy. It encapsulates the neural network model, the problem domain, and the PDE to be solved,
 * along with other relevant settings and options.
 */

class StandardPINNSolver : public PDESolver {
private:
    VanillaNeuralNet net = nullptr;              ///< Shared pointer to the neural network model.

public:
    /**
    * Constructs a new StandardPINNSolver instance.
    *
    * @param args Struct or class containing the necessary configuration parameters.
    *             Must have 'seed' and 'pde_problem' members.
    * @param device The torch::Device on which the computations will be performed (e.g., CPU or CUDA device).
    * @param dtype The torch::Dtype defining the data type for computations (e.g., float or double).
    */
    StandardPINNSolver(const Args& args,
                       torch::Device device,
                       torch::Dtype dtype)
    : PDESolver(args.domain_args,
                args.seed,
                device,
                dtype,
                args.model_args.verbose) {

        net = VanillaNeuralNet(args.model_args, args.domain_args.pde_problem, device_, dtype_);

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
