#pragma once

#include <variant>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>

struct RegularizationArgs {
    float lambda_bnd = 100.0f;      // Boundary condition loss weight.
    float lambda_ic = 100.0f;       // Initial condition loss weight.
    float lambda_res = 1.0f;        // PDE residual loss weight.

    int lambda_upd_interval = 1000; // Interval (epochs) for lambda updates.
    std::string loss_reweight_scheme = "none"; // Loss reweighting scheme: none | grad.

     void print(std::ostream& os = std::cout) const {
        os << "[RegularizationArgs]\n";
        os << "  lambda_bnd             = " << lambda_bnd << "\n";
        os << "  lambda_ic              = " << lambda_ic << "\n";
        os << "  lambda_res             = " << lambda_res << "\n";
        os << "  lambda_upd_interval    = " << lambda_upd_interval << "\n";
        os << "  loss_reweight_scheme   = " << loss_reweight_scheme << "\n";
    }
};

struct LBFGSConfig {
    double lbfgs_min_lr = 1e-5;             // Minimum LBFGS learning rate.
    double lbfgs_max_lr = 1.0;              // Maximum LBFGS learning rate.
    int lbfgs_max_epochs = 50;              // Number of LBFGS epochs.
    int lbfgs_history_size = 50;            // LBFGS history size.
    int lbfgs_max_iter = 1000;              // LBFGS max iterations per step.
    double lbfgs_tolerance_grad = 1e-9;     // LBFGS gradient tolerance.
    double lbfgs_tolerance_change = 1e-9;   // LBFGS change tolerance.
    std::string lbfgs_line_search_fn = "strong_wolfe"; // Line search function.

    void print(std::ostream& os = std::cout) const {
        os << "[LBFGSConfig]\n";
        os << "  lbfgs_min_lr           = " << lbfgs_min_lr << "\n";
        os << "  lbfgs_max_lr           = " << lbfgs_max_lr << "\n";
        os << "  lbfgs_max_epochs       = " << lbfgs_max_epochs << "\n";
        os << "  lbfgs_history_size     = " << lbfgs_history_size << "\n";
        os << "  lbfgs_max_iter         = " << lbfgs_max_iter << "\n";
        os << "  lbfgs_tolerance_grad   = " << lbfgs_tolerance_grad << "\n";
        os << "  lbfgs_tolerance_change = " << lbfgs_tolerance_change << "\n";
        os << "  lbfgs_line_search_fn   = " << lbfgs_line_search_fn << "\n";
    }
};

struct AdamConfig {
    double adam_min_lr = 1e-5;              // Minimum Adam learning rate.
    double adam_max_lr = 0.001;             // Maximum Adam learning rate.
    int adam_max_epochs = 20000;            // Number of Adam epochs.
    std::tuple<double, double> adam_betas = {0.9, 0.999}; // Adam betas.
    double adam_eps = 1e-8;                 // Adam epsilon.

    void print(std::ostream& os = std::cout) const {
        os << "[AdamConfig]\n";
        os << "  adam_min_lr      = " << adam_min_lr << "\n";
        os << "  adam_max_lr      = " << adam_max_lr << "\n";
        os << "  adam_max_epochs  = " << adam_max_epochs << "\n";
        os << "  adam_betas       = (" << std::get<0>(adam_betas) << ", " << std::get<1>(adam_betas) << ")\n";
        os << "  adam_eps         = " << adam_eps << "\n";
    }
};

struct TrainingConfig {
    double lstsq_rcond = -1.0; // -1: source dtype epsilon * max(rows, columns).
    int lstsq_stop_epoch = -1; // Last completed Adam step allowing LS; -1: unlimited.
    bool lstsq_safeguard = true; // Backtrack steady Allen-Cahn LS using nonlinear PDE/boundary loss.
    std::string training_order = "adamonly"; // adamonly | lbfgsonly | adamfirst | lbfgsfirst.
    std::string dtype = "float32";           // Training dtype: float32 | float64.

    // Scheduler
    std::string lr_decaytype = "cosine";    // LR schedule type: cosine | exp | none.
    int lr_decay_interval = 1000;           // Interval for LR decay.
    int lr_warmup_steps = 0;                // Warmup steps.
    int lr_decay_steps = 0;                 // Total decay steps.
    float lr_decay_rate = 0.9f;             // LR decay rate.


    int pred_save_interval = 0;             // Epoch interval for saving predictions.
    bool print_model_trn_info = false;      // Print model parameters during training.
    bool print_lr_sched = false;            // Print learning-rate schedule.
    int print_interval = 1000;              // Logging interval (epochs).
    int test_batch_size = 256;              // Test evaluation chunk size for dim=2 only.

    bool lstsq_updates=false;               // Enable least-squares updates.
    int lstsq_upd_interval=0;               // Interval for least-squares updates.
    int n_lstsq_upds=0;                      // Total number of least-squares updates.
    int n_lstsq_upds_per_epoch=0;            // Updates per epoch (no grad steps between).

    int col_batch_size=0;                 // Adam collocation batch size; 0 uses the full set.
    bool col_batch_shuffle=true;             // Shuffle collocation indices each step.

    int truncate_interval = 1000;           // Interval to truncate coefficients.

    LBFGSConfig lbfgs_args;
    AdamConfig  adam_args;
    RegularizationArgs reglr_args;

    int dim;  // Dimension of input domain (set from Args).
    int seed; // RNG seed (set from Args).

    void print(std::ostream& os = std::cout) const {
        os << "[TrainingConfig]\n";
        os << "  training_order         = " << training_order << "\n";
        os << "  dtype                  = " << dtype << "\n";
        os << "  lr_decaytype           = " << lr_decaytype << "\n";
        os << "  lr_decay_interval      = " << lr_decay_interval << "\n";
        os << "  lr_warmup_steps        = " << lr_warmup_steps << "\n";
        os << "  lr_decay_steps         = " << lr_decay_steps << "\n";
        os << "  lr_decay_rate          = " << lr_decay_rate << "\n";


        os << "  pred_save_interval     = " << pred_save_interval << "\n";
        os << "  print_model_trn_info   = " << std::boolalpha << print_model_trn_info << "\n";
        os << "  print_lr_sched         = " << std::boolalpha << print_lr_sched << "\n";
        os << "  print_interval         = " << print_interval << "\n";

        os << "  lstsq_updates          = " << std::boolalpha << lstsq_updates << "\n";
        os << "  lstsq_upd_interval     = " << lstsq_upd_interval << "\n";
        os << "  n_lstsq_upds           = " << n_lstsq_upds << "\n";
        os << "  n_lstsq_upds_per_epoch = " << n_lstsq_upds_per_epoch << "\n";

        os << "  col_batch_size         = " << col_batch_size << "\n";
        os << "  col_batch_shuffle      = " << std::boolalpha << col_batch_shuffle << "\n";

        os << "  truncate_interval      = " << truncate_interval << "\n";
        os << "  dim                    = " << dim << "\n";
        os << "  seed                   = " << seed << "\n";

        os << "\n";
        adam_args.print(os);
        os << "\n";
        lbfgs_args.print(os);
        os << "\n";
        reglr_args.print(os);
    }
};


struct DomainConfig { // Loaded Domain Args
    bool initial_cond=false;          // Whether initial-condition data is used.
    bool homogeneous_pde=false;       // Whether PDE forcing f is zero.
    std::string pde_problem="poisson";// PDE type: poisson | allencahn | nonss_allencahn | onewaywave_inhomo.
    std::string bnd_typ="dirichlet";  // Boundary type: dirichlet | periodic.
    std::string domain_typ="box";     // Domain type: box.
    std::string point_sampling="equispaced"; // Sampling: equispaced | random.
    int Ncol_pts=1024;                // Number of collocation points.
    int Nbnd_pts=256;                 // Number of boundary points.
    int Ntest_pts=256;                // Number of test points.
    int Nic_pts=256;                  // Number of initial-condition points.
    std::string test_func="test1_k1";    // Manufactured solution selector.

    int dim; // Dimension of input domain (set from Args).
    std::vector<std::vector<double>> domain_bounds; // Per-dim [min,max] bounds.
    std::string data_mode="generate"; // generate | load_all | load_test_only.

    void print(std::ostream& os = std::cout) const {
        os << "[DomainConfig]\n";
        os << "  initial_cond       = " << std::boolalpha << initial_cond << "\n";
        os << "  homogeneous_pde    = " << std::boolalpha << homogeneous_pde << "\n";
        os << "  pde_problem        = " << pde_problem << "\n";
        os << "  bnd_typ            = " << bnd_typ << "\n";
        os << "  domain_typ         = " << domain_typ << "\n";
        os << "  point_sampling     = " << point_sampling << "\n";
        os << "  Ncol_pts           = " << Ncol_pts << "\n";
        os << "  Nbnd_pts           = " << Nbnd_pts << "\n";
        os << "  Ntest_pts          = " << Ntest_pts << "\n";
        os << "  Nic_pts            = " << Nic_pts << "\n";
        os << "  test_func          = " << test_func << "\n";
        os << "  dim                = " << dim << "\n";
        os << "  data_mode          = " << data_mode << "\n";
        os << "  domain_bounds      = [\n";
        for (const auto& bnds : domain_bounds) {
            os << "    [ ";
            for (double val : bnds) os << val << " ";
            os << "]\n";
        }
        os << "  ]\n";
    }
};




struct NeuralNetArgs {
    int depth = 3;                  // Number of hidden layers.
    int width = 32;                 // Width of each hidden layer.
    std::string net_act = "tanh";   // Activation function.
    std::vector<int> layers;        // Explicit layer sizes (computed).

    bool nn_truncate = false;       // Enable coefficient truncation.
    float nn_trunc_thresh = 0.00001f;// Truncation threshold.
    float nn_l1_reg_coeff = 0.0f;    // L1 regularization coefficient.
    float nn_l2_reg_coeff = 0.0f;    // L2 regularization coefficient.

    void print(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(6);
        os << "[NeuralNetArgs]\n";
        os << "  depth              = " << depth << "\n";
        os << "  width              = " << width << "\n";
        os << "  net_act            = " << net_act << "\n";
        os << "  nn_truncate        = " << std::boolalpha << nn_truncate << "\n";
        os << "  nn_trunc_thresh    = " << nn_trunc_thresh << "\n";
        os << "  nn_l1_reg_coeff    = " << nn_l1_reg_coeff << "\n";
        os << "  nn_l2_reg_coeff    = " << nn_l2_reg_coeff << "\n";

        os << "  layers             = [ ";
        for (int l : layers) os << l << " ";
        os << "]\n";
    }
};

enum class BasisGenType {
    TensorProduct
};

inline BasisGenType parseBasisGenType(const std::string& s) {
    if (s == "tensorproduct") return BasisGenType::TensorProduct;
    throw std::invalid_argument("Invalid basis_gen_typ: " + s);
}

inline std::string toString(BasisGenType t) {
    switch (t) {
        case BasisGenType::TensorProduct: return "tensorproduct";
    }
    return "unknown";
}

struct FourierLayerArgs {
    int ell = 2;                             // Max frequency per dimension.
    BasisGenType basis_gen_typ = BasisGenType::TensorProduct; // Basis generation type.
    bool include_zero_mode = true;           // Include zero-frequency mode.
    bool scale_to_period = false;            // Scale inputs to period lengths.
    std::vector<double> fourier_periods;     // Explicit per-axis periods; overrides scale_to_period.
    int periodic_deriv_dim = -1;             // Periodic deriv dimension (-1 uses last dim).
    bool pn_truncate = false;                // Enable truncation of Fourier coeffs.
    float pn_trunc_thresh = 0.00001f;        // Truncation threshold.
    float pn_l1_reg_coeff = 0.0f;            // L1 regularization coefficient.
    float pn_l2_reg_coeff = 0.0f;            // L2 regularization coefficient.

    void print(std::ostream& os = std::cout) const {
        os << "[FourierLayerArgs]\n";
        os << "  ell               = " << ell << "\n";
        os << "  basis_gen_typ     = " << toString(basis_gen_typ) << "\n";
        os << "  include_zero_mode = " << std::boolalpha << include_zero_mode << "\n";
        os << "  scale_to_period   = " << std::boolalpha << scale_to_period << "\n";
        os << "  fourier_periods   = [";
        for (size_t i = 0; i < fourier_periods.size(); ++i) os << (i ? ", " : "") << fourier_periods[i];
        os << "]\n";
        os << "  periodic_deriv_dim = " << periodic_deriv_dim << "\n";
        os << "  pn_truncate       = " << std::boolalpha << pn_truncate << "\n";
        os << "  pn_trunc_thresh   = " << pn_trunc_thresh << "\n";
        os << "  pn_l1_reg_coeff   = " << pn_l1_reg_coeff << "\n";
        os << "  pn_l2_reg_coeff   = " << pn_l2_reg_coeff << "\n";
    }
};

struct FourierNetArgs {
    NeuralNetArgs neuralnet_args;
    FourierLayerArgs fourierlayer_args;

    void print(std::ostream& os = std::cout) const {
        os << "[FourierNetArgs]\n";
        neuralnet_args.print(os);
        os << "\n";
        fourierlayer_args.print(os);
    }
};


struct ModelConfig {
    std::string model_type = "neuralnet"; // neuralnet | fourierlayer | fouriernn.
    std::vector<int> layers;              // Computed layer sizes.

    int input_dim = 2;                    // Input dimension.
    int output_dim = 1;                   // Output dimension.

    bool verbose = false;                 // Enable verbose logging.



    std::variant<NeuralNetArgs, FourierLayerArgs, FourierNetArgs> arch;

    std::vector<std::vector<double>> domain_bounds;

    void print(std::ostream& os = std::cout) const {
        os << "[ModelConfig]\n";
        os << "  model_type             = " << model_type << "\n";
        os << "  input_dim              = " << input_dim << "\n";
        os << "  output_dim             = " << output_dim << "\n";
        os << "  verbose                = " << std::boolalpha << verbose << "\n";

        os << "  domain_bounds        = [\n";
        for (const auto& bnds : domain_bounds) {
            os << "    [ ";
            for (double val : bnds) os << val << " ";
            os << "]\n";
        }
        os << "  ]\n";

        // Print layers
        os << "  layers                 = [ ";
        for (int l : layers) os << l << " ";
        os << "]\n";



        // Print architecture variant
        os << "\n";
        std::visit([&os](const auto& arch_cfg) {
            arch_cfg.print(os);
        }, arch);
    }

    void applyCommon(const ModelConfig& base) {
        input_dim = base.input_dim;
        output_dim = base.output_dim;
        verbose = base.verbose;
        domain_bounds = base.domain_bounds;
    }
};


struct Args {
    int dim=1;                 // Input dimension.
    int seed=1234;             // RNG seed.
    DomainConfig domain_args;  // Domain configuration.
    ModelConfig model_args;    // Model configuration.
    TrainingConfig training_args; // Training configuration.

    void print(std::ostream& os = std::cout) const {
        os << "[Args]\n";
        os << "  dim  = " << dim << "\n";
        os << "  seed = " << seed << "\n";

        os << "\n";
        domain_args.print(os);
        os << "\n";
        model_args.print(os);
        os << "\n";
        training_args.print(os);
    }
};
