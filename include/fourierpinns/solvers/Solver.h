
#include <filesystem>

class Solver {
public:
    Solver() {}

    template<typename Model>
    int saveModel(Model& net, std::string filename_root, std::string datadir) {
        int status=0;
        std::string filename = (std::filesystem::path(datadir) / "models" / ("models_" + filename_root + ".pt")).string();
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        torch::save(net, filename);
        std::cout<<"Model has been saved to "<<filename<<"\n";
        return status;
    }
};

class PDESolver : public Solver {
protected:
    std::shared_ptr<LoadedPDEDomain> domain;    ///< Shared pointer to
    std::shared_ptr<PDE> pde;                   ///< Shared pointer to the PDE object defining the problem.

    int seed;                                   ///< Seed for random number generation, affecting training initialization.
    torch::Device device_; torch::Dtype dtype_;

public:
    PDESolver(const DomainConfig& args,
              int seed_value,
              torch::Device device,
              torch::Dtype dtype,
              bool verbose)
    : Solver(),
      seed(seed_value),
      device_(device),
      dtype_(dtype) {

        torch::manual_seed(static_cast<uint64_t>(seed_value));
        domain = std::make_shared<LoadedPDEDomain>(args.dim,
                                                   args.Ncol_pts,
                                                   args.Nbnd_pts,
                                                   args.Ntest_pts,
                                                   args.Nic_pts,
                                                   args.test_func,
                                                   args.pde_problem,
                                                   args.bnd_typ,
                                                   args.domain_typ,
                                                   args.point_sampling,
                                                   args.homogeneous_pde,
                                                   args.initial_cond,
                                                   args.domain_bounds,
                                                   device_, dtype_);

        pde = std::make_shared<PDE>(args.pde_problem, verbose);
    }

    void loadDomainsFromFile(const std::string& datadir, bool test_only = false) {
        domain->loadDomainsFromFile(datadir, test_only);
    }

    void saveDomainData(const std::string& datadir) const {
        domain->saveDomainData(datadir);
    }

    void generateDomainData(bool gen_test=true, bool gen_bnd=true, bool gen_ic=true, bool gen_col=true) {
        domain->generateDomainData(gen_test, gen_bnd, gen_ic, gen_col);
    }
};
