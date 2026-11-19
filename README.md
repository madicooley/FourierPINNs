# FourierPINNs

This repository contains the C++/LibTorch implementations of FourierPINNs, code accompanying the paper **“Fourier PINNs: From Strong Boundary Conditions to Adaptive Fourier Bases”** by Madison Cooley, Varun Shankar, Robert M. Kirby, and Shandian Zhe.

## Abstract
Interest is rising in Physics-Informed Neural Networks (PINNs) as a mesh-free alternative to traditional numerical solvers for partial differential equations (PDEs).
However, PINNs often struggle to learn high-frequency and multi-scale target solutions.
To tackle this problem, we first study a strong Boundary Condition (BC) version of PINNs for Dirichlet BCs and observe a consistent decline in relative error compared to the standard PINNs.
We then perform a theoretical analysis based on the Fourier transform and convolution theorem.
We find that strong BC PINNs can better learn the amplitudes of high-frequency components of the target solutions.
However, constructing the architecture for strong BC PINNs is difficult for many BCs and domain geometries.
Enlightened by our theoretical analysis, we propose Fourier PINNs --- a simple, general, yet powerful method that augments PINNs with pre-specified, dense Fourier bases.
Our proposed architecture likewise learns high-frequency components better but places no restrictions on the particular BCs or problem domains.
We develop an adaptive learning and basis selection algorithm via alternating neural net basis optimization, Fourier and neural net basis coefficient estimation, and coefficient truncation.
This scheme can flexibly identify the significant frequencies while weakening the nominal frequencies to better capture the target solution's power spectrum.
We show the advantage of our approach through a set of systematic experiments.

[Open Review](https://openreview.net/forum?id=KqRnsEMYLx) [arXiv](https://arxiv.org/pdf/2410.03496)

## Contact

For questions about this work, please email mcooley220 [at] gmail [dot] com.

## Requirements

The instructions below target Ubuntu/Linux with a C++17 compiler. The project
has been built with GCC 11 and CMake 3.22.

Required dependencies:

- Git, CMake, and a C++17 compiler.
- LibTorch, the PyTorch C++ distribution.
- nlohmann/json 3.11.3 and Catch2 3.5.2, downloaded automatically by CMake.
  Catch2 is needed only when tests are enabled.

Python is optional for experiment runners and their tests; it is not
required to run the C++ executable.

## Download and Install Dependencies

Clone the repository into a directory of your choice:

```bash
git clone https://github.com/madicooley/FourierPINNs.git fourierpinns
cd fourierpinns
```

If downloading a ZIP instead, extract it and open a terminal in the extracted
directory containing `CMakeLists.txt`. Run the following project commands from
that directory.

On Ubuntu, install the build tools and development packages:

```bash
sudo apt update
sudo apt install build-essential cmake git unzip
```

On other distributions, install the equivalent packages through your package manager.

Download and extract LibTorch following the
[official installation instructions](https://docs.pytorch.org/cppdocs/installing.html).
Use a CPU distribution for CPU-only runs, or a CUDA distribution compatible
with your NVIDIA driver and the CUDA requirements of that LibTorch release.
Choose a compiler/ABI-compatible distribution. Installing Python's `torch`
package is not required when using the standalone LibTorch archive.

## Configure and Build

Replace `/absolute/path/to/libtorch` with the directory extracted from the
LibTorch archive, containing `include/`, `lib/`, and `share/`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --target fourierpinns -j4
./build/bin/fourierpinns help
```

If LibTorch is not found through `CMAKE_PREFIX_PATH`, point `Torch_DIR` directly
at the directory containing `TorchConfig.cmake`.

Reduce `-j4` to `-j2` or `-j1` if compilation exhausts available memory.
Subsequent builds only require:

```bash
cmake --build build --target fourierpinns -j4
```

The optional `cmake --install build --prefix /path/to/install` command installs
only the CLI. The installed executable still requires LibTorch; this release
does not export a C++ CMake package.

## Run a Small Experiment

This short run checks the neural-network baseline (`neuralnet`) on a generated
1D Poisson problem. It is a setup check, not a full paper experiment:

```bash
./build/bin/fourierpinns train \
  --model=neuralnet \
  --data_mode=generate --datadir=runs/quickstart/ --expid=quickstart \
  --dim=1 --pde_problem=poisson --bnd_typ=dirichlet \
  --point_sampling=equispaced --dim1_min=0 --dim1_max=6.283185307179586 \
  --test_func=test1_k2 --Ncol_pts=128 --Nbnd_pts=16 --Ntest_pts=256 \
  --width=20 --depth=2 --net_act=tanh \
  --training_order=adamonly --adam_max_epochs=100 \
  --seed=1234 --dtype=float32 --write_logs=false
```

The executable automatically uses CUDA when available, otherwise CPU. Select
a visible GPU with `--gpu_device=0`; to hide GPUs for a CPU run, prefix the
command with `CUDA_VISIBLE_DEVICES=""`.

Outputs are saved under `runs/quickstart/`, including model checkpoints,
predictions, statistics, and resolved configuration. Use a unique `datadir`
or `expid` for each run to avoid overwriting results. A trailing `/` in
`datadir` is optional. `train` solves the PDE by default; use
`--solve_pde=false` only for a setup check.

For larger experiments, use the configurations under `experiments/`. See [experiments](experiments/README.md) for more details.

Section-specific setups are also available for
[Section 6.3.1: 1D Poisson](experiments/section_6_3_1/README.md),
[6.3.2: 1D Steady-State Allen-Cahn](experiments/section_6_3_2/README.md),
[6.4.1: 2D Poisson](experiments/section_6_4_1/README.md), and
[6.4.2: 2D Steady-State Allen-Cahn](experiments/section_6_4_2/README.md), and
[6.5: One-Way Wave and Ablations](experiments/section_6_5/README.md). Read the 2D memory
warning before launching full-sized tensor-product experiments.

## Save and Reload Datasets

`gen-data` saves generated coordinates and targets as CPU `.pt` tensors
under `DATADIR/data/`. Use a new run directory: it will not overwrite an
existing `data/` directory.

```bash
./build/bin/fourierpinns gen-data \
  --config=experiments/section_6_3_1/configs/single_scale.json \
  --datadir=runs/reusable_dataset/ --expid=dataset

./build/bin/fourierpinns train \
  --config=experiments/section_6_3_1/configs/single_scale.json \
  --data_mode=load_all --datadir=runs/reusable_dataset/ --expid=reload
```

Use one dataset per directory. `load_all` loads all tensors required by the
selected PDE and boundary conditions; `load_test_only` loads only `XTest`
and `uTest` and regenerates the training tensors. Keep the PDE, domain, and
manufactured solution settings consistent with the saved data. Only `.pt`
files are loaded. Names such as `prefix_XTest.pt` are accepted, but multiple
files for the same required tensor are rejected.

## Run Tests

Tests are enabled by default:

```bash
cmake --build build --target fourierpinns_tests -j4
ctest --test-dir build --output-on-failure
```

To configure without tests or the Catch2 download, add
`-DFOURIERPINNS_ENABLE_TESTING=OFF` to the configure command.

## Layout

- `include/fourierpinns/`: headers grouped into `models`, `basis`, `pdes`,
  `data`, `training`, `solvers`, `config`, and `utils`.
- `src/cli/`: command-line application.
- `src/solvers/`: solver factory implementation.
- `tests/`: unit and integration tests.
- `experiments/`: experiment configurations, scripts, and supporting data.
- `build/`: local build artifacts (ignored).
- `runs/`: experiment outputs (ignored).
