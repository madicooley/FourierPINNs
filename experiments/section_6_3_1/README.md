# Section 6.3.1: 1D Poisson

| Config | Manufactured solution on [0, 2pi] | Selector |
| --- | --- | --- |
| `single_scale.json` | sin(100x) | test1_k100 |
| `multiscale.json` | sin(x) + 0.1 sin(20x) + 0.05 cos(100x) | test2 |
| `product.json` | sin(6x) cos(100x) | test3 |

## Settings

- Hybrid model `fouriernn`, two hidden layers of width 100, tanh, float32.
- 10,000 collocation points and 20,000 test points.
- Adam: exactly 40,000 updates, initial LR 0.001, multiplied by 0.9 every
  1,000 updates. L-BFGS follows, with initial LR 0.01 and tolerances 1e-9.
- Both coefficient groups use L2 penalty alpha/2 times the sum of squared
  active coefficients, with alpha=1e-4; hidden weights are not penalized.
- Pruning threshold 1e-4 for NN output coefficients and Fourier coefficients.
- Raw Fourier basis, with integer frequencies 1 through K; no separate Fourier
  zero mode because the NN output bias supplies the constant mode.

## Training Sequence

1. Run 1,000 joint Adam updates of hidden weights and active output coefficients.
2. Pause Adam. Repeat five times: solve the least-squares (LS) system for output coefficients with hidden
   weights fixed, then permanently mask coefficients below the threshold.
3. Repeat steps 1 and 2 through Adam step 20,000, then run Adam alone to 40,000.
4. Refine all surviving parameters with L-BFGS, without further LS or pruning.

## Commands

From the repository root, rebuild and run the focused tests:

```bash
cmake --build build --target fourierpinns fourierpinns_tests -j4
./build/tests/fourierpinns_tests '[Algorithm1]'
ctest --test-dir build --output-on-failure
```

First run the short pipeline checks (not accuracy benchmarks):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 experiments/section_6_3_1/run.py --smoke --run
```

Print the full sweep before launching it:

```bash
python3 experiments/section_6_3_1/run.py
```

Run a single paper-sized experiment:

```bash
python3 experiments/section_6_3_1/run.py --cases multiscale --ks 200 --seeds 1234 --run
```

Run all three problems, five candidate-frequency cutoffs, and five seeds
(75 sequential runs):

```bash
python3 experiments/section_6_3_1/run.py --run
```

Each run receives a unique output directory and experiment ID. Logs, resolved
configurations, predictions, statistics, and model checkpoints are saved under
`runs/section_6_3_1/`. Use `--output-root` for a different results directory.
Reusing identical arguments will reuse output filenames.

LS logs separate Fourier coefficients, NN output-basis coefficients, and the
NN output bias. `Active/total` counts reflect the masks after pruning, and
`Pruned` counts removals in that solve. Counts describe output coefficients;
pruning them does not physically remove hidden neurons or reduce tensor sizes.
The five largest active coefficients per group are printed by default, with
original basis indices (or sine/cosine mode labels in 1D). Set
`--print_model_trn_info=true` to print all active coefficients. Fourier mode
labels refer to the configured Fourier coordinates, including period scaling.

For a direct CLI invocation:

```bash
./build/bin/fourierpinns train \
  --config=experiments/section_6_3_1/configs/multiscale.json \
  --dim=1 --ell=200 --seed=1234 \
  --datadir=runs/section_6_3_1/manual_multiscale/ --expid=manual_multiscale
```
