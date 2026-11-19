# Section 6.3.2: 1D Steady-State Allen-Cahn

This is the steady-state equation:

```text
u_xx + u * (u^2 - 1) = f(x),  x in [0, 2*pi].
f(x) = u_exact_xx + u_exact * (u_exact^2 - 1).
```

| Case | Exact solution | Selector |
| --- | --- | --- |
| `single_scale` | sin(100x) | `test1_k100` |
| `multiscale` | sin(x) + 0.1 sin(20x) + 0.05 cos(100x) | `test2` |
| `product` | sin(6x) cos(100x) | `test3` |

## Configuration

Paper settings: two hidden layers of width 100, tanh, float32, 10,000 residual
points, 20,000 test points, 40,000 Adam updates followed by L-BFGS, initial
learning rates 0.001 and 0.01, exponential Adam decay by 0.9 every 1,000 updates,
coefficient L2 strength 1e-4 and pruning threshold 1e-4.

## Nonlinear Training Sequence

1. Run 1,000 joint Adam updates.
2. With hidden parameters fixed, repeat five times: evaluate the current full
   prediction u, solve the linear least-squares (LS) system with RHS f - u*(u^2 - 1),
   update output coefficients, then prune small coefficients.
3. Repeat through completed Adam step 20,000 (inclusive): 20 blocks, 100 LS
   passes total. Then continue Adam alone until step 40,000.
4. Refine surviving parameters with L-BFGS (1,000 calls, at most 50 internal
   iterations per call, history 50, tolerances 1e-9), without LS or pruning.

Unlike Poisson, the RHS must be recomputed on every LS pass. These are
fixed-point updates, not exact nonlinear least-squares minimizations; they need
not decrease the nonlinear residual monotonically without safeguards.

`--lstsq_safeguard=true` is enabled by default and applies only to steady-state Allen-Cahn.
Each LS proposal is checked against the full-data weighted nonlinear PDE plus
boundary loss (no regularization or test targets). It tries step sizes 1, 1/2,
..., 2^-20 toward the proposed coefficients, accepting only a finite loss no
larger than before. If none works, parameters are restored and pruning is
skipped. Non-finite LS systems/proposals are rejected before applying them.
Pruning after acceptance is also rolled back, including masks, if it increases
the accepted nonlinear loss. Logs show acceptance/rejection, step size, and
before/after merit. Rejected attempts still consume the LS budget.

This adds residual evaluations and can stall if the fixed-point direction is
not a descent direction; it is not a convergence guarantee. It only guards LS,
not Adam or L-BFGS updates. Poisson LS is unchanged. Check
boundary loss, PDE loss, test error, finite coefficients, and surviving basis
counts. Low residual alone does not establish recovery of the manufactured
solution for a nonlinear PDE. The coefficient L2 penalty is applied during
Adam/L-BFGS, not included as ridge rows in LS.

## Commands

From the repository root, print one command without launching training:

```bash
python3 experiments/section_6_3_2/run.py --cases single_scale --ks 200 --seeds 1234
```

Execute that full run:

```bash
python3 experiments/section_6_3_2/run.py --cases single_scale --ks 200 --seeds 1234 --run
```

Use `--cases multiscale` or `--cases product` for the other solutions. Override remaining numerical settings directly if needed:

```bash
build/bin/fourierpinns train \
  --config=experiments/section_6_3_2/configs/multiscale.json \
  --ell=200 --seed=1234 \
  --expid=s632_multiscale_check --datadir=runs/s632_multiscale_check/
```

Quick checks, not convergence or accuracy verification:

```bash
python3 -m unittest discover -s experiments/section_6_3_2 -p 'test_*.py'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 experiments/section_6_3_2/run.py --smoke --run
```

Smoke uses `K=110`, width 8, 512 residual/16 boundary/1,024 test points, four
Adam steps, two blocks of five LS passes, and two short L-BFGS calls.
Smoke overrides the LS step cutoff to 4 so both short blocks run.

Focused steady-state equation and training regression tests:

```bash
cmake --build build --target fourierpinns_tests -j2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 build/tests/fourierpinns_tests '[AllenCahn]'
```

These check analytic residuals and linear operator bases for all three solutions,
generated physical-coordinate forcing/boundaries,
the chain rule through a known tanh network, and RHS refresh after successive
LS updates.

The default full sweep is 75 runs (3 cases x 5 K values x 5 seeds); restrict
`--cases`, `--ks`, and `--seeds` before adding `--run`. Each run has a separate
directory containing logs, arguments, statistics, predictions, and model files.
Use `--output-root` to avoid overwriting earlier experiments. Full runs can
require several GiB of memory; smoke memory usage is not representative.
