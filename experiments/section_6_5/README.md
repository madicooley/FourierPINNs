# Section 6.5: One-Way Wave

```text
u_t + 10*u_x = v(t,x),    t,x in [0,1]
u(t,x) = sin(pi*x)*cos(10*pi*t) + sin(2*pi*x)*cos(20*pi*t)
u(t,0) = u(t,1) = 0
u(0,x) = sin(pi*x) + sin(2*pi*x)
v = -10*pi*sin(pi*x)*sin(10*pi*t)
    -20*pi*sin(2*pi*x)*sin(20*pi*t)
    +10*pi*cos(pi*x)*cos(10*pi*t)
    +20*pi*cos(2*pi*x)*cos(20*pi*t)
```

Use `pde_problem=onewaywave_inhomo`, `test_func=test7`, `initial_cond=true`.
Spatially 1D, this has two inputs: **column 0 is t and column 1 is x**, so
pass `--dim=2`. Boundary points lie only on spatial edges, not t=1. Initial
conditions are separate constraints at t=0.

## Settings

Table 6 specifies three hidden layers, width 128, tanh, 40,000 training steps,
5,000 collocation points, 256 IC points, 200 boundary points and 100,000 test
points. Adam/L-BFGS start at 0.001/0.01 with exponential decay by 0.9 every
1,000 steps. This setup uses 40,000 Adam steps followed by L-BFGS. Section 6.2
provides float32, L2 coefficient strength 1e-4 and pruning threshold 1e-4.

**Explicit Fourier periods:** the config sets `fourier_periods: [1,2]` in
coordinate order `[t,x]`, independently of the physical domain `[0,1]^2`.
The time factors are `cos(2*pi*k*t)` and the spatial factors `sin/cos(pi*k*x)`.
The exact solution uses time modes 5 and 10 and spatial modes 1 and 2, so
**`K>=10` contains every target factor and their required tensor products**.

The Fourier least-squares (LS) operator block is `P_t + 10*P_x`.
Regression tests compare LS predictions with boundary, IC and PDE forward paths.
No full-sized accuracy claim is made for this setup.

## Commands

Small pipeline check (not an accuracy test):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 experiments/section_6_5/run.py \
  --smoke --test-batch-size=7 --output-root=runs/wave_smoke --run
```

Smoke uses `K=2`, width 4, 64 collocation points, 16 boundary points, 16 IC points,
121 test points, four Adam steps, two five-pass LS blocks and short L-BFGS.
It contains both required spatial modes but omits temporal frequencies 5 and 10.

Print one full-size command without training/allocation:

```bash
python3 experiments/section_6_5/run.py --ks 200 --seeds 1234
```

After checking memory, append `--allow-large-basis --run` to execute it.
Alternatively, `--ks 10 --seeds 1234 --run` tries a smaller hybrid experiment
(441 Fourier columns). This contains the exact target modes, but is not a
full-sized reproduction or a guarantee of optimizer convergence.

The default case is `reg_prune`. Request the full ablation explicitly:

```bash
python3 experiments/section_6_5/run.py \
  --cases reg_prune no_reg_prune reg_no_prune no_reg_no_prune
```

| Case | NN/Fourier L2 strength | NN/Fourier pruning |
| --- | --- | --- |
| `reg_prune` | 1e-4 | enabled, threshold 1e-4 |
| `no_reg_prune` | 0 | enabled, threshold 1e-4 |
| `reg_no_prune` | 1e-4 | disabled |
| `no_reg_no_prune` | 0 | disabled |

LS remains enabled with the same schedule in every case. The runner overrides
the shared `configs/wave.json`. Default sweep: one case x one K x five seeds =
**5 runs**; all four cases = **20 runs**. Smoke defaults to one seed. Output
paths include case/K/seed; use `--output-root` for separate comparison runs.

Direct CLI option names use underscores:

```bash
build/bin/fourierpinns train \
  --config=experiments/section_6_5/configs/wave.json --dim=2 \
  --ell=10 --seed=1234 --test_batch_size=128 \
  --expid=wave_K10_s1234 --datadir=runs/wave_K10_s1234/
```
