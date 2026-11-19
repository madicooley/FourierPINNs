# Section 6.4.2: 2D Steady Allen-Cahn

```text
g(s) = sin(s) + 0.1 sin(20s) + cos(100s)
u(x,y) = g(x) g(y),  (x,y) in [0,2*pi]^2
u_xx + u_yy + u(u^2 - 1) = f(x,y)
f(x,y) = g''(x)g(y) + g(x)g''(y) + u(x,y)(u(x,y)^2 - 1)
g''(s) = -sin(s) - 40 sin(20s) - 10000 cos(100s)
```

The selector is `test6`, with `pde_problem=allencahn`.
The generator supplies exact boundary targets.
Required coordinate frequencies are 1, 20, and 100, including their cross-products.

## Settings

Table 7 specifies width 100, three hidden layers, tanh, 40,000 training steps,
12,000 collocation points, 400 boundary points, 100,000 test points, initial
Adam/L-BFGS learning rates 0.001/0.01, and exponential decay by 0.9 every 1,000
steps. The config uses 40,000 Adam steps followed by L-BFGS. Section 6.2 supplies
float32, coefficient L2 regularization 1e-4 and pruning threshold 1e-4.

`lstsq_safeguard=true` backtracks or rejects least-squares (LS) candidates that worsen the weighted
nonlinear PDE/boundary objective and rolls back harmful pruning. This is a
stability addition.
LS uses unscaled CPU float64 SVD; coefficient regularization applies in Adam/L-BFGS, not LS.

## Commands

Print one full-sized command without allocating training arrays:

```bash
python3 experiments/section_6_4_2/run.py --ks 200 --seeds 1234
```

Run a small pipeline check, including chunked testing, two nonlinear LS blocks,
Adam, L-BFGS and saved predictions:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 experiments/section_6_4_2/run.py --smoke --test-batch-size=7 --run
```

Smoke uses `K=2`, width 4, 64 collocation points, 16 boundary points, 121 test
points and four Adam steps.

After checking available memory, execute one full-sized run explicitly:

```bash
python3 experiments/section_6_4_2/run.py \
  --ks 200 --seeds 1234 --allow-large-basis --run
```

The default sweep uses five seeds (1234-1238).
Use `--output-root` for comparison runs. The direct binary equivalent is:

```bash
build/bin/fourierpinns train \
  --config=experiments/section_6_4_2/configs/multiscale.json \
  --dim=2 --ell=200 --seed=1234 --test_batch_size=256 \
  --expid=s642_multiscale_K200_s1234 \
  --datadir=runs/section_6_4_2/s642_multiscale_K200_s1234/
```

The Python runner's memory guard does not apply to direct binary commands.

## Memory and Verification

The tensor-product basis has `(2*K+1)^2` columns: 160,801 at `K=200`. Test
evaluation is chunked (256 rows by default), about 0.153 GiB per float32 basis
chunk instead of a 59.90 GiB full test cache. All 100,000 test points are retained.
This is not peak memory: temporaries, training bases, derivative bases, LS
float64 copies/workspace, activations and optimizer state cost more. Just one
12,000-row training basis is about 7.19 GiB. Chunked testing does not solve
training/LS memory requirements.

The runner blocks execution above 4,096 Fourier columns without
`--allow-large-basis`. `K<100` omits the exact solution's high frequencies, so
smaller jobs do not represent all target frequencies. The guard remains active
when test evaluation is chunked.

```bash
python3 -m unittest discover -s experiments/section_6_4_2 -p 'test_*.py'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 build/tests/fourierpinns_tests '[Section642]'
```

Tests cover dry runs, overrides, the memory guard, non-finite final metrics,
and the exact solution/forcing/nonzero boundaries. Smoke log checking only
requires finite final metrics, not a convergence threshold.
