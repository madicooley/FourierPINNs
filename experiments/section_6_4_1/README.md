# Section 6.4.1: 2D Poisson

```text
u_xx + u_yy = f(x,y),  (x,y) in [0,2*pi]^2
```

| Case | Exact solution | Selector | Required coordinate frequencies |
| --- | --- | --- | --- |
| `single_scale` | sin(100x) sin(100y) | `test4` | 100 in both axes |
| `multiscale` | sin(6x)cos(20x) + sin(6y)cos(20y) | `test5` | 14 and 26, with constant factors in the other axis |

## Settings

Table 7: three hidden layers, width 100, tanh, 40,000 Adam steps, 12,000
collocation points, 400 boundary points, 100,000 test points, initial Adam/L-BFGS
LRs 0.001/0.01, exponential decay by 0.9 every 1,000 steps. Section 6.2 specifies
float32, coefficient L2 strength 1e-4 and pruning threshold 1e-4. Figure 12
labels FourierPINN `K=200`; the runner defaults to `K=200` and five seeds.

## Critical Memory Limitation

The current implementation uses a full tensor product, not K total features:
with constants enabled the Fourier column count is `(2*K+1)^2`.
The runner prints a partial memory estimate and refuses to execute requests
with more than 4,096 Fourier columns unless `--allow-large-basis` is explicitly
provided. That override is not a recommendation to run `K=200` locally. Smaller
K changes the experiment; `K<100` omits the single-scale target and `K<26` omits
the multiscale target.

## Commands

Dry-run one full-sized experiment (no training/allocation):

```bash
python3 experiments/section_6_4_1/run.py --cases single_scale --ks 200 --seeds 1234
```

Tiny execution checks for both manufactured solutions:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 experiments/section_6_4_1/run.py --smoke --run
```

Smoke uses `K=2`, width 4, 64 residual points, 16 boundary points, 121 test
points, four Adam steps, two blocks of five least-squares (LS) passes, and a
short L-BFGS stage. It deliberately
does not contain the target frequencies and is only a finite-execution check.

For a target-resolving multiscale run, after checking available memory:

```bash
python3 experiments/section_6_4_1/run.py \
  --cases multiscale --ks 26 --seeds 1234 --run
```

This uses 2,809 Fourier columns, fewer than the paper's `K=200` run. A 256-row
test basis chunk is about 2.74 MiB. The runner guard only blocks large column counts and
does not guarantee this smaller job fits in memory or is numerically accurate.

Use runner `--test-batch-size=128` (direct CLI/config `test_batch_size`) to
reduce evaluation memory further.
It does not reduce training-basis or LS memory requirements.
Use `--output-root` to preserve comparison runs. Logs are checked for finite
final metrics, not convergence. Default full sweep: two cases x one K x five
seeds = 10 runs. No full jobs launch unless `--run` is supplied.

```bash
python3 -m unittest discover -s experiments/section_6_4_1 -p 'test_*.py'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 build/tests/fourierpinns_tests '[Section641]'
```
