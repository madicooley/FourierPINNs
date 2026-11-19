# Manufactured Solutions

This file summarizes the built-in manufactured-solution tests and how they map to `test_func` values used by `LoadedPDEDomain::generateDomainData()`.

## 1D Poisson / Allen-Cahn
- `test1_kK` (Sections 6.3.1 and 6.3.2 for k=100):
  - u(x) = sin(kx)
  - Domain: x in [0, 2pi]
  - k is parsed from the suffix (e.g. `test1_k10`)
- `test2`:
  - u(x) = sin(x) + 0.1 sin(20x) + 0.05 cos(100x)
- `test3`:
  - u(x) = sin(6x) cos(100x)

For Allen-Cahn, the forcing is computed as:
- f(x) = u_xx + u (u^2 - 1)

For Poisson, the forcing is:
- f(x) = u_xx

## 2D Poisson (Sec 6.4.1)
- `test4`:
  - u(x,y) = sin(100x) sin(100y)
- `test5`:
  - u(x,y) = sin(6x) cos(20x) + sin(6y) cos(20y)

Poisson forcing is:
- f(x,y) = u_xx + u_yy

## 2D Steady Allen-Cahn (Sec 6.4.2)
- `test6`:
  - u(x,y) = (sin x + 0.1 sin 20x + cos 100x)
           * (sin y + 0.1 sin 20y + cos 100y)

Allen-Cahn forcing is:
- f(x,y) = u_xx + u_yy + u (u^2 - 1)

## 1D One-Way Wave (Sec 6.5)
- `test7`:
  - u(t,x) = sin(pi x) cos(10 pi t) + sin(2 pi x) cos(20 pi t)
  - Dimension ordering is (t, x)
  - Forcing for `pde_problem=onewaywave_inhomo`:
    - v(t,x) = u_t + 10 u_x

Boundary/IC generation notes:
- For the one-way wave, boundary data uses spatial edges x=0 and x=1 across time.
- Initial condition data (`XicTrain`, `uicTrain`) uses t=0 with x in [0,1].
