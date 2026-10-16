# The role of locality and weight sharing in convolutional teacher–student models

Code accompanying the paper by **Francesco Mori** and **Melanie Weber**.

The paper compares a convolutional network (CNN) and a locally connected network (LCN)
that learn a convolutional teacher by online stochastic gradient descent (SGD). In the
limit of large feature dimension D, the learning dynamics reduce to ordinary differential
equations (ODEs) for a few order parameters. This repository implements these ODEs, the
closed-form results of the paper and online SGD, and reproduces every data figure.

## Installation

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib
pip install jupyter                 # optional, for the notebooks
```

## Usage

```bash
bash run_all.sh                             # every figure (about 90 minutes on 8 cores)
bash run_all.sh --quick                     # every figure at low statistics (a few minutes)
python scripts/fig_rescaling.py --replot    # redraw one figure from the cached results
```

Every figure script accepts `--replot` (draw from the results cached in `data/`),
`--quick` (fewer seeds or grid points, output in `figures/quick/`) and `--workers N`
(parallel processes for the scripts that run many independent integrations, all cores by
default). Figures are written as PDF to `figures/`. Each notebook in `notebooks/` shows the
description and parameters of one script and draws its figures from the cached results.

## Figures

| Figure | Content | Script | Notebook | Runtime |
|---|---|---|---|---|
| 2 | P-fold rescaling of the CNN and the LCN | `fig_rescaling.py` | `01_rescaling` | 20 s |
| 3 | plateau and specialization time of the CNN | `fig_plateau.py` | `02_plateau` | 3 min |
| 4 | alignment with the zero-error solutions | `fig_alignment.py` | `03_alignment` | 11 min |
| 5 | asymptotic error | `fig_noise_floor.py` | `04_noise_floor` | 10 s |
| 6, 8, 9 | phase diagram at tuned learning rates | `fig_phase_diagram.py` | `05_phase_diagram` | 20 min |
| 7 | specialization time in further settings (App. E) | `fig_plateau_checks.py` | `06_appendix_plateau_checks` | 9 min |
| 10 | asymptotic error with ReLU (App. J) | `fig_relu.py` | `07_appendix_relu` | 18 min |
| 11 | rescaling with ReLU (App. J) | `fig_relu_rescaling.py` | `08_appendix_relu_rescaling` | 20 min |

Figure 1 is a diagram drawn in the paper. The docstring of each script lists the
parameters of its figure.

## Code

```
cnnlcn/integrals.py      Gaussian integrals I2, J2, I3, I4 of g(z) = erf(z / sqrt 2), ReLU kernel
cnnlcn/model.py          order parameters, generalization error, ODEs of both students, initial states
cnnlcn/solve.py          ODE integration, specialization time, noise floor at finite eta
cnnlcn/sgd.py            online SGD, batched over independent runs
cnnlcn/closed_forms.py   early-time expansion and small-eta noise floors
cnnlcn/style.py          figure style
scripts/                 one script per figure, plus common.py
```

## Conventions

The code uses the notation and units of the paper.

- Inputs `x = (x_0, ..., x_{P-1})`, `x_p` in R^D, with `E[x_{p,a} x_{q,b}] = S_pq δ_ab`,
  `S_pp = 1` and `S_pq = c` for `p ≠ q`. Positions are taken mod P.
- Teacher preactivations `λ_p = (k_t D)^(-1/2) Σ_l w_l · x_{p+l}` with
  `w_l · w_l' / D = T δ_ll'`. Student preactivations `λ̂_p = (k_s D)^(-1/2) Σ_l ŵ_l · x_{p+l}`,
  with one kernel (CNN) or one kernel per position (LCN). Outputs `Σ_p g(λ_p)` and
  `Σ_p g(λ̂_p)`, with `g(z) = erf(z/√2)`.
- Labels `y = f*(x) + σ ξ` with `ξ ~ N(0, 1)` and `σ ≥ 0`. Online SGD on `(y − ŷ)²/2` with
  learning rate `η`, training time `α = t / D`. Generalization error `ε_g = E[(f* − ŷ)²]`.
- Order parameters `R_ll' = ŵ_l · w_l' / (√(k_s k_t) D)` and `Q_ll' = ŵ_l · ŵ_l' / (k_s D)`,
  with position indices for the LCN (`cnnlcn/model.py`).
- Random initial conditions: student weights i.i.d. N(0, 1). `model.random_init` samples
  the resulting order parameters exactly at any D, `model.init_large_D` is the limit
  D → ∞ (R = 0, Q = I / k_s).

## Numerical methods

- ODEs are integrated with LSODA (relative tolerance 1e-8, absolute 1e-11).
- The specialization time is found by event detection on `max_s γ_s = 0.99`.
- The noise floor at finite η is the fixed point of the ODEs near perfect recovery, found
  by least squares with Q constrained to be symmetric.
- Phase diagram: for each student and c, the ODEs are integrated from the D → ∞ initial
  state for 80 learning rates (98 from 1e-5 for P = 8), and the best learning rate at each
  training time is refined by a parabola in (log η, log ε_g).
- SGD computes the generalization error exactly from the weights, through the order
  parameters (erf) or the arc-cosine kernel (ReLU).
