# Bit-in-the-Middle Attack on an Arbiter PUF

**CS771: Introduction to Machine Learning, IIT Kanpur — Assignment 2**
Team Deep_Coders (Group 27): Neeraj Kumar, Shreya Pazare, Shubhangam Raj, Kavita Kumari, Arkajyoti Santra

Recovered the parameters of a 17-bit "bit-in-the-middle" arbiter PUF whose central challenge bit is hidden (a latent variable), by deriving and implementing an alternating-optimization (EM-style) attack that jointly estimates the linear model weights and the missing bits — then extended it to a coupled model where the hidden bit is itself the output of a second, hidden 16-bit PUF, and empirically showed the two learned solutions do **not** agree.

## Problem

A 17-bit arbiter PUF response depends on 17 challenge bits, but only 16 are observable — the middle bit `z_i` is hidden. Given only the 16 visible bits `c_i` and the response `r_i`, the goal is to recover the linear PUF weights `(w, b)` by treating `z_i` as a latent variable and maximizing the likelihood

```
P[r_i, z_i | c_i, w, b] = P[r_i | z_i, c_i, w, b] · P[z_i | c_i, w, b]
```

Two variants are solved:

1. **Simple model (`my_latent`)** — `z_i` has a uniform 0.5 prior (no structure assumed).
2. **Updated / coupled model (`my_latent_updated`)** — `z_i` is not arbitrary; it's itself the response of a hidden 16-bit arbiter PUF `(u, a)` applied to `c_i`, so `P[z_i | c_i, u, a] = σ((2z_i - 1)(u^T φ(c_i) + a))`.

## Approach

Both are solved via **alternating (block coordinate descent) optimization**, since jointly optimizing continuous weights and 8000 discrete latent bits is intractable directly:

**`my_latent`** — 2-block alternation:
- **z-step:** fix `(w, b)`, set `z_i` to whichever of `{0, 1}` gives the higher margin (closed form, monotonic in the sigmoid).
- **(w, b)-step:** fix `z`, fit a standard logistic regression on the 17-dim embedded features `φ(I(c_i, z_i)) → r_i`.
- Alternate to convergence (finite discrete configurations ⇒ guaranteed termination); repeat from 3 different `z⁽⁰⁾` initializations (`z⁽⁰⁾ = y`, random, `z⁽⁰⁾ = 1-y`) and keep the run with the best training log-likelihood.

**`my_latent_updated`** — 3-block alternation:
- **z-step:** fix `(w, b, u, a)`, pick `z_i` maximizing the *sum* of both log-likelihood terms.
- **(w, b)-step:** fix `z`, logistic regression on `φ(I(c_i, z_i)) → r_i`.
- **(u, a)-step:** fix `z`, logistic regression on `φ(c_i) → z_i` (z acts as the label for the hidden 16-bit PUF).
- Same multi-restart strategy as above, keeping the run with the best joint objective.

Both use the standard arbiter-PUF feature map `φ(c) = ∏ⱼ (1 − 2cⱼ)` (cumulative product of sign-flipped bits), implemented via a vectorized reverse-cumprod.

## Key finding: the two models do *not* converge to the same solution

Both functions were run on the same 8000 challenge-response pairs across 12 random seeds (both rely on one unseeded random restart, so results vary run to run — this variability is itself part of the analysis).

| Quantity | Mean | Std. dev. | Range |
|---|---|---|---|
| cosine similarity(w, ŵ) | 0.014 | 0.330 | −0.606 to 0.579 |
| \|b − b̂\| | 0.387 | 0.240 | 0.075 to 0.834 |
| latent alignment fraction | 0.468 | 0.098 | 0.247 to 0.621 |
| train accuracy — `my_latent` | 0.9998 | 0.0001 | 0.9998 to 1.0000 |
| train accuracy — `my_latent_updated` | 0.789 | 0.074 | 0.735 to 0.999 |

*(table reproduced from the report; see [`results/comparison_table.csv`](results/comparison_table.csv) for a re-run on simulated data — [`src/experiments.py`](src/experiments.py) shows the same qualitative pattern)*

**Interpretation:**
- Weight vectors `w` and `ŵ` are effectively **uncorrelated** (mean cosine similarity ≈ 0.01, sign flips across restarts).
- The recovered latent bits agree with the hidden-PUF's predicted sign only **46.8%** of the time — statistically indistinguishable from a coin flip.
- `my_latent`, free to pick any `z_i`, essentially **memorizes** the responses (99.98% train accuracy) — with an unconstrained middle bit, the 17-dim feature space has enough capacity to fit almost any label pattern.
- `my_latent_updated`, constrained to make `z` explainable by a *linearly separable* 16-bit PUF, fits noticeably worse (~79%). This gap is the real signal: if the middle bit truly came from a clean 16-bit arbiter PUF, imposing that structure should cost little accuracy — instead it costs ~21 points, and the two algorithms land in different regions of a non-convex, multi-local-optimum landscape.

Full derivation of the alternating-optimization updates (with the log-likelihood factorization) is in [`report/CS771_Assignment2_Report.pdf`](report/CS771_Assignment2_Report.pdf).

## Repo structure

```
bit-in-the-middle-puf-attack/
├── data/               # public_trn.txt (16 challenge bits + response), if available
├── src/
│   ├── submit.py        # my_latent() and my_latent_updated() (assignment deliverable)
│   └── experiments.py   # reproduces the Part 5 empirical comparison (Table 1) across seeds
├── report/
│   └── CS771_Assignment2_Report.pdf
├── results/             # comparison_table.csv, cosine_similarity.png, train_accuracy.png
├── requirements.txt
└── README.md
```

## Usage

```python
from src.submit import my_latent, my_latent_updated
import numpy as np

X = np.load("data/challenges.npy")   # (n, 16) binary challenge matrix
y = np.load("data/responses.npy")    # (n,) observed responses

w, b, z = my_latent(X, y)                     # simple latent model
w2, b2, u, a = my_latent_updated(X, y)        # coupled hidden-PUF model
```

**Reproduce the empirical comparison (Table 1):**
```bash
python src/experiments.py
```
Looks for `data/public_trn.txt` (16 challenge-bit columns + 1 response column). If not found, it simulates an arbiter-PUF-style bit-in-the-middle dataset with a known ground-truth hidden PUF, so the comparison still runs end-to-end and produces the same qualitative result (uncorrelated weights, ~50% latent alignment, large train-accuracy gap).

## Setup

```bash
pip install -r requirements.txt
```

## Course credit

Project for **CS771: Introduction to Machine Learning**, IIT Kanpur (2024). Shared here for portfolio purposes.
