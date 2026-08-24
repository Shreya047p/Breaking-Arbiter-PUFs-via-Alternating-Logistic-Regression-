"""
experiments.py
==============
Reproduces the empirical comparison (Part 5 / Table 1) from the CS771
Assignment 2 report: "Bit-in-the-Middle Attack" on an arbiter PUF.

Runs my_latent() and my_latent_updated() across multiple random seeds on
challenge-response data, then reports:
    - cosine similarity between the two w vectors (17-dim)
    - |b - b_hat| (bias difference)
    - latent alignment fraction: how often 2*z_i - 1 == sign(u^T phi(c_i) + a)
    - train accuracy of each model against the true response r

If real CRP data is available at data/public_trn.txt (16 challenge-bit
columns + 1 response column, whitespace/comma separated), it is used.
Otherwise, a synthetic 16-bit arbiter-PUF-style dataset is simulated so
the script still runs end-to-end.

Outputs:
    - results/comparison_table.csv
    - results/cosine_similarity.png
    - results/train_accuracy.png

Run:
    python src/experiments.py
"""

import os
import numpy as np

from submit import my_latent, my_latent_updated, _phi, _insert_middle_bit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 12


# --------------------------------------------------------------------------
# 1. Data loading / simulation
# --------------------------------------------------------------------------
def load_or_simulate_data(n=8000, n_bits=16, seed=0):
    path = os.path.join(DATA_DIR, "public_trn.txt")
    if os.path.exists(path):
        print(f"Loading real CRP data from {path} ...")
        raw = np.loadtxt(path)
        X = raw[:, :n_bits].astype(np.float64)
        y = raw[:, n_bits].astype(np.int64)
        return X, y

    print("No real dataset found in data/public_trn.txt — simulating an "
          "arbiter-PUF-style bit-in-the-middle dataset instead.")
    rng = np.random.RandomState(seed)

    # Ground-truth hidden 16-bit arbiter PUF (u*, a*) generating the middle bit z
    u_star = rng.normal(size=n_bits)
    a_star = rng.normal()

    # Ground-truth 17-bit model (w*, b*) generating the response r from [X, z]
    w_star = rng.normal(size=n_bits + 1)
    b_star = rng.normal()

    X = rng.randint(0, 2, size=(n, n_bits)).astype(np.float64)

    phi_X = _phi(X)
    z_prob = 1.0 / (1.0 + np.exp(-(phi_X @ u_star + a_star)))
    z = (rng.uniform(size=n) < z_prob).astype(np.int64)

    X17 = _insert_middle_bit(X, z)
    phi_X17 = _phi(X17)
    r_prob = 1.0 / (1.0 + np.exp(-(phi_X17 @ w_star + b_star)))
    y = (rng.uniform(size=n) < r_prob).astype(np.int64)

    return X, y


# --------------------------------------------------------------------------
# 2. Metrics
# --------------------------------------------------------------------------
def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def train_accuracy_simple(X, y, w, b, z):
    features = _phi(_insert_middle_bit(X, z))
    margin = features @ w + b
    preds = (margin > 0).astype(np.int64)
    return float(np.mean(preds == y))


def train_accuracy_updated(X, y, w, b, u, a):
    # Recover the latent z implied by (u, a), then score responses with (w, b).
    phi_X = _phi(X)
    z_hat = (phi_X @ u + a > 0).astype(np.int64)
    return train_accuracy_simple(X, y, w, b, z_hat), z_hat


def latent_alignment_fraction(z, u, a, X):
    phi_X = _phi(X)
    z_from_uv = (phi_X @ u + a > 0).astype(np.int64)
    agree = (2 * z - 1) == (2 * z_from_uv - 1)
    return float(np.mean(agree))


# --------------------------------------------------------------------------
# 3. Run comparison across seeds
# --------------------------------------------------------------------------
def run_comparison(X, y, n_seeds=N_SEEDS):
    rows = []
    for seed in range(n_seeds):
        np.random.seed(seed)

        w, b, z = my_latent(X, y)
        w2, b2, u, a = my_latent_updated(X, y)

        cos_sim = cosine_similarity(w, w2)
        bias_diff = abs(b - b2)
        align_frac = latent_alignment_fraction(z, u, a, X)
        acc_simple = train_accuracy_simple(X, y, w, b, z)
        acc_updated, _ = train_accuracy_updated(X, y, w2, b2, u, a)

        rows.append({
            "seed": seed,
            "cosine_similarity": cos_sim,
            "bias_diff": bias_diff,
            "latent_alignment_fraction": align_frac,
            "train_acc_simple": acc_simple,
            "train_acc_updated": acc_updated,
        })
        print(f"seed={seed:2d}  cos_sim={cos_sim:+.3f}  |b-bhat|={bias_diff:.3f}  "
              f"align={align_frac:.3f}  acc_simple={acc_simple:.4f}  "
              f"acc_updated={acc_updated:.4f}")

    return rows


def summarize_and_save(rows):
    import csv

    keys = ["cosine_similarity", "bias_diff", "latent_alignment_fraction",
            "train_acc_simple", "train_acc_updated"]

    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved {csv_path}")

    print("\n=== Summary across seeds (mean / std / range) ===")
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f"{k:28s} mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")


def plot_results(rows):
    import matplotlib.pyplot as plt

    seeds = [r["seed"] for r in rows]
    cos_sims = [r["cosine_similarity"] for r in rows]
    acc_simple = [r["train_acc_simple"] for r in rows]
    acc_updated = [r["train_acc_updated"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(seeds, cos_sims, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("cosine_similarity(w, w_hat)")
    ax.set_title("Weight-vector alignment between my_latent and my_latent_updated\n"
                 "across random seeds (near 0 = uncorrelated)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cosine_similarity.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(seeds))
    ax.bar(x - width / 2, np.array(acc_simple) * 100, width, label="my_latent (unconstrained z)")
    ax.bar(x + width / 2, np.array(acc_updated) * 100, width, label="my_latent_updated (constrained z)")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Train accuracy on response r (%)")
    ax.set_title("Train accuracy: unconstrained vs. PUF-constrained latent bit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "train_accuracy.png"), dpi=150)
    plt.close(fig)

    print("Saved cosine_similarity.png and train_accuracy.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    X, y = load_or_simulate_data()
    print(f"\nData: X {X.shape}, y {y.shape}, class balance {y.mean():.3f}\n")

    rows = run_comparison(X, y, n_seeds=N_SEEDS)
    summarize_and_save(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
