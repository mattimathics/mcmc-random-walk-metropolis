"""
Random Walk Metropolis sampler for the Laplace density f(x) = 0.5 * exp(-|x|),
with a Gelman-Rubin R-hat convergence check across step sizes.

Usage:
    python python/metropolis.py

Writes charts to images/ and prints summary numbers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"


def target_log_pdf(x):
    """Log of f(x) = 0.5 * exp(-|x|). Working in logs avoids underflow far from 0."""
    return np.log(0.5) - np.abs(x)


def metropolis(n_samples, step_size, x0=0.0, rng=None):
    """Return a chain of n_samples and the proportion of proposals accepted."""
    rng = np.random.default_rng() if rng is None else rng
    samples = np.empty(n_samples)
    current = x0
    accepted = 0
    proposals = rng.normal(0.0, step_size, n_samples)
    log_u = np.log(rng.uniform(size=n_samples))
    for i in range(n_samples):
        proposal = current + proposals[i]
        if log_u[i] < target_log_pdf(proposal) - target_log_pdf(current):
            current = proposal
            accepted += 1
        samples[i] = current
    return samples, accepted / n_samples


def r_hat(chains):
    """Gelman-Rubin statistic for an array of shape (n_chains, n_samples)."""
    n = chains.shape[1]
    w = chains.var(axis=1, ddof=1).mean()          # within-chain variance
    b = n * chains.mean(axis=1).var(ddof=1)         # between-chain variance
    var_hat = (n - 1) / n * w + b / n
    return np.sqrt(var_hat / w)


def effective_sample_size(chain):
    """ESS using Geyer's initial positive sequence of autocorrelations."""
    x = chain - chain.mean()
    n = len(x)
    f = np.fft.fft(x, 2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    # Sum autocorrelations in adjacent pairs (lags 0+1, 2+3, ...) until a pair
    # turns negative; tau = -1 + 2 * that sum = 1 + 2 * sum over lags >= 1.
    pair_sum = 0.0
    for k in range(0, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        pair_sum += pair
    return n / (2 * pair_sum - 1)


def r_hat_for_step(step_size, n_samples=2000, n_chains=4, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    starts = rng.normal(0.0, 10.0, n_chains)        # dispersed starting points
    chains = np.array([metropolis(n_samples, step_size, x0, rng)[0] for x0 in starts])
    return r_hat(chains)


def main():
    rng = np.random.default_rng(42)
    IMAGES.mkdir(exist_ok=True)

    # Part 1a: one long chain, compared with the true density
    samples, acc = metropolis(10_000, 1.0, rng=rng)
    print(f"Part 1a (N=10,000, s=1): mean {samples.mean():.3f}, SD {samples.std(ddof=1):.3f}, "
          f"acceptance {acc:.1%}  [true mean 0, true SD {np.sqrt(2):.3f}]")

    xs = np.linspace(-8, 8, 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples, bins=60, density=True, color="#9ecae1", edgecolor="white", label="Samples")
    ax.plot(xs, gaussian_kde(samples)(xs), color="#08519c", lw=2, label="Kernel density estimate")
    ax.plot(xs, 0.5 * np.exp(-np.abs(xs)), "r--", lw=2, label="True f(x) = 0.5·exp(-|x|)")
    ax.set(title="Random Walk Metropolis: N = 10,000, s = 1", xlabel="x", ylabel="Density", xlim=(-8, 8))
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "samples_vs_true_density.png", dpi=150)
    plt.close(fig)

    # Part 1b: R-hat across the step-size grid used in the coursework
    grid = np.linspace(0.001, 1, 50)
    r_hats = np.array([r_hat_for_step(s, rng=rng) for s in grid])
    first_ok = grid[np.argmax(r_hats < 1.05)]
    print(f"Part 1b: R-hat {r_hats[0]:.1f} at s=0.001; first below 1.05 at s={first_ok:.3f}; "
          f"all below 1.05 from s={grid[np.max(np.where(r_hats >= 1.05)) + 1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, r_hats, color="#238b45", lw=2)
    ax.axhline(1.05, color="red", ls=":", label="1.05 threshold")
    ax.set(yscale="log", title="Gelman-Rubin R-hat vs step size (4 chains × 2,000 draws)",
           xlabel="Step size s", ylabel="R-hat (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "r_hat_vs_step_size.png", dpi=150)
    plt.close(fig)

    # Extension: why bigger steps help, up to a point. Small steps are almost
    # always accepted but barely move; huge steps are mostly rejected. Effective
    # sample size (ESS) measures how many independent draws the chain is worth.
    wide = np.array([0.1, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 16])
    acc_rates, ess = [], []
    for s in wide:
        runs = [metropolis(20_000, s, rng=rng) for _ in range(3)]
        acc_rates.append(np.mean([a for _, a in runs]))
        ess.append(np.mean([effective_sample_size(c) for c, _ in runs]))
    print("Extension (N=20,000, mean of 3 runs):")
    for s, a, e in zip(wide, acc_rates, ess):
        print(f"  s={s:>5}: acceptance {a:.1%}, ESS {e:,.0f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(wide, ess, "o-", color="#6a51a3", lw=2, label="Effective sample size")
    ax1.set(xscale="log", xlabel="Step size s (log scale)", ylabel="Effective sample size (of 20,000)",
            title="Efficiency vs step size: ESS peaks at a moderate acceptance rate")
    ax2 = ax1.twinx()
    ax2.plot(wide, acc_rates, "s--", color="#fd8d3c", lw=1.5, label="Acceptance rate")
    ax2.set_ylabel("Acceptance rate")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left")
    fig.tight_layout()
    fig.savefig(IMAGES / "efficiency_vs_step_size.png", dpi=150)
    plt.close(fig)

    # Trace plots: what poor and good mixing look like
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, s in zip(axes, [0.05, 1, 16]):
        for x0 in [-10, 10]:
            chain, a = metropolis(2000, s, x0, rng)
            ax.plot(chain, lw=0.7)
        ax.set_title(f"s = {s}", fontsize=10, loc="left")
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Trace plots from two starting points: too small, about right, too large")
    fig.tight_layout()
    fig.savefig(IMAGES / "trace_plots.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
