import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from metropolis import effective_sample_size, metropolis, r_hat  # noqa: E402


def test_sampler_matches_laplace_moments():
    samples, _ = metropolis(50_000, 3.0, rng=np.random.default_rng(0))
    assert abs(samples.mean()) < 0.05
    assert abs(samples.std() - np.sqrt(2)) < 0.05


def test_r_hat_near_one_for_identical_distributions():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 5000))
    assert abs(r_hat(chains) - 1) < 0.01


def test_r_hat_large_for_separated_chains():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 1000)) + np.array([[-10], [0], [10], [20]])
    assert r_hat(chains) > 5


def test_ess_of_independent_draws_close_to_n():
    x = np.random.default_rng(0).normal(size=10_000)
    assert 8_000 < effective_sample_size(x) < 12_000
