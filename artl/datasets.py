"""
Synthetic regression datasets for testing outlier-robust methods.

Each function generates a 2-D grid training set (with injected outliers)
and a random test set (clean), matching the experimental setup of:

    Okuno and Yagishita (2026+), "Outlier-robust neural network training:
    variation regularization meets trimmed loss to prevent functional breakdown"

Available datasets
------------------
make_checkered  : f*(x) = sin(2x₁) cos(2x₂)
make_stripe     : f*(x) = sin(2(x₁ + x₂))

All functions share the same signature and return type (see ``make_checkered``).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# True functions
# ---------------------------------------------------------------------------

def _f_checkered(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * X[:, 0]) * np.cos(2 * X[:, 1])

def _f_stripe(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * (X[:, 0] + X[:, 1]))


# ---------------------------------------------------------------------------
# Generic builder
# ---------------------------------------------------------------------------

def _make_dataset(
    true_fn,
    n_grid: int,
    domain: tuple[float, float],
    noise_std: float,
    outlier_rate: float,
    outlier_value: float,
    n_test: int,
    seed: int | None,
) -> dict:
    rng = np.random.default_rng(seed)

    lb, ub = domain

    # Training grid
    grid = np.linspace(lb, ub, n_grid)
    x1, x2 = np.meshgrid(grid, grid)
    X_train = np.column_stack([x1.ravel(), x2.ravel()]).astype(np.float32)
    y_clean = true_fn(X_train)
    y_train = (y_clean + rng.normal(0, noise_std, size=y_clean.shape)).astype(np.float32)

    # Outlier injection
    n = len(y_train)
    n_out = max(1, round(outlier_rate * n))
    out_idx = rng.choice(n, n_out, replace=False)
    y_train[out_idx] = float(outlier_value)

    # Clean test set
    X_test = rng.uniform(lb, ub, (n_test, 2)).astype(np.float32)
    y_test = true_fn(X_test).astype(np.float32)

    return dict(
        X_train=X_train,
        y_train=y_train,
        out_idx=out_idx,
        X_test=X_test,
        y_test=y_test,
        true_fn=true_fn,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_checkered(
    n_grid: int = 15,
    domain: tuple[float, float] = (0.0, 2 * np.pi),
    noise_std: float = 0.2,
    outlier_rate: float = 0.03,
    outlier_value: float = 5.0,
    n_test: int = 10_000,
    seed: int | None = 0,
) -> dict:
    """
    Checkered dataset:  f*(x) = sin(2x₁) cos(2x₂)

    Parameters
    ----------
    n_grid : int
        Number of grid points per axis.  Training size = n_grid².
    domain : (float, float)
        Shared lower and upper bound for both input dimensions.
    noise_std : float
        Standard deviation of Gaussian noise added to clean targets.
    outlier_rate : float
        Fraction of training samples replaced by outliers.
    outlier_value : float
        Target value assigned to outlier samples.
    n_test : int
        Number of randomly drawn (clean) test samples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X_train  : np.ndarray (n_grid², 2)
        y_train  : np.ndarray (n_grid²,)   — contains outliers
        out_idx  : np.ndarray (n_out,)     — indices of outliers in X_train
        X_test   : np.ndarray (n_test, 2)
        y_test   : np.ndarray (n_test,)    — clean targets
        true_fn  : callable (X) → y
        domain   : (float, float)
    """
    return _make_dataset(
        _f_checkered, n_grid, domain, noise_std,
        outlier_rate, outlier_value, n_test, seed
    )


def make_stripe(
    n_grid: int = 15,
    domain: tuple[float, float] = (0.0, 2 * np.pi),
    noise_std: float = 0.2,
    outlier_rate: float = 0.03,
    outlier_value: float = 5.0,
    n_test: int = 10_000,
    seed: int | None = 0,
) -> dict:
    """
    Stripe dataset:  f*(x) = sin(2(x₁ + x₂))

    Parameters and return value: see ``make_checkered``.
    """
    return _make_dataset(
        _f_stripe, n_grid, domain, noise_std,
        outlier_rate, outlier_value, n_test, seed
    )


# Registry for use in examples
DATASETS = {
    "checkered": make_checkered,
    "stripe":    make_stripe,
}
