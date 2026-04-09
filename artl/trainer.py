"""
ARTLTrainer: outlier-robust neural network trainer.

Minimizes the Augmented and Regularized Trimmed Loss (ARTL):

    F(θ, ξ) = (1/n) ‖r(θ) − ξ‖²  +  λ · C_{k,q}(f_θ)  +  T_h(ξ)

where
    r_i(θ)  = y_i − f_θ(x_i)
    T_h(ξ)  = (1/n) Σ_{i=1}^{h} ξ²_{(i)}   (h smallest ξ_i²)
    C_{k,q} = Σ_j ∫_Ω |∂^k f/∂x_j^k|^q dx  (HOVR, approximated via MC)

Optimizer: Adam (θ and ξ share the same Adam instance).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hovr import compute_hovr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def _parse_bounds(val, J: int, device: str) -> torch.Tensor:
    """
    Accept scalar or array-like and return a (J,) float32 tensor.
    Scalar → same value for all J dimensions.
    Array-like → must have length J.
    """
    if np.isscalar(val) or isinstance(val, (int, float)):
        return torch.full((J,), float(val), dtype=torch.float32, device=device)
    t = torch.tensor(val, dtype=torch.float32, device=device)
    if t.ndim != 1 or t.shape[0] != J:
        raise ValueError(
            f"domain bound must be a scalar or a length-{J} array, "
            f"got shape {tuple(t.shape)}"
        )
    return t


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ARTLTrainer:
    """
    Fits a regression model by minimizing the ARTL objective with Adam.

    Parameters
    ----------
    model : nn.Module
        PyTorch regression model with scalar output per sample.
        Output shape should be (n,) or (n, 1).
        Smooth activations (Sigmoid, Tanh, GELU) are strongly recommended
        because HOVR relies on higher-order automatic differentiation.
    h : float, default 0.95
        Fraction of inlier samples (0 < h ≤ 1).
        h=0.95 → the 5 % largest losses are discarded each step.
    lam : float, default 1e-5
        Regularization strength λ for the HOVR term.
        Set to 0.0 to use trimmed loss only (no smoothness constraint).
    k : int, default 3
        Order of the pure partial derivative used in HOVR.
        k=1 recovers total-variation regularization.
    q : float, default 2.0
        Exponent in the HOVR integrand |∂^k f / ∂x_j^k|^q.
    domain_lb : scalar or array-like, default 0.0
        Lower bound(s) of the rectangular domain Ω for the HOVR integral.
        Scalar → same value for every input dimension.
        Array of length J → per-dimension bounds.
    domain_ub : scalar or array-like, default 1.0
        Upper bound(s) of Ω (same convention as domain_lb).
    n_mc : int, default 10
        Number of Monte-Carlo samples for the HOVR integral approximation.
    lr : float, default 1e-2
        Adam learning rate.
    n_epochs : int, default 2500
        Number of full-data training epochs.
    lr_scheduler : callable or None, default None
        Factory function ``(optimizer) → scheduler``.
        Called once after the Adam instance is created. Example::

            from torch.optim.lr_scheduler import StepLR
            trainer = ARTLTrainer(
                model, ...,
                lr_scheduler=lambda opt: StepLR(opt, step_size=1000, gamma=0.5)
            )
    device : str or None, default None
        ``'cpu'`` or ``'cuda'``. Auto-detected when None.
    verbose : int, default 500
        Print a progress line every *verbose* epochs.  0 = silent.
    random_state : int or None, default None
        Integer seed for the MC sampling RNG (reproducibility).

    Attributes
    ----------
    history_ : list of float
        ARTL loss value at each epoch (populated after ``fit``).
    xi_ : torch.Tensor, shape (n,)
        Optimized auxiliary variable ξ (populated after ``fit``).
    """

    def __init__(
        self,
        model: nn.Module,
        h: float = 0.95,
        lam: float = 1e-5,
        k: int = 3,
        q: float = 2.0,
        domain_lb=0.0,
        domain_ub=1.0,
        n_mc: int = 10,
        lr: float = 1e-2,
        n_epochs: int = 2500,
        lr_scheduler=None,
        device: str | None = None,
        verbose: int = 500,
        random_state: int | None = None,
    ):
        if not (0.0 < h <= 1.0):
            raise ValueError(f"h must be in (0, 1], got {h}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")
        if n_mc < 1:
            raise ValueError(f"n_mc must be >= 1, got {n_mc}")

        self.h = h
        self.lam = lam
        self.k = k
        self.q = q
        self.domain_lb = domain_lb
        self.domain_ub = domain_ub
        self.n_mc = n_mc
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_scheduler_fn = lr_scheduler
        self.verbose = verbose
        self.random_state = random_state

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.xi_: torch.Tensor | None = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trimmed_loss(xi: torch.Tensor, h_int: int) -> torch.Tensor:
        """T_h(ξ) = (1/n) · sum of the h smallest ξ_i²"""
        n = xi.shape[0]
        sorted_sq = torch.sort(xi ** 2)[0]
        return sorted_sq[:h_int].sum() / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "ARTLTrainer":
        """
        Train the model on (X, y).

        Parameters
        ----------
        X : array-like, shape (n, J)
        y : array-like, shape (n,)

        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X = _to_tensor(X, device=self.device)
        y = _to_tensor(y, device=self.device)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1-D with length n = X.shape[0]")

        n, J = X.shape
        h_int = max(1, round(self.h * n))

        lb = _parse_bounds(self.domain_lb, J, self.device)
        ub = _parse_bounds(self.domain_ub, J, self.device)
        if (lb >= ub).any():
            raise ValueError(
                "domain_lb must be strictly less than domain_ub in every dimension"
            )

        # ξ ∈ ℝⁿ — jointly optimized with θ via Adam
        xi = torch.zeros(n, device=self.device, requires_grad=True)

        opt = torch.optim.Adam(
            list(self.model.parameters()) + [xi], lr=self.lr
        )

        scheduler = None
        if self.lr_scheduler_fn is not None:
            scheduler = self.lr_scheduler_fn(opt)

        self.history_ = []
        self.model.train()
        w = len(str(self.n_epochs))

        for epoch in range(self.n_epochs):
            opt.zero_grad()

            pred = self.model(X)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            r = y - pred

            # (1) data fit:  (1/n) ‖r − ξ‖²
            data_loss = ((r - xi) ** 2).mean()

            # (2) trimmed loss on ξ:  T_h(ξ)
            trim_loss = self._trimmed_loss(xi, h_int)

            # (3) HOVR:  λ · C_{k,q}(f_θ)
            if self.lam > 0.0:
                hovr_loss = self.lam * compute_hovr(
                    self.model, lb, ub, self.k, self.q, self.n_mc
                )
            else:
                hovr_loss = torch.tensor(0.0, device=self.device)

            loss = data_loss + trim_loss + hovr_loss
            loss.backward()
            opt.step()

            if scheduler is not None:
                scheduler.step()

            self.history_.append(loss.item())

            if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
                print(
                    f"[{epoch+1:{w}}/{self.n_epochs}] "
                    f"loss={loss.item():.4f}  "
                    f"data={data_loss.item():.4f}  "
                    f"trim={trim_loss.item():.4f}  "
                    f"hovr(stochastic)={hovr_loss.item():.4f}"
                )

        self.xi_ = xi.detach()
        self.model.eval()
        return self

    def predict(self, X) -> np.ndarray:
        """
        Parameters
        ----------
        X : array-like, shape (m, J)

        Returns
        -------
        y_pred : np.ndarray, shape (m,)
        """
        X = _to_tensor(X, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
        return pred.cpu().numpy()

    def score(self, X, y) -> float:
        """Predictive MSE on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean((y_pred - np.asarray(y, dtype=np.float32)) ** 2))

    def robust_validation_score(self, X, y) -> float:
        """
        Trimmed loss T_h evaluated on (X, y).
        Useful for robust cross-validation / hyper-parameter selection.
        """
        X = _to_tensor(X, device=self.device)
        y = _to_tensor(y, device=self.device)
        m = X.shape[0]
        h_int = max(1, round(self.h * m))
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            sorted_sq = torch.sort((y - pred) ** 2)[0]
            score = sorted_sq[:h_int].sum() / m
        return score.item()
