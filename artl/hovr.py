"""
Higher-Order Variation Regularization (HOVR) computation.

Approximates:
    C_{k,q}(f_θ) = Σ_j ∫_Ω |∂^k f_θ(x) / ∂x_j^k|^q dx

via Monte Carlo sampling over Ω.
"""

import torch


def compute_kth_pure_partial(out, z, j, k):
    """
    Compute the k-th pure partial derivative ∂^k f / ∂x_j^k
    evaluated at the MC samples z.

    Args:
        out : (M,) tensor — model output f(z) at MC samples
        z   : (M, J) tensor with requires_grad=True — MC sample points
        j   : int — dimension index
        k   : int — derivative order

    Returns:
        (M,) tensor of ∂^k f(z_m) / ∂x_j^k for each sample m
    """
    deriv = out
    for _ in range(k):
        grads = torch.autograd.grad(
            deriv.sum(),
            z,
            create_graph=True,
            retain_graph=True,
        )[0]  # (M, J)
        deriv = grads[:, j]  # pick dimension j → (M,)
    return deriv


def compute_hovr(model, lb, ub, k, q, n_mc):
    """
    Approximate HOVR via Monte Carlo:

        C_{k,q}(f_θ) ≈ vol(Ω) * (1/M) Σ_m Σ_j |∂^k f_θ(z_m)/∂x_j^k|^q

    where z_m ~ Uniform(Ω) i.i.d.

    Args:
        model : nn.Module with scalar output per sample
        lb    : (J,) tensor — lower bounds of Ω per dimension
        ub    : (J,) tensor — upper bounds of Ω per dimension
        k     : int — order of derivative
        q     : float — exponent
        n_mc  : int — number of MC samples M

    Returns:
        Scalar tensor (the HOVR value, with gradient graph for θ)
    """
    J = lb.shape[0]
    device = lb.device
    vol = (ub - lb).prod()

    z = torch.rand(n_mc, J, device=device) * (ub - lb) + lb
    z.requires_grad_(True)

    out = model(z)
    if out.dim() > 1:
        out = out.squeeze(-1)  # (M,)

    hovr = torch.zeros(1, device=device, dtype=out.dtype).squeeze()

    for j in range(J):
        deriv_j = compute_kth_pure_partial(out, z, j, k)
        hovr = hovr + (deriv_j.abs() ** q).mean() * vol

    return hovr
