"""
loss_evidential.py
==================
Phase 2A — Evidential regression loss functions based on the Normal-Inverse-Gamma
(NIG) prior (Amini et al., 2020).

Provides:
    - nig_nll:  Negative log-likelihood of the Student-t marginal.
    - nig_reg:  KL divergence regulariser toward the prior.
    - ErrorScaledEvidentialLoss:  Combined loss with error-scaled KL penalty.
"""

import torch
from torch import nn
import math


def nig_nll(y: torch.Tensor,
            mu: torch.Tensor,
            v: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of the NIG marginal (Student-t).

    Parameters
    ----------
    y     : (B,) true targets
    mu    : (B,) predicted mean
    v     : (B,) virtual evidence for the mean  (> 0)
    alpha : (B,) shape of the Inverse-Gamma      (> 1)
    beta  : (B,) scale of the Inverse-Gamma       (> 0)

    Returns
    -------
    nll : (B,) per-sample negative log-likelihood
    """
    residual = (y - mu) ** 2
    nll = (
        0.5 * torch.log(math.pi / v)
        - alpha * torch.log(beta)
        + (alpha + 0.5) * torch.log(beta + 0.5 * v * residual)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def nig_reg(y: torch.Tensor,
            mu: torch.Tensor,
            v: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor) -> torch.Tensor:
    """
    KL divergence regulariser encouraging the posterior to collapse to the
    prior when the prediction is wrong.

    Simplified form: (|y - mu| · v · (2α + v)) / (v · β)
    which penalises unwarranted evidence (high v, high α) when the residual
    is large.

    Returns
    -------
    reg : (B,) per-sample regularisation term
    """
    residual_abs = torch.abs(y - mu)
    reg = residual_abs * (2.0 * v + alpha)
    return reg


class ErrorScaledEvidentialLoss(nn.Module):
    """
    Combined evidential regression loss with error-scaled KL penalty.

        L = NLL  +  coeff · |y − μ| · KL

    The error-scaling penalises the model severely when it makes a large
    prediction error while expressing low uncertainty (Soleimany et al., 2021).

    Parameters
    ----------
    coeff : float
        Weight on the error-scaled KL term (default 0.1).
    """

    def __init__(self, coeff: float = 0.1):
        super().__init__()
        self.coeff = coeff

    def forward(self,
                y: torch.Tensor,
                mu: torch.Tensor,
                v: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        loss : scalar, mean over the batch
        """
        nll = nig_nll(y, mu, v, alpha, beta)
        reg = nig_reg(y, mu, v, alpha, beta)
        error = torch.abs(y - mu).detach()          # stop gradient on scaling

        loss = nll + self.coeff * error * reg
        return loss.mean()
