"""Losses of Sec. 6 (Eqs. loss_cls, loss_dir, loss_alp, loss_total)."""

import torch
import torch.nn.functional as F

from sid_common import N_CLASSES, NOISE_CLASS
from sidnet import PATTERNS

LABEL_SMOOTHING = 0.05
HARD_CLASSES = ["NS", "EW", "NSEW", "NSE", "NSW", "NEW", "SEW"]


def class_weights():
    """w_k = 2.0 for the hardest classes, 1.0 for all others (both regimes)."""
    w = torch.ones(N_CLASSES)
    for name in HARD_CLASSES:
        w[NOISE_CLASS[name]] = 2.0
    return w


def direction_targets(y):
    """b(y) in {0,1}^4: which of N, S, E, W are suppressed in the true class."""
    return PATTERNS.to(y.device)[y]


def loss_cls(logits, y, w, eps=LABEL_SMOOTHING):
    """-sum_k w_k y~_k log p_k, with y~ = 1 - eps + eps/16 for the true class, eps/16 otherwise."""
    logp = F.log_softmax(logits.float(), -1)
    target = torch.full_like(logp, eps / N_CLASSES)
    target.scatter_(1, y[:, None], 1 - eps + eps / N_CLASSES)
    return -(w.to(logp.device)[None] * target * logp).sum(-1).mean()


def loss_dir(s_hat, y):
    """BCE between the fused suppression probabilities s_hat and b(y)."""
    return F.binary_cross_entropy(s_hat.float().clamp(1e-6, 1 - 1e-6), direction_targets(y))


def loss_alp(alpha_hat, alpha_true, y):
    """MSE over the suppressed directions only; left out when there are none."""
    active = direction_targets(y).bool()
    if active.sum() == 0:
        return None
    return F.mse_loss(alpha_hat.float()[active], alpha_true.float()[active])


def total_loss(logits, s_hat, alpha_hat, y, alpha_true, w, b_cls, b_dir, b_alp):
    l_cls = loss_cls(logits, y, w)
    l_dir = loss_dir(s_hat, y)
    l_alp = loss_alp(alpha_hat, alpha_true, y)
    total = b_cls * l_cls + b_dir * l_dir
    if l_alp is not None:
        total = total + b_alp * l_alp
    parts = {"cls": l_cls.item(), "dir": l_dir.item(), "alp": 0.0 if l_alp is None else l_alp.item()}
    return total, parts
