"""CORAL (Sun & Saenko 2016) and DANN gradient-reversal components for Exp3/Exp4."""
import numpy as np, torch, torch.nn as nn


def _sqrtm(C, eps=1e-6):
    v, Q = np.linalg.eigh(C); v = np.maximum(v, eps)
    return Q @ np.diag(np.sqrt(v)) @ Q.T


def _invsqrtm(C, eps=1e-6):
    v, Q = np.linalg.eigh(C); v = np.maximum(v, eps)
    return Q @ np.diag(1.0 / np.sqrt(v)) @ Q.T


def coral_fit(Xs, Xt, eps=1e-6):
    """Return transform A mapping SOURCE features toward TARGET covariance. Fit uses source-train and target-train only."""
    d = Xs.shape[1]
    Cs = np.cov(Xs.T) + eps * np.eye(d)
    Ct = np.cov(Xt.T) + eps * np.eye(d)
    return _invsqrtm(Cs) @ _sqrtm(Ct)


def coral_apply(X, A):
    return (X @ A).astype(np.float32)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainDiscriminator(nn.Module):
    def __init__(self, emb=32, n_domains=2, lam=0.5, hidden=64):
        super().__init__()
        self.lam = lam
        self.net = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(), nn.Linear(hidden, n_domains))

    def forward(self, z):
        return self.net(GradReverse.apply(z, self.lam))
