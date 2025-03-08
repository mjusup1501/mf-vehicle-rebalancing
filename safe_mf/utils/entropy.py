from typing import Union
import torch
import numpy as np


def entropy(mu: Union[torch.Tensor, np.ndarray], w: Union[torch.Tensor, np.ndarray] = None):
    dtype = 'torch'
    if isinstance(mu, np.ndarray):
        dtype = 'numpy'
        mu = torch.tensor(mu)
    if isinstance(w, np.ndarray):
        w = torch.tensor(w)
    if w is None:
        w = torch.ones_like(mu)
    mu = mu.flatten()
    w = w.flatten()
    mu_ = mu.clone()
    entropy = -torch.sum(w * torch.log(mu_ + 1e-10) * mu_)
    if dtype == 'numpy':
        return entropy.detach().numpy() 
    return entropy


def entropic_constraint(mu: torch.Tensor, c: float, w: torch.Tensor = None):
    return entropy(mu, w) - c


def max_entropy(num_cells: int):
    return torch.log(torch.tensor(num_cells))

# Kulback-Leibler divergence
def kl_divergence(true: torch.Tensor, pred: torch.Tensor):
    true = true / true.sum(dim=1)
    pred = pred / pred.sum(dim=1)
    return torch.sum(true * (torch.log(true + 1e-10) - torch.log(pred + 1e-10)), dim=1)

# Jensen-Shannon divergence
def js_divergence(p, q):
    p = p / p.sum(dim=1)
    q = q / q.sum(dim=1)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
