from collections import deque, namedtuple
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
import numpy as np


class MatchingDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

def concat_inputs(
    states: torch.Tensor,
    mu: torch.Tensor,
    rho: torch.Tensor,
):
    if mu.shape[0] == 1:
        mu = mu.expand(states.shape[0], -1)
    if rho.shape[0] == 1:
        rho = rho.expand(states.shape[0], -1)
    return torch.cat([states, mu, rho], dim=1)


def normalize_states(
    states: torch.Tensor, state_space: Tuple[float, float]
) -> torch.Tensor:
    # min-max normalization
    return (states - state_space[0]) / (state_space[1] - state_space[0])


def normalize_distributions(
    distributions: torch.Tensor,
    p: int = 1,
    dim: int = -1,
) -> torch.Tensor:
    if len(distributions.shape) == 1:
        distributions = distributions.reshape(1, -1)
    return torch.nn.functional.normalize(distributions, dim=dim, p=p)


def denormalize_actions(
    actions: torch.Tensor, action_space: Tuple[float, float]
) -> torch.Tensor:
    return (actions * (action_space[1] - action_space[0]) / 2) + (
        action_space[1] + action_space[0]
    ) / 2


def normalize_inputs(
    states: torch.Tensor,
    mu: torch.Tensor,
    rho: torch.Tensor,
    state_space: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states_ = normalize_states(states, state_space)
    mu_ = normalize_distributions(mu)
    rho_ = normalize_distributions(rho)
    return (states_, mu_, rho_)

def agg_by_labels(samples, labels, n_samples, agg='sum', p=1):
    ''' select mean(samples), count() from samples group by labels order by labels asc '''
    if isinstance(samples, np.ndarray):
        dtype = 'numpy'
        samples = torch.from_numpy(samples).float()
        labels = torch.from_numpy(labels).long()
    weight = torch.zeros(n_samples, samples.shape[0]).to(samples.device) # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1
    if agg == 'mean':
        weight = normalize_distributions(weight, p=p, dim=1) # l1 normalization
    agg = weight @ samples
    if dtype == 'numpy':
        agg = agg.numpy()

    return agg