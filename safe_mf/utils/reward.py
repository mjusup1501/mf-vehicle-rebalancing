import torch
import numpy as np
from safe_mf.utils.data import normalize_distributions
from safe_mf.utils.entropy import js_divergence
from typing import Callable

def lifted_reward(
    mu_repositioned: torch.Tensor,
    mu_matched: torch.Tensor,
    mu_cruising: torch.Tensor,
    mu_repositioned_next: torch.Tensor,
    mu_matched_next: torch.Tensor,
    mu_cruising_next: torch.Tensor,
    demand_matrix: torch.Tensor,
    weight_matrix: torch.Tensor,
    max_barrier: float,
    barrier_lambda: float,
    constraint_function: Callable = None,
) -> float:
    dtype = 'torch'
    if isinstance(mu_repositioned, np.ndarray):
        dtype = 'numpy'
        mu_repositioned = torch.from_numpy(mu_repositioned)
        mu_matched = torch.from_numpy(mu_matched)
        mu_cruising = torch.from_numpy(mu_cruising)
        mu_repositioned_next = torch.from_numpy(mu_repositioned_next)
        mu_matched_next = torch.from_numpy(mu_matched_next)
        mu_cruising_next = torch.from_numpy(mu_cruising_next)
        demand_matrix = torch.from_numpy(demand_matrix)
        weight_matrix = torch.from_numpy(weight_matrix)
    repositioning_penalty = 0.0
    repositioning_bonus = 0.0
    cruising_penalty = 0.0
    js_penalty = 1.
    mu_available = mu_matched + mu_cruising #+ next_mu_repositioned
    mu_available = normalize_distributions(mu_available, p=1)
    demand_matrix = normalize_distributions(demand_matrix, p=1)
    a = mu_matched.sum(), js_divergence(mu_available, demand_matrix).squeeze(0)
    # REWARD 0
    main_reward = (
        mu_matched.sum() 
        - repositioning_penalty * mu_repositioned.sum()
        - cruising_penalty * mu_cruising.sum()
        - js_penalty * js_divergence(mu_available, demand_matrix).squeeze(0)
    )
    # Normalize main reward because it is in range [-1, 1]
    main_reward = (main_reward + 1) / 2
    if constraint_function is None:
        if dtype == 'numpy':
            return main_reward.detach().numpy()
        return main_reward
    else:
        mu_available_next = mu_matched_next + mu_cruising_next #+ next_mu_repositioned
        mu_available_next = normalize_distributions(mu_available_next, p=1)
        constraint_value = constraint_function(mu_available_next, weight_matrix)
        barrier = torch.nan_to_num(
            torch.log(constraint_value),
            nan=-10,
            neginf=-10,
        )
        if barrier.item() > 0:
            barrier = barrier / max_barrier
        barrier = barrier_lambda * barrier
        reward = (main_reward + barrier) / 2
        if dtype == 'numpy':
            return reward.detach().numpy()
        return reward
        