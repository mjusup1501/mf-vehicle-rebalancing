import torch
import torchist
import numpy as np
from safe_mf.utils.utils import states_to_cell
from safe_mf.utils.distributions import shifted_uniform
from safe_mf.utils.distributions import TruncatedNormal
from safe_mf.utils.utils import states_to_cell, cells_to_index, index_to_cell


def perturb_cell_centers(cell_centers, linspace_x, linspace_y, device='cpu'):
    cells = states_to_cell(cell_centers, linspace_x, linspace_y)
    idx_x, idx_y = cells[:, 0], cells[:, 1]
    x = shifted_uniform(linspace_x[idx_x], linspace_x[idx_x + 1], device=device)
    y = shifted_uniform(linspace_y[idx_y], linspace_y[idx_y + 1], device=device)
    if isinstance(cell_centers, torch.Tensor):
        states = torch.stack([x, y], axis=1)
    elif isinstance(cell_centers, np.ndarray):
        states = np.stack([x, y], axis=1)

    return states


def add_noise_to_states(states, linspace_x, linspace_y, control_std):
    dtype = 'torch'
    if isinstance(states, np.ndarray):
        dtype = 'numpy'
        states = torch.from_numpy(states)
        linspace_x = torch.from_numpy(linspace_x)
        linspace_y = torch.from_numpy(linspace_y)
    with torch.no_grad():
        zeros = torch.zeros_like(states)
        std = torch.ones_like(states) * control_std
        cells = states_to_cell(states, linspace_x, linspace_y)
        cells_x = cells[:, 0]
        cells_y = cells[:, 1]
        lower_bound_x = linspace_x[cells_x].unsqueeze(1)
        lower_bound_y = linspace_y[cells_y].unsqueeze(1)
        upper_bound_x = linspace_x[cells_x + 1].unsqueeze(1)
        upper_bound_y = linspace_y[cells_y + 1].unsqueeze(1)
        lower_bound = torch.cat((lower_bound_x, lower_bound_y), dim=1)
        lower_bound = lower_bound - states
        upper_bound = torch.cat((upper_bound_x, upper_bound_y), dim=1)
        upper_bound = upper_bound - states
        tnorm = TruncatedNormal(loc=zeros, scale=std, a=lower_bound, b=upper_bound)
        noise = tnorm.sample()
        del (
            zeros, std, lower_bound, upper_bound, 
            lower_bound_x, lower_bound_y, 
            upper_bound_x, upper_bound_y
        )
        noisy_states = states + noise
    if dtype == 'numpy':
        return noisy_states.detach().numpy()
    
    return noisy_states


def move_vehicles_to_destinations(destination_idx, num_intervals, linspace_x, linspace_y, device='cpu'):
    dtype = 'torch'
    if isinstance(destination_idx, np.ndarray):
        dtype = 'numpy'
        destination_idx = torch.from_numpy(destination_idx)
        linspace_x = torch.from_numpy(linspace_x)
        linspace_y = torch.from_numpy(linspace_y)
    # Choose demand state uniformly from a cell
    idx_x, idx_y = index_to_cell(destination_idx, num_intervals)
    # Possibly add Gaussian noise instead of the uniform choice
    # Or even just move to a cell center
    x = shifted_uniform(linspace_x[idx_x], linspace_x[idx_x + 1], device=device)
    y = shifted_uniform(linspace_y[idx_y], linspace_y[idx_y + 1], device=device)
    next_states = torch.stack([x, y], dim=1)
    if dtype == 'numpy':
        return next_states.detach().numpy()

    return next_states


def get_matched_states(states, transition_matrix, num_intervals, linspace_x, linspace_y):
    dtype = 'torch'
    if isinstance(states, np.ndarray):
        device = 'cpu'
        dtype = 'numpy'
        states = torch.from_numpy(states)
        transition_matrix = torch.from_numpy(transition_matrix)
        linspace_x = torch.from_numpy(linspace_x)
        linspace_y = torch.from_numpy(linspace_y)
    else:
        device = states.device
    cells = states_to_cell(states, linspace_x, linspace_y)
    origins_idx = cells_to_index(cells, num_intervals)
    transition_probs = transition_matrix[origins_idx, :]
    destinations_idx = torch.multinomial(transition_probs, num_samples=1).squeeze(1)
    next_states = move_vehicles_to_destinations(destinations_idx, num_intervals, linspace_x, linspace_y, device)
    if dtype == 'numpy':
        return next_states.detach().numpy()

    return next_states


def get_cruising_states(states, linspace_x, linspace_y, control_std, mf_transition: bool=False):
    dtype = 'torch'
    if isinstance(states, np.ndarray):
        dtype = 'numpy'
        states = torch.from_numpy(states)
        linspace_x = torch.from_numpy(linspace_x)
        linspace_y = torch.from_numpy(linspace_y)
    next_states = states
    if not mf_transition:
        next_states = add_noise_to_states(next_states, linspace_x, linspace_y, control_std)
    if dtype == 'numpy':
        return next_states.detach().numpy()

    return next_states


def mf_deterministic_step(
    ra_states: torch.Tensor,
    num_agents: int,
    num_intervals: int,
    lower_bound: float,
    upper_bound: float,
    device: str='cpu'
) -> torch.Tensor:
    """
    REQUIRES CPU
    """
    dtype = 'torch'
    if isinstance(ra_states, np.ndarray):
        dtype = 'numpy'
        ra_states = torch.from_numpy(ra_states)
    ra_states = ra_states.cpu()
    mu = torchist.histogramdd(
        ra_states,
        bins=num_intervals,
        low=lower_bound,
        upp=upper_bound,
    )
    mu = mu.to(device).float().reshape(1, -1)
    mu = mu / num_agents
    ra_states = ra_states.to(device)
    if dtype == 'numpy':
        return mu.detach().numpy()

    return mu