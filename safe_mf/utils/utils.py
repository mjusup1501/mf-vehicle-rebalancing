import re
import torch
import numpy as np


def find_best_ckpt(policy_ckpt_dir):
    best_episodes = list(policy_ckpt_dir.glob("policy_best*[0-9].pt"))
    if not best_episodes:
        return "policy_final.pt"
    best_episode = 0
    for episode in best_episodes:
        episode = int(re.findall(r'\d+', episode.name)[0])
        if episode > best_episode:
            best_episode = episode
    best_ckpt = f"policy_best{best_episode}.pt"

    return best_ckpt


def find_last_ckpt(policy_ckpt_dir):
    final_episode = list(policy_ckpt_dir.glob("policy_final.pt"))
    if final_episode:
        return "policy_final.pt"
    last_episodes = list(policy_ckpt_dir.glob("policy*[0-9].pt"))
    last_episode = 0
    for episode in last_episodes:
        episode = int(re.findall(r'\d+', episode.name)[0])
        if episode > last_episode:
            last_episode = episode
    last_ckpt = f"policy{last_episode}.pt"

    return last_ckpt


def find_last(dir, prefix, suffix):
    last_episodes = list(dir.glob(f"{prefix}*[0-9].{suffix}"))
    last_episode = 0
    for episode in last_episodes:
        episode = int(re.findall(r'\d+', episode.name)[0])
        if episode > last_episode:
            last_episode = episode
    last_ckpt = f"{prefix}{last_episode}.{suffix}"

    return last_ckpt


def extract_digits(string):
    # Use regular expression to find digits
    digits = re.findall(r'\d+', string)
    return [int(d) for d in digits][0]


def find_exact(dir, prefix, suffix):
    episodes = list(dir.glob(f"{prefix}*[0-9].{suffix}"))
    if not episodes:
        return None
    episode = 0
    for ep in episodes:
        ep = int(re.findall(r'\d+', ep.name)[0])
        if ep > episode:
            episode = ep
    ckpt = f"{prefix}{episode}.{suffix}"

    return ckpt


def index_to_cell(index, n):
    i = index // n
    j = index % n
    return (i, j)


def cells_to_index(cells, n):
    if isinstance(cells, tuple):
        idx = cells[0] * n + cells[1]
        return int(idx)
    if cells.ndim == 2:
        idx = cells[:, 0] * n + cells[:, 1]
    elif cells.ndim == 3:
        idx = cells[:, :, 0] * n + cells[:, :, 1]
    if isinstance(idx, torch.Tensor):
        return idx.long()
    if isinstance(idx, np.ndarray):
        return idx.astype(int)


def states_to_cell(states, linspace_x, linspace_y):
    # Use digitize to find the indices of the cells that contains given states
    if isinstance(states, torch.Tensor):
        buckets_x = torch.bucketize(states[:, 0], linspace_x) - 1
        buckets_y = torch.bucketize(states[:, 1], linspace_y) - 1
        cells = torch.stack([buckets_x, buckets_y], dim=1)
    else:
        buckets_x = np.digitize(states[:, 0], linspace_x, right=True) - 1
        buckets_y = np.digitize(states[:, 1], linspace_y, right=True) - 1
        cells = np.column_stack([buckets_x, buckets_y])

    # Bucketize returns -1 for states equal to 0 so we overwrite it
    mask_x = cells[:, 0] == -1
    cells[:, 0][mask_x] = 0
    mask_y = cells[:, 1] == -1
    cells[:, 1][mask_y] = 0

    return cells


def get_neighboring_cells(cell, radius, N):
    """ Get the indices of the neighboring cells of one cell.
    
    Args:
        cell: a two-tuple represent the index of the cell.
        radius: the radius of the neighborhood.
        N: the length of the grid.
    
    Returns:
        A list of neighboring cells' indices.
    """
    i, j = cell
    neighbors = []
    # cell itself is included
    for x in range(i - radius, i + radius + 1):
        for y in range(j - radius, j + radius + 1):
            if 0 <= x < N and 0 <= y < N:  # Check if the neighbor is within the grid boundaries
                neighbors.append((x, y))
    
    return neighbors


def reverse_min_max(x, min, max, min_data, max_data):
    # Scale the data
    x_scaled = (x - min_data) / (max_data - min_data)
    # Reverse scaling
    x_reversed = x_scaled * (max - min) + min

    return x_reversed



