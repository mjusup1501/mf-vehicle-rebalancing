import math
from time import time
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torchist
import numpy as np

from safe_mf.envs.env import Env
from safe_mf.models.policy import MFPolicy
from safe_mf.utils.utils import index_to_cell
from safe_mf.utils.transition import (
    perturb_cell_centers,
    add_noise_to_states, 
    get_matched_states, 
    get_cruising_states,
    mf_deterministic_step
)
from safe_mf.utils.reward import lifted_reward
from safe_mf.utils.data import normalize_distributions
from safe_mf.utils.entropy import js_divergence, max_entropy, entropy
from torch.distributions import Normal
from safe_mf.utils.distributions import shifted_uniform
import torch
from safe_mf.models.matching_simulators import (
    GlobalMatchingProcess, 
    LocalMatchingProcess,
)
from safe_mf.models.matching_models import (
    CellLevelApproximation,
    OptimalTransport,
)
from safe_mf.models.statistical_model import StatisticalMatching #, GNNStatisticalMatching


class VehicleRepositioningSimultaneous(Env):
    def __init__(
        self,
        num_cells: int,
        demand_matrix: torch.Tensor,
        origin_destination_matrix: torch.Tensor,
        weight_matrix: torch.Tensor,
        control_std: float,
        mu_init_type: str = 'uniform',
        barrier_lambda: float = 0.001,
        constraint_function: Callable = None,
        max_barrier: float = None,
        state_space: Tuple[float, float] = (0.0, 1.0),
        action_space: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),     
        mf_matching_cfg: Mapping[str, Any] = None,
        real_world_matching_cfg: Mapping[str, Any] = None,
        exec_type: str = "train",
        num_agents: int = 1,
        reward_type: str = 'current_mu',
        matching_target: str = 'regression',
    ) -> None:
        super().__init__(state_dim=2, num_cells=num_cells, action_dim=2, device=device)
        if demand_matrix.ndim == 1:
            self.demand_matrix = demand_matrix.unsqueeze(0)
        else:
            self.demand_matrix = demand_matrix.unsqueeze(1)
        self.mu_init_type = mu_init_type
        if mu_init_type == 'uniform':
            self.mu_init = torch.ones(size=(self.num_cells,), device=self.device).reshape(1, -1) / self.num_cells
        elif mu_init_type == 'demand':
            if self.demand_matrix.ndim == 2:
                self.mu_init = self.demand_matrix.clone()
            else:
                self.mu_init = self.demand_matrix[-1].clone()
                self.demand_matrix = self.demand_matrix[:-1]
        if weight_matrix.ndim == 1:
            self.weight_matrix = weight_matrix.unsqueeze(0)
        else:
            self.weight_matrix = weight_matrix.unsqueeze(1)
        self.valid_cells_mask = self.weight_matrix.bool().flatten()
        self.origin_destination_matrix = origin_destination_matrix
        # When all rows in the demand matrix are 0 we want agents to stay in the current cell
        # Check if I already have this logic during input preprocessing
        mask = torch.all(self.origin_destination_matrix == 0, dim=-1).nonzero()
        if self.origin_destination_matrix.ndim == 2:
            self.origin_destination_matrix[mask[:, 0], mask[:, 0]] = 1.
        else:
            self.origin_destination_matrix[mask[:, 0], mask[:, 1], mask[:, 1]] = 1.
        self.state_space = state_space
        self.action_space = action_space
        self.barrier_lambda = barrier_lambda
        self.num_intervals = int(math.sqrt(self.num_cells))
        self.control_std = control_std if control_std else None
        self.exec_type = exec_type
        self.num_agents = num_agents
        self.reward_type = reward_type
        self.matching_target = matching_target
        self.constraint_function = constraint_function
        self.max_barrier = max_barrier
        self.linspace_x = torch.linspace(self.state_space[0], self.state_space[1], self.num_intervals + 1, device=device)
        self.linspace_y = torch.linspace(self.state_space[0], self.state_space[1], self.num_intervals + 1, device=device)
        if self.control_std is None:
            self.normal_control = 0.0
        else:
            self.normal_control = Normal(loc=0.0, scale=self.control_std)

        with torch.no_grad():
            self.cell_centers_1d = (
                torch.arange(0, self.num_intervals, 1, device=self.device)
                .reshape(1, -1)
                / self.num_intervals
            ) + 0.5 / self.num_intervals

            self.cell_centers = torch.cartesian_prod(*[self.cell_centers_1d.squeeze(0)] * self.state_dim)
        padding = torch.tensor([10 ** 4], device=self.device).unsqueeze(0)
        upper = self.cell_centers_1d + (0.5 / self.num_intervals)
        lower = self.cell_centers_1d - (0.5 / self.num_intervals)
        first = lower[0][0].reshape(padding.shape)
        last = upper[0][-1].reshape(padding.shape)
        self.lower = torch.cat([-padding, lower, last], dim=1)
        self.upper = torch.cat([first, upper, padding], dim=1)

        # Defining lambda functions
        self.perturb_cell_centers = lambda cell_centers: perturb_cell_centers(cell_centers, self.linspace_x, self.linspace_y, self.device)
        self.get_matched_states = (
            lambda states, transition_matrix: 
            get_matched_states(states, transition_matrix, self.num_intervals, self.linspace_x, self.linspace_y)
        )
        self.get_cruising_states = (
            lambda states, mf_transition: 
            get_cruising_states(states, self.linspace_x, self.linspace_y, self.control_std, mf_transition)
        )
        self.add_noise_to_states = lambda states: add_noise_to_states(states, self.linspace_x, self.linspace_y, self.control_std)
        self.lifted_reward = (
            lambda 
            mu_repositioned, mu_matched, mu_cruising, 
            mu_repositioned_next, mu_matched_next, mu_cruising_next, 
            demand_matrix:
            lifted_reward(
                mu_repositioned, mu_matched, mu_cruising,
                mu_repositioned_next, mu_matched_next, mu_cruising_next, 
                demand_matrix, self.weight_matrix, self.max_barrier, 
                self.barrier_lambda, self.constraint_function
            )
        )
        self.mf_deterministic_step = lambda ra_states: mf_deterministic_step(
            ra_states, self.num_agents, self.num_intervals, self.state_space[0], self.state_space[1], self.device
        )

        self.mf_matching_type = mf_matching_cfg.get('type') if mf_matching_cfg is not None else None
        self.mf_matching = self._get_matching_process(mf_matching_cfg)
        self.real_world_matching_type = real_world_matching_cfg.get('type') if real_world_matching_cfg is not None else self.mf_matching_type
        self.real_world_matching = self._get_matching_process(real_world_matching_cfg) if real_world_matching_cfg is not None else self.mf_matching

    def _get_matching_process(self, config, ckpt=None):
        if config is None:
            return None
        matching_type = config.pop('type')
        ckpt = config.pop('checkpoint', None)
        if matching_type == 'cell_level_approximation':
            matching_process = CellLevelApproximation(
                linspace_x=self.linspace_x, 
                linspace_y=self.linspace_y,
                num_intervals=self.num_intervals,
                weight_matrix=self.weight_matrix,
                device=self.device
            )
        elif matching_type == 'optimal_transport':
            matching_process = OptimalTransport(
                num_cells=self.num_cells,
                num_intervals=self.num_intervals,
                linspace_x=self.linspace_x,
                linspace_y=self.linspace_y,
                weight_matrix=self.weight_matrix,
                **config
            )
        elif matching_type == 'local_matching_simulator':
            matching_process = LocalMatchingProcess(
                num_cells=self.num_cells,
                num_intervals=self.num_intervals,
                control_std=self.control_std,
                linspace_x=self.linspace_x,
                linspace_y=self.linspace_y,
                weight_matrix=self.weight_matrix,
                **config
            )
        elif matching_type == 'global_matching_simulator':
            matching_process = GlobalMatchingProcess(
                num_cells=self.num_cells,
                num_intervals=self.num_intervals,
                control_std=self.control_std,
                linspace_x=self.linspace_x,
                linspace_y=self.linspace_y,
                weight_matrix=self.weight_matrix,
                **config
            )
        elif matching_type == 'statistical_model':
            if ckpt is not None:
                matching_process = torch.load(ckpt, map_location=self.device)
            else:
                matching_process = StatisticalMatching(
                    state_dim=self.state_dim,
                    num_cells=self.num_cells,
                    linspace_x=self.linspace_x,
                    linspace_y=self.linspace_y,
                    target_type=self.matching_target,
                    weight_matrix=self.weight_matrix,
                    device=self.device,
                    **config,
                )      

        return matching_process
    

    def reset(self) -> None:
        self.inference_time = 0.0
        self.mu = self.mu_init.clone()
        # Initialize mu from demand matrix
        self.mu *= self.weight_matrix
        self.mu = normalize_distributions(self.mu, p=1)
        # Initialize representative agents from mu
        ra_state_cells = torch.multinomial(
                                self.mu, 
                                num_samples=self.num_agents, 
                                replacement=True
                            ).to(self.device).flatten()
        ra_cells_x, ra_cells_y = index_to_cell(ra_state_cells, self.num_intervals)
        # Uniformly generate (x,y) coordinates within cells
        low_x = self.linspace_x[ra_cells_x].unsqueeze(1)
        low_y = self.linspace_y[ra_cells_y].unsqueeze(1)
        high_x = self.linspace_x[ra_cells_x + 1].unsqueeze(1)
        high_y = self.linspace_y[ra_cells_y + 1].unsqueeze(1)
        low = torch.cat((low_x, low_y), dim=1)
        high = torch.cat((high_x, high_y), dim=1)
        self.ra_states = shifted_uniform(
                            low=low, 
                            high=high,
                            size=(self.num_agents, self.state_dim), 
                            device=self.device
                        )
    
    
    def get_repositioned_states(
            self, states, mu, demand_matrix, policy, exploration, 
            mf_transition: bool=False, step: Optional[int]=None
    ):
        start_time = time()
        actions = policy(states, mu, demand_matrix, exploration, step)
        self.inference_time += time() - start_time
        repositioning_actions = actions[:, :self.action_dim]
        next_states = states + repositioning_actions
        # Should we add noise before or after clipping?
        next_states = torch.clamp(next_states, self.state_space[0], self.state_space[1])
        if not mf_transition:
            next_states = self.add_noise_to_states(next_states)

        return actions, next_states


    def step(
        self,
        policy: MFPolicy,
        step: Optional[int] = None,
        exploration: Optional[float] = None,
        policy_optimization: bool = False,
        train_matching_model: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu
        act_dim = self.action_dim
        if self.origin_destination_matrix.ndim == 2:
            origin_destination_matrix = self.origin_destination_matrix
        else:
            origin_destination_matrix = self.origin_destination_matrix[step]
        if self.demand_matrix.ndim == 2:
            demand_matrix = self.demand_matrix
        else:
            demand_matrix = self.demand_matrix[step]
        if self.exec_type == 'train':
            if policy_optimization:
                matching_process = self.mf_matching
            else:
                matching_process = self.real_world_matching
            if step == 0:
                self.mf_states = self.perturb_cell_centers(self.cell_centers)
            mf_control_actions, mf_repositioned_states_next = self.get_repositioned_states(
                self.mf_states.clone(), mu, demand_matrix, policy, exploration, mf_transition=True, step=step
            )
            mf_cruising_states_next = self.get_cruising_states(self.mf_states.clone(), mf_transition=True)
            mf_controlled_proportions = mf_control_actions[:, act_dim:act_dim + 1].reshape(1, -1)
            a4 = mf_controlled_proportions.sum()
            mu_available = (1 - mf_controlled_proportions) * mu
            mf_matched_proportions = (
                matching_process
                .get_matched_proportions(mu_available, demand_matrix)
                .reshape(1, -1)
            )
            if isinstance(mf_matched_proportions, np.ndarray):
                mf_matched_proportions = (
                    torch.from_numpy(mf_matched_proportions)
                    .type(mu_available.dtype)
                    .reshape(mf_controlled_proportions.shape)
                    .to(self.device)
                )
            mu_repositioned = mf_controlled_proportions * mu
            # a1 = mf_matched_proportions.sum()
            mu_matched = (1 - mf_controlled_proportions) * mf_matched_proportions * mu
            mu_cruising = (1 - mf_controlled_proportions) * (1 - mf_matched_proportions) * mu
            mu_repositioned_next = self._step_probabilistic(mf_repositioned_states_next, mu_repositioned)
            mu_matched_next = mu_matched @ origin_destination_matrix
            mu_cruising_next = self._step_probabilistic(mf_cruising_states_next, mu_cruising)
            if train_matching_model:
                valid_mf_states = self.mf_states[self.valid_cells_mask].clone()
                mf_matching_probs = (
                        matching_process
                        .get_matching_probabilities(
                            states=valid_mf_states, 
                            matched_proportions=mf_matched_proportions
                        )
                        .reshape(-1, 1)
                    )
                if isinstance(mf_matching_probs, np.ndarray):
                    mf_matching_probs = (
                        torch.from_numpy(mf_matching_probs)
                        .type(mu_available.dtype)
                        .to(self.device)
                    )
                if self.matching_target == 'classification':
                    mf_matched_uniform = torch.rand(size=mf_matching_probs.shape, device=self.device)
                    mask_matched = mf_matched_uniform <= mf_matching_probs
                    mf_matching_probs = mask_matched.float()
        elif self.exec_type == 'eval':
            matching_process = self.real_world_matching
            with torch.no_grad():
                ra_states = self.ra_states
                ra_control_actions, ra_repositioned_states_next = (
                    self.get_repositioned_states(ra_states, mu, demand_matrix, policy, exploration, mf_transition=False, step=step)
                )
                ra_controlled_probs = ra_control_actions[:, act_dim:act_dim + 1].flatten()
                controlled_uniform = torch.rand(size=ra_controlled_probs.shape, device=self.device)
                mask_controlled = controlled_uniform <= ra_controlled_probs
                ra_repositioned_states = ra_states[mask_controlled]
                ra_repositioned_states_next = ra_repositioned_states_next[mask_controlled]
                ra_available_states = ra_states[~mask_controlled]
                requests, vehicles, pickup_distances = (
                    matching_process.simulate(ra_available_states, demand_matrix)
                )
                if isinstance(requests, np.ndarray):
                    requests = (
                        torch.from_numpy(requests)
                        .type(ra_states.dtype)
                        .to(self.device)
                    )
                    vehicles = (
                        torch.from_numpy(vehicles)
                        .type(ra_states.dtype)
                        .to(self.device)
                    )
                    pickup_distances = (
                        torch.from_numpy(pickup_distances)
                        .type(ra_states.dtype)
                        .to(self.device)
                    )
                matched_mask = vehicles[:, 0] == 1
                ra_matched_states = vehicles[matched_mask, -2:]
                ra_cruising_states = vehicles[~matched_mask, -2:]
                ra_matched_states_next = self.get_matched_states(ra_matched_states, origin_destination_matrix)
                ra_cruising_states_next = self.get_cruising_states(ra_cruising_states, mf_transition=False)
                mu_repositioned = self.mf_deterministic_step(ra_repositioned_states)
                mu_matched = self.mf_deterministic_step(ra_matched_states)
                mu_cruising = self.mf_deterministic_step(ra_cruising_states)
                mu_repositioned_next = self.mf_deterministic_step(ra_repositioned_states_next)
                mu_matched_next = self.mf_deterministic_step(ra_matched_states_next)
                mu_cruising_next = self.mf_deterministic_step(ra_cruising_states_next)
                ra_states_next = torch.cat(
                    [ra_repositioned_states_next, ra_matched_states_next, ra_cruising_states_next], 
                    axis=0
                )
                self.ra_states = ra_states_next
        mu_next = mu_repositioned_next + mu_matched_next + mu_cruising_next
        a5 = mu.sum(), mu_repositioned.sum(), mu_matched.sum(), mu_cruising.sum()
        self.mu = mu_next
        lifted_reward = self.lifted_reward(
            mu_repositioned, mu_matched, mu_cruising, 
            mu_repositioned_next, mu_matched_next, mu_cruising_next,
            demand_matrix
        )
        if not train_matching_model:
            valid_mf_states = None
            demand_matrix = None
            mf_matching_probs = None
        if self.exec_type == 'train':
            if policy_optimization:
                return lifted_reward
            else:
                return (
                    valid_mf_states, demand_matrix, mf_matching_probs,
                    mu_repositioned, mu_matched, mu_cruising,
                    mu_repositioned_next, mu_matched_next, mu_cruising_next, 
                    mf_matched_proportions, lifted_reward
                )
        elif self.exec_type == 'eval':
            return (
                    ra_repositioned_states, ra_matched_states, ra_cruising_states,
                    mu_repositioned, mu_matched, mu_cruising,
                    ra_repositioned_states_next, ra_matched_states_next, ra_cruising_states_next,
                    mu_repositioned_next, mu_matched_next, mu_cruising_next, 
                    lifted_reward, requests, vehicles, pickup_distances
                )
            

    def _step_probabilistic(
        self,
        next_states: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        ERF based
        """
        joint_probs = torch.zeros(size=[self.num_cells for _ in range(2)], device=self.device)
        probs = []
        for i in range(self.state_dim):
            next_states_projection = next_states[:, i:i+1]
            probs.append(self._apply_truncated_cdf(next_states_projection))
        cols = 0
        for i in range(self.num_intervals):
            joint_probs[:, cols:cols + self.num_intervals] = probs[0][:, i:i + 1] * probs[1]
            cols += self.num_intervals
        multiplied_probs = (joint_probs.T * mu).sum(dim=-1)
        next_mu = multiplied_probs.reshape(-1, self.num_cells).sum(dim=0)

        return next_mu.reshape(1, -1)


    def _apply_truncated_cdf(
        self, next_states_projection: torch.Tensor
    ) -> torch.Tensor:
        cdf = (
                self.normal_control.cdf(self.upper - next_states_projection) 
                - self.normal_control.cdf(self.lower - next_states_projection)
            )
        w = cdf[:, 1:-1] / cdf[:, 1:-1].sum(dim=1, keepdim=True)
        cdf = cdf[:, 1:-1] + (cdf[:, 0].unsqueeze(1) + cdf[:, -1].unsqueeze(1)) * w
        cdf /= cdf.sum(dim=1, keepdim=True)
        
        return cdf


    def _apply_erf(
            self, lower: torch.Tensor, mean: torch.Tensor, upper: torch.Tensor, noise
        ) -> torch.Tensor:
            upper[0][-1] += 10 ** 4 
            lower[0][0] -= 10 ** 4
            return 0.5 * (
                torch.erf((upper - mean) / (noise * math.sqrt(2)))
                - torch.erf((lower - mean) / (noise * math.sqrt(2)))
            )


    def _apply_cdf(
        self, lower: torch.Tensor, mean: torch.Tensor, upper: torch.Tensor, noise
    ) -> torch.Tensor:
        upper[0][-1] += 10 ** 4 
        lower[0][0] -= 10 ** 4

        return self.normal.cdf(upper - mean) - self.normal.cdf(lower - mean)