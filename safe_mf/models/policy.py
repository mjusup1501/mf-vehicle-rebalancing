import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from safe_mf.utils.distributions import TruncatedNormal, shifted_uniform
from safe_mf.utils.data import (
    denormalize_actions,
    normalize_states,
)
 

class RandomPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_space: Tuple[float, float],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_space = action_space
        self.device = device

    def forward(
        self, 
        states: torch.Tensor, 
        mu: torch.Tensor, 
        demand: torch.Tensor,
        exploration: Optional[float] = None, 
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes the action given the state and the mean field

        Args:
            mu (torch.Tensor): [num_cells]
            exploration (bool): for consistency with MFPolicy

        Returns:
            torch.Tensor: [num_cells * action_dim]
        """
        size = (states.shape[0], self.action_dim)
        low = self.action_space[0]
        high = self.action_space[1]
        repositioning_actions = shifted_uniform(low, high, size, device=self.device)
        size = (states.shape[0], 1)
        proportion_actions = shifted_uniform(0, 1, size, device=self.device)
        actions = torch.cat((repositioning_actions, proportion_actions), dim=1)

        return actions
    

class DummyPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device = device

    def forward(
        self, 
        states: torch.Tensor, 
        mu: torch.Tensor, 
        demand: torch.Tensor,
        exploration: Optional[float] = None, 
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes the action given the state and the mean field

        Args:
            mu (torch.Tensor): [num_cells]
            exploration (bool): for consistency with MFPolicy

        Returns:
            torch.Tensor: [num_cells * action_dim]
        """
        size = (states.shape[0], self.action_dim)
        repositioning_actions = torch.zeros(size, device=self.device)
        size = (states.shape[0], 1)
        proportion_actions = torch.zeros(size, device=self.device)
        actions = torch.cat((repositioning_actions, proportion_actions), dim=1)

        return actions
    

class TanhScaled(nn.Module):
    def __init__(self, scaling_factor):
        super(TanhScaled, self).__init__()
        self.tanh = nn.Tanh()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        tanh_output = self.tanh(x)
        scaled_output = self.scaling_factor * tanh_output
        return scaled_output


class MFPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_cells: int,
        action_dim: int,
        hidden_dims: List[int],
        state_space: Tuple[float, float],
        action_space: Tuple[float, float],
        non_stationary: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        assert len(hidden_dims) > 0
        self.num_cells = num_cells
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_space = state_space
        self.action_space = action_space
        self.non_stationary = non_stationary
        self.device = device
        final_dim = action_dim + 1 # +1 for probability of controller action
        ns_dim = 1 if self.non_stationary else 0
        self.model = [nn.Linear(state_dim + 2 * num_cells + ns_dim, hidden_dims[0])]
        dims = hidden_dims + [final_dim]
        # Tracking running stats negatively impacts training if policy is not set to eval mode
        # We set track_running_stats=False to safeguard against forgetting to set the policy to eval mode
        for i in range(len(dims) - 1): 
            self.model += [nn.LeakyReLU(), nn.BatchNorm1d(dims[i]), nn.Linear(dims[i], dims[i + 1])]
        self.model += [nn.Tanh()] 
        self.model = nn.Sequential(*self.model)
        self.reset_weights()


    def reset_weights(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                # If batch normalization comes before the activation, use i + 2!
                if i + 1 < len(self.model):
                    activation = self.model[i + 1]
                    if isinstance(activation, nn.ReLU):
                        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                        nn.init.zeros_(layer.bias)
                    elif isinstance(activation, nn.LeakyReLU):
                        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                        nn.init.zeros_(layer.bias)
                    elif isinstance(activation, nn.Tanh):
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                        nn.init.zeros_(layer.bias)
                    elif isinstance(activation, nn.Sigmoid):
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
                        nn.init.zeros_(layer.bias)
                else: # Tha last layer is nn.Linear
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)


    def forward(
        self, 
        states: torch.Tensor, 
        mu: torch.Tensor, 
        demand: torch.Tensor,
        exploration: Optional[float] = None, 
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes the actions given the mean field

        Args:
            states (torch.Tensor): [m, state_dim]
            mu (torch.Tensor): [n, num_cells]

        Returns:
            torch.Tensor: [m, act_dim] or [n, act_dim + state_dim]
        """
        if states.shape[0] == mu.shape[0]:
            if self.non_stationary and step is not None:
                steps = torch.tensor(step, device=self.device).expand(states.shape[0], 1)
            else:
                steps = torch.empty(0, device=self.device)
            inputs = torch.cat(
                (
                    normalize_states(states, self.state_space),
                    mu,
                    demand,
                    steps
                ),
                dim=1,
            )
        elif mu.shape[0] == 1:
            if self.non_stationary and step is not None:
                steps = torch.tensor(step, device=self.device).expand(states.shape[0], 1)
            else:
                steps = torch.empty(0, device=self.device)
            inputs = torch.cat(
                (
                    normalize_states(states, self.state_space),
                    mu.expand(states.shape[0], -1),
                    demand.expand(states.shape[0], -1),
                    steps
                ),
                dim=1,
            )
        else:
            norm_states = normalize_states(states, self.state_space)
            norm_states = norm_states.repeat(mu.shape[0], 1, 1)
            norm_mu = mu
            norm_mu = norm_mu.unsqueeze(1).repeat(1, norm_states.shape[1], 1)
            norm_demand = demand
            norm_demand = norm_demand.unsqueeze(1).repeat(1, norm_states.shape[1], 1)
            if self.non_stationary and step is not None:
                steps = torch.tensor(step, device=self.device).expand(1, norm_states.shape[1], 1)
            else:
                steps = torch.empty(0, device=self.device)
            inputs = torch.cat((norm_states, norm_mu, norm_demand, steps), dim=2)
            inputs = inputs.reshape(-1, inputs.shape[2])
        outputs = self.model(inputs)
        # Scale [-1, 1] to [-action_space, action_space]
        # Assumes symmetric action space around 0
        act_dim = self.action_dim
        repositioning_actions = outputs[:, :act_dim]
        repositioning_actions = denormalize_actions(repositioning_actions, self.action_space)
        proportion_actions = outputs[:, act_dim: act_dim+1]
        proportion_actions = denormalize_actions(proportion_actions, (0, 1))
        actions = torch.cat((repositioning_actions, proportion_actions), dim=1)

        if exploration:
            zeros = torch.zeros_like(actions)
            std = torch.ones_like(actions) * exploration
            lower_bound_repositioning = self.action_space[0] * torch.ones_like(repositioning_actions)
            upper_bound_repositioning = self.action_space[1] * torch.ones_like(repositioning_actions)
            lower_bound_proportion = torch.zeros_like(proportion_actions)
            upper_bound_proportion = torch.ones_like(proportion_actions)
            lower_bound = torch.cat((lower_bound_repositioning, lower_bound_proportion), dim=1)
            lower_bound = lower_bound - actions
            upper_bound = torch.cat((upper_bound_repositioning, upper_bound_proportion), dim=1)
            upper_bound = upper_bound - actions
            tnorm = TruncatedNormal(loc=zeros, scale=std, a=lower_bound, b=upper_bound)
            noise = tnorm.sample()
            actions = actions + noise
        
        return actions