from copy import deepcopy
from typing import Any, Mapping, Optional
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import wandb
from safe_mf.envs.env import Env

from safe_mf.models.policy import MFPolicy
from safe_mf.utils.distributions import shifted_uniform

class GradientDescent:
    def __init__(
        self,
        env: Env,
        policy_cfg: Mapping[str, Any] = None,
        num_epochs: int = 1_000,
        non_stationary: bool = False,
        reset_weights_until_episode: int = 0,
        explore_until_episode: int = 0,
        exploration_decay: Optional[float] = None,
        patience: int = 1,
        min_improvement: int = 0.001,
        checkpoint: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.env = env
        self.num_epochs = num_epochs
        self.state_dim = self.env.state_dim
        self.num_cells = self.env.num_cells
        self.action_dim = self.env.action_dim
        self.device = device
        self.state_space = self.env.state_space
        self.action_space = self.env.action_space
        self.exploration_decay = exploration_decay
        self.non_stationary = non_stationary
        self.reset_weights_until_episode = reset_weights_until_episode
        self.explore_until_episode = explore_until_episode
        self.patience = patience
        self.min_improvement = min_improvement
        self.train_counter = 0
        if checkpoint is not None:
            self.policy = torch.load(checkpoint, map_location=self.device)
            self.policy.device = self.device
        else:
            self.policy = MFPolicy(
                self.state_dim,
                self.num_cells,
                self.action_dim,
                policy_cfg["hidden_dims"],
                self.state_space,
                self.action_space,
                self.non_stationary,
                self.device
            ).to(self.device)

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=policy_cfg["lr"],
            weight_decay=policy_cfg["weight_decay"],
        )


    def train(
        self,
        horizon: int,
    ) -> MFPolicy:
        if self.train_counter < self.reset_weights_until_episode:
            self.policy.reset_weights()
        if self.train_counter >= self.explore_until_episode:
            self.exploration_decay = 0.
        best_model = deepcopy(self.policy.state_dict())
        best_loss = -torch.inf
        early_stopper = EarlyStopper(patience=self.patience, min_improvement=self.min_improvement)
        for epoch in tqdm(range(self.num_epochs), desc='Policy optimization'):
            self.policy.eval()
            # self.policy.train()
            self.policy_optimizer.zero_grad()
            self.env.reset()
            rewards = []
            for step in range(horizon): 
                exploration = np.exp(-self.exploration_decay * epoch) if self.exploration_decay else None
                reward = self.env.step(
                    self.policy,
                    step,
                    exploration=exploration,
                    policy_optimization=True
                )
                rewards.append(reward)
            self.policy.train()
            loss = -torch.stack(rewards).sum()
            current_loss = -loss.item()
            loss.backward()
            gradient_norms = []
            # for name, param in self.policy.named_parameters():
            #     if param.grad is not None:
            #         gradient_norm = torch.norm(param.grad, 2).item()
            #         gradient_norms.append(gradient_norm)
            # a = min(gradient_norms), max(gradient_norms)
            # Layerwise normalization
            # for param in self.policy.parameters():
            #     torch.nn.utils.clip_grad_norm_(
            #         [param],max_norm=1.0, 
            #         norm_type=2, error_if_nonfinite=True
            #     )
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=1.0, 
                norm_type=2, error_if_nonfinite=True
                )
            # gradient_norms = []
            # for name, param in self.policy.named_parameters():
            #     if param.grad is not None:
            #         gradient_norm = torch.norm(param.grad, 2).item()
            #         gradient_norms.append(gradient_norm)
            # # b = min(gradient_norms), max(gradient_norms)
            # if not np.isfinite(max(gradient_norms)) or max(gradient_norms) < 1e-6:
            #     print("Vanishing gradients")
            #     sys.exit(1)
            self.policy_optimizer.step()
            # Early stopping
            if (epoch - 1) % 100 == 0:
                if early_stopper.early_stop(current_loss, best_loss):
                    break
            if current_loss >= best_loss:
                best_model = deepcopy(self.policy.state_dict())
                best_loss = current_loss
            if epoch % 1 == 0:
                wandb.log({"training_episode_reward": current_loss})  
        self.policy.load_state_dict(best_model)
        self.train_counter += 1
        self.policy.eval()

        return self.policy


class EarlyStopper:
    def __init__(self, patience=1, min_improvement=0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.early_stop_loss = -torch.inf

    def early_stop(self, loss, best_loss):
        if self.early_stop_loss > 0:
            requirement = self.early_stop_loss * (1 + self.min_improvement)
        else:
            requirement = self.early_stop_loss * (1 - self.min_improvement)
        if loss > requirement:
            self.early_stop_loss = best_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False