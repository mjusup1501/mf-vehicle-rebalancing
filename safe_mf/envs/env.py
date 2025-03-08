from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
import torch
from safe_mf.models.policy import MFPolicy


class Env(ABC):
    def __init__(
        self,
        state_dim: int,
        num_cells: int,
        action_dim: int,
        device=torch.device,
    ) -> None:
        self.state_dim = state_dim
        self.num_cells = num_cells
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, 
        policy: MFPolicy, 
        step: Optional[int] = None, 
        statistical_matching: bool = False,
        policy_optimization: bool = False
    ) -> Tuple[torch.Tensor, float]:
        raise NotImplementedError
