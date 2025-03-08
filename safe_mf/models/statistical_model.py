from copy import deepcopy
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from safe_mf.utils.utils import states_to_cell, cells_to_index
from safe_mf.utils.data import concat_inputs, MatchingDataset


class CustomClampLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(CustomClampLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        clamped_values = torch.clamp(x, self.min_val, self.max_val)
        return clamped_values


def custom_clamp_hook(module, grad_input, grad_output):
    # During backward pass, use the original unclamped values
    return grad_input
	

class NALU(nn.Module):
    def __init__(self, in_dim, out_dim, e=1e-5):
        super(NALU, self).__init__()
        self.e = e
        self.G = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.W = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('Gbias', None)
        self.register_parameter('Wbias', None)
        self.nac = NeuralAccumulator(in_dim, out_dim)
        nn.init.xavier_uniform_(self.G)
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        a = self.nac(x)
        g = torch.sigmoid(nn.functional.linear(x, self.G, self.Gbias))
        m = torch.exp(nn.functional.linear(torch.log(torch.abs(x) + self.e), self.W, self.Wbias))
        out = g*a + (1-g)*m
        return out
	

class NeuralAccumulator(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(NeuralAccumulator, self).__init__()
		self.W1 = nn.Parameter(torch.Tensor(out_dim, in_dim))
		self.W2 = nn.Parameter(torch.Tensor(out_dim, in_dim))
		# self.register_parameter('bias', None)
		self.W = nn.Parameter(torch.tanh(self.W1) * torch.sigmoid(self.W2))
		self.model = nn.Linear(in_dim, out_dim, bias=False)
		self.reset_weights()
        

	def reset_weights(self) -> None:
		nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('tanh'))
		nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('sigmoid'))
		self.model.weight = self.W


	def forward(self, x):
		# out = nn.functional.linear(x, self.W, self.bias)
		return self.model(x)


def nn_initialization(model):
	for i, layer in enumerate(model):
		if isinstance(layer, nn.Linear):
			# If batch normalization comes before the activation, use i + 2!
			if i + 1 < len(model):
				activation = model[i + 1]
			else:
				activation = None
			if isinstance(activation, nn.ReLU):
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
			elif isinstance(activation, nn.LeakyReLU):
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(activation, nn.PReLU):
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in')
			elif isinstance(activation, nn.Tanh):
				nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
			elif isinstance(activation, nn.Sigmoid):
				nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
			else: # Tha last layer is nn.Linear
				nn.init.xavier_uniform_(layer.weight)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)
		elif isinstance(layer, nn.BatchNorm1d):
			nn.init.ones_(layer.weight)
			nn.init.zeros_(layer.bias)
		# elif isinstance(layer, GATv2Conv):
		# 	layer.reset_parameters()
	
	
class NeuralCellLevelApproximation(nn.Module):
	def __init__(self, in_dim, out_dim, hidden_dims: List[int] = [1], eps=1e-10):
		super(NeuralCellLevelApproximation, self).__init__()
		self.eps = eps
		self.G = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.Sigmoid())
		self.nac = NeuralAccumulator(in_dim, out_dim)
		self.M = []
		dims = [in_dim] + hidden_dims
		for i in range(len(dims) - 1):
			self.M += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
		self.M += [nn.Linear(hidden_dims[-1], out_dim)]
		self.M = nn.Sequential(*self.M)
		self.clamp = CustomClampLayer(min_val=0., max_val=1.)
		self.clamp.requires_grad_(False)
		self.reset_weights()


	def reset_weights(self) -> None:
		self.nac.reset_weights()
		nn_initialization(self.M)
		nn_initialization(self.G)

	def forward(self, x):
		# gate
		g = self.G(x)
		# division
		d = self.nac(torch.log(x + self.eps))
		d = torch.exp(d)
		# min
		m = self.M(x)
		out = g*d + (1-g)*m
		return self.clamp(out)
	

class MSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))
	

class NeuralNetwork(nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dims: List[int],
		out_dim: int = 1,
		target_type: Optional[str] = 'regression',
		min_out: Optional[float] = 0.,
		max_out: Optional[float] = 1.
	) -> None:
		super().__init__()
		dims = [in_dim] + hidden_dims
		self.model = []
		for i in range(len(dims) - 1):
			self.model += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(), nn.BatchNorm1d(dims[i + 1])]
		self.model += [nn.Linear(hidden_dims[-1], out_dim)]
		if target_type == 'regression':
			self.model += [CustomClampLayer(min_val=min_out, max_val=max_out)]
			self.model[-1].requires_grad_(False)
		elif target_type == 'classification':
			self.model += [nn.Sigmoid()]
		self.model = nn.Sequential(*self.model)
		self.reset_weights()


	def reset_weights(self) -> None:
		nn_initialization(self.model)


	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.model(inputs)
	

class StatisticalMatching(nn.Module):
	def __init__(
			self, 
			state_dim, 
			num_cells, 
			linspace_x,
			linspace_y,
			input_type: str = "grid",
			target_type: str = "regression",
			num_nets: int = 1, 
			hidden_dims: List[int] = [16, 16],
			num_epochs: int = 100,
			buffer_size = 1_000,
			batch_size: int = 64,
			holdout: float = 0.9,
			lr: float = 0.005, 
			weight_decay: float = 0.0005,
			dim_reduction: Optional[str] = None, 
			reset_weights_until_episode: int = 0,
			patience: int = 30,
			min_improvement: float = 0.005,
			weight_matrix: Optional[torch.Tensor] = None,
			device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.state_dim = state_dim
		self.num_cells = num_cells
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.input_type = input_type
		self.target_type = target_type
		self.num_nets = num_nets
		self.hidden_dims = hidden_dims
		self.num_epochs = num_epochs
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.holdout = holdout
		self.lr = lr
		self.weight_decay = weight_decay
		self.dim_reduction = dim_reduction
		self.reset_weights_until_episode = reset_weights_until_episode
		self.patience = patience
		self.min_improvement = min_improvement
		self.weight_matrix = weight_matrix.flatten()
		self.device = device
		self.num_intervals = int(math.sqrt(self.num_cells))
		self.train_counter = 0
		with torch.no_grad():
			self.cell_centers_1d = (
				torch.arange(0, self.num_intervals, 1, device=self.device)
				.reshape(1, -1)
				/ self.num_intervals
			) + 0.5 / self.num_intervals

			self.cell_centers = torch.cartesian_prod(*[self.cell_centers_1d.squeeze(0)] * self.state_dim)
		if self.input_type == "grid":
			in_dim=2 * self.num_cells + 1
		elif self.input_type == "cell":
			in_dim=2
		elif self.input_type == "cell_with_index":
			in_dim=3
		out_dim=1
		min_out=0. 
		max_out=1.
		self.models = nn.ModuleList(
			[
				NeuralNetwork(
					in_dim=in_dim,
					out_dim=out_dim, 
					hidden_dims=hidden_dims, 
					target_type=self.target_type, 
					min_out=min_out, 
					max_out=max_out
				) 
				for _ in range(self.num_nets)
			]
		).to(device)

		(
            self.saved_states,
            self.saved_mus,
            self.saved_rhos,
            self.saved_matching_probs,
        ) = ([], [], [], [])

		self.optimizers = [
			torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
			for model in self.models
		]


	def reset_weights(self) -> None:
		for model in self.models:
			model.reset_weights()
	
	
	def get_matching_probabilities(self, 
			states: torch.Tensor,
			mu_available: torch.Tensor, 
			rho: torch.Tensor,
	) -> torch.Tensor:
		return self.forward(states, mu_available, rho)


	def get_matched_proportions(self, 
			mu_available: torch.Tensor, 
			rho: torch.Tensor
	) -> torch.Tensor:
		# Should we use perturbed cell centers? Or maybe even average over a huge sample?
		return self.forward(self.cell_centers, mu_available, rho)


	def forward(self, 
			states: torch.Tensor,
			mu_available: torch.Tensor, 
			rho: torch.Tensor,
	) -> torch.Tensor:
		state_cells = states_to_cell(states, self.linspace_x, self.linspace_y)
		state_idx = cells_to_index(state_cells, self.num_intervals).flatten()
		if "cell" in self.input_type:
			mu_available = mu_available[torch.arange(mu_available.size(0)), state_idx].reshape(-1, 1)
			rho = rho[torch.arange(rho.size(0)), state_idx].reshape(-1, 1)
		inputs = self.preprocess_inputs(state_idx.reshape(-1, 1), mu_available, rho, self.dim_reduction) 
		outputs = []
		for model in self.models:
			output = model(inputs)
			if self.weight_matrix is not None:
				non_zero_rho = self.weight_matrix[state_idx].reshape(-1, 1)
				output = output * non_zero_rho
			outputs.append(output)
		outputs = torch.cat(outputs, dim=1).mean(dim=1).reshape(-1, 1)

		return outputs
	

	def preprocess_inputs(
            self, state_idx: torch.Tensor, mu: torch.Tensor, 
            rho: torch.Tensor, dim_reduction: Optional[str] = None
        ) -> torch.Tensor:
		# Placeholder for dim reduction
		if dim_reduction is not None:
			pass 
		if self.input_type == "grid":
			inputs = concat_inputs(state_idx, mu, rho)
		elif self.input_type == "cell":
			inputs = torch.cat([mu, rho], dim=1)
		elif self.input_type == "cell_with_index":
			inputs = concat_inputs(state_idx, mu, rho)

		return inputs
	
	
	def train(
        self,
        states: torch.Tensor,
        mus: torch.Tensor,
        rhos: torch.Tensor,
        matching_probs: torch.Tensor,
    ):
		if self.train_counter < self.reset_weights_until_episode:
			self.reset_weights()
		# State cells are enough given the design of matching models and simulators
		state_cells = states_to_cell(states, self.linspace_x, self.linspace_y)
		state_idx = cells_to_index(state_cells, self.num_intervals).flatten()
		if "cell" in self.input_type:
			mus = mus[torch.arange(mus.size(0)), state_idx].reshape(-1, 1)
			rhos = rhos[torch.arange(rhos.size(0)), state_idx].reshape(-1, 1)
		self.saved_states.append(state_idx.reshape(-1, 1))
		self.saved_mus.append(mus)
		self.saved_rhos.append(rhos)
		matching_probs = matching_probs.reshape(-1, 1)
		self.saved_matching_probs.append(matching_probs)
		if len(self.saved_states) > self.buffer_size:
			self.saved_states.pop(0)
			self.saved_mus.pop(0)
			self.saved_rhos.pop(0)
			self.saved_matching_probs.pop(0)
		num_samples = len(self.saved_states) * len(self.saved_states[0]) 
		if self.batch_size <= 8 and num_samples >= 1_500:
			self.batch_size = 16
		elif self.batch_size <= 16 and num_samples >= 3_000:
			self.batch_size = 32
		elif self.batch_size <= 32 and num_samples >= 6_000:
			self.batch_size = 64
		elif self.batch_size <= 64 and num_samples >= 12_000:
			self.batch_size = 128
		elif self.batch_size <= 128 and num_samples >= 24_000:
			self.batch_size = 256
		elif self.batch_size <= 256 and num_samples >= 48_000:
			self.batch_size = 512

		states = torch.cat(self.saved_states, dim=0)
		mus = torch.cat(self.saved_mus, dim=0)
		rhos = torch.cat(self.saved_rhos, dim=0)
		matching_probs = torch.cat(self.saved_matching_probs, dim=0)
		inputs = self.preprocess_inputs(states, mus, rhos, self.dim_reduction)
		# In the previous paper reduction="sum"
		if self.target_type == "regression":
			# loss = nn.MSELoss(reduction="mean")
			loss = MSLELoss(reduction="mean")
		elif self.target_type == "classification":
			loss = nn.BCELoss(reduction="mean")

		split_point = int(self.holdout * len(inputs))
		for model, opt in zip(self.models, self.optimizers):
			idx = torch.randperm(len(inputs))
			bidx = torch.poisson(torch.ones(split_point)).to(int).to(self.device)
			train_loader = DataLoader(
				MatchingDataset(
					torch.repeat_interleave(inputs[idx][:split_point], bidx, dim=0),
					torch.repeat_interleave(matching_probs[idx][:split_point], bidx, dim=0),
				),
				shuffle=True,
				batch_size=self.batch_size,
				drop_last=True
			)
			val_loader = DataLoader(
				MatchingDataset(
					inputs[idx][split_point:], matching_probs[idx][split_point:]
				),
				batch_size=self.batch_size,
				drop_last=True
			)
			best_val_loss = torch.inf
			best_train_loss = torch.inf
			best_model = deepcopy(model.state_dict())
			early_stopper = EarlyStopper(self.patience, self.min_improvement)
			for t in tqdm(range(self.num_epochs), desc='Statistical model optimization'):
				train_loss = 0.0
				model.train()
				for batch_inputs, batch_targets in train_loader:
					batch_outputs = model(batch_inputs)
					batch_loss = loss(
						batch_outputs.flatten(),
						batch_targets.flatten(),
					)
					# batch_loss = torch.tensor(batch_loss, requires_grad=True)
					# Regularization penalty for min_logvar and max_logvar
					opt.zero_grad()
					batch_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
					opt.step()
					train_loss += batch_loss.item()
				val_loss = 0.0
				model.eval()
				for batch_inputs, batch_targets in val_loader:
					with torch.no_grad():
						batch_outputs = model(batch_inputs)
						batch_loss = loss(
							batch_outputs.flatten(),
							batch_targets.flatten(),
						)
						val_loss += batch_loss.item()
				if early_stopper.early_stop(val_loss):             
					break
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_model = deepcopy(model.state_dict())
			model.load_state_dict(best_model)
		self.train_counter += 1


class EarlyStopper:
    def __init__(self, patience=1, min_improvement=0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.min_val_loss = torch.inf

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss * (1 - self.min_improvement):
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False