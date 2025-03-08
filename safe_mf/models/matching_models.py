from copy import deepcopy
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from tqdm import tqdm

from safe_mf.utils.utils import states_to_cell, cells_to_index, get_neighboring_cells
from safe_mf.utils.data import concat_inputs, MatchingDataset



class CellLevelApproximation(object):
	def __init__(
			self, 
			linspace_x, 
			linspace_y, 
			num_intervals, 
			weight_matrix: Optional[torch.Tensor] = None,
			device: torch.device = torch.device("cpu"),
	):
		self.weight_matrix = weight_matrix
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.num_intervals = num_intervals
		self.device = device


	def get_matching_probabilities(
			self, 
			states: torch.Tensor,
			mu_available: torch.Tensor, 
			rho: torch.Tensor,
	) -> torch.Tensor:
		matched_proportions = self.get_matched_proportions(mu_available, rho)
		state_cells = states_to_cell(states, self.linspace_x, self.linspace_y)
		states_idx = cells_to_index(state_cells, self.num_intervals)
		if matched_proportions.shape[0] == 1:
			matching_probabilities = matched_proportions[:, states_idx]
		else:
			matching_probabilities = matched_proportions[states_idx]

		return matching_probabilities


	def get_matched_proportions(
			self, 
			mu_available: torch.Tensor, 
			rho: torch.Tensor
	) -> torch.Tensor:
		one = torch.tensor(1, device=self.device)
		mu_ = torch.log(mu_available + 1e-10)
		demand_ = torch.log(rho + 1e-10)
		matched_proportions = torch.min(one, torch.exp(demand_ - mu_))
		# Necessary logic to avoid 0/0 = 1
		mask = mu_available > 0
		matched_proportions = matched_proportions * mask
		if self.weight_matrix is not None:
			matched_proportions = matched_proportions * self.weight_matrix

		return matched_proportions
      

class OptimalTransport(object):
	def __init__(
			self,
			num_cells,
			num_intervals,
			linspace_x,
			linspace_y,
			num_matching_cell_neighbors=1,
			cruising_cost=2, 
			probs_scaling_exp=1e15,
			cost_scaling_exp=1e5,
			weight_matrix: Optional[torch.Tensor] = None,
	):
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x.cpu().numpy()
		self.linspace_y = linspace_y.cpu().numpy()
		self.matching_radius = num_matching_cell_neighbors
		self.cruising_cost = cruising_cost
		self.probs_scaling_exp = probs_scaling_exp
		self.cost_scaling_exp = cost_scaling_exp
		if isinstance(weight_matrix, torch.Tensor):
			weight_matrix = weight_matrix.detach().cpu().numpy()
		self.weight_matrix = weight_matrix


	def get_matching_probabilities(
			self, 
			states: torch.Tensor,
			mu_available: Optional[torch.Tensor] = None, 
			rho: Optional[torch.Tensor] = None,
			matched_proportions: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		if isinstance(mu_available, torch.Tensor):
			mu_available = mu_available.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		if mu_available is not None:
			if isinstance(states, torch.Tensor):
				states = states.detach().cpu().numpy()
		if isinstance(states, np.ndarray) and matched_proportions is not None:
			matched_proportions = matched_proportions.detach().cpu().numpy()
		if matched_proportions is None:
			matched_proportions = self.get_matched_proportions(mu_available, rho)
		state_cells = states_to_cell(states, self.linspace_x, self.linspace_y)
		states_idx = cells_to_index(state_cells, self.num_intervals)
		if matched_proportions.shape[0] == 1:
			matching_probabilities = matched_proportions[:, states_idx]
		else:
			matching_probabilities = matched_proportions[states_idx]

		return matching_probabilities


	def get_matched_proportions(
			self, 
			mu_available: torch.Tensor, 
			rho: torch.Tensor
	) -> torch.Tensor:
		if isinstance(mu_available, torch.Tensor):
			mu_available = mu_available.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		mu_available = mu_available.flatten()
		rho = rho.flatten()
		graph = self.create_2_norm_neighbor_graph(mu_available, rho)
		# flow_cost, flow_dict = nx.network_simplex(graph)
		flow_dict = nx.min_cost_flow(graph)
		matched_proportions = self._compute_matched_proportions_from_flowdict(flow_dict)
		if self.weight_matrix is not None:
			matched_proportions = matched_proportions * self.weight_matrix

		return matched_proportions


	def _compute_matched_proportions_from_flowdict(self, flow_dict):
		matched_proportions = np.ndarray(self.num_cells)
		for i in range(self.num_intervals):
			for j in range(self.num_intervals):
				cell_idx = cells_to_index((i,j), self.num_intervals)
				supply = flow_dict['source'][f'mu_{i}_{j}']
				cruising = flow_dict[f'mu_{i}_{j}']['sink']
				if supply == 0:
					matched_proportions[cell_idx] = 0
				else:
					matched_proportions[cell_idx] = 1 - cruising / supply

		return matched_proportions


	def _scale_up(self, probs):
		total_mass = probs.sum()
		probs = (probs * self.probs_scaling_exp).astype(int)
		error = int(self.probs_scaling_exp * total_mass) - probs.sum()
		# avoid negative error which could be cause by rounding errors
		error = np.maximum(0, error)
		probs[-1] += error
		probs = probs.astype(int)
            
		return probs
	

	def create_2_norm_neighbor_graph(self, mu, rho):
		# scale up mu and rho
		mu = self._scale_up(mu)
		rho = self._scale_up(rho)
		# compute residual
		residual = abs(int(mu.sum() - rho.sum()))
		# avoid negative residual which could be caused by rounding errors
		# residual = np.maximum(0, residual)

		graph = nx.DiGraph()
		if mu.sum() <= rho.sum():
			# Source and sink nodes
			graph.add_node("source", demand=-rho.sum())
			graph.add_node("sink", demand=rho.sum())
			# edge for residual flows
			graph.add_edge(
				"source",
				"sink", 
				weight=-1,
				capacity=residual
			)
		elif mu.sum() > rho.sum():
			graph.add_node("source", demand=-mu.sum())
			graph.add_node("sink", demand=rho.sum())
			graph.add_node("dummy_sink", demand=residual)
			graph.add_edge(
				"source",
				"dummy_sink", 
				weight=-1,
				capacity=residual
			) 
		for i in range(self.num_intervals):
			for j in range(self.num_intervals):
				graph.add_node(f"mu_{i}_{j}")
				graph.add_node(f"rho_{i}_{j}")
				cell_idx = cells_to_index((i,j), self.num_intervals)
				graph.add_edge(
					"source", 
					f'mu_{i}_{j}',
					weight=0,
					capacity=mu[cell_idx]
				)
				graph.add_edge(
					f'rho_{i}_{j}',
					"sink",
					weight=0,
					capacity=rho[cell_idx]
				)
		for i in range(self.num_intervals):
			for j in range(self.num_intervals):
				start = f"mu_{i}_{j}"
				neighbors = get_neighboring_cells((i, j), self.matching_radius, self.num_intervals)
				for (k,l) in neighbors:
					end = f"rho_{k}_{l}"
					dist = np.sqrt((i - k) ** 2 + (j - l) ** 2)
					# For 2 norm the cost is also scaled up
					graph.add_edge(
						start,
						end,
						weight=int(dist * self.cost_scaling_exp),
						capacity=np.inf
					)
		for i in range(self.num_intervals):
			for j in range(self.num_intervals):
				# For 2 norm the cost is also scaled up
				graph.add_edge(
					f'mu_{i}_{j}',
					'sink',
					weight=int(self.cruising_cost * self.cost_scaling_exp),
					capacity=np.inf
				)

		return graph




