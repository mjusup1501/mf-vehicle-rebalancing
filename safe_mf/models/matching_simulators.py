import math
from typing import Any
import torch
import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from safe_mf.utils.utils import index_to_cell, cells_to_index, states_to_cell
from typing import Tuple, List, Optional
from tqdm import tqdm


class GlobalMatchingProcess:
	def __init__(
			self, 
			num_vehicles, 
			num_rounds,
			num_cells, 
			num_intervals,
			control_std,  
			linspace_x, 
			linspace_y,
			weight_matrix=None,
			num_matching_cell_neighbors=0, 
			waiting_rounds=1
		):
		'''
        arguments:
            num_vehicles: number of vehicles
            num_passengers: number of passengers
            num_cells: number of cells used to discretize the space
            num_matching_cell_neighbors: radius measured in number of cell half-diagonals
        '''
		self.num_vehicles = num_vehicles
		self.num_rounds = num_rounds
		self.waiting_rounds = waiting_rounds
		self.control_std = control_std
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.weight_matrix = weight_matrix
		if isinstance(linspace_x, torch.Tensor):
			self.linspace_x = linspace_x.detach().cpu().numpy()
		if isinstance(linspace_y, torch.Tensor):
			self.linspace_y = linspace_y.detach().cpu().numpy()
		if isinstance(weight_matrix, torch.Tensor):
			self.weight_matrix = weight_matrix.flatten().detach().cpu().numpy()
		if torch.cuda.is_available():
			self.linspace_x_cuda = torch.tensor(linspace_x).cuda()
			self.linspace_y_cuda = torch.tensor(linspace_y).cuda()
		else:
			self.linspace_x_cuda = linspace_x
			self.linspace_y_cuda = linspace_y
		diameter = 1 / self.num_intervals
		# Multiply by the number of cells in a row/column
		num_neighbors = 1 + 2 * num_matching_cell_neighbors
		self.matching_radius = (diameter * num_neighbors) / 2

	
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
			linspace_x = self.linspace_x
			linspace_y = self.linspace_y
		elif isinstance(states, torch.Tensor):
			linspace_x = self.linspace_x_cuda
			linspace_y = self.linspace_y_cuda
		if matched_proportions is None:
			matched_proportions = self.get_matched_proportions(mu_available, rho)
		state_cells = states_to_cell(states, linspace_x, linspace_y)
		states_idx = cells_to_index(state_cells, self.num_intervals)
		if matched_proportions.shape[0] == 1:
			matching_probabilities = matched_proportions[:, states_idx]
		else:
			matching_probabilities = matched_proportions[states_idx]

		return matching_probabilities


	def _simulate_matching(self, requests_simulator, vehicles_simulator):
		pickup_distances = []
		for round in range(self.num_rounds):
			# print(f'Round {round}')
			requests = requests_simulator.get_pending_requests(round)
			vehicles = vehicles_simulator.get_available_vehicles()
			# return self.matching(requests, vehicles)
			matched_request_ids, matched_vehicle_ids, pickup_distance = (
				self.matching(requests, vehicles)
			)
			requests_simulator.update_pending_requests(matched_request_ids, round)
			vehicles_simulator.update_available_vehicles(matched_vehicle_ids, round)
			pickup_distances.append(pickup_distance)
		matched_proportions = vehicles_simulator.get_matched_proportions()
		if self.weight_matrix is not None:
			matched_proportions = matched_proportions * self.weight_matrix
		requests = requests_simulator.matched
		vehicles = vehicles_simulator.vehicles[:, -4:]
		pickup_distances = np.concatenate(pickup_distances)

		return matched_proportions, requests, vehicles, pickup_distances
	
	
	def simulate(self, ra_states, rho):
		if isinstance(ra_states, torch.Tensor):
			ra_states = ra_states.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		rho = rho.flatten()
		num_requests = int(rho.sum() * self.num_vehicles)
		cruising_std = self.control_std / math.sqrt(self.num_rounds)
		requests_simulator = GlobalRequestSimulator(
			rho, num_requests, self.num_rounds, self.num_cells, self.num_intervals, 
			self.linspace_x, self.linspace_y, self.waiting_rounds
		)
		vehicles_simulator = GlobalVehicleSimulator(
			self.num_cells, self.num_intervals, self.linspace_x, self.linspace_y, cruising_std,
			ra_states=ra_states
		)
		_, requests, vehicles, pickup_distances = self._simulate_matching(requests_simulator, vehicles_simulator)

		return requests, vehicles, pickup_distances


	def get_matched_proportions(self, mu_available, rho):
		'''
		Simulate matching process

		arguments:
			mu_available: distribution of available vehicles (doesn't need to sum to 1)
			rho: distribution of requests (must sum to 1)
			num_rounds: num of matching rounds

		returns:
			m: matching rate vector – for each cell in [0,1]
		'''
		if isinstance(mu_available, torch.Tensor):
			mu_available = mu_available.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		mu_available = mu_available.flatten()
		rho = rho.flatten()
		num_requests = int(rho.sum() * self.num_vehicles)
		cruising_std = self.control_std / math.sqrt(self.num_rounds)
		requests_simulator = GlobalRequestSimulator(
			rho, num_requests, self.num_rounds, self.num_cells, self.num_intervals, 
			self.linspace_x, self.linspace_y, self.waiting_rounds
		)
		vehicles_simulator = GlobalVehicleSimulator(
			self.num_cells, self.num_intervals, self.linspace_x, self.linspace_y, cruising_std,
			mu_available=mu_available, num_vehicles=self.num_vehicles, 
		)
		matched_proportions, _, _, _ = self._simulate_matching(requests_simulator, vehicles_simulator)
		
		return matched_proportions


	def matching(self, requests, vehicles):
		'''
		Simulate one matching round

		arguments:
			requests: array of requests (request_id, x, y)
			vehicles: array of vehicles (vehicle_id, x, y)

		returns:
			request_ids: array of ids of matched requests
			vehicle_ids: array of ids of matched vehicles
			pickup_distances: array of pickup distance pickup distances
		'''
		request_ids = requests[:, 0].astype(int)
		vehicle_ids = vehicles[:, 0].astype(int)
		request_coordinates = requests[:, -2:]
		vehicle_coordinates = vehicles[:, -2:]
		distance_matrix = cdist(request_coordinates, vehicle_coordinates)
		# Trick to avoid infeasible matching
		threshold = 10
		distance_matrix[distance_matrix > self.matching_radius] = threshold
		try:
			matched_requests, matched_vehicles = linear_sum_assignment(distance_matrix)
			pickup_distances = distance_matrix[matched_requests, matched_vehicles]
			# Trick to avoid infeasible matching – keep only those within the radius
			valid_matches = pickup_distances != threshold
			matched_requests = matched_requests[valid_matches]
			matched_vehicles = matched_vehicles[valid_matches]
			pickup_distances = pickup_distances[valid_matches]
			matched_request_ids = request_ids[matched_requests]
			matched_vehicle_ids = vehicle_ids[matched_vehicles]
			valid_request_coordinates = request_coordinates[matched_requests]
			valid_vehicle_coordinates = vehicle_coordinates[matched_vehicles]
			pickup_distances = np.column_stack((pickup_distances, valid_request_coordinates, valid_vehicle_coordinates))

			return matched_request_ids, matched_vehicle_ids, pickup_distances
		except ValueError: 
			# If the assignment is infeasible, return empty arrays to represents no matches
			# If we had some iterative matching, we could return partial matches which 
			# are not supported by LP-solvers
			return np.array([], dtype=bool), np.array([], dtype=bool), np.array([])


class GlobalRequestSimulator:
	def __init__(self, rho, num_requests, num_rounds, num_cells, num_intervals, linspace_x, linspace_y, waiting_rounds=1):
		self.rho = rho
		self.num_requests = num_requests
		self.num_rounds = num_rounds
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.waiting_rounds = waiting_rounds
		self.pending = np.ones(self.num_requests, dtype=bool)
		self.init_requests(self.rho)
		

	def init_requests(self, rho):
		'''
		Initialize location of requests
		'''
		request_ids = np.arange(self.num_requests)
		# We need to normalize rho to make it a distribution
		rho = rho / rho.sum()
		# Determine a cell for each request given the demand distribution rho
		request_cells = np.random.choice(self.num_cells, size=self.num_requests, p=rho)
		# Reshape passenger cell idx to (x,y) cell idx
		cells_x, cells_y = index_to_cell(request_cells, self.num_intervals)
		# Uniformly generate (x,y) coordinates within cells
		x = np.random.uniform(self.linspace_x[cells_x], self.linspace_x[cells_x + 1])
		y = np.random.uniform(self.linspace_y[cells_y], self.linspace_y[cells_y + 1])
		# Generate request times
		request_round = np.random.randint(self.num_rounds, size=self.num_requests)  
		self.requests = np.column_stack((request_ids, request_round, x, y))
		matched_round = -1 * np.ones(self.num_requests)
		self.matched = np.zeros(self.num_requests)
		self.matched = np.column_stack((self.matched, request_round, matched_round, x, y))


	def get_pending_requests(self, round):
		'''
		Generate arrival requests
		'''
		self.waiting_upper_bound_mask = round <= self.requests[:, 1] + self.waiting_rounds - 1
		rounds_mask = (
			self.waiting_upper_bound_mask &
			(self.requests[:, 1] <= round)
		)
		requests = self.requests[self.pending & rounds_mask]

		return requests
	

	def update_pending_requests(self, matched_ids, round):
		'''
		Update requests status


		arguments:
			matched_ids: index of matched vehicles in available vehicles
		'''
		# update availability
		self.pending[matched_ids] = False
		self.pending[~self.waiting_upper_bound_mask] = False
		self.matched[matched_ids, 0] = True
		self.matched[matched_ids, 2] = round


class GlobalVehicleSimulator:
	def __init__(self, num_cells, num_intervals, linspace_x, linspace_y, cruising_std=1, mu_available=None, ra_states=None, num_vehicles=None):
		# Mask that keeps track of available vehicles
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.cruising_std = cruising_std
		# Compute the number of available vehicles
		# It's just a proportion of the total number of vehicles
		# because mu_available might not sum up to 1
		if mu_available is not None:
			prob_mass = mu_available.sum()
			num_available = int(num_vehicles * prob_mass)
			# We need to normalize mu_available to make it a distribution
			# We are applying it to the available vehicles only to keep the scale intact
			mu_available = mu_available / prob_mass
			self.mf_init_vehicles(mu_available, num_available)
		if ra_states is not None:
			num_available = ra_states.shape[0]
			self.ra_init_vehicles(ra_states, num_available)
		

	def mf_init_vehicles(self, mu_available, num_available):
		'''
		Initialize vehicles
		'''
		vehicle_ids = np.arange(num_available)
		matched = np.zeros(num_available)
		matched_round = -1 * np.ones(num_available)
		# Determine a cell for each vehicle given the distribution rho
		# Remember the initial cells for computing matched_proportions
		self.vehicle_cells = np.random.choice(self.num_cells, size=num_available, p=mu_available)
		# Reshape vehicle cell idx to (x,y) cell idx
		self.cells_x, self.cells_y = index_to_cell(self.vehicle_cells, self.num_intervals)
		# Uniformly generate (x,y) coordinates within cells
		x = np.random.uniform(self.linspace_x[self.cells_x], self.linspace_x[self.cells_x + 1])
		y = np.random.uniform(self.linspace_y[self.cells_y], self.linspace_y[self.cells_y + 1])
		self.vehicles = np.column_stack((vehicle_ids, matched, matched_round, x, y))


	def ra_init_vehicles(self, ra_states, num_available):
		'''
		Initialize vehicles
		'''
		vehicle_ids = np.arange(num_available)
		matched = np.zeros(num_available)
		matched_round = -1 * np.ones(num_available)
		# Determine a cell for each vehicle
		# Remember the initial cells for computing matched_proportions
		self.vehicle_cells = states_to_cell(ra_states, self.linspace_x, self.linspace_y)
		self.vehicle_cells = cells_to_index(self.vehicle_cells, self.num_intervals)
		# Reshape vehicle cell idx to (x,y) cell idx
		self.cells_x, self.cells_y = index_to_cell(self.vehicle_cells, self.num_intervals)
		# Uniformly generate (x,y) coordinates within cells
		x = ra_states[:, 0]
		y = ra_states[:, 1]
		self.vehicles = np.column_stack((vehicle_ids, matched, matched_round, x, y))


	def get_matched_proportions(self):
		'''
		Compute matched proportions for each cell
		'''
		matched = self.get_matched_mask()
		data = np.column_stack((self.vehicle_cells, matched))
		matched_proportions = np.array(
			[data[data[:, 0] == i][:, 1].mean() 
			for i in range(self.num_cells)]
		)
		matched_proportions = np.nan_to_num(matched_proportions, nan=0)
		
		return matched_proportions

	def get_matched_mask(self):
		'''
		Generate available vehicles mask
		'''
		return self.vehicles[:, 1].flatten().astype(bool)


	def get_available_vehicles(self):
		'''
		Generate available vehicles 
		'''
		available = ~self.get_matched_mask()
		return self.vehicles[available]


	def update_available_vehicles(self, matched_ids, round):
		'''
		Update vehicle availability (and locations)


		arguments:
			matched_ids: index of matched vehicles in available vehicles
		'''
		# update availability
		self.vehicles[matched_ids, 1] = True
		self.vehicles[matched_ids, 2] = round
		# We should clip the coordinates to avoid vehicles leaving the cell 
		available = ~self.get_matched_mask()
		x = self.vehicles[available, -2]
		cells_x = self.cells_x[available]
		lower_bound_x = self.linspace_x[cells_x] - x
		upper_bound_x = self.linspace_x[cells_x + 1] - x
		y = self.vehicles[available, -1]
		cells_y = self.cells_y[available]
		lower_bound_y = self.linspace_y[cells_y] - y
		upper_bound_y = self.linspace_y[cells_y + 1] - y
		lower_bound = np.column_stack([lower_bound_x, lower_bound_y])
		upper_bound = np.column_stack([upper_bound_x, upper_bound_y])
		self.vehicles[available, -2:] += truncnorm(lower_bound, upper_bound, 0, self.cruising_std).rvs()
		

class LocalMatchingProcess:
	def __init__(
			self, 
			num_vehicles, 
			num_requests,
			num_rounds,
			num_cells, 
			num_intervals, 
			control_std,
			linspace_x, 
			linspace_y,
			weight_matrix=None, 
			waiting_rounds=1
		):
		'''
        arguments:
            num_vehicles: number of vehicles
            num_passengers: number of passengers
            num_cells: number of cells used to discretize the space
        '''
		self.num_vehicles = num_vehicles
		self.num_requests = num_requests
		self.num_rounds = num_rounds
		self.waiting_rounds = waiting_rounds
		self.control_std = control_std
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.weight_matrix = weight_matrix
		if isinstance(linspace_x, torch.Tensor):
			self.linspace_x = linspace_x.detach().cpu().numpy()
		if isinstance(linspace_y, torch.Tensor):
			self.linspace_y = linspace_y.detach().cpu().numpy()
		if isinstance(weight_matrix, torch.Tensor):
			self.weight_matrix = weight_matrix.flatten().detach().cpu().numpy()

	
	def get_matching_probabilities(
			self, 
			states: torch.Tensor,
			mu_available: torch.Tensor, 
			rho: torch.Tensor,
	) -> torch.Tensor:
		if isinstance(states, torch.Tensor):
			states = states.detach().cpu().numpy()
		if isinstance(mu_available, torch.Tensor):
			mu_available = mu_available.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		matched_proportions = self.get_matched_proportions(mu_available, rho)
		state_cells = states_to_cell(states, self.linspace_x, self.linspace_y)
		states_idx = cells_to_index(state_cells, self.num_intervals)
		if matched_proportions.shape[0] == 1:
			matching_probabilities = matched_proportions[:, states_idx]
		else:
			matching_probabilities = matched_proportions[states_idx]

		return matching_probabilities


	def get_matched_proportions(self, mu_available, rho):
		'''
		Simulate matching process

		arguments:
			mu_available: distribution of available vehicles (doesn't need to sum to 1)
			rho: distribution of requests (must sum to 1)
			num_rounds: num of matching rounds

		returns:
			m: matching rate vector – for each cell in [0,1]
		'''
		if isinstance(mu_available, torch.Tensor):
			mu_available = mu_available.detach().cpu().numpy()
		if isinstance(rho, torch.Tensor):
			rho = rho.detach().cpu().numpy()
		mu_available = mu_available.flatten()
		rho = rho.flatten()
		cruising_std = self.control_std / math.sqrt(self.num_rounds)
		request_simulator = LocalRequestSimulator(
			rho, self.num_requests, self.num_rounds, self.num_cells, self.num_intervals, 
			self.linspace_x, self.linspace_y, self.waiting_rounds
		)
		vehicle_simulator = LocalVehicleSimulator(
			mu_available, self.num_vehicles, self.num_cells, self.num_intervals, self.linspace_x, self.linspace_y, cruising_std
		)
		for round in range(self.num_rounds):
			requests = request_simulator.get_pending_requests(round)
			vehicles = vehicle_simulator.get_available_vehicles()
			# return self.matching(requests, vehicles)
			matched_request_ids, matched_vehicle_ids, pickup_distances = (
				self.matching(requests, vehicles)
			)
			request_simulator.update_pending_requests(matched_request_ids)
			vehicle_simulator.update_available_vehicles(matched_vehicle_ids)
		matched_proportions = vehicle_simulator.get_matched_proportions()
		if self.weight_matrix is not None:
			matched_proportions = matched_proportions * self.weight_matrix

		return matched_proportions


	def matching(self, requests, vehicles):
		'''
		Simulate one matching round

		arguments:
			requests: array of requests (request_id, x, y)
			vehicles: array of vehicles (vehicle_id, x, y)

		returns:
			request_ids: array of ids of matched requests
			vehicle_ids: array of ids of matched vehicles
			pickup_distances: array of pickup distance pickup distances
		'''
		matched_request_ids = np.array([], dtype=int)
		matched_vehicle_ids = np.array([], dtype=int)
		pickup_distances = np.array([])
		for cell in range(self.num_cells):
			cell_requests = requests[requests[:, 1].astype(int) == cell]
			cell_vehicles = vehicles[vehicles[:, 1].astype(int) == cell]
			cell_request_ids, cell_vehicle_ids, cell_pickup_distances = self._matching(
				cell_requests, cell_vehicles
			)
			matched_request_ids = np.append(matched_request_ids, cell_request_ids)
			matched_vehicle_ids = np.append(matched_vehicle_ids, cell_vehicle_ids)
			pickup_distances = np.append(pickup_distances, cell_pickup_distances)

		return matched_request_ids, matched_vehicle_ids, pickup_distances


	def _matching(self, requests, vehicles):
		request_ids = requests[:, 0].astype(int)
		vehicle_ids = vehicles[:, 0].astype(int)
		request_coordinates = requests[:, -2:]
		vehicle_coordinates = vehicles[:, -2:]
		distance_matrix = cdist(request_coordinates, vehicle_coordinates)
		try:
			matched_requests, matched_vehicles = linear_sum_assignment(distance_matrix)
			pickup_distances = distance_matrix[matched_requests, matched_vehicles]
			matched_request_ids = request_ids[matched_requests]
			matched_vehicle_ids = vehicle_ids[matched_vehicles]

			return matched_request_ids, matched_vehicle_ids, pickup_distances
		except ValueError:
			# If the assignment is infeasible, return empty arrays to represents no matches
			# If we had some iterative matching, we could return partial matches which 
			# are not supported by LP-solvers
			# In general, local matching should be infeasible less often than global matching
			return np.array([], dtype=bool), np.array([], dtype=bool), np.array([])
		

class LocalRequestSimulator:
	def __init__(self, rho, num_requests, num_rounds, num_cells, num_intervals, linspace_x, linspace_y, waiting_rounds=1):
		self.rho = rho
		self.num_requests = num_requests
		self.num_rounds = num_rounds
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.waiting_rounds = waiting_rounds
		self.pending = np.ones(self.num_requests, dtype=bool)
		self.init_requests()
		

	def init_requests(self):
		'''
		Initialize location of requests
		'''
		request_ids = np.arange(self.num_requests)
		# Determine a cell for each request given the demand distribution rho
		request_cells = np.random.choice(self.num_cells, size=self.num_requests, p=self.rho)
		# Reshape passenger cell idx to (x,y) cell idx
		cells_x, cells_y = index_to_cell(request_cells, self.num_intervals)
		# Uniformly generate (x,y) coordinates within cells
		x = np.random.uniform(self.linspace_x[cells_x], self.linspace_x[cells_x + 1])
		y = np.random.uniform(self.linspace_y[cells_y], self.linspace_y[cells_y + 1])
		# Generate request times
		round = np.random.randint(self.num_rounds, size=self.num_requests)  
		self.requests = np.column_stack((request_ids, request_cells, round, x, y))


	def get_pending_requests(self, round):
		'''
		Generate arrival requests
		'''
		waiting_upper_bound_mask = round <= self.requests[:, 2] + self.waiting_rounds - 1
		self.pending[~waiting_upper_bound_mask] = False
		rounds_mask = (
			waiting_upper_bound_mask &
			(self.requests[:, 2] <= round)
		)
		requests = self.requests[self.pending & rounds_mask]

		return requests
	

	def update_pending_requests(self, matched_ids):
		'''
		Update requests status


		arguments:
			matched_ids: index of matched vehicles in available vehicles
		'''
		# update availability
		self.pending[matched_ids] = False


class LocalVehicleSimulator:
	def __init__(self, mu_available, num_vehicles, num_cells, num_intervals, linspace_x, linspace_y, cruising_std=1):
		# Compute the number of available vehicles
		# It's just a proportion of the total number of vehicles
		# because mu_available might not sum up to 1
		self.num_available = int(num_vehicles * mu_available.sum())
		# We need to normalize mu_available to make it a distribution
		# We are applying it to the available vehicles only
		self.mu_available = mu_available / mu_available.sum()
		# Mask that keeps track of available vehicles
		self.available = np.ones(self.num_available, dtype=bool)
		self.num_cells = num_cells
		self.num_intervals = num_intervals
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.cruising_std = cruising_std
		self.init_vehicles()
		

	def init_vehicles(self):
		'''
		Initialize vehicles
		'''
		vehicle_ids = np.arange(self.num_available)
		# Determine a cell for each vehicle given the distribution rho
		# Unlike global matching, crusing could take vehicles outside of the cell
		vehicle_cells = np.random.choice(self.num_cells, size=self.num_available, p=self.mu_available)
		# Reshape vehicle cell idx to (x,y) cell idx
		self.cells_x, self.cells_y = index_to_cell(vehicle_cells, self.num_intervals)
		# Uniformly generate (x,y) coordinates within cells
		x = np.random.uniform(self.linspace_x[self.cells_x], self.linspace_x[self.cells_x + 1])
		y = np.random.uniform(self.linspace_y[self.cells_y], self.linspace_y[self.cells_y + 1])
		self.vehicles = np.column_stack((vehicle_ids, vehicle_cells, x, y))


	def get_matched_proportions(self):
		'''
		Compute matched proportions for each cell
		'''
		data = np.column_stack((self.vehicles[:, 1], 1 - self.available))
		matched_proportions = np.array(
			[data[data[:, 0] == i][:, 1].mean() 
			for i in range(self.num_cells)]
		)
		matched_proportions = np.nan_to_num(matched_proportions, nan=0)
		
		return matched_proportions


	def get_available_vehicles(self):
		'''
		Generate available vehicles 
		'''
		return self.vehicles[self.available]


	def update_available_vehicles(self, matched_ids):
		'''
		Update vehicle availability (and locations)


		arguments:
			matched_ids: index of matched vehicles in available vehicles
		'''
		# update availability
		self.available[matched_ids] = False
		# We should clip the coordinates to avoid vehicles leaving the cell 
		x = self.vehicles[self.available, -2]
		cells_x = self.cells_x[self.available]
		lower_bound_x = self.linspace_x[cells_x] - x
		upper_bound_x = self.linspace_x[cells_x + 1] - x
		self.vehicles[self.available, -2] += truncnorm(lower_bound_x, upper_bound_x, 0, self.cruising_std).rvs()
		y = self.vehicles[self.available, -1]
		cells_y = self.cells_y[self.available]
		lower_bound_y = self.linspace_y[cells_y] - y
		upper_bound_y = self.linspace_y[cells_y + 1] - y
		self.vehicles[self.available, -1] += truncnorm(lower_bound_y, upper_bound_y, 0, self.cruising_std).rvs()

