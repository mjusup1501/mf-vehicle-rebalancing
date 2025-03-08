import numpy as np
import multiprocessing
from time import time
from tqdm import tqdm
from safe_mf.models.matching_simulators import GlobalMatchingProcess
from safe_mf.models.lp_rebalancers import StaticRebalancer, RealTimeRebalancer
from safe_mf.utils.utils import states_to_cell, cells_to_index
from safe_mf.utils.data import agg_by_labels
from safe_mf.utils.transition import (
    perturb_cell_centers as perturb_cells,
    get_matched_states as get_matched, 
    get_cruising_states as get_cruising,
    mf_deterministic_step as mf_step
)
from safe_mf.utils.reward import lifted_reward as lreward
from safe_mf.utils.entropy import entropic_constraint, max_entropy, entropy as entropy_func


def compute_entropy(mu, weight_matrix):
    entropy = entropy_func(mu, weight_matrix)

    return entropy[None]


def compute_constraint_violation(mu, weight_matrix, constraint_function):
    constraint_violation = constraint_function(mu, weight_matrix)

    return constraint_violation[None]


def generate_states_from_counts(ra_cnt, cell_centers, perturb_cell_centers):
    ra_states = np.repeat(cell_centers, ra_cnt, axis=0)
    ra_states = perturb_cell_centers(ra_states)

    return ra_states


def generate_counts_from_states(ra_states, linspace_x, linspace_y, num_cells, num_intervals):
    cells = states_to_cell(ra_states, linspace_x, linspace_y)
    idxs = cells_to_index(cells, num_intervals)
    ra_cnt = agg_by_labels(np.ones(idxs.shape[0]), idxs, num_cells).astype(int)

    return ra_cnt


def round_repositioning_proposal(proposal, threshold=0.1):
    """ Round up initial repositioning proposal to ensure all entries are integers.

    Args:
        proposal: LP decisions of shape (NUM_CELLS, NUM_CELLS);
        threshold: threshold for the non-integer part
    
    Returns:
        rounded: rounded proposals
    """

    rounded = np.where(proposal - np.floor(proposal) < threshold, np.floor(proposal), np.ceil(proposal))
    return rounded


# function for determining the repositioning decisions
def determine_from_LP_solution(supply, repositioning_proposal, num_cells):
    """ Determines the available vehicles and repositioning vehicles for each cell.

    Args:
        supply: supply vector of shape (NUM_CELLS,);
        repositioning_proposal: LP decisions of shape (NUM_CELLS, NUM_CELLS);
    
    Returns:
        available: available vehicle vector of shape (NUM_CELLS,);
        repositioning_final: final repositioning decision of shape (NUM_CELLS, NUM_CELLS).
    """
    # initialize repositioning decision
    repositioning_od_matrix = np.zeros_like(repositioning_proposal)
    
    for i in range(num_cells):
        # copy proposal if supply is sufficient
        if np.sum(repositioning_proposal[i,:]) <= supply[i]:
            repositioning_od_matrix[i,:] = repositioning_proposal[i,:]
        # otherwise use all supply and propotionally scale proposal as final
        else:
            scale = supply[i] / np.sum(repositioning_proposal[i,:])
            # floor all elements to ensure the sum does not surpass supply
            repositioning_od_matrix[i,:] = np.floor(scale * repositioning_proposal[i,:])

    ra_available_cnt = supply - repositioning_od_matrix.sum(axis=1)
    ra_repositioned_cnt = repositioning_od_matrix.sum(axis=1)
    ra_repositioned_next_cnt = repositioning_od_matrix.sum(axis=0)

    return ra_available_cnt.astype(int), ra_repositioned_cnt.astype(int), ra_repositioned_next_cnt.astype(int)


def execute_episode(
        mu_initial_cnt, 
        horizon, 
        demand_matrix, 
        weight_matrix,
        demand_matrix_cnt, 
        origin_destination_matrix, 
        num_cells, 
        num_intervals, 
        num_agents, 
        rebalancer,
        optimizer_cfg,
        cell_centers,
        perturb_cell_centers,
        matching_process,
        get_matched_states,
        get_cruising_states,
        lifted_reward,
        constraint_function,
        linspace_x,
        linspace_y,
        mf_deterministic_step,
        max_num_requests,
    ):
    inference_time = 0.0
    mus_cnt = [mu_initial_cnt]
    ( 
        episode_ra_repositioned_states, episode_ra_matched_states, episode_ra_cruising_states, 
        episode_mus_repositioned, episode_mus_matched, episode_mus_cruising, 
        episode_lifted_rewards, episode_requests, episode_vehicles, episode_pickup_distances,
        episode_entropies, episode_constraint_violations
    ) = ([],[],[],[],[],[],[],[],[],[],[],[])
    for step in tqdm(range(horizon), desc='Policy rollout'):
        # extract supply
        mu_cnt = mus_cnt[-1]
        # import data
        demand = demand_matrix[step]
        demand_cnt = demand_matrix_cnt[step]
        od_matrix = origin_destination_matrix[step]
        rebalancer_type = optimizer_cfg.get('rebalancer_type')
        desired_count_pct = optimizer_cfg.get('desired_count_pct', None)
        max_iter = optimizer_cfg.get('max_iter', 30_000)
        if rebalancer_type == 'static':
            rebalancer.lamb.value = demand_cnt
            rebalancer.P.value = od_matrix
            start_time = time() 
            rebalancer.problem.solve(verbose=1, solver='OSQP', max_iter=max_iter)
            inference_time += time() - start_time
            lp_solution_raw = rebalancer.alpha.value
        elif rebalancer_type == 'real-time':
            ra_excessive_cnt = np.maximum(mu_cnt - demand_cnt, 0)
            ra_incoming = (od_matrix * (np.repeat(demand_cnt[:,np.newaxis], num_cells, 1))).sum(axis=0)
            ra_desired_cnt = (desired_count_pct * mu_initial_cnt).astype(int) - ra_incoming
            ra_desired_cnt = np.maximum(ra_desired_cnt, 0)
            rebalancer.v_ex.value = ra_excessive_cnt.astype(int)
            rebalancer.v_d.value = ra_desired_cnt.astype(int)
            start_time = time() 
            rebalancer.problem.solve(verbose=True, solver='OSQP', max_iter=max_iter)
            inference_time += time() - start_time
            lp_solution_raw = rebalancer.num.value

        lp_solution = round_repositioning_proposal(lp_solution_raw, threshold=0.1)
        ra_available_cnt, ra_repositioned_cnt, ra_repositioned_next_cnt = determine_from_LP_solution(mu_cnt, lp_solution, num_cells)
        ra_available_states = generate_states_from_counts(ra_available_cnt, cell_centers, perturb_cell_centers)
        ra_repositioned_states = generate_states_from_counts(ra_repositioned_cnt, cell_centers, perturb_cell_centers)
        ra_repositioned_states_next = generate_states_from_counts(ra_repositioned_next_cnt, cell_centers, perturb_cell_centers)
        requests, vehicles, pickup_distances = (
            matching_process.simulate(ra_available_states, demand)
        )
        matched_mask = vehicles[:, 0] == 1
        ra_matched_states = vehicles[matched_mask, -2:]
        ra_cruising_states = vehicles[~matched_mask, -2:]
        ra_matched_states_next = get_matched_states(ra_matched_states, od_matrix)
        ra_cruising_states_next = get_cruising_states(ra_cruising_states, mf_transition=False)
        ra_matched_next_cnt = generate_counts_from_states(ra_matched_states_next, linspace_x, linspace_y, num_cells, num_intervals)
        ra_cruising_next_cnt = generate_counts_from_states(ra_cruising_states_next, linspace_x, linspace_y, num_cells, num_intervals)
        mu_cnt_next = ra_repositioned_cnt + ra_matched_next_cnt + ra_cruising_next_cnt
        mus_cnt.append(mu_cnt_next)

        mu_repositioned = mf_deterministic_step(ra_repositioned_states)
        mu_matched = mf_deterministic_step(ra_matched_states)
        mu_cruising = mf_deterministic_step(ra_cruising_states)
        mu_repositioned_next = mf_deterministic_step(ra_repositioned_states_next)
        mu_matched_next = mf_deterministic_step(ra_matched_states_next)
        mu_cruising_next = mf_deterministic_step(ra_cruising_states_next)
        reward = lifted_reward(
            mu_repositioned, mu_matched, mu_cruising, 
            mu_repositioned_next, mu_matched_next, mu_cruising_next,
            demand
        )

        dummy_rows = -1 * np.ones((num_agents - ra_repositioned_states.shape[0], ra_repositioned_states.shape[1]))
        ra_repositioned_states = np.concatenate((ra_repositioned_states, dummy_rows), axis=0)   
        dummy_rows = -1 * np.ones((num_agents - ra_matched_states.shape[0], ra_matched_states.shape[1]))
        ra_matched_states = np.concatenate((ra_matched_states, dummy_rows), axis=0)
        dummy_rows = -1 * np.ones((num_agents - ra_cruising_states.shape[0], ra_cruising_states.shape[1]))
        ra_cruising_states = np.concatenate((ra_cruising_states, dummy_rows), axis=0)      
        dummy_rows = -1 * np.ones((num_agents - vehicles.shape[0], vehicles.shape[1]))
        vehicles = np.concatenate((vehicles, dummy_rows), axis=0)
        dummy_rows = -1 * np.ones((num_agents - pickup_distances.shape[0], pickup_distances.shape[1]))
        pickup_distances = np.concatenate((pickup_distances, dummy_rows), axis=0)
        dummy_rows = -1 * np.ones((max_num_requests - requests.shape[0], requests.shape[1]))
        requests = np.concatenate((requests, dummy_rows), axis=0)
        episode_ra_repositioned_states.append(ra_repositioned_states)
        episode_ra_matched_states.append(ra_matched_states)
        episode_ra_cruising_states.append(ra_cruising_states)
        episode_mus_repositioned.append(mu_repositioned)
        episode_mus_matched.append(mu_matched)
        episode_mus_cruising.append(mu_cruising)
        episode_lifted_rewards.append(reward[None])
        episode_requests.append(requests)
        episode_vehicles.append(vehicles)
        episode_pickup_distances.append(pickup_distances)
        mu_available_next = mu_matched_next + mu_cruising_next
        mu_available_next /= mu_available_next.sum()
        episode_entropies.append(compute_entropy(mu_available_next, weight_matrix))
        if constraint_function is not None:
            episode_constraint_violations.append(compute_constraint_violation(mu_available_next, weight_matrix, constraint_function))
    # Postprocessing for visualization
    # Add the final step to the list
    dummy_rows = -1 * np.ones((num_agents - ra_repositioned_states_next.shape[0], ra_repositioned_states_next.shape[1]))
    ra_repositioned_states_next = np.concatenate((ra_repositioned_states_next, dummy_rows), axis=0)
    dummy_rows = -1 * np.ones((num_agents - ra_matched_states_next.shape[0], ra_matched_states_next.shape[1]))
    ra_matched_states_next = np.concatenate((ra_matched_states_next, dummy_rows), axis=0)
    dummy_rows = -1 * np.ones((num_agents - ra_cruising_states_next.shape[0], ra_cruising_states_next.shape[1]))
    ra_cruising_states_next = np.concatenate((ra_cruising_states_next, dummy_rows), axis=0)
    episode_ra_repositioned_states.append(ra_repositioned_states_next)
    episode_ra_matched_states.append(ra_matched_states_next)
    episode_ra_cruising_states.append(ra_cruising_states_next)
    episode_mus_repositioned.append(mu_repositioned_next)
    episode_mus_matched.append(mu_matched_next)
    episode_mus_cruising.append(mu_cruising_next)
    # Add dummy values to align the dimensions
    reward = np.zeros_like(reward)
    episode_lifted_rewards.append(reward[None])
    mu_available_next = mu_matched_next + mu_cruising_next
    mu_available_next /= mu_available_next.sum()
    episode_entropies.append(compute_entropy(mu_available_next, weight_matrix))
    if constraint_function is not None:
        episode_constraint_violations.append(compute_constraint_violation(mu_available_next, weight_matrix, constraint_function))

    return (
        episode_ra_repositioned_states, episode_ra_matched_states, 
        episode_ra_cruising_states, episode_mus_repositioned, 
        episode_mus_matched, episode_mus_cruising, 
        episode_lifted_rewards, episode_requests, 
        episode_vehicles, episode_pickup_distances,
        episode_entropies, episode_constraint_violations,
        inference_time
    )


def evaluate_benchmark(
        n_repeats,
        horizon,
        num_agents, 
        demand_matrix, 
        weight_matrix,
        origin_destination_matrix,
        state_dim,
        num_cells,
        num_intervals,
        control_std,
        state_space,
        real_world_matching_cfg,
        optimizer_cfg,
        results_dir,
        evaluations_count,
        barrier_lambda,
        max_entropy_ratio
):
    demand_matrix_cnt = (num_agents * demand_matrix).round().astype(int)
    mu_initial_cnt = demand_matrix_cnt[-1]
    demand_matrix = demand_matrix[:-1]
    demand_matrix_cnt = demand_matrix_cnt[:-1]
    max_demand = np.sum(demand_matrix, axis=1).max()
    max_num_requests_1 = int(max_demand * num_agents)
    max_num_requests_2 = demand_matrix_cnt.sum(axis=1).max() 
    max_num_requests = max(max_num_requests_1, max_num_requests_2)
    residual = num_agents - mu_initial_cnt.sum()
    idx = np.where(mu_initial_cnt > 0)[0]
    idx = np.random.choice(idx, size=abs(residual), replace=False)
    if residual >= 0:
        mu_initial_cnt[idx] += 1
    else:
        mu_initial_cnt[idx] -= 1
    linspace_x = np.linspace(state_space[0], state_space[1], num_intervals + 1)
    linspace_y = np.linspace(state_space[0], state_space[1], num_intervals + 1)
    cell_centers_1d = (
        np.arange(0, num_intervals, 1)
        .reshape(1, -1)
        / num_intervals
    ) + 0.5 / num_intervals
    cell_centers_x, cell_centers_y = np.meshgrid(*[cell_centers_1d.squeeze(0)] * state_dim, indexing='ij')
    cell_centers = np.column_stack((cell_centers_x.ravel(), cell_centers_y.ravel()))
    matching_type = real_world_matching_cfg.pop('type')
    if matching_type == 'global_matching_simulator':
        matching_process = GlobalMatchingProcess(
            num_cells=num_cells,
            num_intervals=num_intervals,
            control_std=control_std,
            linspace_x=linspace_x,
            linspace_y=linspace_y,
            weight_matrix=weight_matrix,
            **real_world_matching_cfg
        )
    perturb_cell_centers = lambda cell_centers: perturb_cells(cell_centers, linspace_x, linspace_y)
    get_matched_states = (
        lambda states, transition_matrix: 
        get_matched(states, transition_matrix, num_intervals, linspace_x, linspace_y)
    )
    get_cruising_states = (
        lambda states, mf_transition: 
        get_cruising(states, linspace_x, linspace_y, control_std, mf_transition)
    )
    mf_deterministic_step = lambda ra_states: mf_step(ra_states, num_agents, num_intervals, state_space[0], state_space[1])
    if max_entropy_ratio is not None:
        # We don't use all cells, but only those where we have demand
        if weight_matrix is not None:
            num_non_zero_cells = weight_matrix.sum()
        else:
            num_non_zero_cells = num_cells
        max_entropy_ = max_entropy(num_non_zero_cells).detach().numpy()
        max_barrier = np.log(max_entropy_ - max_entropy_ratio * max_entropy_)
        constraint_function = lambda mu, w: entropic_constraint(
                mu, max_entropy_ratio * max_entropy_, w
            )
    else:
        constraint_function = None
        max_barrier = None
    lifted_reward = (
            lambda 
            mu_repositioned, mu_matched, mu_cruising, 
            mu_repositioned_next, mu_matched_next, mu_cruising_next, 
            demand_matrix:
            lreward(
                mu_repositioned, mu_matched, mu_cruising,
                mu_repositioned_next, mu_matched_next, mu_cruising_next, 
                demand_matrix, weight_matrix, max_barrier, 
                barrier_lambda, constraint_function
            )
        )

    ( 
        visualization_ra_repositioned_states, visualization_ra_matched_states,
        visualization_ra_cruising_states,
        visualization_mus_repositioned,
        visualization_mus_matched, visualization_mus_cruising,
        visualization_lifted_rewards,
        visualization_requests, visualization_vehicles,
        visualization_pickup_distances,
        visualization_entropies, visualization_constraint_violations
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])
    print('=' * 15 + f"Episode {evaluations_count}" + '=' * 15)   
    rebalancer_type = optimizer_cfg.get('rebalancer_type')
    if rebalancer_type == 'static':
        rebalancer = StaticRebalancer(num_cells)
    elif rebalancer_type == 'real-time':
        rebalancer = RealTimeRebalancer(num_cells)
    episode_input = (
        mu_initial_cnt, 
        horizon, 
        demand_matrix, 
        weight_matrix,
        demand_matrix_cnt, 
        origin_destination_matrix, 
        num_cells, num_intervals, 
        num_agents, 
        rebalancer,
        optimizer_cfg,
        cell_centers,
        perturb_cell_centers,
        matching_process,
        get_matched_states,
        get_cruising_states,
        lifted_reward,
        constraint_function,
        linspace_x,
        linspace_y,
        mf_deterministic_step,
        max_num_requests,
    )
    # num_processes = multiprocessing.cpu_count()  # Use all available CPUs
    # pool = multiprocessing.Pool(processes=num_processes)
    # input_data = [episode_input] * n_repeats
    # episode_results = pool.starmap(execute_episode, episode_input)
    inference_times = np.zeros(n_repeats)
    for i in range(n_repeats):
        ( 
            episode_ra_repositioned_states, episode_ra_matched_states, 
            episode_ra_cruising_states, episode_mus_repositioned, 
            episode_mus_matched, episode_mus_cruising, 
            episode_lifted_rewards, episode_requests, 
            episode_vehicles, episode_pickup_distances,
            episode_entropies, episode_constraint_violations,
            episode_inference_times
        ) = execute_episode(*episode_input)
        inference_times[i] = episode_inference_times
        visualization_ra_repositioned_states.append(np.stack(episode_ra_repositioned_states))
        visualization_ra_matched_states.append(np.stack(episode_ra_matched_states))
        visualization_ra_cruising_states.append(np.stack(episode_ra_cruising_states))
        visualization_mus_repositioned.append(np.stack(episode_mus_repositioned).squeeze(1))
        visualization_mus_matched.append(np.stack(episode_mus_matched).squeeze(1))
        visualization_mus_cruising.append(np.stack(episode_mus_cruising).squeeze(1))
        visualization_lifted_rewards.append(np.stack(episode_lifted_rewards))
        visualization_requests.append(np.stack(episode_requests))
        visualization_vehicles.append(np.stack(episode_vehicles))
        visualization_pickup_distances.append(np.stack(episode_pickup_distances))
        visualization_entropies.append(np.stack(episode_entropies))
        if constraint_function is not None:
            visualization_constraint_violations.append(np.stack(episode_constraint_violations))
        np.save(results_dir / f'inference_times{evaluations_count}', inference_times)
        np.save(results_dir / f"ra_repositioned_states{evaluations_count}", np.stack(visualization_ra_repositioned_states))
        np.save(results_dir / f"ra_matched_states{evaluations_count}", np.stack(visualization_ra_matched_states))
        np.save(results_dir / f"ra_cruising_states{evaluations_count}", np.stack(visualization_ra_cruising_states))
        np.save(results_dir / f"mus_repositioned{evaluations_count}", np.stack(visualization_mus_repositioned))
        np.save(results_dir / f"mus_matched{evaluations_count}", np.stack(visualization_mus_matched))
        np.save(results_dir / f"mus_cruising{evaluations_count}", np.stack(visualization_mus_cruising))
        np.save(results_dir / f"lifted_rewards{evaluations_count}", visualization_lifted_rewards)
        np.save(results_dir / f"entropies{evaluations_count}", visualization_entropies)
        if constraint_function is not None:
            np.save(results_dir / f"constraint_violations{evaluations_count}", visualization_constraint_violations)
        np.save(results_dir / f"requests{evaluations_count}", visualization_requests)
        np.save(results_dir / f"vehicles{evaluations_count}", visualization_vehicles)
        np.save(results_dir / f"pickup_distances{evaluations_count}", visualization_pickup_distances)
        total_reward = np.stack(visualization_lifted_rewards).sum()
        avg_matched = np.stack(visualization_mus_matched)[:, :-1, :].sum(axis=-1).mean()
        avg_cruising = np.stack(visualization_mus_cruising)[:, :-1, :].sum(axis=-1).mean()
        print('=' * 15 + f"Episode {evaluations_count}" + '=' * 15)    
        print(f"avg_step_reward: {total_reward / ((i + 1) * horizon)}")
        print(f"avg_step_matched: {avg_matched}")
        print(f"avg_step_cruising: {avg_cruising}")