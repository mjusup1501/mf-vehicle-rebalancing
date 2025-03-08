import os

from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
import numpy as np
import math
import torch
import wandb
from copy import deepcopy
import re
import gc

from safe_mf.envs.vehicle_repositioning_simultaneous import VehicleRepositioningSimultaneous
from safe_mf.models.gradient_descent import GradientDescent
from safe_mf.models.policy import MFPolicy, RandomPolicy, DummyPolicy
from safe_mf.utils.entropy import entropic_constraint, max_entropy, entropy as entropy_func
from safe_mf.utils.data import normalize_distributions
from safe_mf.utils.utils import find_best_ckpt, find_last_ckpt, find_last, find_exact, extract_digits
from safe_mf.models.bechmarks import evaluate_benchmark


class SafeMFMARL:
    def __init__(
        self,
        env_name: str,
        framework: str,
        num_cells: int,
        control_std: float,
        max_entropy_ratio: float,
        barrier_lambda: float,
        optimizer_cfg: Mapping[str, Any],
        device: str,
        logdir: str,
        eval_logdir: str=None,
        mu_init: str = "uniform",
        input_data_path: str = None,
        mf_matching_cfg: Mapping[str, Any] = None,
        real_world_matching_cfg: Mapping[str, Any] = None,
        exec_type: str = "train",
        num_agents: int = 1,
        reward_type: str = 'current_mu',
        matching_target: str = 'regression',
    ) -> None:
        self.env_name = env_name
        self.framework = framework
        self.device = torch.device(device)
        self.optimizer_cfg = optimizer_cfg
        self.num_agents = num_agents
        self.evaluations_count = 0
        self.logdir = logdir
        self.eval_logdir = logdir
        input_data_path = Path(input_data_path)
        if exec_type == 'train':
            self.results_dir = logdir / "data" / exec_type
        elif exec_type in {'eval', 'lp_eval'}:
            pct_uniform = input_data_path.name
            input_type = input_data_path.parent.name
            pct_uniform = re.findall(r'\d+pct-u', pct_uniform)[0]
            self.results_dir = eval_logdir / input_type / pct_uniform
        os.makedirs(self.results_dir, exist_ok=True)
        if mf_matching_cfg is not None:
            self.mf_matching_type = mf_matching_cfg['type']
            if self.mf_matching_type == 'statistical_model':
                self.mf_matching_ckpt_dir = logdir / "checkpoints" / "matching_model"
                os.makedirs(self.mf_matching_ckpt_dir, exist_ok=True)
                mf_matching_ckpt = mf_matching_cfg.get('checkpoint', None)
                if mf_matching_ckpt is not None:
                    self.evaluations_count = max(self.evaluations_count, int(re.findall(r'\d+', mf_matching_ckpt)[0]))
                    mf_matching_cfg['checkpoint'] = self.mf_matching_ckpt_dir / mf_matching_ckpt
        self.exec_type = exec_type
        if self.exec_type == 'train':
            self.policy_ckpt_dir = logdir / "checkpoints" / "policy"
            os.makedirs(self.policy_ckpt_dir, exist_ok=True)
            policy_ckpt = self.optimizer_cfg.get('checkpoint', None)
            if policy_ckpt is not None:
                if policy_ckpt == 'policy_best.pt':
                    policy_ckpt = find_best_ckpt(self.policy_ckpt_dir)
                if policy_ckpt == 'policy.pt':
                    policy_ckpt = find_last_ckpt(self.policy_ckpt_dir)
                last_episode = re.findall(r'\d+', policy_ckpt)
                if last_episode:
                    self.evaluations_count = int(last_episode[0])
                else:
                    self.evaluations_count = 0
                self.optimizer_cfg['checkpoint'] = self.policy_ckpt_dir / policy_ckpt
        elif self.exec_type == 'lp_eval':
            pass
        elif (
            (self.exec_type == 'eval') 
            & (self.framework in {'mfc', 'mfrl'})
            & (optimizer_cfg['type'] != 'dummy_policy')
        ):
            best_reward = float('-inf')
            best_run = None
            best_folder = None
            model = self.framework
            constr = max_entropy_ratio
            if constr is None:
                constr = 0
            else:
                constr = int(constr * 100)
            uniform = re.findall(r'\d+', self.logdir.name)
            if uniform:
                uniform = int(uniform[0])
            elif constr == 0:
                uniform = 100
            else:
                uniform = 0
            runs_folder = self.logdir
            for folder in os.listdir(runs_folder):
                if (model in folder) & (f"_{uniform}u" in folder) & (f"_{constr}pct" in folder): 
                    ckpt_folder  = runs_folder / folder / 'checkpoints' / 'policy'
                    best_ckpt = find_best_ckpt(ckpt_folder)
                    best_ckpt_path = ckpt_folder / best_ckpt
                    if best_ckpt == "policy_final.pt":
                        reward = find_last(best_ckpt_path, 'lifted_rewards', 'npy')
                    else:
                        episode = extract_digits(best_ckpt)
                        reward = f"lifted_rewards{episode}.npy"
                    reward_folder = runs_folder / folder / 'data' / 'train'
                    lifted_rewards = np.load(reward_folder / reward)
                    avg_reward = lifted_rewards[:, :-1].mean()
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_run = best_ckpt_path
                        best_folder = folder
            print(f'Best checkpoint is: {best_run}')
            print(f'Best average reward is: {best_reward}')
            self.optimizer_cfg['checkpoint'] = best_run
            # Store the name of the run used for the evaluation
            with open(self.results_dir / best_folder, 'w') as file: pass
        if self.framework == 'lp':
            self.demand_matrix = np.load(input_data_path / 'demand_matrix.npy')
            self.weight_matrix = np.load(input_data_path / 'weight_matrix.npy')
            self.origin_destination_matrix = np.load(input_data_path / 'origin_destination_matrix.npy')
            self.state_space = (0.0, 1.0)
            self.state_dim = len(self.state_space)
            self.num_cells = num_cells
            self.num_intervals = int(math.sqrt(self.num_cells))
            self.control_std = control_std
            self.real_world_matching_cfg = real_world_matching_cfg
            self.optimizer_cfg = optimizer_cfg
            self.barrier_lambda = barrier_lambda
            self.max_entropy_ratio = max_entropy_ratio
        elif self.env_name == "vehicle-repositioning-simultaneous":
            assert input_data_path is not None, "input_data_path must be specified for vehicle repositioning environment"
            demand_matrix = np.load(input_data_path / 'demand_matrix.npy')
            demand_matrix = torch.from_numpy(demand_matrix).float().to(device)
            origin_destination_matrix = np.load(input_data_path / 'origin_destination_matrix.npy')
            origin_destination_matrix = torch.from_numpy(origin_destination_matrix).float().to(device)
            self.weight_matrix = np.load(input_data_path / 'weight_matrix.npy')
            self.weight_matrix = torch.from_numpy(self.weight_matrix).float().to(device)
            if max_entropy_ratio is not None:
                # We don't use all cells, but only those where we have demand
                if self.weight_matrix is not None:
                    num_non_zero_cells = self.weight_matrix.sum().item()
                else:
                    num_non_zero_cells = num_cells
                a = max_entropy(num_non_zero_cells), max_entropy(num_cells)
                max_entropy_ = max_entropy(num_non_zero_cells)
                max_barrier = torch.log(max_entropy_ - max_entropy_ratio * max_entropy_)
                self.constraint_function = lambda mu, w: entropic_constraint(
                        mu, max_entropy_ratio * max_entropy_, w
                    )
            else:
                self.constraint_function = None
                max_barrier = None
            self.env = VehicleRepositioningSimultaneous(
                num_cells,
                demand_matrix,
                origin_destination_matrix,
                self.weight_matrix,
                control_std,
                mu_init,
                barrier_lambda,
                self.constraint_function,
                max_barrier,
                device=device,
                mf_matching_cfg=mf_matching_cfg,
                real_world_matching_cfg=real_world_matching_cfg,
                exec_type=self.exec_type,
                num_agents=self.num_agents,
                reward_type=reward_type,
                matching_target=matching_target,
            )

        if self.exec_type == 'train':
            self.policy_type = self.optimizer_cfg.pop('type')
            self.optimizer = GradientDescent(
                self.env, 
                device=self.device,
                **self.optimizer_cfg
            )


    def _warmup(
        self,
        n_transitions: int,
        horizon: int
    ): pass


    def _compute_entropy(self, mu: torch.Tensor) -> torch.Tensor:
        entropy = entropy_func(mu, self.weight_matrix)

        return entropy.unsqueeze(0)
    

    def _compute_constraint_violation(self, mu: torch.Tensor) -> torch.Tensor:
        constraint_violation = self.constraint_function(mu, self.weight_matrix)

        return constraint_violation.unsqueeze(0)
    

    def _evaluate(
        self,
        horizon: int,
        policy: MFPolicy,
        train_matching_model: bool=False,
        n_repeats: int=10,
    ):
        inference_times = np.zeros(n_repeats)
        max_demand = torch.sum(self.env.demand_matrix, dim=(1, 2)).max()
        max_num_requests = int(max_demand * self.num_agents)
        ( 
            train_ra_states, train_mus_available, train_demands, train_matching_probs,
            visualization_ra_repositioned_states, visualization_ra_matched_states,
            visualization_ra_cruising_states,
            visualization_mus, visualization_mus_repositioned,
            visualization_mus_matched, visualization_mus_cruising,
            visualization_mf_matched_proportions, visualization_lifted_rewards,
            visualization_constraint_violations, visualization_entropies,
            visualization_requests, visualization_vehicles,
            visualization_pickup_distances
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],)
        print('=' * 15 + f"Episode {self.evaluations_count}" + '=' * 15)   
        with torch.no_grad():
            policy.eval()
            if self.exec_type == 'train':
                for i in tqdm(range(n_repeats), desc='Train repeat'):
                    self.env.reset()
                    ( 
                        episode_ra_states, episode_mus_available,
                        episode_demands, episode_ra_matching_outputs, 
                        episode_mus_repositioned, episode_mus_matched, episode_mus_cruising, 
                        episode_mf_matched_proportions, episode_lifted_rewards,
                        episode_entropies, episode_constraint_violations 
                    ) = ([], [], [], [], [], [], [], [], [], [], [])   
                    for step in tqdm(range(horizon), desc='Policy rollout'):
                        (
                            ra_states, demands, ra_matching_outputs,
                            mu_repositioned, mu_matched, mu_cruising,
                            mu_repositioned_next, mu_matched_next, 
                            mu_cruising_next, mf_matched_proportions, 
                            lifted_reward
                        ) = self.env.step(
                            policy, 
                            step, 
                            exploration=None,
                            train_matching_model=train_matching_model
                        )
                        if train_matching_model:
                            episode_ra_states.append(ra_states)
                            mu_available = mu_matched + mu_cruising
                            episode_mus_available.append(mu_available)
                            episode_demands.append(demands)
                            episode_ra_matching_outputs.append(ra_matching_outputs)
                        episode_mus_repositioned.append(mu_repositioned)
                        episode_mus_matched.append(mu_matched)
                        episode_mus_cruising.append(mu_cruising)
                        episode_mf_matched_proportions.append(mf_matched_proportions)
                        episode_lifted_rewards.append(lifted_reward.unsqueeze(0))
                        mu_available_next = mu_matched_next + mu_cruising_next
                        mu_available_next = normalize_distributions(mu_available_next, p=1)
                        episode_entropies.append(self._compute_entropy(mu_available_next))
                        if self.constraint_function is not None:
                            episode_constraint_violations.append(self._compute_constraint_violation(mu_available_next))
                    if train_matching_model:
                        train_ra_states.extend(episode_ra_states)
                        train_mus_available.extend(episode_mus_available)
                        train_demands.extend(episode_demands)
                        train_matching_probs.extend(episode_ra_matching_outputs)
                    # Postprocessing for visualization
                    # Add the final step to the list
                    episode_mus_repositioned.append(mu_repositioned_next)
                    episode_mus_matched.append(mu_matched_next)
                    episode_mus_cruising.append(mu_cruising_next)
                    visualization_mus_repositioned.append(torch.stack(episode_mus_repositioned).squeeze(1).cpu().numpy())
                    visualization_mus_matched.append(torch.stack(episode_mus_matched).squeeze(1).cpu().numpy())
                    visualization_mus_cruising.append(torch.stack(episode_mus_cruising).squeeze(1).cpu().numpy())
                    # Add dummy values to align the dimensions
                    mf_matched_proportions = torch.zeros_like(mf_matched_proportions)
                    episode_mf_matched_proportions.append(mf_matched_proportions.unsqueeze(0))
                    visualization_mf_matched_proportions.append(torch.stack(episode_lifted_rewards).cpu().numpy())
                    lifted_reward = torch.zeros_like(lifted_reward)
                    episode_lifted_rewards.append(lifted_reward.unsqueeze(0))
                    visualization_lifted_rewards.append(torch.stack(episode_lifted_rewards).cpu().numpy())
                    # Compute entropy after the last step
                    mu_available_next = mu_matched_next + mu_cruising_next
                    mu_available_next = normalize_distributions(mu_available_next, p=1)
                    episode_entropies.append(self._compute_entropy(mu_available_next))
                    visualization_entropies.append(torch.stack(episode_entropies).cpu().numpy())
                    if self.constraint_function is not None:
                        episode_constraint_violations.append(self._compute_constraint_violation(mu_available_next))
                        visualization_constraint_violations.append(torch.stack(episode_constraint_violations).cpu().numpy())
                    del ( 
                        episode_mus_repositioned, 
                        episode_mus_matched, episode_mus_cruising, 
                        episode_mf_matched_proportions, episode_lifted_rewards,
                        episode_entropies, episode_constraint_violations 
                    ) 
                    if train_matching_model:
                        del (
                            episode_ra_states, episode_mus_available,
                            episode_demands, episode_ra_matching_outputs
                        )
                    gc.collect()
                visualization_mus_repositioned = np.stack(visualization_mus_repositioned)
                np.save(self.results_dir / f"mus_repositioned{self.evaluations_count}", visualization_mus_repositioned)
                visualization_mus_matched = np.stack(visualization_mus_matched)
                np.save(self.results_dir / f"mus_matched{self.evaluations_count}", visualization_mus_matched)
                visualization_mus_cruising = np.stack(visualization_mus_cruising)
                np.save(self.results_dir / f"mus_cruising{self.evaluations_count}", visualization_mus_cruising)
                visualization_mus = visualization_mus_repositioned + visualization_mus_matched + visualization_mus_cruising
                visualization_mus = np.stack(visualization_mus)
                np.save(self.results_dir / f"mus{self.evaluations_count}", visualization_mus)
                np.save(self.results_dir / f"mf_matched_proportions{self.evaluations_count}", visualization_mf_matched_proportions)
                np.save(self.results_dir / f"lifted_rewards{self.evaluations_count}", visualization_lifted_rewards)
                np.save(self.results_dir / f"entropies{self.evaluations_count}", visualization_entropies)
                if self.constraint_function is not None:
                    np.save(self.results_dir / f"constraint_violations{self.evaluations_count}", visualization_constraint_violations)
                total_reward = np.stack(visualization_lifted_rewards).sum()
                avg_matched = visualization_mus_matched.squeeze(0)[:-1].sum(axis=1).mean()
                avg_cruising = visualization_mus_cruising.squeeze(0)[:-1].sum(axis=1).mean()
                wandb.log({"avg_episode_reward": total_reward / n_repeats})
                wandb.log({"avg_step_reward": total_reward / (n_repeats * horizon)}) 
                print(f"avg_step_reward: {total_reward / (n_repeats * horizon)}")
                print(f"avg_step_matched: {avg_matched}")
                print(f"avg_step_cruising: {avg_cruising}")
                total_entropy = np.stack(visualization_entropies).sum()
                wandb.log({"avg_step_entropy": total_entropy / (n_repeats * horizon)})
                if self.constraint_function is not None:
                    min_constraint_violation = np.inf
                    num_constraint_violations = 0
                    for min_constraint_violations in visualization_constraint_violations:
                        min_constraint_violation = min(min_constraint_violation, min(min_constraint_violations))
                        num_constraint_violations += sum(min_constraint_violations < 0)
                    wandb.log({"min_constraint_violation": min_constraint_violation})  
                    wandb.log({"avg_episode_constraint_violations": num_constraint_violations / n_repeats})  
                del ( 
                    visualization_mus, visualization_mus_repositioned,
                    visualization_mus_matched, visualization_mus_cruising,
                    visualization_mf_matched_proportions, visualization_lifted_rewards,
                    visualization_constraint_violations, visualization_entropies
                )
                gc.collect()
            elif self.exec_type == 'eval':
                for i in tqdm(range(n_repeats), desc='Eval repeat'):
                    self.env.reset()
                    ( 
                        episode_ra_repositioned_states, episode_ra_matched_states, episode_ra_cruising_states, 
                        episode_mus_repositioned, episode_mus_matched, episode_mus_cruising, 
                        episode_lifted_rewards, episode_requests, episode_vehicles, episode_pickup_distances,
                        episode_entropies, episode_constraint_violations 
                    ) = ([],[],[],[],[],[],[],[],[],[],[],[])
                    for step in tqdm(range(horizon), desc='Policy rollout'):
                        (
                            ra_repositioned_states, ra_matched_states, ra_cruising_states,
                            mu_repositioned, mu_matched, mu_cruising,
                            ra_repositioned_states_next, ra_matched_states_next, ra_cruising_states_next,
                            mu_repositioned_next, mu_matched_next, mu_cruising_next, 
                            lifted_reward, requests, vehicles, pickup_distances
                        ) = self.env.step(
                                policy, 
                                step, 
                                exploration=None
                            ) 
                        # Add dummy values to align the dimensions
                        dummy_rows = -1 * torch.ones(self.num_agents - ra_repositioned_states.shape[0], ra_repositioned_states.shape[1], device=self.device)
                        ra_repositioned_states = torch.cat((ra_repositioned_states, dummy_rows), dim=0)   
                        dummy_rows = -1 * torch.ones(self.num_agents - ra_matched_states.shape[0], ra_matched_states.shape[1], device=self.device)
                        ra_matched_states = torch.cat((ra_matched_states, dummy_rows), dim=0)
                        dummy_rows = -1 * torch.ones(self.num_agents - ra_cruising_states.shape[0], ra_cruising_states.shape[1], device=self.device)
                        ra_cruising_states = torch.cat((ra_cruising_states, dummy_rows), dim=0)      
                        dummy_rows = -1 * torch.ones(self.num_agents - vehicles.shape[0], vehicles.shape[1], device=self.device)
                        vehicles = torch.cat((vehicles, dummy_rows), dim=0)
                        dummy_rows = -1 * torch.ones(self.num_agents - pickup_distances.shape[0], pickup_distances.shape[1], device=self.device)
                        pickup_distances = torch.cat((pickup_distances, dummy_rows), dim=0) 
                        dummy_rows = -1 * torch.ones(max_num_requests - requests.shape[0], requests.shape[1], device=self.device)
                        requests = torch.cat((requests, dummy_rows), dim=0)
                        episode_ra_repositioned_states.append(ra_repositioned_states)
                        episode_ra_matched_states.append(ra_matched_states)
                        episode_ra_cruising_states.append(ra_cruising_states)
                        episode_mus_repositioned.append(mu_repositioned)
                        episode_mus_matched.append(mu_matched)
                        episode_mus_cruising.append(mu_cruising)
                        episode_lifted_rewards.append(lifted_reward.unsqueeze(0))
                        episode_requests.append(requests)
                        episode_vehicles.append(vehicles)
                        episode_pickup_distances.append(pickup_distances)
                        mu_available_next = mu_matched_next + mu_cruising_next
                        mu_available_next = normalize_distributions(mu_available_next, p=1)
                        episode_entropies.append(self._compute_entropy(mu_available_next))
                        if self.constraint_function is not None:
                            episode_constraint_violations.append(self._compute_constraint_violation(mu_available_next))
                    inference_times[i] = self.env.inference_time
                    # Postprocessing for visualization
                    # Add the final step to the list
                    dummy_rows = -1 * torch.ones(self.num_agents - ra_repositioned_states_next.shape[0], ra_repositioned_states_next.shape[1], device=self.device)
                    ra_repositioned_states_next = torch.cat((ra_repositioned_states_next, dummy_rows), dim=0)
                    dummy_rows = -1 * torch.ones(self.num_agents - ra_matched_states_next.shape[0], ra_matched_states_next.shape[1], device=self.device)
                    ra_matched_states_next = torch.cat((ra_matched_states_next, dummy_rows), dim=0)
                    dummy_rows = -1 * torch.ones(self.num_agents - ra_cruising_states_next.shape[0], ra_cruising_states_next.shape[1], device=self.device)
                    ra_cruising_states_next = torch.cat((ra_cruising_states_next, dummy_rows), dim=0)
                    episode_ra_repositioned_states.append(ra_repositioned_states_next)
                    episode_ra_matched_states.append(ra_matched_states_next)
                    episode_ra_cruising_states.append(ra_cruising_states_next)
                    visualization_ra_repositioned_states.append(torch.stack(episode_ra_repositioned_states).cpu().numpy())
                    visualization_ra_matched_states.append(torch.stack(episode_ra_matched_states).cpu().numpy())
                    visualization_ra_cruising_states.append(torch.stack(episode_ra_cruising_states).cpu().numpy())
                    episode_mus_repositioned.append(mu_repositioned_next)
                    episode_mus_matched.append(mu_matched_next)
                    episode_mus_cruising.append(mu_cruising_next)
                    visualization_mus_repositioned.append(torch.stack(episode_mus_repositioned).squeeze(1).cpu().numpy())
                    visualization_mus_matched.append(torch.stack(episode_mus_matched).squeeze(1).cpu().numpy())
                    visualization_mus_cruising.append(torch.stack(episode_mus_cruising).squeeze(1).cpu().numpy())
                    # Add dummy values to align the dimensions
                    lifted_reward = torch.zeros_like(lifted_reward)
                    episode_lifted_rewards.append(lifted_reward.unsqueeze(0))
                    visualization_lifted_rewards.append(torch.stack(episode_lifted_rewards).cpu().numpy())
                    # Compute entropy after the last step
                    mu_available_next = mu_matched_next + mu_cruising_next
                    mu_available_next = normalize_distributions(mu_available_next, p=1)
                    episode_entropies.append(self._compute_entropy(mu_available_next))
                    visualization_entropies.append(torch.stack(episode_entropies).cpu().numpy())
                    if self.constraint_function is not None:
                        episode_constraint_violations.append(self._compute_constraint_violation(mu_available_next))
                        visualization_constraint_violations.append(torch.stack(episode_constraint_violations).cpu().numpy())
                    visualization_requests.append(torch.stack(episode_requests).cpu().numpy())
                    visualization_vehicles.append(torch.stack(episode_vehicles).cpu().numpy())
                    visualization_pickup_distances.append(torch.stack(episode_pickup_distances).cpu().numpy())
                    del ( 
                        episode_ra_repositioned_states, episode_ra_matched_states, 
                        episode_ra_cruising_states, episode_mus_repositioned, 
                        episode_mus_matched, episode_mus_cruising, 
                        episode_lifted_rewards, episode_requests, 
                        episode_vehicles, episode_pickup_distances,
                        episode_entropies, episode_constraint_violations 
                    ) 
                    gc.collect()
                    np.save(self.results_dir / f'inference_times{self.evaluations_count}', inference_times)
                    np.save(self.results_dir / f"ra_repositioned_states{self.evaluations_count}", np.stack(visualization_ra_repositioned_states))
                    np.save(self.results_dir / f"ra_matched_states{self.evaluations_count}", np.stack(visualization_ra_matched_states))
                    np.save(self.results_dir / f"ra_cruising_states{self.evaluations_count}", np.stack(visualization_ra_cruising_states))
                    np.save(self.results_dir / f"mus_repositioned{self.evaluations_count}", np.stack(visualization_mus_repositioned))
                    np.save(self.results_dir / f"mus_matched{self.evaluations_count}", np.stack(visualization_mus_matched))
                    np.save(self.results_dir / f"mus_cruising{self.evaluations_count}", np.stack(visualization_mus_cruising))
                    np.save(self.results_dir / f"lifted_rewards{self.evaluations_count}", visualization_lifted_rewards)
                    np.save(self.results_dir / f"entropies{self.evaluations_count}", visualization_entropies)
                    if self.constraint_function is not None:
                        np.save(self.results_dir / f"constraint_violations{self.evaluations_count}", visualization_constraint_violations)
                    np.save(self.results_dir / f"requests{self.evaluations_count}", visualization_requests)
                    np.save(self.results_dir / f"vehicles{self.evaluations_count}", visualization_vehicles)
                    np.save(self.results_dir / f"pickup_distances{self.evaluations_count}", visualization_pickup_distances)
                    total_reward = np.stack(visualization_lifted_rewards).sum()
                    avg_matched = np.stack(visualization_mus_matched)[:, :-1, :].sum(axis=-1).mean()
                    avg_cruising = np.stack(visualization_mus_cruising)[:, :-1, :].sum(axis=-1).mean()
                    wandb.log({"avg_episode_reward": total_reward / (i + 1)})
                    wandb.log({"avg_step_reward": total_reward / ((i + 1) * horizon)})
                    print('=' * 15 + f"Episode {self.evaluations_count}" + '=' * 15)    
                    print(f"avg_step_reward: {total_reward / ((i + 1) * horizon)}")
                    print(f"avg_step_matched: {avg_matched}")
                    print(f"avg_step_cruising: {avg_cruising}")
                    total_entropy = np.stack(visualization_entropies).sum()
                    wandb.log({"avg_step_entropy": total_entropy / ((i + 1) * horizon)})
                    if self.constraint_function is not None:
                        min_constraint_violation = np.inf
                        num_constraint_violations = 0
                        for min_constraint_violations in visualization_constraint_violations:
                            min_constraint_violation = min(min_constraint_violation, min(min_constraint_violations))
                            num_constraint_violations += sum(min_constraint_violations < 0)
                        wandb.log({"min_constraint_violation": min_constraint_violation})  
                        wandb.log({"avg_episode_constraint_violations": num_constraint_violations / (i + 1)})  
                del ( 
                    visualization_ra_repositioned_states,
                    visualization_ra_matched_states, visualization_ra_cruising_states,
                    visualization_mus, visualization_mus_repositioned,
                    visualization_mus_matched, visualization_mus_cruising,
                    visualization_mf_matched_proportions, visualization_lifted_rewards,
                    visualization_requests, visualization_vehicles, visualization_pickup_distances,
                    visualization_constraint_violations, visualization_entropies
                )
                gc.collect()
        if train_matching_model:
            num_agents = ra_states.shape[0] 
            ra_states = torch.cat(train_ra_states, dim=0).to(self.device)
            mus_available = torch.cat(train_mus_available, dim=0).to(self.device)
            mus_available = mus_available.repeat_interleave(num_agents, dim=0)
            demands = torch.cat(train_demands, dim=0).to(self.device)
            demands = demands.repeat_interleave(num_agents, dim=0)
            matching_probs = torch.cat(train_matching_probs, dim=0).to(self.device)
            self.env.mf_matching.train(ra_states, mus_available, demands, matching_probs)
            del (
                train_ra_states, train_mus_available, train_demands, train_matching_probs,
                ra_states, mus_available, demands, matching_probs
            )
            gc.collect()

        return total_reward / n_repeats


    def _run_mfc(self, horizon: int, n_repeats: int=10):
        # initial_policy = self.optimizer.policy
        # _ = self._evaluate(horizon, initial_policy, n_repeats=n_repeats)
        opt_policy = self.optimizer.train(horizon)
        torch.save(opt_policy, self.policy_ckpt_dir / "policy_final.pt")
        _ = self._evaluate(horizon, opt_policy, n_repeats=n_repeats)


    def _run_mfrl(
        self,
        horizon: int,
        warmup_steps: int,
        n_episodes: int,
        n_repeats: int = 10,
    ):
        # self._warmup(warmup_steps, horizon)
        opt_policy = self.optimizer.policy
        best_policy = opt_policy
        best_reward = -torch.inf
        # torch.save(self.env.mf_matching, f"{self.mf_matching_ckpt_dir}/matching_model{self.evaluations_count}.pt")
        # torch.save(opt_policy, f"{self.policy_ckpt_dir}/policy{self.evaluations_count}.pt")
        # torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_best{self.evaluations_count}.pt")
        for _ in range(n_episodes):
            # prev_matching_model = deepcopy(self.env.mf_matching)
            avg_reward = self._evaluate(
                horizon,
                opt_policy,
                train_matching_model=True,
                n_repeats=n_repeats,
            )
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_policy = deepcopy(opt_policy)
                torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_best{self.evaluations_count}.pt")
                # torch.save(prev_matching_model, f"{self.mf_matching_ckpt_dir}/matching_model_best{self.evaluations_count}.pt")
            self.evaluations_count += 1
            # torch.save(self.env.mf_matching, f"{self.mf_matching_ckpt_dir}/matching_model{self.evaluations_count}.pt")
            opt_policy = self.optimizer.train(horizon)
            # torch.save(opt_policy, f"{self.policy_ckpt_dir}/policy{self.evaluations_count}.pt")
        # Checking if the final policy is the best
        avg_reward = self._evaluate(horizon, opt_policy, n_repeats=n_repeats)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_policy = deepcopy(opt_policy)
            torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_best{self.evaluations_count}.pt")
            # torch.save(self.env.mf_matching, f"{self.mf_matching_ckpt_dir}/matching_model_best{self.evaluations_count}.pt")
        # torch.save(self.env.mf_matching, f"{self.mf_matching_ckpt_dir}/matching_model_final.pt")
        torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_final.pt")
    

    def run(
        self,
        n_episodes: int,
        horizon: int,
        warmup_steps: int,
        n_repeats: int = 4,
    ) -> None:
        if self.exec_type == "train":
            if self.framework == "mfc":
                self._run_mfc(horizon, n_repeats)
            else:
                self._run_mfrl(
                    horizon,
                    warmup_steps,
                    n_episodes,
                    n_repeats
                )
        elif self.exec_type == "eval":
            policy_type = self.optimizer_cfg['type']
            if policy_type == 'mf_policy':
                policy_ckpt = self.optimizer_cfg['checkpoint']
                opt_policy = torch.load(policy_ckpt, map_location=self.device)
            elif policy_type == 'dummy_policy':
                act_dim = self.env.action_dim
                opt_policy = DummyPolicy(action_dim=act_dim, device=self.device)
            opt_policy.device = self.device
            _ = self._evaluate(horizon, opt_policy, n_repeats=n_repeats)
        elif self.exec_type == 'lp_eval':
            evaluate_benchmark(
                n_repeats,
                horizon,
                num_agents=self.num_agents, 
                demand_matrix=self.demand_matrix, 
                weight_matrix=self.weight_matrix,
                origin_destination_matrix=self.origin_destination_matrix,
                state_dim=self.state_dim,
                num_cells=self.num_cells,
                num_intervals=self.num_intervals,
                control_std=self.control_std,
                state_space=self.state_space,
                real_world_matching_cfg=self.real_world_matching_cfg,
                optimizer_cfg=self.optimizer_cfg,
                results_dir=self.results_dir,
                evaluations_count=self.evaluations_count,
                barrier_lambda=self.barrier_lambda,
                max_entropy_ratio=self.max_entropy_ratio,
            )