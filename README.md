# Mean-Field Reinforcement Learning for Ride-Sourcing Vehicle Rebalancing under Accessibility Constraint

To install required packages, run `pip install -e .`

In `configs` directory you can find training and evaluation config files for MFRL and MFRL algorithms.
In `checkpoints` directory you can find associated policy checkpoints.

To train an experiment you must modify the config file:
1. Set a *logdir* field if you want to overwrite the default log directory
2. You must provide the demand matrix, origin-destination matrix and weight matrix (indicates green areas, rivers etc.) and modify *input_data_path* in a config file.
3. Optionally modify other fields, including *entity*, for wandb logging
4. Execute `python safe_mf/main.py --config path/to/config.yaml`

To evaluate an experiment:
1. Set a *eval_logdir* field if you want to overwrite the default log directory
2. You must provide the demand matrix, origin-destination matrix and weight matrix (indicates green areas, rivers etc.) and modify *input_data_path* in a config file.
3. Optionally modify other fields, including *entity*, for wandb logging
4. For MFRL and MFC set *logdir* to the policy where training model is stored
5. Execute `python safe_mf/main.py --config path/to/config.yaml`