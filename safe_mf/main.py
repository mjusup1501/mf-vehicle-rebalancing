import os
import sys
import warnings
import yaml
import torch
from datetime import datetime
from safe_mf.utils.parser import parser
from pathlib import Path
import torch

import wandb
from safe_mf.alg.safe_mf_marl import SafeMFMARL

warnings.filterwarnings("ignore")
CONFIG_EXCLUDE_KEYS = ["logging"]

parser.add_argument("--config", type=str)

def main():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) 
    logdir = config['logdir'].split('/')[-1] if config['logdir'] is not None else None
    eval_logdir = config.get('eval_logdir', None)
    if eval_logdir is not None:
        eval_logdir = eval_logdir.split('/')[-1]
    input_folder = config['model']['input_data_path']
    if logdir is not None:
        config['logdir'] = logdir
    config['model']['input_data_path'] = input_folder
    if eval_logdir is not None:
        config['eval_logdir'] = eval_logdir
    run_id = Path(args.config).stem 
    if config["run_id"] is None:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = f"{run_id}_{now}"
    else:
        suffix = config["run_id"]
        run_id = f"{run_id}_{suffix}"
    wandb_id = f"{run_id}_{config['exec_type']}"
    if logdir is not None:
        logdir = Path(config["logdir"])
    exec_type=config.get("exec_type", "train")
    if exec_type == 'train':
        if logdir is not None:
            logdir = Path(config["logdir"]) / run_id
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
    if eval_logdir is not None:
        eval_logdir = Path(config["eval_logdir"]) / run_id
        os.makedirs(eval_logdir, exist_ok=True)

    wandb.init(
        project="mf-vehicle-rebalancing",
        id=wandb_id,
        dir=eval_logdir if eval_logdir is not None else logdir,
        entity=config.get('entity', 'user'),
        save_code=False,
        config=config,
        config_exclude_keys=CONFIG_EXCLUDE_KEYS,
        mode=config.get("logging", "disabled"),
    )

    if config['device'] == 'cuda:0' and torch.cuda.is_available():
        device = config["device"]
    else:
        device = "cpu"
    print(f"Using device: {device}")

    alg = SafeMFMARL(
            env_name=config.get("env_name", "vehicle-repositioning-simultaneous"),
            **config["model"], 
            device=device,
            logdir=logdir,
            eval_logdir=eval_logdir,
            exec_type=exec_type,
    )

    alg.run(**config["training"])

if __name__ == "__main__":
    main()
