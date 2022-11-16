from experiments.learning.rl_singleagent_optimal_incremental import train_agents
from experiments.learning.utils.second_order_opt import sample_ppo_params
from experiments.learning.utils.load_config import load_config

from omegaconf import OmegaConf
from omegaconf import DictConfig
import optuna
import logging
import sys


def objective(trial):
    ppo_params = sample_ppo_params(trial)
    cfg = load_config(__file__)
    # cfg = {**ppo_params, **drone_params}
    cfg = OmegaConf.merge(cfg, ppo_params)
    cfg["tag"] = f"short_timesteps_retain_policy_optuna_{trial.number}"
    reward = train_agents(cfg)
    return reward


if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study()

    study.optimize(objective, n_trials=50)

    print(study.best_params)
