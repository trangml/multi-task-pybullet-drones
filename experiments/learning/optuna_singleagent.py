from experiments.learning.rl_singleagent import train_loop
from experiments.learning.utils.hyperparams_opt import sample_ppo_params
from experiments.learning.utils.hyperparams_opt import sample_drone_params
from experiments.learning.utils.load_config import load_config

from omegaconf import OmegaConf
from omegaconf import DictConfig
import optuna


def objective(trial):
    ppo_params = sample_ppo_params(trial)
    drone_params = sample_drone_params(trial)
    cfg = load_config(__file__)
    #cfg = {**ppo_params, **drone_params}
    cfg = OmegaConf.merge(cfg, ppo_params, drone_params)
    cfg["tag"] = f"optuna_{trial.number}"
    reward = train_loop(cfg)
    return reward


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print(study.best_params)