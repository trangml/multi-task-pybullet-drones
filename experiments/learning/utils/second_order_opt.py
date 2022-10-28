from typing import Any, Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from torch import nn as nn


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    gradient_weight = trial.suggest_categorical(
        "gradient_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    hessian_approx = trial.suggest_categorical(
        "hessian_approx", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    return {
        "ppo": {"gradient_weight": gradient_weight, "hessian_approx": hessian_approx}
    }


HYPERPARAMS_SAMPLER = {
    "ppo": sample_ppo_params,
}
