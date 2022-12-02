import os
import random
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def average_weights(models):
    """Average weights of models"""
    new_model = models[0].state_dict()
    for key in new_model.keys():
        for i in range(1, len(models)):
            new_model[key] += models[i].state_dict()[key]
        new_model[key] /= len(models)
    return new_model
