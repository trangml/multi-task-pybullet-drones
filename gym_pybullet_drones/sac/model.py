import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, action_space=None):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_size, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log = nn.Linear(hidden_dim, action_size)
        self.apply(self.init_weights)

        if action_space is not None:
            self.scaling = torch.FloatTensor((action_space.high - action_space.low) / 2)
            self.bias = torch.FloatTensor((action_space.high + action_space.low) / 2)
        else:
            self.scaling = torch.FloatTensor([1, 1])
            self.bias = torch.FloatTensor([0, 0])

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scaling + self.bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.scaling * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.scaling + self.bias
        return action, log_prob, mean

    def to(self, device):
        self.scaling = self.scaling.to(device)
        self.bias = self.bias.to(device)
        return super(Policy, self).to(device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim):
        super(QNetwork, self).__init__()
        # Define Q1 architecture and Q2 architecture
        self.linear1 = nn.Linear(state_size + action_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(state_size + action_size, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        # Q1 architecture
        x = torch.cat([state, action], 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        # Q2 architecture
        y = torch.cat([state, action], 2)
        y = F.relu(self.linear4(y))
        y = F.relu(self.linear5(y))
        y = self.linear6(y)

        return x, y
