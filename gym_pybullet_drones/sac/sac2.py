import gym
import numpy as np
import random
import itertools
import matplotlib
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

class SAC():
    def __init__(
            self,
            base_kwargs,

            env,
            policy,
            initial_exploration_policy,
            qf1,
            qf2,
            vf,
            pool,

            lr=3e-3,
            scale_reward=1,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,
    ):
        self._env = env
        self._policy = policy
        self._initial_exploration_policy = initial_exploration_policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._vf = vf
        self._pool = pool

        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior


        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()




class FourierBasis:
    def __init__(self, d, ranges, order=3):
        nterms = pow(order + 1.0, d)
        self.numTerms = int(nterms)
        self.order = order
        self.ranges = np.array(ranges)
        iter = itertools.product(range(order + 1), repeat=d)
        self.c = np.array([list(map(int, x)) for x in iter])

    def phi(self, features):
        basisFeatures = np.array([self.scale(features[i], i) for i in range(len(features))])
        return np.cos(np.pi * np.dot(self.c, basisFeatures))

    def scale(self, value, pos):
        if self.ranges[0, pos] == self.ranges[1, pos]:
            return 0.0
        else:
            return (value - self.ranges[0, pos]) / (self.ranges[1, pos] - self.ranges[0, pos])

def main():
    env = gym.make("CartPole-v0")
    env.np_random.seed(42)
    num_actions = env.action_space.n
    num_obs = env.observation_space.shape[0]

    num_episodes = 1000
    # step size
    alphas = [0.01]
    for alpha in alphas:
        epsilon = 0.01
        # Discount factor
        gamma = 0.9
        order = 3

        total_rewards = []
        x_axis = np.arange(0, num_episodes)
        # Initialize a q-function linear function approximation using the Fourier Basis
        lims = [[-5, -5, -5, -5], [5, 5, 5, 5]]
        q_function_1 = FourierBasis(num_obs, lims, order=order)
        weights_1 = np.zeros((num_actions, q_function_1.numTerms))
        q_function_2 = FourierBasis(num_obs, lims, order=order)
        weights_2 = np.zeros((num_actions, q_function_2.numTerms))
        for episode in range(num_episodes):
            timestep = 0
            state = env.reset()
            rewards = 0
            done = False
            while not done:
                # Select actions using a fixed stochastic policy
                # 0-right 1-down 2-left 3-up
                if np.random.random() > epsilon:
                    possible = np.dot(weights, q_function.phi(state))
                    action = np.argmax(possible)
                else:
                    action = np.random.randint(0, num_actions)
                next_state, reward, done, _ = env.step(action)
                pred = max(np.dot(weights, q_function.phi(next_state)))
                td_error = (
                    reward
                    + gamma * (pred)
                    - np.dot(weights, q_function.phi(state))[action]
                )
                # Update the q-function
                a1 = alpha * (td_error) * q_function.phi(state)
                weights[action] += a1
                state = next_state
                rewards += reward
                timestep = timestep + 1
            print(f"\nTotal reward: {rewards}")
            total_rewards.append(rewards)
        fig, ax = plt.subplots()
        ax.plot(x_axis, total_rewards, label="Total Rewards")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.set_title(r"Q-Learning for Cart-Pole")
        ax.legend(loc="best")
        plt.show()

if __name__ == "__main__":
    main()
