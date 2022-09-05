import gym
import pickle


class NotVecNormalize(gym.Wrapper):
    """
    Normal Gym wrapper that replicates the functionality of Stable Baselines3's VecNormalize wrapper
    for non VecEnvs (i.e. `gym.Env`) in production.
    """

    def __init__(self, env, path):
        super().__init__(env)

        with open(path, "rb") as file_handler:
            self.vec_normalize = pickle.load(file_handler)

    def reset(self):
        observation = self.env.reset()
        return self.vec_normalize.normalize_obs(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.vec_normalize.normalize_obs(observation)
        reward = self.vec_normalize.normalize_reward(reward)
        return observation, reward, done, info
