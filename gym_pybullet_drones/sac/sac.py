import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gym_pybullet_drones.sac.experience_replay import ReplayMemory, HERBuffer
from gym_pybullet_drones.sac.model import QNetwork, Policy


class Pytorch_SAC:
    def __init__(
        self,
        env,
        num_inputs,
        action_space,
        batch_size=32,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        hidden_size=256,
        lr_actor=1e-4,
        lr_critic=1e-3,
        lr_policy=1e-3,
        target_update_interval=1,
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_policy = lr_policy

        self.target_update_interval = target_update_interval

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(
            self.device
        )
        # Update the target network to be the critic network
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy = Policy(num_inputs, action_space.shape[0], hidden_size, action_space).to(
            self.device
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)

        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, updates):
        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_states_actions, next_states_log_probs, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_states_actions)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_states_log_probs
            )
            next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Update policy
        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic)
        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, total_episodes=10000, log_interval=100):
        # for each episode
        rewards_history = []
        updates = 0
        ave_reward = 0
        for i in range(total_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            # select an action and observe new state
            states = []
            next_states = []
            actions = []
            rewards = []
            dones = []
            # go through all the steps of the episode
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                total_reward += reward[0]
            ave_reward += total_reward
            if i % log_interval == 0 and i is not 0:
                print(
                    "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                        i, total_reward, ave_reward / log_interval
                    )
                )
                rewards_history.append(ave_reward / log_interval)
                ave_reward = 0
            # now go through and add transitions to the replay buffer
            for step in range(len(states)):
                self.memory.push(
                    states[step], actions[step], rewards[step], next_states[step], dones[step]
                )
                # once done, sample minibatch from memory and update
                if len(self.memory) > self.batch_size:
                    updates += 1
                    critic_1_loss, critic_2_loss, policy_loss = self.update(updates)
                # #### HINDSIGHT EXPERIENCE REPLAY ####
                # # sample additional goals from the states
                # for goal in goals:
                #     # get supplementary reward
                #     # r' = r(s, a, g')
                #     # Store transition in memory
                #     self.memory.push(state, action, reward, goal, done)
        return rewards_history

    # Save model parameters
    def save(self, filename):
        print("Saving models to {}".format(filename))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            },
            filename,
        )

    # Load model parameters
    def load(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
