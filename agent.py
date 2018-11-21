import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from memory import ReplayBuffer
from actor_critic import Actor, Critic


from torch.distributions import Normal

TRANSFER_RATE = 1e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTOR_LR = 1e-3
VALUE0_LR = 1e-3
VALUE1_LR = 1e-3

class Agent():
    def __init__(self, state_size, action_size, action_sigma=0.1, memory_size=1000000, batch=128, sigma=0.2, noise_clip=0.5, gamma = 0.99, update_frequency=2, seed=0):
        '''
        TD3 Agent
        :param state_size: State Dimension
        :param action_size: Action dimension
        :param action_sigma: standard deviation of the noise to be added to the action
        :param memory_size:
        :param batch:
        :param sigma: Standard deviation of the noise to be added to the target function (Chapter 5.3 of TD3 Paper)
        :param noise_clip: How much noise to allow
        :param gamma:
        :param update_frequency:
        :param seed:
        '''

        self.state_size = state_size
        self.action_size = action_size

        self.action_sigma = action_sigma
        self.sigma = sigma
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.seed = seed

        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.critic0 = Critic(self.state_size, self.action_size).to(device)
        #second Critic as described in the paper
        # https: // arxiv.org / pdf / 1802.09477.pdf
        self.critic1 = Critic(self.state_size, self.action_size).to(device)

        self.target_actor = Actor(self.state_size, self.action_size).to(device)
        self.target_critic0 = Critic(self.state_size, self.action_size).to(device)
        # second Critic as described in the paper
        # https: // arxiv.org / pdf / 1802.09477.pdf
        self.target_critic1 = Critic(self.state_size, self.action_size).to(device)

        self.memory = ReplayBuffer(memory_size, batch, seed=seed)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic0_optimizer = Adam(self.critic0.parameters(), lr=VALUE0_LR)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=VALUE1_LR)

        self.soft_update(self.actor, self.target_actor, 1)
        self.soft_update(self.critic0, self.target_critic0, 1)
        self.soft_update(self.critic1, self.target_critic1, 1)

    def act(self, state, epsilon=True):

        state = torch.from_numpy(np.asarray(state)).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state).cpu().data.numpy()
        self.actor.train()

        if epsilon:
            #if we want to inject some noise
            noise = np.random.normal(0, self.action_sigma, action.shape[0])
            action += noise

        return action

    def update(self, step):
        '''
        #https: // arxiv.org / pdf / 1802.09477.pdf
        the function is very similar to typical DDPG algorithm, except for
        1) we have 2 critics to update
        2) we take the min of the 2 values critics output
        3) Has modified Target network with noise injected into it (Chapter 5.3 of the paper)
        4) We delay updating the actor by certain steps

        :param step: how often to update the actor
        :return:
        '''

        state, action, reward, next_state, done = self.memory.sample()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        next_state_action = self.target_actor(next_state)

        #sample a random noise
        noise = Normal(torch.zeros(self.action_size), self.sigma).sample()
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip).to(device)

        next_state_action += noise

        target_Q0 = self.target_critic0(next_state, next_state_action)
        target_Q1 = self.target_critic1(next_state, next_state_action)
        target_Q  = torch.min(target_Q0, target_Q1)

        target_value = reward + self.gamma * target_Q * (1.0 - done)

        expected_Q0 = self.critic0(state, action)
        expected_Q1 = self.critic1(state, action)

        critic_0_loss = F.mse_loss(expected_Q0, target_value.detach())
        critic_1_loss = F.mse_loss(expected_Q1, target_value.detach())

        self.critic0_optimizer.zero_grad()
        critic_0_loss.backward()
        self.critic0_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic1_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss

        #as mentioned in the paper, we delay updating the actor network.


        if step % self.update_frequency == 0:

            actor_loss = self.critic0.forward(state, self.actor.forward(state))
            actor_loss = -actor_loss.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # ----------------------- update target networks ------------------- #
            self.soft_update(self.critic0, self.target_critic0, TRANSFER_RATE)
            self.soft_update(self.critic1, self.target_critic1, TRANSFER_RATE)
            self.soft_update(self.actor, self.target_actor, TRANSFER_RATE)

    def soft_update(self, local_model, target_model, tao):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tao * local_param.data + (1.0 - tao) * target_param.data)

    def add_to_memory(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)
