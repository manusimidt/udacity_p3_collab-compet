import numpy as np
import random
from collections import namedtuple, deque

import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

"""
For some reason the pytorch function .to(device) takes forever to execute.
Probably some issue between my cuda version and the pytorch version (0.4.0) that unityagents requires
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   59C    P0    28W / 130W |    679MiB /  5941MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class Agent:
    """ This class represents the reinforcement learning agent """

    def __init__(self, state_size: int, action_size: int,
                 gamma: float = 0.99, lr_actor: float = 0.001, lr_critic: float = 0.003,
                 weight_decay: float = 0.0001, tau: float = 0.001,
                 buffer_size: int = 100000, batch_size: int = 64):
        """
        :param state_size: how many states does the agent get as input (input size of neural networks)
        :param action_size: from how many actions can the agent choose
        :param gamma: discount factor
        :param lr_actor: learning rate of the actor network
        :param lr_critic: learning rate of the critic network
        :param weight_decay:
        :param tau: soft update parameter
        :param buffer_size: size of replay buffer
        :param batch_size: size of learning batch (mini-batch)
        """
        self.tau = tau
        self.action_size = action_size
        self.state_size = state_size
        pass

    def step(self, experience: tuple):
        pass

    def act(self, state, add_noise: bool = True):
        return np.clip(np.random.randn(2, self.action_size), 1, -1)

    def learn(self, experiences):
        pass

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class AgentDuo:
    """ This class stores the two agents and their common replay buffer """

    def __init__(self, agent1: Agent, agent2: Agent, buffer_size: int, batch_size: int):
        self.agent1 = agent1
        self.agent2 = agent2


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# Copied from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
