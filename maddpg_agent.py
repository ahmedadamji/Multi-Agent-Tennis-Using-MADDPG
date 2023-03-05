import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # How often to update the network weights.
UPDATE_TIMES = 5        # How many times to update the network weights per update.

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of parallel agents
        """
        # Initialize the agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size,  action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Update Frequency parameters
        self.update_every = UPDATE_EVERY
        self.update_times = UPDATE_TIMES
        # Records the number of time steps that have occured since the last update
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones, agent):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Incrementing the number of time steps passed
        self.t_step += 1
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # Update the network weights every update_every time steps
            if (self.t_step >= self.update_every):
                # Update the network weights update_times times
                for i in range(self.update_times):
                    # Sample a batch of experiences from the replay buffer
                    experiences = self.memory.sample()
                    # Learn from the experiences
                    self.learn(experiences, GAMMA, agent)
                # Reset the number of time steps passed
                self.t_step = 0

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        # Converting the state to a tensor
        states = torch.from_numpy(states).float().to(device)
        # Setting the network to evaluation mode
        self.actor_local.eval()
        # Initializing the actions tensor
        actions = np.zeros((self.num_agents, self.action_size))
        # Disabling gradient calculation
        with torch.no_grad():
            # Fetching the action of each parallel agent
            actions = self.actor_local(states).cpu().data.numpy()

        # Setting the network back to training mode
        self.actor_local.train()
        if add_noise:
            # Adding noise to the actions to encourage exploration
            actions += np.array(self.noise.sample())
        # Clipping the actions to the range [-1, 1]
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Unpacking the experiences
        states, actions, rewards, next_states, dones = experiences
        # Splitting the states and next_states into agent 1 and agent 2 states
        agent1_states = states[:, :self.state_size]
        agent2_states = states[:, self.state_size:]
        agent1_next_states = next_states[:, :self.state_size]
        agent2_next_states = next_states[:, self.state_size:]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        if agent == 0:
            # Fetching the actions of the first agent
            actions_next = self.actor_target(agent1_next_states)
            actions_next = torch.cat((actions_next, actions[:, self.action_size:]), dim=1)
        else:
            # Fetching the actions of the second agent
            actions_next = self.actor_target(agent2_next_states)
            actions_next = torch.cat((actions[:, :self.action_size], actions_next), dim=1)
        
        # Fetching the Q values of the next states for finding the Q targets
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        # The done flag is used to ensure that the Q_actual value is only added
        # to the reward if the episode has not terminated for any of the agents
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        # Computing the gradients
        critic_loss.backward()
        # Clipping the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        # Updating the weights
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss

        # Construct action prediction vector relative to each agent
        if agent == 0:
            # Fetching the actions of the first agent
            actions_pred = self.actor_local(agent1_states)
            actions_pred = torch.cat((actions_pred, actions[:, 2:]), dim=1)
        else:
            # Fetching the actions of the second agent
            actions_pred = self.actor_local(agent2_states)
            actions_pred = torch.cat((actions[:, :2], actions_pred), dim=1)

        # Compute the loss for optimizing the actor
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        # Computing the gradients
        actor_loss.backward()
        # Clipping the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        # Updating the weights
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update modelself.reset() parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.15):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
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

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
        self.seed = random.seed(seed)
    
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
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)