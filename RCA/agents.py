from re import T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

class CQLAgent():
    def __init__(self, state_size, action_size, hidden_dim=100,
                 device='cpu', tau=1e-3, alpha=5.0, gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma

        self.net = DQN(state_size=state_size, action_size=action_size,
                       hidden_dim=hidden_dim).to(self.device)
        self.target_net = DQN(state_size=state_size, action_size=action_size,
                              hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.net.eval()
            with torch.no_grad():
                action_values = self.net(state)
            self.net.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        
        return action
    
    def learn(self, data):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states = data
        actions = actions.reshape((-1, 1))
        rewards = rewards.reshape((-1, 1))
        with torch.no_grad():
            Q_target_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma*Q_target_next)
        Q_a_s = self.net(states)     
        Q_expected = Q_a_s.gather(1, actions)

        cql_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_expected.mean()
        bellman_loss = F.mse_loss(Q_expected, Q_targets)

        total_loss = self.alpha*cql_loss + 0.5*bellman_loss

        total_loss.backward()
        clip_grad_norm_(self.net.parameters(), 1)
        self.optimizer.step()
        
        self.soft_update(self.net, self.target_net)

        return total_loss.detach().item(), cql_loss.detach().item(), bellman_loss.detach().item()

    def soft_update(self, q_net, target_net):
        for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(self.tau*q_param.data + (1-self.tau)*target_param.data)


class DQNAgent():
    def __init__(self, state_size, action_size, hidden_dim=100,
                 device='cpu', tau=1e-3, gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.net = DQN(state_size=state_size, action_size=action_size,
                       hidden_dim=hidden_dim).to(self.device)
        self.target_net = DQN(state_size=state_size, action_size=action_size,
                              hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.net.eval()
            with torch.no_grad():
                action_values = self.net(state)
            self.net.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        
        return action
    
    def learn(self, data):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states = data
        actions = actions.reshape((-1, 1))
        rewards = rewards.reshape((-1, 1))
        with torch.no_grad():
            Q_target_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma*Q_target_next)
        
        Q_expected = self.net(states).gather(1, actions)
        
        criterion = nn.SmoothL1Loss()
        bellman_loss = criterion(Q_expected, Q_targets)
        # for MSE loss, uncomment below
        #bellman_loss = F.mse_loss(Q_expected, Q_targets)
        
        total_loss = bellman_loss

        total_loss.backward()
        clip_grad_norm_(self.net.parameters(), 1)
        self.optimizer.step()

        self.soft_update(self.net, self.target_net)

        return total_loss.detach().item()
    
    def soft_update(self, q_net, target_net):
        for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(self.tau*q_param.data + (1-self.tau)*target_param.data)

    

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim):
        super(DQN, self).__init__()
        self.input_shape = state_size
        self.output_shape = action_size
        self.head = nn.Linear(self.input_shape, hidden_dim)
        self.ff_1 = nn.Linear(hidden_dim, hidden_dim)
        self.ff_2 = nn.Linear(hidden_dim, self.output_shape)

    def forward(self, x):

        x = F.relu(self.head(x))
        x = F.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out