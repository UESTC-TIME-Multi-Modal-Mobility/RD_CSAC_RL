# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, MLP
from torchrl.modules.distributions import TanhNormal
import numpy as np
from copy import deepcopy

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-20, max_log_std=20):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SAC:
    def __init__(self, cfg, observation_spec, action_spec, device):
        self.cfg = cfg
        self.device = device
        
        # Get dimensions
        if hasattr(observation_spec, 'observation'):
            state_dim = observation_spec['observation'].shape[-1]
        elif len(observation_spec.shape) > 0:
            state_dim = observation_spec.shape[-1]
        else:
            state_dim = observation_spec['observation'].shape[-1]
        action_dim = action_spec.shape[-1]
        
        # Store dimensions for later use
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks with improved architecture
        self.actor = ActorNetwork(state_dim, action_dim, ).to(device)
        self.critic1 = CriticNetwork(state_dim, action_dim, ).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim,).to(device)
        
        # Target networks
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        
        # Temperature parameter - start with a reasonable value
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2, device=device)))
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -float(action_dim)
        
        # Parameters
        self.gamma = getattr(cfg, 'gamma', 0.99)
        self.action_limit = getattr(cfg.actor, 'action_limit', 2.0)
        
        # Improved optimizers with different learning rates
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic.learning_rate)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=getattr(cfg, 'alpha_learning_rate', 3e-4))

    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            mu, log_std = self.actor(state)
            
            if deterministic:
                # For evaluation, use deterministic action
                action = torch.tanh(mu)
            else:
                # For training, sample from the policy
                std = log_std.exp()
                normal = torch.distributions.Normal(mu, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
            
            # Apply action limit
            action = action * self.action_limit
            
            return action.squeeze(0) if action.shape[0] == 1 else action

    def train(self, replay_buffer, batch_size=200, tau=0.005):
        """SAC training step with improved stability"""
        if len(replay_buffer) < batch_size:
            return {}, {}
            
        # Sample batch
        batch = replay_buffer.sample(batch_size).to(self.device)
        
        states = batch['observation'].to(self.device)
        actions = batch['action'].to(self.device) / self.action_limit  # Normalize actions
        rewards = batch['next', 'reward'].squeeze(-1).to(self.device)
        next_states = batch['next', 'observation'].to(self.device)
        dones = (
            batch['next', 'terminated'].squeeze(-1).to(torch.bool)
            | batch['next', 'truncated'].squeeze(-1).to(torch.bool)
        ).float().to(self.device)
        
        # ============ Update Critics ============
        with torch.no_grad():
            # Sample next actions using current policy
            next_mu, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mu, next_std)
            next_x_t = next_normal.rsample()
            next_actions = torch.tanh(next_x_t)
            
            # Calculate log probabilities with correction for tanh
            next_log_probs = next_normal.log_prob(next_x_t) - torch.log(1 - next_actions.pow(2) + 1e-6)
            next_log_probs = next_log_probs.sum(-1)
            
            # Target Q values
            next_q1 = self.critic1_target(next_states, next_actions).squeeze(-1)
            next_q2 = self.critic2_target(next_states, next_actions).squeeze(-1)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute target with entropy regularization
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
        
        # Current Q values
        q1 = self.critic1(states, actions).squeeze(-1)
        q2 = self.critic2(states, actions).squeeze(-1)
        
        # Critic losses
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        # Update critics
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)  # Gradient clipping
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)  # Gradient clipping
        self.critic2_optim.step()
        
        # ============ Update Actor ============
        # Resample actions for actor update
        mu, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        actions_new = torch.tanh(x_t)
        
        # Calculate log probabilities
        log_probs = normal.log_prob(x_t) - torch.log(1 - actions_new.pow(2) + 1e-6)
        log_probs = log_probs.sum(-1)
        
        # Q values for new actions
        q1_new = self.critic1(states, actions_new).squeeze(-1)
        q2_new = self.critic2(states, actions_new).squeeze(-1)
        q_min = torch.min(q1_new, q2_new)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q_min).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
        self.actor_optim.step()
        
        # ============ Update Temperature ============

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Update temperature
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        # ============ Soft Update Target Networks ============
        self._soft_update(self.critic1_target, self.critic1, tau)
        self._soft_update(self.critic2_target, self.critic2, tau)
        
        # Return loss info
        loss_info = {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
        }
        
        reward_info = {
            'reward_mean': rewards.mean().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'log_prob_mean': log_probs.mean().item(),
        }
        
        return loss_info, reward_info

    def _soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
