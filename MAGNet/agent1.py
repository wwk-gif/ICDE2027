"""
Genuine Multi-Agent Reinforcement Learning Controller for MAGNet.

Each slice has its own autonomous agent with:
  - A neural network policy (Actor) that maps observations → actions
  - A value network (Critic) for state-value estimation
  - Inter-agent communication via multi-head attention
  - Online A2C (Advantage Actor-Critic) learning

This replaces the previous "fake" multi-agent system where each "agent"
was just a 3-number vector perturbed by random noise.

Architecture:
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Agent 0  │  │ Agent 1  │  │ Agent 2  │   (one per slice)
  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │
  │ │ObsEnc│ │  │ │ObsEnc│ │  │ │ObsEnc│ │
  │ └──┬───┘ │  │ └──┬───┘ │  │ └──┬───┘ │
  │ ┌──┴───┐ │  │ ┌──┴───┐ │  │ ┌──┴───┐ │
  │ │Actor │ │  │ │Actor │ │  │ │Actor │ │
  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │
  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │
  │ │Critic│ │  │ │Critic│ │  │ │Critic│ │
  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
            ┌─────────┴──────────┐
            │ Cross-Agent        │
            │ Attention Module   │  (inter-agent communication)
            └─────────┬──────────┘
                      │
              ┌───────┴───────┐
              │  Consensus    │
              │  Aggregation  │  → global loss weights
              └───────────────┘
"""

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Neural Network Modules ====================

class ObservationEncoder(nn.Module):
    """Encodes per-slice observations into a compact state (lightweight MLP, no GRU)."""
    def __init__(self, obs_dim, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.fc(obs)


class ActorNetwork(nn.Module):
    """Policy network (lightweight): outputs mean and log-std for Gaussian actions."""
    def __init__(self, state_dim, action_dim=3, hidden_dim=16):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state):
        shared = self.net(state)
        mean = self.mean_head(shared)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample(self, state):
        """
        Sample action from Gaussian policy.
        Returns: action, log_prob, entropy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterized
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(self, state, action):
        """Evaluate log_prob and entropy for a given action."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value network (lightweight): estimates state value V(s)."""
    def __init__(self, state_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class CrossAgentAttention(nn.Module):
    """
    Lightweight multi-head attention for inter-agent communication.

    Each agent attends to all others to coordinate loss weights across slices.
    Reduced dims + single head for speed while keeping genuine multi-agent coordination.
    """
    def __init__(self, state_dim=16, dropout=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.q_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.k_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.out_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, agent_states):
        """
        agent_states: [n_agents, state_dim]
        Returns: [n_agents, state_dim] with cross-agent information
        """
        n_agents, D = agent_states.shape
        q = self.q_proj(agent_states)
        k = self.k_proj(agent_states)
        v = self.v_proj(agent_states)

        scale = D ** -0.5
        attn = F.softmax((q @ k.T) * scale, dim=-1)
        attn = self.dropout(attn)
        out = self.out_proj(attn @ v)

        return agent_states + out  # residual


# ==================== Per-Agent RL Agent ====================

class RLAgent(nn.Module):
    """
    A single RL agent controlling one slice (lightweight actor-critic).
    No GRU — state is encoded from current observation only.
    """
    def __init__(self, obs_dim, state_dim=16, action_dim=3):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.obs_encoder = ObservationEncoder(obs_dim, state_dim)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim=16)
        self.critic = CriticNetwork(state_dim, hidden_dim=16)

    def get_state(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.obs_encoder(obs).squeeze(0)

    def act(self, state, deterministic=False):
        if deterministic:
            mean, _ = self.actor.forward(state.unsqueeze(0))
            action = mean.squeeze(0)
            log_prob = torch.tensor(0.0)
        else:
            action, log_prob, _ = self.actor.sample(state.unsqueeze(0))
            action = action.squeeze(0)
        value = self.critic(state.unsqueeze(0)).squeeze(0)
        return action, log_prob, value

    def evaluate(self, state, action):
        log_prob, entropy = self.actor.evaluate(state.unsqueeze(0), action.unsqueeze(0))
        value = self.critic(state.unsqueeze(0)).squeeze(0)
        return log_prob.squeeze(0), entropy.squeeze(0), value

    def evaluate_batch(self, states, actions):
        log_prob, entropy = self.actor.evaluate(states, actions)
        value = self.critic(states)
        return log_prob, entropy, value


# ==================== Multi-Agent Controller ====================

class MultiAgentController:
    """
    Genuine multi-agent reinforcement learning controller.

    Manages N agents (one per slice), coordinates them via attention,
    and updates policies using A2C with a cooperative reward signal.

    Features:
      - Neural network policies (actor-critic) per agent
      - Cross-agent attention for coordination
      - Online learning with experience replay
      - Observation encoding with temporal context
      - Entropy regularization for exploration
    """

    def __init__(
        self,
        n_agents=3,
        action_dim=3,
        obs_dim=3,
        state_dim=16,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.95,
        entropy_coef=0.01,
        device='cpu',
    ):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Create lightweight agents (one per slice, each with own actor-critic)
        self.agents = nn.ModuleList([
            RLAgent(obs_dim, state_dim, action_dim)
            for _ in range(n_agents)
        ])

        # Cross-agent communication (genuine multi-agent coordination)
        self.attention = CrossAgentAttention(state_dim)

        # Optimizers
        actor_params = []
        critic_params = []
        for agent in self.agents:
            actor_params += list(agent.obs_encoder.parameters())
            actor_params += list(agent.actor.parameters())
            critic_params += list(agent.critic.parameters())
        actor_params += list(self.attention.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic)

        # Stores for online learning
        self.states_buffer = []       # list of [n_agents, state_dim]
        self.actions_buffer = []      # list of [n_agents, action_dim]
        self.log_probs_buffer = []    # list of [n_agents]
        self.values_buffer = []       # list of [n_agents]
        self.rewards_buffer = []      # list of [n_agents]
        self.dones_buffer = []        # list of bool

        # For observation history
        self.obs_history = [deque(maxlen=5) for _ in range(n_agents)]

        # Action scaling (map from [-1,1] to [w_min, w_max])
        self.w_min = 0.3
        self.w_max = 2.0

        # Mode
        self.train_mode = True

        # Store current state and prior action for consistency
        self.current_states = None
        self.current_actions_raw = np.ones((n_agents, action_dim)) * 0.0
        self.last_actions = np.ones((n_agents, action_dim)) * 0.5
        self.prev_ari = None
        self.episode_count = 0

    def to(self, device):
        """Move all modules to device."""
        self.device = device
        self.agents = self.agents.to(device)
        self.attention = self.attention.to(device)
        return self

    def train(self):
        self.train_mode = True
        for agent in self.agents:
            agent.train()

    def eval(self):
        self.train_mode = False
        for agent in self.agents:
            agent.eval()

    def _build_observation(self, agent_idx, per_slice_errors=None, ari=None, loss_val=None):
        """
        Build observation vector for a specific agent.

        Each observation includes:
          - Normalized reconstruction error for this slice
          - Current ARI (global, shared)
          - Accumulated loss
          - (Optional) previous action
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Feature 0: per-slice error (normalized)
        if per_slice_errors is not None and agent_idx < len(per_slice_errors):
            obs[0] = np.clip(per_slice_errors[agent_idx] / 10.0, 0, 10)

        # Feature 1: ARI or reward signal
        if ari is not None:
            obs[1] = np.clip(ari, 0, 1)
        else:
            obs[1] = 0.0

        # Feature 2: normalized loss value
        if loss_val is not None:
            obs[2] = np.clip(abs(float(loss_val)) / 10.0, 0, 10)

        # Add to history and use averaged observation
        self.obs_history[agent_idx].append(obs)
        if len(self.obs_history[agent_idx]) > 0:
            avg_obs = np.mean(list(self.obs_history[agent_idx]), axis=0)
        else:
            avg_obs = obs

        return torch.from_numpy(avg_obs).float().to(self.device)

    def select_actions(self):
        """
        Each agent observes its state, communicates, and proposes actions.

        Returns:
            actions: [n_agents, action_dim] in range [w_min, w_max]
        """
        # If no current_states (first call), use simple warm-up actions
        if self.current_states is None:
            actions_raw = np.random.uniform(-0.5, 0.5, size=(self.n_agents, self.action_dim))
            self.current_actions_raw = actions_raw
            scaled = self._scale_actions(actions_raw)
            self.last_actions = scaled
            return scaled.copy()

        with torch.no_grad():
            # Cross-agent communication
            communicated_states = self.attention(self.current_states)  # [n_agents, D]

            actions_raw = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
            log_probs = np.zeros(self.n_agents, dtype=np.float32)
            values = np.zeros(self.n_agents, dtype=np.float32)

            for i, agent in enumerate(self.agents):
                state_i = communicated_states[i]
                if self.train_mode:
                    action, log_prob, value = agent.act(state_i, deterministic=False)
                else:
                    action, log_prob, value = agent.act(state_i, deterministic=True)

                actions_raw[i] = action.cpu().numpy()
                log_probs[i] = log_prob.cpu().item()
                values[i] = value.cpu().item()

            self.current_actions_raw = actions_raw

            # Store for learning
            if self.train_mode:
                self.states_buffer.append(self.current_states.cpu().numpy().copy())
                self.actions_buffer.append(actions_raw.copy())
                self.log_probs_buffer.append(log_probs.copy())
                self.values_buffer.append(values.copy())

            # Scale actions from (-1, 1) to [w_min, w_max]
            scaled = self._scale_actions(actions_raw)
            self.last_actions = scaled
            return scaled.copy()

    def observe(self, per_slice_obs):
        """
        Receive per-slice observations to build states for next action selection.

        Args:
            per_slice_obs: array of shape [n_agents] with per-slice metrics
        """
        if per_slice_obs is None:
            return

        per_slice_obs = np.asarray(per_slice_obs)

        states = []
        for i in range(self.n_agents):
            obs_i = self._build_observation(
                i,
                per_slice_errors=per_slice_obs,
                ari=self.prev_ari if hasattr(self, 'prev_ari') else None,
                loss_val=per_slice_obs[i] if i < len(per_slice_obs) else 0.0,
            )
            state_i = self.agents[i].get_state(obs_i)
            states.append(state_i)

        with torch.no_grad():
            states_tensor = torch.stack(states)  # [n_agents, state_dim]
        self.current_states = states_tensor

    def update(self, rewards, evaluate_fn=None):
        """
        Update all agent policies using batched A2C (vectorized for speed).

        Args:
            rewards: scalar or array of shape [n_agents]
            evaluate_fn: (deprecated) kept for backward compatibility
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.size == 1:
            rewards = np.repeat(rewards, self.n_agents)

        self.rewards_buffer.append(rewards.copy())
        self.dones_buffer.append(False)

        # Only update after accumulating enough transitions (every 5 steps)
        if len(self.rewards_buffer) < 5:
            return

        n_steps = len(self.rewards_buffer)

        # --- Batch all data ---
        # states: [n_steps, n_agents, state_dim] -> [n_steps * n_agents, state_dim]
        states_t = torch.from_numpy(np.stack(self.states_buffer)).float().to(self.device)
        actions_t = torch.from_numpy(np.stack(self.actions_buffer)).float().to(self.device)
        values_t = torch.from_numpy(np.stack(self.values_buffer)).float().to(self.device)
        rewards_buf = np.stack(self.rewards_buffer)

        NS, NA, SD = states_t.shape  # n_steps, n_agents, state_dim
        states_flat = states_t.reshape(NS * NA, SD)
        actions_flat = actions_t.reshape(NS * NA, self.action_dim)
        values_flat = values_t.reshape(NS * NA)

        # --- Compute returns (reward-to-go) ---
        returns = np.zeros((NS, NA), dtype=np.float32)
        running_return = np.zeros(NA, dtype=np.float32)
        for t in range(NS - 1, -1, -1):
            running_return = rewards_buf[t] + self.gamma * running_return
            returns[t] = running_return
        returns_t = torch.from_numpy(returns).float().to(self.device).reshape(NS * NA)
        advantages_t = returns_t - values_flat
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # --- Batched cross-agent attention ---
        # Process each timestep's agents as a batch
        communicated_flat = []
        for t in range(NS):
            comm_t = self.attention(states_t[t])  # [n_agents, state_dim]
            communicated_flat.append(comm_t)
        communicated_flat = torch.cat(communicated_flat, dim=0)  # [n_steps * n_agents, state_dim]

        # --- Batched evaluate: all agents together ---
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0

        for i, agent in enumerate(self.agents):
            # Each agent processes all timesteps at once
            agent_mask = torch.arange(NA, device=self.device) == i
            idxs = torch.nonzero(
                agent_mask.repeat(NS)
            ).squeeze(-1)

            if idxs.numel() == 0:
                continue

            states_i = communicated_flat[idxs]     # [n_steps, state_dim]
            actions_i = actions_flat[idxs]          # [n_steps, action_dim]
            adv_i = advantages_t[idxs]              # [n_steps]
            ret_i = returns_t[idxs]                 # [n_steps]

            # Evaluate in batch
            log_prob, entropy, value = agent.evaluate_batch(states_i, actions_i)

            actor_loss_total += -(log_prob * adv_i.detach()).mean() - self.entropy_coef * entropy.mean()
            critic_loss_total += F.mse_loss(value, ret_i)
            entropy_total += entropy.mean()

        # Average over agents
        actor_loss = actor_loss_total / self.n_agents
        critic_loss = critic_loss_total / self.n_agents

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self._actors_parameters(), 1.0)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 1.0)
        self.critic_optimizer.step()

        self._clear_buffers()

    def _actors_parameters(self):
        """Get all parameters for actor optimization."""
        params = []
        for agent in self.agents:
            params += list(agent.obs_encoder.parameters())
            params += list(agent.actor.parameters())
        params += list(self.attention.parameters())
        return params

    def _clear_buffers(self):
        self.states_buffer = []
        self.actions_buffer = []
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []

    def _scale_actions(self, actions_raw):
        """Scale actions from [-1, 1] to [w_min, w_max]."""
        actions_raw = np.asarray(actions_raw)
        # Map tanh output (-1, 1) → (w_min, w_max)
        scaled = self.w_min + (self.w_max - self.w_min) * (actions_raw + 1.0) / 2.0
        return np.clip(scaled, self.w_min, self.w_max)

    def _unscale_actions(self, actions_scaled):
        """Reverse the scaling for critic evaluation."""
        actions_scaled = np.asarray(actions_scaled)
        raw = 2.0 * (actions_scaled - self.w_min) / (self.w_max - self.w_min) - 1.0
        return np.clip(raw, -1.0, 1.0)

    def save(self, path):
        torch.save({
            'agents': self.agents.state_dict(),
            'attention': self.attention.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(checkpoint['agents'])
        self.attention.load_state_dict(checkpoint['attention'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
