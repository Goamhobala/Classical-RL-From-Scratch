"""
Pure Actor-Critic with no PPO implementation from scratch. Some of the code is repeated here. That's intentional
because I want to get more reps in to get used to the patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import copy
from torch.distributions import Categorical
from agents.TrainLogger import TrainLogger

class MLP(nn.Module):
    # just a basic MLP
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
env_id = "LunarLander-v3"
env = gym.make(env_id, render_mode=None)
 
EPOCHS = 500
GAMMA = 0.99
LR = 3e-4
VALUE_COEF = 1
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

log = TrainLogger()

actor = MLP(obs_dim, n_actions).to(device)
critic = MLP(obs_dim, 1).to(device)

optimiser_actor = optim.Adam(actor.parameters(), lr=LR)
optimiser_critic = optim.Adam(critic.parameters(), lr=LR)
returns_log = []
for epoch in range(EPOCHS):
    state, _ = env.reset()
    terminated = truncated = False
    ep_return = 0.0
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    step = 0
    while not (terminated or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = Categorical(logits=actor(state_tensor)).sample().item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_obs.append(state)
        ep_actions.append(action)
        ep_rewards.append(reward)
        state = next_state
        
        ep_return += reward
        step += 1
    
    R = 0.0
    returns = []
    for r in reversed(ep_rewards):
        R = r + GAMMA * R
        returns.append(R)
    returns.reverse()
    
    b_states = torch.as_tensor(np.array(ep_obs), dtype=torch.float32, device=device)
    b_actions = torch.as_tensor(ep_actions, dtype=torch.int64, device=device)
    b_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    
    values = critic(b_states).squeeze(-1)
    
    advantages = (b_returns - values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    value_loss = F.mse_loss(values, b_returns)
    pg_loss = (-Categorical(logits=actor(b_states)).log_prob(b_actions) * advantages).mean()
    
    loss = pg_loss + VALUE_COEF * value_loss
    
    optimiser_actor.zero_grad()
    optimiser_critic.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM
    )
    optimiser_actor.step()
    optimiser_critic.step()
    
    ev = 1 - (b_returns - values.detach()).var() / (b_returns.var() + 1e-8)

    log.log(ep_return=ep_return,
            pg_loss=pg_loss.item(),
            v_loss=value_loss.item(),
            expl_var=ev.item())
    
    returns_log.append(ep_return)
    if epoch % 20 == 0:
        print(f"ep {epoch:>4} | avg20 {np.mean(returns_log[-20:]):7.1f} | "
              f"expl.var {ev.item():5.2f} | "
              f"steps {step}")

    


torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
               "a2c.pt")
log.plot("training.png")
log.save_csv("training.csv")
env.close()