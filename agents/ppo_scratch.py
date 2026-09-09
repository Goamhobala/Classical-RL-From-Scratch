"""
PPO implemenation from scratch with actor critic. Some of the code here is repeated on purpose because I'm 
trying to get more reps in to get used to the pattern
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Normal
from TrainLogger import TrainLogger

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
    
class PPOObjective(nn.Module):
    def __init__(self, clip_epsilon=0.2, ):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        
    def forward(self, logp_new, logp_old, advantages):
        ratio = torch.exp(logp_new - logp_old) # log(A) - log(B) = log(A/B)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate = torch.min(ratio * advantages, clip_ratio * advantages)
        # mean over the minibatch, and NEGATED: we want to maximise the surrogate but
        # the optimiser minimises whatever we hand it.
        return -surrogate.mean()
       

env_id = "LunarLander-v3"
env = gym.make(env_id, render_mode="human", continuous=True)
 
TOTAL_UPDATES = 500
GAMMA = 0.99
GAE_LAMBDA = 0.95
MINIBATCH_SIZE = 64
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.0
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048
CLIP_EPS = 0.2
UPDATE_EPOCHS = 10


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

log = TrainLogger()

actor = MLP(obs_dim, act_dim).to(device)
critic = MLP(obs_dim, 1).to(device)

log_std = nn.Parameter(torch.zeros(act_dim, device=device))

ppo_objective = PPOObjective(clip_epsilon=CLIP_EPS)

optimiser = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()) + [log_std],
    lr=LR,
)


state, _ = env.reset()
ep_return = 0.0
returns_log = []

# rollout loop
for update in range(TOTAL_UPDATES):

    # fresh buffers every update
    buf_obs     = torch.zeros(ROLLOUT_STEPS, obs_dim, device=device)
    buf_actions = torch.zeros(ROLLOUT_STEPS, act_dim, device=device)
    buf_logp    = torch.zeros(ROLLOUT_STEPS, device=device)
    buf_rewards = torch.zeros(ROLLOUT_STEPS, device=device)
    buf_dones   = torch.zeros(ROLLOUT_STEPS, device=device)
    buf_values  = torch.zeros(ROLLOUT_STEPS, device=device)

    completed = []

    for t in range(ROLLOUT_STEPS):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            # no_grad because nothing here is trained 
            mean = actor(state_t)
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1) 
            value = critic(state_t).squeeze(-1)

        env_action = np.clip(action.cpu().numpy(), -1.0, 1.0)
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated # merge into one flag

        buf_obs[t]     = state_t
        buf_actions[t] = action # we store the unclipped version
        buf_logp[t]    = log_prob
        buf_values[t]  = value
        buf_rewards[t] = float(reward)
        buf_dones[t]   = float(done)

        ep_return += reward
        state = next_state

        if done:
            completed.append(ep_return)
            ep_return = 0.0
            state, _ = env.reset()

    with torch.no_grad():
        next_value = critic(
            torch.as_tensor(state, dtype=torch.float32, device=device)
        ).squeeze(-1)

# Im using GAE for updating the advantage function. This is not technically part of PPO, it's a different paper. 
# An alternative would be MC
    advantages = torch.zeros_like(buf_rewards)
    last_gae = 0.0
    # gae_1 = buf_rewards[0] + GAMMA * buf_values[1] - buf_values[0]
    for t in reversed(range(ROLLOUT_STEPS)):

        next_nonterminal = 1.0 - buf_dones[t]
        next_val = next_value if t == ROLLOUT_STEPS - 1 else buf_values[t + 1]
        delta = buf_rewards[t] + GAMMA * next_val * next_nonterminal - buf_values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
        advantages[t] = last_gae

    # the critic's regression target
    returns = advantages + buf_values
    b_inds = np.arange(ROLLOUT_STEPS)
    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, ROLLOUT_STEPS, MINIBATCH_SIZE):
            minibatch_idx = b_inds[start:start + MINIBATCH_SIZE]
            minibatch_obs = buf_obs[minibatch_idx]

            dist = Normal(actor(minibatch_obs), log_std.exp())
            logp_new = dist.log_prob(buf_actions[minibatch_idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            new_values = critic(minibatch_obs).squeeze(-1)

            #standardise the minibatch advantage
            minibatch_adv = advantages[minibatch_idx]
            minibatch_adv = (minibatch_adv - minibatch_adv.mean()) / (minibatch_adv.std() + 1e-8)

            pg_loss = ppo_objective(logp_new, buf_logp[minibatch_idx], minibatch_adv)
            value_loss = F.mse_loss(new_values, returns[minibatch_idx])

            loss = pg_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()) + [log_std],
                MAX_GRAD_NORM,
            )
            optimiser.step()


    # ---------------------------------------------------------- stage 5: log and report
    ev = 1 - (returns - buf_values).var() / (returns.var() + 1e-8)

    if completed:
        returns_log.extend(completed)
    avg20 = float(np.mean(returns_log[-20:])) if returns_log else 0.0
    upd_ret = float(np.mean(completed)) if completed else avg20

    log.log(ep_return=upd_ret,
            pg_loss=pg_loss.item(),
            v_loss=value_loss.item(),
            entropy=entropy.item(),
            expl_var=ev.item(),
            std=log_std.exp().mean().item())

    if update % 10 == 0:
        print(f"upd {update:>4} | steps {(update + 1) * ROLLOUT_STEPS:>8} | "
              f"avg20 {avg20:7.1f} | expl.var {ev.item():5.2f} | "
              f"std {log_std.exp().mean().item():4.2f} | eps {len(completed)}")

torch.save({"actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "log_std": log_std.detach().cpu()}, "ppo.pt")
log.plot("ppo_training.png")
log.save_csv("ppo_training.csv")
env.close()
