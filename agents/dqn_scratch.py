import gymnasium as gym
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import copy

env_id = "LunarLander-v3"
env = gym.make(env_id, render_mode=None)
obs, info = env.reset(seed=43)
print(obs)
EPOCHS=500


action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
print(terminated)

class EnvBuffer:
    def __init__(self, max_size, obs_dim):
        self.max_size = max_size
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros(max_size, dtype=np.int64)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.index = 0
        self.fill = 0
        
    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.index % self.max_size] = obs
        self.action[self.index % self.max_size] = action
        self.reward[self.index % self.max_size] = reward
        self.next_obs[self.index % self.max_size] = next_obs
        self.done[self.index % self.max_size] = done
        self.index += 1
        self.fill = min(self.fill + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0,self.fill, batch_size)
        return (self.obs[indices],
                self.action[indices],
                self.reward[indices],
                self.next_obs[indices],
                self.done[indices])
        

class MLP(nn.Module):
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
    


GAMMA, EPS_MIN, EPS_DECAY = 0.99, 0.05, 0.995
BATCH_SIZE, LEARN_START, TARGET_SYNC = 64, 1000, 1000

epsilon = 1.0
buffer = EnvBuffer(max_size=100_000, obs_dim=env.observation_space.shape[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(env.observation_space.shape[0], env.action_space.n).to(device)
target = copy.deepcopy(model)                      # needs: import copy
for p in target.parameters():
    p.requires_grad_(False)
optimiser = optim.Adam(model.parameters(), lr=5e-4)

step_count = 0
returns = []

for epoch in range(EPOCHS):
    state, _ = env.reset()
    terminated = truncated = False
    ep_return = 0.0

    while not terminated and not truncated:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = model(s).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.push(state, action, reward, next_obs, float(terminated))
        state = next_obs
        ep_return += reward
        step_count += 1

        #train
        if buffer.fill >= LEARN_START:
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(BATCH_SIZE)
            b_s  = torch.as_tensor(b_s,  device=device)
            b_a  = torch.as_tensor(b_a,  device=device)
            b_r  = torch.as_tensor(b_r,  device=device)
            b_ns = torch.as_tensor(b_ns, device=device)
            b_d  = torch.as_tensor(b_d,  device=device)

            q_taken = model(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                y = b_r + GAMMA * model(b_ns).max(dim=1).values * (1.0 - b_d)

            loss = F.mse_loss(q_taken, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if step_count % TARGET_SYNC == 0:
            target.load_state_dict(model.state_dict())

    returns.append(ep_return)
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    if epoch % 20 == 0:
        print(f"ep {epoch:>4} | eps {epsilon:.3f} | avg20 {np.mean(returns[-20:]):7.1f} | steps {step_count}")
    

torch.save(model.state_dict(), "dqn2013.pt")