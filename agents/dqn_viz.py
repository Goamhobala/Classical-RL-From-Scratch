# dqn_viz.py
import gymnasium as gym, torch
from agents.dqn_scratch import MLP   # or paste the class

env = gym.make("LunarLander-v3", render_mode="human")
model = MLP(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load("dqn2013.pt", map_location="cpu"))
model.eval()

for ep in range(5):
    state, _ = env.reset()
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        with torch.no_grad():
            action = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        state, r, terminated, truncated, _ = env.step(action)
        total += r
    print(f"episode {ep}: {total:.1f}")

env.close()