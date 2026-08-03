import argparse
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents import QLearningAgent, SARSAAgent
from discretize import Discretizer

# CartPole-v1 observation: [cart position, cart velocity, pole angle, pole angular velocity]
# Position/angle bounds are the episode-termination limits; velocity bounds are
# unbounded in theory so we clip to a practical range. Tune bins to trade off
# state-space size vs. resolution.
DISCRETIZE_BOUNDS = {
    "low": [-2.4, -3.0, -0.2095, -3.5],
    "high": [2.4, 3.0, 0.2095, 3.5],
    "bins": [6, 6, 12, 12],
}

AGENTS = {
    "sarsa": SARSAAgent,
    "qlearning": QLearningAgent,
}


def make_agent(name, n_actions, **kwargs):
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}, choose from {list(AGENTS)}")
    discretizer = Discretizer(**DISCRETIZE_BOUNDS)
    agent = AGENTS[name](n_states=discretizer.shape, n_actions=n_actions, **kwargs)
    return agent, discretizer


def train(agent_name, n_episodes=2000, log_dir=None, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent, discretizer = make_agent(agent_name, env.action_space.n)
    writer = SummaryWriter(log_dir) if log_dir else None

    episode_rewards = []
    recent_rewards = deque(maxlen=100)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = discretizer(obs)
        action = agent.select_action(state)

        total_reward = 0.0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretizer(next_obs)
            next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)

        if writer:
            writer.add_scalar("reward", total_reward, episode)
            writer.add_scalar("epsilon", agent.epsilon, episode)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:5d} | "
                f"avg reward (last 100): {np.mean(recent_rewards):6.1f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    if writer:
        writer.close()

    return agent, episode_rewards


def plot_rewards(rewards, window=50, title="Training reward"):
    rewards = np.array(rewards)
    plt.plot(rewards, alpha=0.3, label="raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(len(rewards) - len(smoothed), len(rewards)),
            smoothed,
            label=f"{window}-ep moving avg",
        )
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=list(AGENTS), help="Which algorithm to train")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log dir")
    args = parser.parse_args()

    trained_agent, rewards = train(
        args.agent, n_episodes=args.episodes, log_dir=args.log_dir, render=args.render
    )
    plot_rewards(rewards, title=f"{args.agent} on CartPole-v1")
