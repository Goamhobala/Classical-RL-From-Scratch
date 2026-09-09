import argparse
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents import QLearningAgent, SARSAAgent
from tasks import TASKS, make_env, n_states, state_fn

AGENTS = {
    "sarsa": SARSAAgent,
    "qlearning": QLearningAgent,
}


def make_agent(agent_name, task_name, env, **kwargs):
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent {agent_name!r}, choose from {list(AGENTS)}")
    return AGENTS[agent_name](
        n_states=n_states(task_name, env), n_actions=env.action_space.n, **kwargs
    )


def train(agent_name, task_name, n_episodes=2000, log_dir=None, render=False):
    env = make_env(task_name, render=render)
    agent = make_agent(agent_name, task_name, env)
    writer = SummaryWriter(log_dir) if log_dir else None

    episode_rewards = []
    recent_rewards = deque(maxlen=100)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = state_fn(task_name, obs)
        action = agent.select_action(state)

        total_reward = 0.0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_fn(task_name, next_obs)
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
    parser.add_argument("--task", choices=list(TASKS), default="cartpole")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log dir")
    args = parser.parse_args()

    trained_agent, rewards = train(
        args.agent,
        args.task,
        n_episodes=args.episodes,
        log_dir=args.log_dir,
        render=args.render,
    )
    plot_rewards(rewards, title=f"{args.agent} on {args.task}")
