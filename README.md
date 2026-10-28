# RL

Tabular SARSA / Q-learning on CartPole-v1 (gymnasium).
My implementations of PPO, A2C with PPO, DQN can be found in /agents.

```bash
uv run python train.py qlearning --episodes 2000
uv run python train.py sarsa --episodes 2000
```

Options:

- `--render` — watch the episodes live.
- `--log-dir runs/qlearning` — log reward/epsilon to TensorBoard
  (`tensorboard --logdir runs`).
