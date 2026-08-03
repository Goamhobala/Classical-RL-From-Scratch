# RL

Tabular SARSA / Q-learning on CartPole-v1 (gymnasium).

## Structure

- `discretize.py` — `Discretizer` bins CartPole's 4D continuous observation
  (cart position, cart velocity, pole angle, pole angular velocity) into a
  discrete state tuple, since tabular Q-tables require finite states.
- `agents/base.py` — `TabularAgent`: Q-table init, epsilon-greedy action
  selection, epsilon decay. Shared by both algorithms.
- `agents/sarsa.py` / `agents/q_learning.py` — each has an `update()` stub
  with a docstring giving the update rule, left as `NotImplementedError` to
  fill in yourself. That's the one line where the two algorithms actually
  differ (on-policy `Q(s', a')` vs. off-policy `max Q(s', ·)`).
- `train.py` — episode loop, epsilon decay, TensorBoard logging, and a
  reward plot.

## Usage

Implement the update rule in `agents/sarsa.py` or `agents/q_learning.py`,
then run:

```bash
uv run python train.py qlearning --episodes 2000
uv run python train.py sarsa --episodes 2000
```

Options:

- `--render` — watch the episodes live.
- `--log-dir runs/qlearning` — log reward/epsilon to TensorBoard
  (`tensorboard --logdir runs`).
