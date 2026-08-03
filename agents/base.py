import numpy as np


class TabularAgent:
    """Shared scaffolding for tabular TD control agents.

    Handles the Q-table, epsilon-greedy action selection, and epsilon decay.
    Subclasses (SARSA, Q-learning) only need to implement `update`, since
    that's the one line where the two algorithms actually differ.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((*n_states, n_actions))

    def select_action(self, state):
        """Epsilon-greedy: random action with prob. epsilon, else greedy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state):
        """Greedy action w.r.t. the current Q-table (no exploration)."""
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, next_action, done):
        """Update self.q_table[state][action] using the transition.

        Implement in subclass — this is the SARSA vs. Q-learning update rule.
        """
        raise NotImplementedError
