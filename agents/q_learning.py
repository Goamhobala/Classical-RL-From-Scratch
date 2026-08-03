from .base import TabularAgent


class QLearningAgent(TabularAgent):
    """Off-policy TD control.

    TODO: implement the Q-learning update rule:

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

    Unlike SARSA, this bootstraps off the greedy action in `next_state`
    regardless of which action is actually taken next (`next_action` is
    passed in for a consistent interface with SARSA, but Q-learning doesn't
    need it — see `self.greedy_action` / `np.max` on `self.q_table[next_state]`).
    Remember to zero out the bootstrap term when `done` is True.
    """

    def update(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError("Implement the Q-learning update rule here")
