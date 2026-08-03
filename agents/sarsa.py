from .base import TabularAgent


class SARSAAgent(TabularAgent):
    """On-policy TD control.

    TODO: implement the SARSA update rule:

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))

    `next_action` is the action actually selected in `next_state` by the
    behavior policy (epsilon-greedy) — that's what makes this on-policy.
    Remember to zero out the bootstrap term when `done` is True (there is
    no Q(s', a') beyond a terminal state).
    """

    def update(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError("Implement the SARSA update rule here")
