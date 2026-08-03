import numpy as np


class Discretizer:
    """Bins a continuous Box observation into a discrete state tuple.

    CartPole's observation (cart position, cart velocity, pole angle, pole
    angular velocity) is continuous, but tabular SARSA/Q-learning need a
    finite state space. This clips each dimension to [low, high] and buckets
    it into `bins[i]` equal-width intervals.
    """

    def __init__(self, low, high, bins):
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)
        self.bins = np.array(bins, dtype=np.int64)
        self.edges = [
            np.linspace(lo, hi, n + 1)[1:-1]
            for lo, hi, n in zip(self.low, self.high, self.bins)
        ]

    def __call__(self, obs):
        obs = np.clip(obs, self.low, self.high)
        return tuple(
            int(np.digitize(x, edges)) for x, edges in zip(obs, self.edges)
        )

    @property
    def shape(self):
        """Shape of the resulting discrete state space, e.g. (6, 6, 12, 12)."""
        return tuple(int(b) for b in self.bins)
