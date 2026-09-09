import gymnasium as gym

from discretize import Discretizer

# Each task pairs a gymnasium env id with how to turn its observation into a
# Q-table index. Box (continuous) observations get a Discretizer; Discrete
# observations (already a bare int, e.g. FrozenLake) are used as-is.
TASKS = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "env_kwargs": {},
        # [cart position, cart velocity, pole angle, pole angular velocity]
        "discretizer": Discretizer(
            low=[-2.4, -3.0, -0.2095, -3.5],
            high=[2.4, 3.0, 0.2095, 3.5],
            bins=[6, 6, 12, 12],
        ),
    },
    "mountaincar": {
        "env_id": "MountainCar-v0",
        "env_kwargs": {},
        # [position, velocity]
        "discretizer": Discretizer(
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            bins=[20, 20],
        ),
    },
    "frozenlake": {
        "env_id": "FrozenLake-v1",
        # Slippery ice: the chosen action only happens 1/3 of the time, the
        # rest you slide sideways - the "snowy" part that makes this harder
        # than a plain gridworld. Set to False for a deterministic warm-up.
        "env_kwargs": {"is_slippery": True},
        "discretizer": None,  # observation is already a Discrete int
    },
}


def make_env(task_name, render=False):
    spec = TASKS[task_name]
    kwargs = dict(spec["env_kwargs"])
    if render:
        kwargs["render_mode"] = "human"
    return gym.make(spec["env_id"], **kwargs)


def state_fn(task_name, obs):
    discretizer = TASKS[task_name]["discretizer"]
    if discretizer is not None:
        return discretizer(obs)
    return int(obs)


def n_states(task_name, env):
    discretizer = TASKS[task_name]["discretizer"]
    if discretizer is not None:
        return discretizer.shape
    return (env.observation_space.n,)
