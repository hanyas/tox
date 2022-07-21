from jax.config import config
config.update("jax_enable_x64", True)

import tox
from time import time

import jax.numpy as jnp
from jax import block_until_ready

from tox.algos import ilqr
from tox.utils import Trajectory

import matplotlib.pyplot as plt


env, env_params = tox.make("LQR-v0")

init_reference = Trajectory(
    state=jnp.zeros((env.horizon + 1, env.state_dim)),
    action=jnp.zeros((env.horizon, env.action_dim)),
)

init_policy = ilqr.LinearPolicy(
    K=jnp.zeros((env.horizon, env.action_dim, env.state_dim)),
    kff=jnp.zeros((env.horizon, env.action_dim)),
)

options = ilqr.Hyperparameters()

start = time()
policy, reference, _ = ilqr.solver(
    env, env_params, init_policy, init_reference, options
)

episode = ilqr.rollout(
    env,
    env_params,
    policy,
    reference,
    1.0
)
block_until_ready(episode)
end = time()
print("Execution Time:", end - start)

state, action, next_state = episode

plt.subplot(3, 1, 1)
plt.plot(state[:, 0].T)
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1].T)
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[:, 1].T)
plt.ylabel("u")
plt.xlabel("t")
plt.show()
