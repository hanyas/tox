import tox
from time import time

import jax.random as jr
from jax import block_until_ready

from jax.config import config

from tox.algos import riccati
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

rng = jr.PRNGKey(1337)
env, env_params = tox.make("LQR-v0")

start = time()
policy = riccati.solver(env, env_params)

rng_episodes = jr.split(rng, 100)
episodes = riccati.rollout(
    rng_episodes,
    env,
    env_params,
    policy,
)
block_until_ready(episodes)
end = time()
print("Execution Time:", end - start)

state, action, next_state = episodes

plt.subplot(3, 1, 1)
plt.plot(state[:, :, 0].T)
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[:, :, 1].T)
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[:, :, 1].T)
plt.ylabel("u")
plt.xlabel("t")
plt.show()
