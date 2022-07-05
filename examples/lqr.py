import tox
from time import time

import jax.numpy as jnp
import jax.random as jr
from jax import block_until_ready

from jax.config import config

from tox.algos import riccati
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


rng = jr.PRNGKey(1337)
env, env_params = tox.make("LQR-v0")
nb_steps = 100

state_ref = jnp.zeros((nb_steps + 1, env_params.state_dim))
action_ref = jnp.zeros((nb_steps, env_params.action_dim))

start = time()
policy = riccati.solver(env, env_params, state_ref, action_ref)

rng_episodes = jr.split(rng, 100)
episodes = riccati.rollout(
    rng_episodes,
    env,
    env_params,
    policy,
    nb_steps,
)
block_until_ready(episodes)
end = time()
print("Execution Time:", end - start)

state, action, next_state, cost = episodes

plt.subplot(3, 1, 1)
plt.plot(state[:, :, 0].T)
plt.ylabel('x1')
plt.subplot(3, 1, 2)
plt.plot(state[:, :, 1].T)
plt.ylabel('x2')
plt.subplot(3, 1, 3)
plt.plot(action[:, :, 1].T)
plt.ylabel('u')
plt.xlabel('t')
plt.show()