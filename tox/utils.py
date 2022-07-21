from typing import NamedTuple
import jax.numpy as jnp


class Trajectory(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray

    @property
    def final(self):
        return self.state[-1]

    @property
    def transient(self):
        return Trajectory(self.state[:-1], self.action)
