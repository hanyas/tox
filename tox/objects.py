from typing import NamedTuple, Callable, Tuple
import jax.numpy as jnp


class Box(NamedTuple):
    low: jnp.ndarray
    high: jnp.ndarray
    shape: Tuple

    def __call__(self, func) -> Callable:
        return lambda *args: jnp.clip(func(*args), self.low, self.high)

    def contains(self, x: jnp.ndarray) -> bool:
        range_cond = jnp.logical_and(
            jnp.all(x >= self.low), jnp.all(x <= self.high)
        )
        return range_cond

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.low, self.high)


class Trajectory(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray

    @property
    def final(self):
        return self.state[-1]

    @property
    def transient(self):
        return Trajectory(self.state[:-1], self.action)

    @property
    def horizon(self):
        return len(self.action)


class QuadraticFinalCost(NamedTuple):
    Cxx: jnp.ndarray
    cx: jnp.ndarray
    c0: float


class QuadraticTransientCost(NamedTuple):
    Cxx: jnp.ndarray
    Cuu: jnp.ndarray
    Cxu: jnp.ndarray
    cx: jnp.ndarray
    cu: jnp.ndarray
    c0: float


class LinearDynamics(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray
