from typing import NamedTuple, Callable, Tuple
import jax.numpy as jnp


class Box:

    def __init__(self, low: jnp.ndarray, high: jnp.ndarray, shape: Tuple):
        self.low = low
        self.high = high
        self.shape = shape

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.low, self.high)

    def __call__(self, func) -> Callable:
        return lambda *args: self.clip(func(*args))


class Trajectory(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray

    @property
    def horizon(self):
        return len(self.action)

    @property
    def final(self):
        return self.state[-1]

    @property
    def transient(self):
        return Trajectory(self.state[:-1], self.action)


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
    f0: jnp.ndarray
