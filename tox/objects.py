from typing import NamedTuple, Callable, Tuple
import jax.numpy as jnp


class Box:
    def __init__(self, low: jnp.ndarray, high: jnp.ndarray, shape: Tuple):
        self.low = low
        self.high = high
        self.shape = shape

    def __call__(self, func) -> Callable:
        return lambda *args: self.clip(func(*args))

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.low, self.high)


class BeliefTrajectory(NamedTuple):
    mean: jnp.ndarray
    variance: jnp.ndarray
    action: jnp.ndarray

    @property
    def horizon(self):
        return len(self.action)

    @property
    def final(self):
        return self.mean[-1], self.variance[-1]

    @property
    def transient(self):
        return BeliefTrajectory(
            self.mean[:-1], self.variance[:-1], self.action
        )


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
    c0: jnp.ndarray


class LinearDynamics(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray
