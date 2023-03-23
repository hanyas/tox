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


class LinearGaussianDynamics(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray
    sigma: jnp.ndarray


class QuadraticDynamics(NamedTuple):
    fxx: jnp.ndarray
    fuu: jnp.ndarray
    fxu: jnp.ndarray
    fx: jnp.ndarray
    fu: jnp.ndarray
    f0: jnp.ndarray


class BeliefTrajectory(NamedTuple):
    belief: jnp.ndarray
    action: jnp.ndarray

    @property
    def horizon(self):
        return len(self.action)

    @property
    def final(self):
        return self.belief[-1]

    @property
    def transient(self):
        return BeliefTrajectory(self.belief[:-1], self.action)


class QuadraticFinalBeliefCost(NamedTuple):
    Cbb: jnp.ndarray  # Q
    cb: jnp.ndarray  # q
    c0: float  # q0


class QuadraticTransientBeliefCost(NamedTuple):
    Cbb: jnp.ndarray  # Q
    Cuu: jnp.ndarray  # R
    Cbu: jnp.ndarray  # P
    cb: jnp.ndarray  # q
    cu: jnp.ndarray  # r
    c0: float  # p0


class LinearBeliefDynamics(NamedTuple):
    gb: jnp.ndarray  # F
    gu: jnp.ndarray  # G

    Wb: jnp.ndarray  # Fi
    Wu: jnp.ndarray  # Gi

    W: jnp.ndarray
