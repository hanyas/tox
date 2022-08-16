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


class QuadraticDynamics(NamedTuple):
    fxx: jnp.ndarray
    fuu: jnp.ndarray
    fxu: jnp.ndarray
    fx: jnp.ndarray
    fu: jnp.ndarray
    f0: jnp.ndarray


class Belief(NamedTuple):
    bel_mu: jnp.ndarray
    bel_cov: jnp.ndarray


class BeliefTrajectory(NamedTuple):
    bel_mu: jnp.ndarray
    bel_cov: jnp.ndarray
    action: jnp.ndarray

    @property
    def horizon(self):
        return len(self.action)

    @property
    def final(self):
        return Belief(self.bel_mu[-1], self.bel_cov[-1])

    @property
    def transient(self):
        return BeliefTrajectory(self.bel_mu[:-1], self.bel_cov[:-1], self.action)


class QuadraticFinalBeliefCost(NamedTuple):
    Cxx: jnp.ndarray  # Q
    cx: jnp.ndarray  # q
    cs: jnp.ndarray  # p
    c0: float  # q0


class QuadraticTransientBeliefCost(NamedTuple):
    Cxx: jnp.ndarray  # Q
    Cuu: jnp.ndarray  # R
    Cxu: jnp.ndarray  # P
    cx: jnp.ndarray  # q
    cu: jnp.ndarray  # r
    cs: jnp.ndarray  # p
    c0: float  # q0


class LinearBeliefDynamics(NamedTuple):
    # Mean-mean wrt ref-mean and ref-action
    fx: jnp.ndarray  # F
    fu: jnp.ndarray  # G

    # Mean-variance wrt ref-mean, ref-var, and ref-action
    Wx: jnp.ndarray  # X
    Ws: jnp.ndarray  # Y
    Wu: jnp.ndarray  # Z

    # Variance wrt ref-mean, ref-var, and ref-action
    Px: jnp.ndarray  # T
    Ps: jnp.ndarray  # U
    Pu: jnp.ndarray  # V

    W: jnp.ndarray   # y
