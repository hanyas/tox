import abc
from typing import Tuple

import jax.random as jr
import jax.numpy as jnp

from flax import struct

from tox.envs.spaces import Box


@struct.dataclass
class Initial:
    mu: jnp.ndarray
    sigma: jnp.ndarray


@struct.dataclass
class Params:
    # discretization
    simulation_step: float
    downsampling: int

    # dimensions
    state_dim: int
    action_dim: int

    state_shape: Tuple
    action_shape: Tuple

    # limits
    state_low: jnp.ndarray
    state_high: jnp.ndarray

    state_space: Box
    observation_space: Box
    action_space: Box

    # initial state
    init_dist: Initial


class Environment(abc.ABC):
    @property
    def default_params(self) -> Params:
        return Params()

    @abc.abstractmethod
    def dynamics(
        self, state: jnp.ndarray, action: jnp.ndarray, params: Params
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def dynamics_noise(
        self, state: jnp.ndarray, action: jnp.ndarray, params: Params
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self, state: jnp.ndarray, params: Params) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_noise(
        self, state: jnp.ndarray, action: jnp.ndarray, params: Params
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def cost(
        self, state: jnp.ndarray, action: jnp.ndarray, params: Params
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        key: jr.PRNGKey,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, key: jr.PRNGKey, params: Params) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
