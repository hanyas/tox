import abc
from typing import TypeVar, Tuple

import jax.random as jr
import jax.numpy as jnp


Parameters = TypeVar("Parameters")


class Environment(abc.ABC):
    @abc.abstractmethod
    def __init__(self, step, downsampling, horizon, state_dim, action_dim):

        # discretization
        self.simulation_step: float = step
        self.downsampling: int = downsampling

        # horizon
        self.horizon: int = horizon

        # dimensions
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.state_shape: Tuple = (state_dim,)
        self.action_shape: Tuple = (action_dim,)

    @property
    def default_params(self) -> Parameters:
        return Parameters()

    @abc.abstractmethod
    def dynamics(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def dynamics_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def final_cost(self, state: jnp.ndarray, params: Parameters) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        key: jr.PRNGKey,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, key: jr.PRNGKey, params: Parameters) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
