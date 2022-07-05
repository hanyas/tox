import abc
from typing import TypeVar

import jax.random as jr
import jax.numpy as jnp


Params = TypeVar('Params')


class Environment(abc.ABC):

    @property
    def default_params(self) -> Params:
        return Params()

    @abc.abstractmethod
    def dynamics(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def dynamics_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def final_cost(self, state: jnp.ndarray, params: Params) -> float:
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
