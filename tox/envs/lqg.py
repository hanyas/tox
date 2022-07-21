from typing import NamedTuple

from tox.envs.environment import StochasticEnv
from tox.envs.spaces import Box

import jax.numpy as jnp
import jax.random as jr
from jax.lax import fori_loop


class InitialDistribution(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray


class Parameters(NamedTuple):
    # cost
    goal: jnp.ndarray = jnp.array([10.0, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e0]))

    # dynamics
    A: jnp.ndarray = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B: jnp.ndarray = jnp.array([[0.0], [1.0]])
    c: jnp.ndarray = jnp.array([0.0, 0.0])
    sigma: jnp.ndarray = 5e-3 * jnp.eye(2)

    # initial state
    init_dist: InitialDistribution = InitialDistribution(
        mean=jnp.array([0.0, 0.0]), cov=1e-2 * jnp.eye(2)
    )


class LinearQuadraticGaussian(StochasticEnv):
    """Linear quadratic Gaussian"""

    def __init__(self, step=0.01, downsampling=10, horizon=100):
        super().__init__(
            step, downsampling, horizon, state_dim=2, action_dim=1
        )

        # limits
        self.state_space: Box = Box(
            low=jnp.ones(self.state_shape) * jnp.finfo(jnp.float32).min,
            high=jnp.ones(self.state_shape) * jnp.finfo(jnp.float32).max,
            shape=self.state_shape,
        )

        self.observation_space: Box = self.state_space

        self.action_space: Box = Box(
            low=jnp.ones(self.action_shape) * jnp.finfo(jnp.float32).min,
            high=jnp.ones(self.action_shape) * jnp.finfo(jnp.float32).max,
            shape=self.action_shape,
        )

    @property
    def default_params(self) -> Parameters:
        return Parameters()

    def dynamics(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        def evolve(i, arg):
            x, u, p = arg

            def f(x, u, p):
                _u = self.action_space.clip(u)
                return p.A @ x + p.B @ _u + p.c

            dt = self.simulation_step

            k1 = f(x, u, p)
            k2 = f(x + 0.5 * dt * k1, u, p)
            k3 = f(x + 0.5 * dt * k2, u, p)
            k4 = f(x + dt * k3, u, p)

            xn = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xn = self.state_space.clip(xn)

            return [xn, u, p]

        return fori_loop(
            lower=0,
            upper=self.downsampling,
            body_fun=evolve,
            init_val=[state, action, params],
        )[0]

    def dynamics_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        return params.sigma

    def observation(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def observation_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> float:
        c = (state - params.goal).transpose() @ params.state_cost @ (
            state - params.goal
        ) + action.transpose() @ params.action_cost @ action
        return c * (self.simulation_step * self.downsampling)

    def final_cost(self, state: jnp.ndarray, params: Parameters) -> float:
        c = (
            (state - params.goal).transpose()
            @ params.final_state_cost
            @ (state - params.goal)
        )
        return c * (self.simulation_step * self.downsampling)

    def step(
        self,
        key: jr.PRNGKey,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Parameters,
    ) -> jnp.ndarray:
        return jr.multivariate_normal(
            key=key,
            mean=self.dynamics(state, action, params),
            cov=self.dynamics_noise(state, action, params),
        )

    def reset(self, key: jr.PRNGKey, params: Parameters) -> jnp.ndarray:
        return jr.multivariate_normal(
            key=key,
            mean=params.init_dist.mean,
            cov=params.init_dist.cov,
        )
