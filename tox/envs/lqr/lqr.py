from typing import Tuple

from functools import partial

from tox.envs.environment import Environment
from tox.envs.spaces import Box

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jax.lax import fori_loop

from flax import struct


@struct.dataclass
class Initial:
    mu: jnp.ndarray
    sigma: jnp.ndarray


@struct.dataclass
class Params:
    # discretization
    simulation_step: float = struct.field(pytree_node=False, default=0.01)
    downsampling: int = struct.field(pytree_node=False, default=10)

    # horizon
    horizon: int = struct.field(pytree_node=False, default=100)

    # dimensions
    state_dim: int = struct.field(pytree_node=False, default=2)
    action_dim: int = struct.field(pytree_node=False, default=1)

    state_shape: Tuple = struct.field(pytree_node=False, default=(2,))
    action_shape: Tuple = struct.field(pytree_node=False, default=(1,))
    # limits
    state_space: Box = struct.field(
        pytree_node=False,
        default=Box(
            low=jnp.ones((2,)) * jnp.finfo(jnp.float32).min,
            high=jnp.ones(2,) * jnp.finfo(jnp.float32).max,
            shape=(2,),
        ),
    )

    observation_space: Box = struct.field(
        pytree_node=False, default=state_space
    )

    action_space: Box = struct.field(
        pytree_node=False,
        default=Box(
            low=jnp.ones((1,)) * jnp.finfo(jnp.float32).min,
            high=jnp.ones((1,)) * jnp.finfo(jnp.float32).max,
            shape=(1,),
        ),
    )

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
    init_dist: Initial = Initial(
        mu=jnp.array([0.0, 0.0]), sigma=1e-2 * jnp.eye(2)
    )


class LinearQuadratic(Environment):
    """Linear quadratic system"""

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> Params:
        return Params()

    def dynamics(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        def evolve(i, arg):
            x, u, p = arg

            def f(x, u, p):
                u = p.action_space.clip(u)
                return p.A @ x + p.B @ u + p.c

            dt = p.simulation_step

            k1 = f(x, u, p)
            k2 = f(x + 0.5 * dt * k1, u, p)
            k3 = f(x + 0.5 * dt * k2, u, p)
            k4 = f(x + dt * k3, u, p)

            xn = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xn = p.state_space.clip(xn)

            return [xn, u, p]

        return fori_loop(
            lower=0,
            upper=params.downsampling,
            body_fun=evolve,
            init_val=[state, action, params],
        )[0]

    def dynamics_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        return params.sigma

    def observation(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def observation_noise(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> float:
        c = (state - params.goal).transpose() @ params.state_cost @ (
            state - params.goal
        ) + action.transpose() @ params.action_cost @ action
        return c * (params.simulation_step * params.downsampling)

    def final_cost(self, state: jnp.ndarray, params: Params) -> float:
        c = (
            (state - params.goal).transpose()
            @ params.final_state_cost
            @ (state - params.goal)
        )
        return c * (params.simulation_step * params.downsampling)

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        key: jr.PRNGKey,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: Params,
    ) -> jnp.ndarray:
        return jr.multivariate_normal(
            key=key,
            mean=self.dynamics(state, action, params),
            cov=self.dynamics_noise(state, action, params),
        )

    @partial(jit, static_argnums=(0,))
    def reset(self, key: jr.PRNGKey, params: Params) -> jnp.ndarray:
        return jr.multivariate_normal(
            key=key,
            mean=params.init_dist.mu,
            cov=params.init_dist.sigma,
        )
