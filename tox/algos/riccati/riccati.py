from functools import partial

import jax.numpy as jnp
import jax.random as jr

from jax import jacobian as jac
from jax import hessian as hess
from jax import jacfwd

from jax import jit, vmap
from jax.lax import scan

from flax import struct

from tox.envs import Environment, Params


@struct.dataclass
class QuadraticCost:
    Cxx: jnp.ndarray
    Cuu: jnp.ndarray
    Cxu: jnp.ndarray

    cx: jnp.ndarray
    cu: jnp.ndarray


@struct.dataclass
class LinearDynamics:
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray


@struct.dataclass
class LinearPolicy:
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(self, x: jnp.ndarray, t: int) -> jnp.ndarray:
        return self.K[t] @ x + self.kff[t]


@partial(vmap, in_axes=(None, None, 0, 0))
def _second_order_cost(
    env: Environment,
    env_params: Params,
    state_ref: jnp.ndarray,
    action_ref: jnp.ndarray,
) -> QuadraticCost:
    Cxx = 0.5 * hess(env.cost, 0)(state_ref, action_ref, env_params)
    Cuu = 0.5 * hess(env.cost, 1)(state_ref, action_ref, env_params)
    Cxu = 0.5 * jac(jac(env.cost, 0), 1)(state_ref, action_ref, env_params)

    cx = (
        jac(env.cost, 0)(state_ref, action_ref, env_params)
        - hess(env.cost, 0)(state_ref, action_ref, env_params) @ state_ref
        - 2.0
        * jac(jac(env.cost, 0), 1)(state_ref, action_ref, env_params)
        @ action_ref
    )
    cu = (
        jac(env.cost, 1)(state_ref, action_ref, env_params)
        - hess(env.cost, 1)(state_ref, action_ref, env_params) @ action_ref
        - 2.0
        * state_ref.transpose()
        @ jac(jac(env.cost, 0), 1)(state_ref, action_ref, env_params)
    )

    return QuadraticCost(Cxx, Cuu, Cxu, cx, cu)


@partial(vmap, in_axes=(None, None, 0, 0))
def _first_order_dynamics(
    env: Environment,
    env_params: Params,
    state_ref: jnp.ndarray,
    action_ref: jnp.ndarray,
) -> LinearDynamics:
    A = jacfwd(env.dynamics, 0)(state_ref, action_ref, env_params)
    B = jacfwd(env.dynamics, 1)(state_ref, action_ref, env_params)
    c = (
        env.dynamics(state_ref, action_ref, env_params)
        - A @ state_ref
        - B @ action_ref
    )

    return LinearDynamics(A, B, c)


def _backward_pass(
    quadratic_cost: QuadraticCost, linear_dynamics: LinearDynamics
) -> LinearPolicy:
    Cxx, Cuu, Cxu, cx, cu = (
        quadratic_cost.Cxx,
        quadratic_cost.Cuu,
        quadratic_cost.Cxu,
        quadratic_cost.cx,
        quadratic_cost.cu,
    )
    A, B, c = linear_dynamics.A, linear_dynamics.B, linear_dynamics.c

    def backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B, c = params
        Vxx, vx = carry

        Qxx = Cxx + A.transpose() @ Vxx @ A
        Quu = Cuu + B.transpose() @ Vxx @ B
        Qux = Cxu.transpose() + B.transpose() @ Vxx @ A

        qx = cx + 2.0 * A.transpose() @ Vxx @ c + A.transpose() @ vx
        qu = cu + 2.0 * B.transpose() @ Vxx @ c + B.transpose() @ vx

        Quu_inv = jnp.linalg.inv(Quu)

        K = -Quu_inv @ Qux
        kff = -0.5 * Quu_inv @ qu

        Vxx = Qxx + Qux.transpose() * K
        vx = qx + 2.0 * kff.transpose() @ Qux

        return [Vxx, vx], [K, kff]

    K, kff = scan(
        f=backwards,
        init=[Cxx[-1], cx[-1]],
        xs=(Cxx[:-1], Cuu[:-1], Cxu[:-1], cx[:-1], cu[:-1], A, B, c),
        reverse=True,
    )[1]

    return LinearPolicy(K, kff)


@partial(jit, static_argnums=(0,))
def solver(
    env: Environment,
    env_params: Params,
    state_ref: jnp.ndarray,
    action_ref: jnp.ndarray,
) -> LinearPolicy:

    # get quadratic cost around ref traj
    _action_ref = jnp.vstack(
        (action_ref, jnp.zeros((1, action_ref.shape[-1])))
    )
    quadratic_cost = _second_order_cost(
        env, env_params, state_ref, _action_ref
    )

    # get linear dynamics around ref traj
    _state_ref = state_ref[:-1]
    linear_dynamics = _first_order_dynamics(
        env, env_params, _state_ref, action_ref
    )

    return _backward_pass(quadratic_cost, linear_dynamics)


@partial(jit, static_argnums=(1, 4))
@partial(vmap, in_axes=(0, None, None, None, None))
def rollout(
    rng: jr.PRNGKey,
    env: Environment,
    env_params: Params,
    policy: LinearPolicy,
    horizon: int,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    rng, rng_reset = jr.split(rng, 2)
    state = env.reset(rng_reset, env_params)

    def episode(carry, time):
        rng, state = carry

        rng, rng_step = jr.split(rng, 2)
        action = policy(state, time)
        cost = env.cost(state, action, env_params)
        next_state = env.step(rng_step, state, action, env_params)

        return [rng, next_state], [state, action, next_state, cost]

    return scan(
        f=episode,
        init=[rng, state],
        xs=jnp.arange(horizon),
    )[1]
