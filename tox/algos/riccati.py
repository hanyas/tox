from typing import NamedTuple

from functools import partial

import jax.numpy as jnp
import jax.random as jr

from jax import jacobian as jac
from jax import hessian as hess
from jax import jacfwd

from jax import jit, vmap
from jax.lax import scan

from tox.envs import Environment, Parameters
from tox.utils import Trajectory


class FinalQuadraticCost(NamedTuple):
    Cxx: jnp.ndarray
    cx: jnp.ndarray


class QuadraticCost(NamedTuple):
    Cxx: jnp.ndarray
    Cuu: jnp.ndarray
    Cxu: jnp.ndarray
    cx: jnp.ndarray
    cu: jnp.ndarray


class LinearDynamics(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(self, time: int, state: jnp.ndarray) -> jnp.ndarray:
        return self.K[time] @ state + self.kff[time]


def _second_order_final_cost(
    env: Environment,
    env_params: Parameters,
    state: jnp.array,
) -> FinalQuadraticCost:
    Cxx = 0.5 * hess(env.final_cost, 0)(state, env_params)

    cx = (
        jac(env.final_cost, 0)(state, env_params)
        - hess(env.final_cost, 0)(state, env_params) @ state
    )

    return FinalQuadraticCost(Cxx, cx)


@partial(vmap, in_axes=(None, None, 0))
def _second_order_cost(
    env: Environment,
    env_params: Parameters,
    reference: Trajectory,
) -> QuadraticCost:
    Cxx = 0.5 * hess(env.cost, 0)(
        reference.state, reference.action, env_params
    )
    Cuu = 0.5 * hess(env.cost, 1)(
        reference.state, reference.action, env_params
    )
    Cxu = 0.5 * jac(jac(env.cost, 0), 1)(
        reference.state, reference.action, env_params
    )

    cx = (
        jac(env.cost, 0)(reference.state, reference.action, env_params)
        - hess(env.cost, 0)(reference.state, reference.action, env_params)
        @ reference.state
        - jac(jac(env.cost, 0), 1)(
            reference.state, reference.action, env_params
        )
        @ reference.action
    )
    cu = (
        jac(env.cost, 1)(reference.state, reference.action, env_params)
        - hess(env.cost, 1)(reference.state, reference.action, env_params)
        @ reference.action
        - reference.state.transpose()
        @ jac(jac(env.cost, 0), 1)(
            reference.state, reference.action, env_params
        )
    )

    return QuadraticCost(Cxx, Cuu, Cxu, cx, cu)


@partial(vmap, in_axes=(None, None, 0))
def _first_order_dynamics(
    env: Environment,
    env_params: Parameters,
    reference: Trajectory,
) -> LinearDynamics:
    A = jacfwd(env.dynamics, 0)(reference.state, reference.action, env_params)
    B = jacfwd(env.dynamics, 1)(reference.state, reference.action, env_params)
    c = env.dynamics(reference.state, reference.action, env_params) - (A @ reference.state + B @ reference.action)
    return LinearDynamics(A, B, c)


def _backward_pass(
    final_quadratic_cost: FinalQuadraticCost,
    quadratic_cost: QuadraticCost,
    linear_dynamics: LinearDynamics,
) -> LinearPolicy:

    fCxx, fcx = final_quadratic_cost.Cxx, final_quadratic_cost.cx

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

        Vxx = Qxx + Qux.transpose() @ K
        vx = qx + 2.0 * kff.transpose() @ Qux

        return [Vxx, vx], [K, kff]

    K, kff = scan(
        f=backwards,
        init=[fCxx, fcx],
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B, c),
        reverse=True,
    )[1]

    return LinearPolicy(K, kff)


@partial(jit, static_argnums=(0,))
def solver(
    env: Environment,
    env_params: Parameters,
    reference: Trajectory,
) -> LinearPolicy:

    # Reference not really needed in LQR.
    # We use it to show the general case of linearizing
    # around a trajectory to retrieve dynamics and cost.

    # get quadratic cost around reference
    final_quadratic_cost = _second_order_final_cost(
        env, env_params, reference.final
    )

    quadratic_cost = _second_order_cost(env, env_params, reference.transient)

    # get linear dynamics around reference
    linear_dynamics = _first_order_dynamics(
        env, env_params, reference.transient
    )

    return _backward_pass(
        final_quadratic_cost, quadratic_cost, linear_dynamics
    )


@partial(jit, static_argnums=(1,))
@partial(vmap, in_axes=(0, None, None, None))
def rollout(
    rng: jr.PRNGKey,
    env: Environment,
    env_params: Parameters,
    policy: LinearPolicy,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    rng, rng_reset = jr.split(rng, 2)
    state = env.reset(rng_reset, env_params)

    def episode(carry, time):
        rng, state = carry

        rng, rng_step = jr.split(rng, 2)
        action = policy(time, state)
        next_state = env.step(rng_step, state, action, env_params)

        return [rng, next_state], [state, action, next_state]

    return scan(
        f=episode,
        init=[rng, state],
        xs=jnp.arange(env.horizon),
    )[1]
