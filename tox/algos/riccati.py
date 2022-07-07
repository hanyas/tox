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
class Trajectory:
    state: jnp.ndarray
    action: jnp.ndarray

    @property
    def final(self):
        return self.state[-1]

    @property
    def transient(self):
        return Trajectory(self.state[:-1], self.action)


@struct.dataclass
class FinalQuadraticCost:
    Cxx: jnp.ndarray
    cx: jnp.ndarray


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

    def __call__(self, time: int, state: jnp.ndarray) -> jnp.ndarray:
        return self.K[time] @ state + self.kff[time]


def _second_order_final_cost(
    env: Environment,
    env_params: Params,
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
    env_params: Params,
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
    env_params: Params,
    reference: Trajectory,
) -> LinearDynamics:
    A = jacfwd(env.dynamics, 0)(reference.state, reference.action, env_params)
    B = jacfwd(env.dynamics, 1)(reference.state, reference.action, env_params)
    c = (
        env.dynamics(reference.state, reference.action, env_params)
        - A @ reference.state
        - B @ reference.action
    )

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
    env_params: Params,
) -> LinearPolicy:

    # Create a reference trajectory to extract matrices through auto-diff.
    # This operation is not necessary, one could just read the matrices out,
    # but we want to highlight the general case using JAX.

    reference = Trajectory(
        state=jnp.zeros((env_params.horizon + 1, env_params.state_dim)),
        action=jnp.zeros((env_params.horizon, env_params.action_dim)),
    )

    # get quadratic cost around ref traj
    final_quadratic_cost = _second_order_final_cost(
        env, env_params, reference.final
    )

    quadratic_cost = _second_order_cost(env, env_params, reference.transient)

    # get linear dynamics around ref traj
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
    env_params: Params,
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
        xs=jnp.arange(env_params.horizon),
    )[1]
