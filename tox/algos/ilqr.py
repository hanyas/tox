from typing import Any, NamedTuple

from functools import partial

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as onp

from jax import jacobian as jac
from jax import hessian as hess
from jax import jacfwd

from jax import jit, vmap
from jax.lax import scan, cond

from tox.envs import DeterministicEnv, Parameters
from tox.utils import Trajectory


class FinalQuadraticCost(NamedTuple):
    Cxx: jnp.ndarray
    cx: jnp.ndarray


class TransientQuadraticCost(NamedTuple):
    Cxx: jnp.ndarray
    Cuu: jnp.ndarray
    Cxu: jnp.ndarray
    cx: jnp.ndarray
    cu: jnp.ndarray


class LinearDynamics(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(
        self, t: int, state: jnp.ndarray, reference: Trajectory, alpha: float
    ) -> jnp.ndarray:
        return (
            reference.action[t]
            + alpha * self.kff[t]
            + self.K[t] @ (state - reference.state[t])
        )


@partial(jit, static_argnums=(0,))
def _second_order_final_cost(
    env: DeterministicEnv,
    env_params: Parameters,
    state: jnp.array,
) -> FinalQuadraticCost:
    Cxx = hess(env.final_cost, 0)(state, env_params)
    cx = jac(env.final_cost, 0)(state, env_params)

    return FinalQuadraticCost(Cxx, cx)


@partial(jit, static_argnums=(0,))
@partial(vmap, in_axes=(None, None, 0))
def _second_order_transient_cost(
    env: DeterministicEnv,
    env_params: Parameters,
    reference: Trajectory,
) -> TransientQuadraticCost:
    Cxx = hess(env.cost, 0)(reference.state, reference.action, env_params)
    Cuu = hess(env.cost, 1)(reference.state, reference.action, env_params)
    Cxu = jac(jac(env.cost, 0), 1)(
        reference.state, reference.action, env_params
    )

    cx = jac(env.cost, 0)(reference.state, reference.action, env_params)
    cu = jac(env.cost, 1)(reference.state, reference.action, env_params)

    return TransientQuadraticCost(Cxx, Cuu, Cxu, cx, cu)


@partial(jit, static_argnums=(0,))
@partial(vmap, in_axes=(None, None, 0))
def _first_order_dynamics(
    env: DeterministicEnv,
    env_params: Parameters,
    reference: Trajectory,
) -> LinearDynamics:
    A = jacfwd(env.dynamics, 0)(reference.state, reference.action, env_params)
    B = jacfwd(env.dynamics, 1)(reference.state, reference.action, env_params)
    return LinearDynamics(A, B)


@jit
def _backward_pass(
    final_quadratic_cost: FinalQuadraticCost,
    transient_quadratic_cost: TransientQuadraticCost,
    linear_dynamics: LinearDynamics,
    lmbda: float,
) -> (LinearPolicy, Any, Any):

    fCxx, fcx = final_quadratic_cost.Cxx, final_quadratic_cost.cx

    Cxx, Cuu, Cxu, cx, cu = (
        transient_quadratic_cost.Cxx,
        transient_quadratic_cost.Cuu,
        transient_quadratic_cost.Cxu,
        transient_quadratic_cost.cx,
        transient_quadratic_cost.cu,
    )
    A, B = linear_dynamics.A, linear_dynamics.B

    delta_V = jnp.zeros((2,))

    def backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B = params

        Vxx, vx, dv = carry

        Qxx = Cxx + A.transpose() @ Vxx @ A
        Quu = Cuu + B.transpose() @ Vxx @ B
        Qux = (Cxu + A.transpose() @ Vxx @ B).transpose()

        qx = cx + A.transpose() @ vx
        qu = cu + B.transpose() @ vx

        Quu_reg = (
            Cuu + B.transpose() @ (Vxx + lmbda * jnp.eye(Vxx.shape[0])) @ B
        )
        Qux_reg = (
            Cxu + A.transpose() @ (Vxx + lmbda * jnp.eye(Vxx.shape[0])) @ B
        ).transpose()

        feasability = jnp.all(jnp.linalg.eigvals(Quu_reg) > 0.0)

        def backpass_not_feasible(Vxx, vx, delta_V, Quu, Qux, qu, qx):
            K, kff = jnp.zeros_like(Qux), jnp.zeros_like(qu)
            return [Vxx, vx, delta_V], [K, kff, delta_V, False]

        def backpass_feasible(Vxx, vx, delta_V, Quu, Qux, qu, qx):
            Quu_inv = jnp.linalg.inv(Quu_reg)

            K = -Quu_inv @ Qux_reg
            kff = -Quu_inv @ qu

            Vxx = (
                Qxx + K.transpose() @ Quu @ K + K.T @ Qux + Qux.transpose() @ K
            )
            Vxx = 0.5 * (Vxx + Vxx.transpose())

            delta_V += jnp.array([kff.T @ qu, 0.5 * kff.T @ Quu @ kff])

            vx = (
                qx
                + K.transpose() @ Quu @ kff
                + K.T @ qu
                + Qux.transpose() @ kff
            )

            return [Vxx, vx, delta_V], [K, kff, delta_V, True]

        return cond(
            feasability,
            backpass_feasible,
            backpass_not_feasible,
            *[Vxx, vx, dv, Quu, Qux, qu, qx]
        )

    K, kff, dv, feasible = scan(
        f=backwards,
        init=[fCxx, fcx, delta_V],
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B),
        reverse=True,
    )[1]

    return LinearPolicy(K, kff), delta_V, feasible


# Dataclasses are not included in pytrees
@dataclass
class Hyperparameters:
    alphas: onp.ndarray = onp.power(10.0, onp.linspace(0, -3, 11))
    lmbda: float = 1.0
    d_lmbda: float = 1.0
    min_lmbda: float = 1e-6
    max_lmbda: float = 1e6
    mult_lmbda: float = 1.6
    tol_fun: float = 1e-6
    tol_grad: float = 1e-4
    min_improv: float = 0.0
    max_iter: int = 100


def solver(
    env: DeterministicEnv,
    env_params: Parameters,
    policy: LinearPolicy,
    reference: Trajectory,
    options: Hyperparameters,
) -> (LinearPolicy, Trajectory):
    trace = []

    last_total_cost = None

    for alpha in options.alphas:
        _state, _action, _total_cost = rollout(
            env, env_params, policy, reference, alpha
        )
        if jnp.all(_state < 1e8):
            reference = Trajectory(_state, _action)
            last_total_cost = _total_cost
            break
        else:
            print("Initial trajectory diverges")

    trace.append(last_total_cost)

    for _ in range(options.max_iter):
        # get quadratic cost around reference
        final_quadratic_cost = _second_order_final_cost(
            env, env_params, reference.final
        )

        transient_quadratic_cost = _second_order_transient_cost(
            env, env_params, reference.transient
        )

        # get linear dynamics around reference
        linear_dynamics = _first_order_dynamics(
            env, env_params, reference.transient
        )

        backpass_feasible = False
        _next_policy, delta_V = None, None
        while not backpass_feasible:
            _next_policy, delta_V, feasible = _backward_pass(
                final_quadratic_cost,
                transient_quadratic_cost,
                linear_dynamics,
                options.lmbda,
            )
            if not jnp.all(feasible):
                # increase lmbda
                options.d_lmbda = jnp.maximum(
                    options.d_lmbda * options.mult_lmbda, options.mult_lmbda
                )
                options.lmbda = jnp.maximum(
                    options.lmbda * options.d_lmbda, options.min_lmbda
                )
                if options.lmbda > options.max_lmbda:
                    print("Maximum regularization reached")
                    break
                else:
                    continue
            else:
                backpass_feasible = True

        # terminate if gradient too small
        _grad_norm = jnp.mean(
            jnp.max(
                jnp.abs(_next_policy.kff) / (jnp.abs(reference.action) + 1.0),
                axis=1,
            )
        )
        if _grad_norm < options.tol_grad and options.lmbda < 1e-5:
            options.d_lmbda = jnp.minimum(
                options.d_lmbda / options.mult_lmbda, 1.0 / options.mult_lmbda
            )
            options.lmbda = (
                options.lmbda
                * options.d_lmbda
                * (options.lmbda > options.min_lmbda)
            )
            print("Gradient tolerance reached")
            break

        _next_reference = None
        _total_cost, _total_cost_diff = None, None

        # execute a forward pass
        fwdpass_feasible = False
        if backpass_feasible:
            for alpha in options.alphas:
                # apply on actual system
                _state, _action, _total_cost = rollout(
                    env, env_params, _next_policy, reference, alpha
                )

                _next_reference = Trajectory(_state, _action)

                # check return improvement
                _total_cost_diff = last_total_cost - _total_cost
                _expected_improvement = (
                    1.0 * alpha * (delta_V[0] + alpha * delta_V[1])
                )
                _improvment = _total_cost_diff / _expected_improvement
                if _improvment >= options.min_improv:
                    fwdpass_feasible = True
                    break

        # accept or reject
        if fwdpass_feasible:
            # decrease lmbda
            options.d_lmbda = jnp.minimum(
                options.d_lmbda / options.mult_lmbda, 1.0 / options.mult_lmbda
            )
            options.lmbda = (
                options.lmbda
                * options.d_lmbda
                * (options.lmbda > options.min_lmbda)
            )

            policy = _next_policy
            reference = _next_reference
            last_return = _total_cost

            trace.append(last_return)

            # terminate if reached objective tolerance
            if _total_cost_diff < options.tol_fun:
                print("Objective tolerance reached")
                break
        else:
            # increase lmbda
            options.d_lmbda = jnp.maximum(
                options.d_lmbda * options.mult_lmbda, options.mult_lmbda
            )
            options.lmbda = jnp.maximum(
                options.lmbda * options.d_lmbda, options.min_lmbda
            )
            if options.lmbda > options.max_lmbda:
                break
            else:
                continue

    return policy, reference, trace


@partial(jit, static_argnums=(0, 4))
def rollout(
    env: DeterministicEnv,
    env_params: Parameters,
    policy: LinearPolicy,
    reference: Trajectory,
    alpha: float,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    state = env.reset(env_params)

    def episode(carry, time):
        state = carry

        action = policy(time, state, reference, alpha)
        cost = env.cost(state, action, env_params)
        next_state = env.step(state, action, env_params)

        return next_state, [next_state, action, cost]

    next_state, action, cost = scan(
        f=episode,
        init=state,
        xs=jnp.arange(env.horizon),
    )[1]

    state = jnp.vstack((state, next_state))
    total_cost = jnp.sum(cost) + env.final_cost(state[-1], env_params)
    return state, action, total_cost
