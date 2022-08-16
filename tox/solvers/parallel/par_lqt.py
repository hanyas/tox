import jax
import jax.numpy as jnp

from tox.objects import parPolicy, QuadraticFinalCost, QuadraticTransientCost, LinearDynamics, Trajectory

from tox.solvers.parallel.params_and_operators import value_function_associative_params, \
    value_function_associative_params_final, value_function_operator_rev, policy_parameters, \
    trajectory_associative_params_initial, trajectory_operator, _trajectory_associative_params, action_fn


def parBackwardPass(quadratic_final_cost: QuadraticFinalCost,
                    quadratic_transient_cost: QuadraticTransientCost,
                    linear_dynamics: LinearDynamics,
                    reference: Trajectory):

    generic_params = value_function_associative_params(quadratic_transient_cost, linear_dynamics)
    final_params = value_function_associative_params_final(quadratic_final_cost)
    associative_params = jax.tree_map(lambda x, y: jnp.concatenate([x, y[None, ...]]),
                                      generic_params,
                                      final_params)
    _, _, _, vx, Vxx = jax.lax.associative_scan(value_function_operator_rev, associative_params, reverse=True)
    K, Kv, Kc, res = policy_parameters(quadratic_transient_cost, linear_dynamics, Vxx[1:], vx[1:])
    policy = parPolicy(K, Kv, Kc)

    return vx, Vxx, policy, res


def parForwardPass(xs_init: jnp.ndarray,
                   linear_dynamics: LinearDynamics,
                   policy: parPolicy,
                   vx):

    K, Kv, Kc = policy.K, policy.Kv, policy.Kc
    F, c, L = linear_dynamics.A, linear_dynamics.c, linear_dynamics.B

    initial_linear = LinearDynamics(F[0], L[0], c[0])
    initial_policy = parPolicy(K[0], Kv[0], Kc[0])
    initial_params = trajectory_associative_params_initial(initial_linear, initial_policy, vx[1], xs_init)

    generic_linear = LinearDynamics(F[1:], L[1:], c[1:])
    generic_policy = parPolicy(K[1:], Kv[1:], Kc[1:])
    generic_params = _trajectory_associative_params(generic_linear, generic_policy, vx[2:])
    trajectory_associative_params = jax.tree_map(lambda x, y: jnp.concatenate([x[None, ...], y]),
                                                 initial_params,
                                                 generic_params)
    Fs, cs = jax.lax.associative_scan(trajectory_operator, trajectory_associative_params)
    state = jnp.concatenate([xs_init[None, ...], cs], axis=0)
    action = action_fn(policy, state[:-1], vx[1:], c)

    return state, action
