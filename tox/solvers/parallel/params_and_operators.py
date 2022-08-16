import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import vmap

from tox.objects import LinearDynamics, QuadraticFinalCost, \
    QuadraticTransientCost, parPolicy


@vmap
def value_function_associative_params(quadratic_transient_cost: QuadraticTransientCost,
                                      linear_dynamics: LinearDynamics,
                                      ):

    Fs, cs, Ls = linear_dynamics.A, linear_dynamics.c, linear_dynamics.B
    Xs, Us = quadratic_transient_cost.Cxx, quadratic_transient_cost.Cuu
    As = Fs
    bs = cs
    Cs = Ls @ jlinalg.solve(Us, Ls.T)
    etas = quadratic_transient_cost.cx
    Js = Xs

    return As, bs, Cs, etas, Js


def value_function_associative_params_final(quadratic_final_cost: QuadraticFinalCost):

    XT = quadratic_final_cost.Cxx
    dim_x = XT.shape[0]
    A = jnp.zeros((dim_x, dim_x))
    b = jnp.zeros(dim_x)
    C = jnp.zeros((dim_x, dim_x))
    eta = quadratic_final_cost.cx
    J = XT

    return A, b, C, eta, J


def value_function_operator(elem1, elem2):
    """
    Associative operator described in Lemma 10 of
    "Temporal Parallelisation of Dynamic Programming and Linear Quadratic Control" paper.
    https://arxiv.org/abs/2104.03186

    Parameters:
    -------
    elem1: tuple of array
        A_{kj}, b_{kj}, C_{kj}, eta_{kj}, J_{kj}
    elem2: tuple of array
        A_{ji}, b_{ji}, C_{ji}, eta_{ji}, J_{ji}
    Returns
    -------
    elem12: tuple of array
        A_{ki}, b_{ki}, C_{ki}, eta_{ki}, J_{ki}
    """

    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    dim_x = b1.shape[0]

    temp1 = jnp.eye(dim_x) + C1 @ J2
    temp2 = jnp.eye(dim_x) + J2 @ C1

    temp3 = jlinalg.solve(temp1.T, A2.T).T
    temp4 = jlinalg.solve(temp2.T, A1).T

    A = temp3 @ A1
    b = temp3 @ (b1 + C1 @ eta2) + b2
    C = temp3 @ C1 @ A2.T + C2
    eta = temp4 @ (eta2 - J2 @ b1) + eta1
    J = temp4 @ J2 @ A1 + J1

    return A, b, C, eta, J


@vmap
def value_function_operator_rev(elem1, elem2):
    return value_function_operator(elem2, elem1)


def trajectory_associative_params_initial(linear_dynamics: LinearDynamics,
                                          policy: parPolicy,
                                          v,
                                          xs):

    K, Kv, Kc = policy.K, policy.Kv, policy.Kc
    F, c, L = linear_dynamics.A, linear_dynamics.c, linear_dynamics.B
    Fts = F - L @ K
    cts = c + L @ Kv @ v - L @ Kc @ c

    Ft = jnp.zeros(F.shape)
    ct = Fts @ xs + cts

    return Ft, ct


@vmap
def _trajectory_associative_params(linear_dynamics: LinearDynamics,
                                   policy: parPolicy,
                                   v):
    """
    Associative parameters described in Lemma 14 of
    "Temporal Parallelisation of Dynamic Programming and Linear Quadratic Control" paper.
    https://arxiv.org/abs/2104.03186

    Parameters:
    -------

    Returns
    -------
    elem12: tuple of array
         Ft, ct
    """

    K, Kv, Kc = policy.K, policy.Kv, policy.Kc
    F, c, L = linear_dynamics.A, linear_dynamics.c, linear_dynamics.B
    Ft = F - L @ K
    ct = c + L @ Kv @ v - L @ Kc @ c

    return Ft, ct


@vmap
def trajectory_operator(elem1, elem2):
    """
    Associative operator described in Lemma 13 of
    "Temporal Parallelisation of Dynamic Programming and Linear Quadratic Control" paper.
    https://arxiv.org/abs/2104.03186

    Parameters:
    -------
    elem1: tuple of array
        Ft_{kj}, ct_{kj}
    elem2: tuple of array
        Ft_{ji}, ct_{ji}, C_{ji}, eta_{ji}, J_{ji}
    Returns
    -------
    elem12: tuple of array
        Ft_{ki}, ct_{ki}
    """
    Ft1, ct1 = elem1
    Ft2, ct2 = elem2

    Ft = Ft2 @ Ft1
    ct = Ft2 @ ct1 + ct2

    return Ft, ct


@vmap
def policy_parameters(quadratic_transient_cost: QuadraticTransientCost,
                      linear_dynamics: LinearDynamics,
                      S,
                      v):
    X, U = quadratic_transient_cost.Cxx, quadratic_transient_cost.Cuu
    F, c, L = linear_dynamics.A, linear_dynamics.c, linear_dynamics.B

    temp = L.T @ S @ L + U
    Kv = jlinalg.solve(temp, L.T)
    Kc = Kv @ S
    K = Kc @ F
    res = Kv @ v - Kc @ c

    return K, Kv, Kc, res


@vmap
def action_fn(policy, state, v, c):
    K, Kv, Kc = policy
    u = - K @ state + Kv @ v - Kc @ c
    return u