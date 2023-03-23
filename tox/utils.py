from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import fori_loop


def unflatten_gaussian(state: jnp.ndarray, state_dim: int):
    mu = state[:state_dim]
    tril = jnp.zeros((state_dim, state_dim))
    chol = tril.at[jnp.tril_indices(state_dim)].set(state[state_dim:])
    return mu, chol


def flatten_gaussian(mu: jnp.ndarray, chol: jnp.ndarray, state_dim: int):
    state = jnp.hstack((mu, chol[jnp.tril_indices(state_dim)]))
    return state


def wrap_angle(x: float) -> float:
    # wrap angle between [0, 2*pi]
    return x % (2.0 * jnp.pi)


def symmetrize(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (x + x.T)


def runge_kutta(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
    ode: Callable,
    step: float,
):
    k1 = ode(state, action, time)
    k2 = ode(state + 0.5 * step * k1, action, time)
    k3 = ode(state + 0.5 * step * k2, action, time)
    k4 = ode(state + step * k3, action, time)
    return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def discretize_stochastic_dynamics(
    ode: Callable, simulation_step: float, downsampling: int
):
    def dynamics(
        state: jnp.ndarray,
        action: jnp.ndarray,
        time: int,
    ):
        def _step(t, state):
            next_state = runge_kutta(
                state,
                action,
                time + t * simulation_step,
                ode,
                simulation_step,
            )
            return next_state

        return fori_loop(
            lower=0,
            upper=downsampling,
            body_fun=_step,
            init_val=state,
        )

    return dynamics


def discretize_dynamics(
    ode: Callable, simulation_step: float, downsampling: int
):
    def dynamics(
        state: jnp.ndarray,
        action: jnp.ndarray,
        time: int,
    ):
        def _step(t, state):
            next_state = runge_kutta(
                state,
                action,
                time + t * simulation_step,
                ode,
                simulation_step,
            )
            return next_state

        return fori_loop(
            lower=0,
            upper=downsampling,
            body_fun=_step,
            init_val=state,
        )

    return dynamics


# https://github.com/AdrienCorenflos/sqrt-parallel-smoothers
def tria(A):
    return qr(A.T).T


@jax.custom_jvp
def qr(A: jnp.ndarray):
    """The JAX provided implementation is not parallelizable using VMAP. As a consequence, we have to rewrite it..."""
    return _qr(A)


# @partial(jax.jit, static_argnums=(1,))
def _qr(A: jnp.ndarray, return_q=False):
    m, n = A.shape
    min_ = min(m, n)
    if return_q:
        Q = jnp.eye(m)

    for j in range(min_):
        # Apply Householder transformation.
        v, tau = _householder(A[j:, j])

        H = jnp.eye(m)
        H = H.at[j:, j:].add(-tau * (v[:, None] @ v[None, :]))

        A = H @ A
        if return_q:
            Q = H @ Q  # noqa

    R = jnp.triu(A[:min_, :min_])
    if return_q:
        return Q[:n].T, R  # noqa
    else:
        return R


def _householder(a):
    if a.dtype == jnp.float64:
        eps = 1e-9
    else:
        eps = 1e-7

    alpha = a[0]
    s = jnp.sum(a[1:] ** 2)
    cond = s < eps

    def if_not_cond(v):
        t = (alpha**2 + s) ** 0.5
        v0 = jax.lax.cond(
            alpha <= 0, lambda _: alpha - t, lambda _: -s / (alpha + t), None
        )
        tau = 2 * v0**2 / (s + v0**2)
        v = v / v0
        v = v.at[0].set(1.0)
        return v, tau

    return jax.lax.cond(cond, lambda v: (v, 0.0), if_not_cond, a)


def qr_jvp_rule(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    q, r = _qr(x, True)
    m, n = x.shape
    min_ = min(m, n)
    if m < n:
        dx = dx[:, :m]
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(q.T, dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - qt_dx_rinv_lower.T  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    do = do + jnp.eye(min_, dtype=do.dtype) * (
        qt_dx_rinv - jnp.real(qt_dx_rinv)
    )
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return r, dr


qr.defjvp(qr_jvp_rule)
