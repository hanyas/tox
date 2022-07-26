from typing import Callable

import jax.numpy as jnp


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
