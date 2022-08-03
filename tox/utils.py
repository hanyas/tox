from typing import Callable

import jax.numpy as jnp
from jax.lax import fori_loop


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


def discretize_dynamics(ode: Callable, simulation_step: float, downsampling: int):
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
