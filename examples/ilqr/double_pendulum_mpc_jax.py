from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jax import block_until_ready

from tox.objects import Box
from tox.utils import discretize_dynamics, wrap_angle
from tox.solvers import ilqr as ilqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e4, 1e4, 1e4, 1e4]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost = jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0]))
    action_cost = jnp.diag(jnp.array([1e-3, 1e-3]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def double_pendulum(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    # https://underactuated.mit.edu/multibody.html#section1

    g = 9.81
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    k1, k2 = 1e-3, 1e-3

    th1, th2, dth1, dth2 = state
    u1, u2 = action

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    s12 = jnp.sin(th1 + th2)

    # inertia
    M = jnp.array(
        [
            [
                (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * c2,
                m2 * l2**2 + m2 * l1 * l2 * c2,
            ],
            [
                m2 * l2**2 + m2 * l1 * l2 * c2,
                m2 * l2**2
            ],
        ]
    )

    # Corliolis
    C = jnp.array(
        [
            [
                0.0,
                -m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2
            ],
            [
                0.5 * m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2,
                -0.5 * m2 * l1 * l2 * dth1 * s2,
            ],
        ]
    )

    # gravity
    tau = -g * jnp.array(
        [
            (m1 + m2) * l1 * s1 + m2 * l2 * s12,
            m2 * l2 * s12
        ]
    )

    B = jnp.eye(2)

    u1 = u1 - k1 * dth1
    u2 = u2 - k2 * dth2

    u = jnp.hstack([u1, u2])
    v = jnp.hstack([dth1, dth2])

    a = jnp.linalg.solve(M, tau + B @ u - C @ v)

    return jnp.hstack((v, a))


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=double_pendulum, simulation_step=simulation_step, downsampling=downsampling
)

state_dim = 4
action_dim = 2

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=-50.0 * jnp.ones((action_dim,)),
    high=50.0 * jnp.ones((action_dim,)),
    shape=(action_dim,),
)

init_state = jnp.array(
    [
        wrap_angle(-0.01),
        wrap_angle(0.01),
        -0.01,
        0.01,
    ]
)
goal_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

nb_steps = 100
horizon = 50

key = jr.PRNGKey(747)
key, control_key = jr.split(key, 2)
control = 1e-2 * jr.normal(control_key, shape=(horizon, action_dim))

options = ilqr.Hyperparameters(max_iter=250)

start = clock.time()
state, action, cost = ilqr.exact_mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    control,
    action_space,
    horizon,
    nb_steps,
    options,
)
block_until_ready(state)
end = clock.time()
print("Compilation + Execution Time:", end - start)

plt.subplot(6, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("q1")
plt.subplot(6, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("q2")
plt.subplot(6, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("dq1")
plt.subplot(6, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("dq2")
plt.subplot(6, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u1")
plt.subplot(6, 1, 6)
plt.plot(action[:, 1])
plt.ylabel("u2")
plt.xlabel("t")
plt.show()
