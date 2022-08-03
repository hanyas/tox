from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)

import jax.numpy as jnp
import jax.random as jr
from jax import block_until_ready

from tox.objects import Trajectory, Box
from tox.utils import discretize_dynamics, wrap_angle
from tox.solvers import ilqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e4, 1e4, 1e4, 1e4]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3],
        )
    )
    c = (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e1, 1e0, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-1]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3],
        )
    )
    c = (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += action.T @ action_cost @ action
    return c


def acrobot(state: jnp.ndarray, action: jnp.ndarray, time: int) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#section1

    g = 9.81
    l1, l2 = 1.0, 2.0
    m1, m2 = 1.0, 1.0
    k2 = 1e-3

    lc1 = l1 / 2.0
    lc2 = l2 / 2.0

    I1 = m1 * l1**2 / 3.0
    I2 = m2 * l2**2 / 3.0

    th1, th2, dth1, dth2 = state
    u2 = action

    th1, th2 = state[:2]
    dth1, dth2 = state[2:]

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    s12 = jnp.sin(th1 + th2)

    # inertia
    M = jnp.array(
        [
            [
                I1 + I2 + m2 * l1**2 + 2.0 * m2 * l1 * lc2 * c2,
                I2 + m2 * l1 * lc2 * c2,
            ],
            [I2 + m2 * l1 * lc2 * c2, I2],
        ]
    )

    # Corliolis
    C = jnp.array(
        [
            [-2.0 * m2 * l1 * lc2 * s2 * dth2, -m2 * l1 * lc2 * s2 * dth2],
            [
                m2 * l1 * lc2 * s2 * dth1,
                0.0,
            ],
        ]
    )

    # gravity
    tau = -g * jnp.array(
        [m1 * lc1 * s1 + m2 * (l1 * s1 + lc2 * s12), m2 * lc2 * s12]
    )

    B = jnp.array([[0.0, 0.0], [0.0, 1.0]])

    u2 = u2 - k2 * dth2

    u = jnp.hstack([0.0, u2])
    v = jnp.hstack([dth1, dth2])

    a = jnp.linalg.solve(M, tau + B @ u - C @ v)

    return jnp.hstack((v, a))


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=acrobot, simulation_step=simulation_step, downsampling=downsampling
)

state_dim = 4
action_dim = 1

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=-100.0 * jnp.ones((action_dim,)),
    high=100.0 * jnp.ones((action_dim,)),
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

key = jr.PRNGKey(123)

key, policy_key = jr.split(key, 2)
policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

options = ilqr.Hyperparameters(max_iter=250, tol_fun=1e-2)

start = clock.time()
state, action = ilqr.mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    reference,
    options,
    nb_steps,
)
block_until_ready(state)
end = clock.time()
print("Compilation + Execution Time:", end - start)

plt.subplot(5, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("q1")
plt.subplot(5, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("q2")
plt.subplot(5, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("dq1")
plt.subplot(5, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("dq2")
plt.subplot(5, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()

init_state = jnp.array(
    [
        wrap_angle(0.01),
        wrap_angle(-0.01),
        0.01,
        -0.01,
    ]
)

key, policy_key = jr.split(key, 2)
policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

start = clock.time()
state, action = ilqr.mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    reference,
    options,
    nb_steps,
)
block_until_ready(state)
end = clock.time()
print("Execution Time:", end - start)

plt.subplot(5, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("q1")
plt.subplot(5, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("q2")
plt.subplot(5, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("dq1")
plt.subplot(5, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("dq2")
plt.subplot(5, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
