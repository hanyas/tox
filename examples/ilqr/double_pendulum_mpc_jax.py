from typing import Callable
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)

import jax.numpy as jnp
import jax.random as jr

from jax import jit
from jax import block_until_ready
from jax.lax import fori_loop, scan

from tox.objects import Trajectory, Box
from tox.utils import runge_kutta
from tox.solvers import ilqr
from tox.solvers.ilqr import Hyperparameters, LinearPolicy

import time as clock
import matplotlib.pyplot as plt


simulation_step = 0.01
downsampling = 5

state_dim = 4
action_dim = 2


def wrap_angle(x):
    # wrap angle between [0, 2*pi]
    return x % (2.0 * jnp.pi)


def final_cost(state: jnp.ndarray) -> float:
    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e4, 1e4, 1e4, 1e4]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = (_wrapped - goal).T @ final_state_cost @ (_wrapped - goal)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> float:

    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    state_cost: jnp.ndarray = jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-3, 1e-3]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = (_wrapped - goal).T @ state_cost @ (_wrapped - goal)
    c += action.T @ action_cost @ action
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


def dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    def _step(t, state):
        next_state = runge_kutta(
            state,
            action,
            time + t * simulation_step,
            double_pendulum,
            simulation_step,
        )
        return next_state

    return fori_loop(
        lower=0,
        upper=downsampling,
        body_fun=_step,
        init_val=state,
    )


@partial(jit, static_argnums=(0, 1, 2, -1))
def mpc_rollout(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    init_state: jnp.ndarray,
    options: Hyperparameters,
    nb_steps: int,
) -> (jnp.ndarray, jnp.ndarray):
    def mpc_step(carry, args):
        policy, reference, state = carry

        next_policy, next_reference = ilqr.jax_solver(
            final_cost,
            transient_cost,
            dynamics,
            state_space,
            policy,
            action_space,
            reference,
            state,
            options,
        )

        action = next_reference.action[0]
        next_state = state_space.clip(dynamics(state, action, 0))

        return (next_policy, next_reference, next_state), (next_state, action)

    _, (state, action) = scan(
        mpc_step, init=(policy, reference, init_state), xs=jnp.arange(nb_steps)
    )

    state = jnp.vstack((init_state, state))
    return state, action


nb_steps = 100
horizon = 50

key = jr.PRNGKey(747)
key, policy_key = jr.split(key, 2)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

init_state = jnp.array(
    [
        wrap_angle(-0.01),
        wrap_angle(0.01),
        -0.01,
        0.01,
    ]
)
options = ilqr.Hyperparameters(max_iter=250)

start = clock.time()
state, action = mpc_rollout(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    policy,
    action_space,
    reference,
    init_state,
    options,
    nb_steps,
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

key, policy_key = jr.split(key, 2)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

init_state = jnp.array(
    [
        wrap_angle(0.01),
        wrap_angle(-0.01),
        0.01,
        -0.01,
    ]
)

start = clock.time()
state, action = mpc_rollout(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    policy,
    action_space,
    reference,
    init_state,
    options,
    nb_steps,
)
block_until_ready(state)
end = clock.time()
print("Execution Time:", end - start)

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
