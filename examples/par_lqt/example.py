import jax.numpy as jnp
from examples.par_lqt.model import model_parameters
from tox.helpers import linearize_dynamics, quadratize_final_cost, quadratize_transient_cost
from tox.solvers.parallel.par_lqt import parBackwardPass, parForwardPass
from matplotlib import pyplot as plt


final_cost, transient_cost, dynamics, state_space, action_space, reference, goal_state = model_parameters()
time = jnp.linspace(0, reference.horizon, reference.horizon + 1)
quadratic_final_cost = quadratize_final_cost(final_cost, goal_state, reference.final)
quadratic_transient_cost = quadratize_transient_cost(transient_cost, goal_state, reference.transient, time[:-1])
linear_dynamics = linearize_dynamics(dynamics, state_space, reference.transient, time[:-1])
xs_init = jnp.zeros((2,))


vx, Vxx, policy, res = parBackwardPass(quadratic_final_cost, quadratic_transient_cost, linear_dynamics, reference)
state, action = parForwardPass(xs_init, linear_dynamics, policy, vx)


plt.subplot(3, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
