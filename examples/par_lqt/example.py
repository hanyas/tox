from examples.par_lqt.model import model_parameters
from tox.helpers import linearization, quadratize_final_cost_par, quadratize_transient_cost_par

from tox.solvers.parallel.par_lqt import parBackwardPass, parForwardPass
import jax.numpy as jnp
from matplotlib import pyplot as plt

# model
final_cost, transient_cost, dynamics, state_space, action_space, reference = model_parameters()
quadratic_final_cost = quadratize_final_cost_par(final_cost, reference.final)
quadratic_transient_cost = quadratize_transient_cost_par(transient_cost, reference.transient)
linear_dynamics = linearization(dynamics, state_space, reference.transient)
xs_init = jnp.zeros((2,))
#
# Parallel lqt
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
