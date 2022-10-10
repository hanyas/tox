from typing import Callable

import jax.scipy
from jax import numpy as jnp
from jax import jacfwd, vmap

from jax import scipy as jsc
from jax.scipy.linalg import solve_triangular

from tox.objects import Box
from tox.utils import symmetrize, tria


def kalman_filter_dynamics(
    dynamics_mean: Callable,
    dynamics_noise: Callable,
    state_space: Box,
    observation_mean: Callable,
    observation_noise: Callable,
    observation_space: Box,
):
    dynamics_mean = state_space(dynamics_mean)
    observation_mean = observation_space(observation_mean)

    def belief_dynamics(
        bel_mu: jnp.ndarray,
        bel_cov: jnp.ndarray,
        action: jnp.ndarray,
        time: int,
    ):

        A = jacfwd(dynamics_mean, 0)(bel_mu, action, time)
        H = jacfwd(observation_mean, 0)(dynamics_mean(bel_mu, action, time), time)

        dyn_cov = dynamics_noise(bel_mu, action, time)
        obs_cov = observation_noise(dynamics_mean(bel_mu, action, time), time)

        G = symmetrize(A @ bel_cov @ A.T + dyn_cov)
        S = H @ G @ H.T + obs_cov
        K = jnp.linalg.solve(S, H @ G.T).T

        bel_mu_mu = dynamics_mean(bel_mu, action, time)
        bel_mu_cov = symmetrize(K @ S @ K.T)
        bel_cov = symmetrize(G - K @ S @ K.T)

        return bel_mu_mu, bel_mu_cov, bel_cov

    return belief_dynamics


def cubature_filter_dynamics(
    dynamics_mean: Callable,
    dynamics_noise: Callable,
    state_space: Box,
    observation_mean: Callable,
    observation_noise: Callable,
    observation_space: Box,
):
    def cubature_weights(dim):
        w = jnp.ones(shape=(2 * dim,)) / (2 * dim)
        xi = jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0) * jnp.sqrt(
            dim
        )
        return w, xi

    def sigma_points(mu, cov):
        dim = mu.shape[0]
        chol = jax.scipy.linalg.cholesky(cov)

        w, xi = cubature_weights(dim)
        xs = mu[None, :] + jnp.dot(chol, xi.T).T
        return xs, w

    def belief_dynamics(
        bel_mu: jnp.ndarray,
        bel_cov: jnp.ndarray,
        action: jnp.ndarray,
        time: int,
    ):

        _dynamics_mean = vmap(
            state_space(dynamics_mean), in_axes=(0, None, None)
        )
        dyn_cov = dynamics_noise(bel_mu, action, time)

        Xi, Wx = sigma_points(bel_mu, bel_cov)
        Xn = _dynamics_mean(Xi, action, time)

        mx = jnp.einsum("n,nd->d", Wx, Xn)
        Px = dyn_cov + jnp.einsum(
            "n,nd,nk->dk", Wx, Xn - mx[None], Xn - mx[None]
        )

        _observation_mean = vmap(
            observation_space(observation_mean), in_axes=(0, None)
        )
        obs_cov = observation_noise(mx, time)

        Xi, Wx = sigma_points(mx, Px)
        Yn = _observation_mean(Xi, time)

        my = jnp.einsum("n,nd->d", Wx, Yn)
        S = obs_cov + jnp.einsum(
            "n,nd,nk->dk", Wx, Yn - my[None], Yn - my[None]
        )
        C = jnp.einsum("n,nd,nk->dk", Wx, Xn - mx[None], Yn - my[None])

        # K = C @ jnp.linalg.inv(S)
        K = jnp.linalg.solve(S, C.T).T

        bel_mu_mu = mx
        bel_mu_cov = symmetrize(K @ S @ K.T)
        bel_cov = symmetrize(Px - K @ S @ K.T)

        return bel_mu_mu, bel_mu_cov, bel_cov

    return belief_dynamics


def kalman_filter(
    dynamics_mean: Callable,
    dynamics_noise: Callable,
    state_space: Box,
    observation_mean: Callable,
    observation_noise: Callable,
    observation_space: Box,
    bel_mu: jnp.ndarray,
    bel_cov: jnp.ndarray,
    action: jnp.ndarray,
    measurement: jnp.ndarray,
    time: int,
):
    _dynamics_mean = state_space(dynamics_mean)
    _observation_mean = observation_space(observation_mean)

    A = jacfwd(_dynamics_mean, 0)(bel_mu, action, time)
    H = jacfwd(_observation_mean, 0)(_dynamics_mean(bel_mu, action, time), time)

    dyn_cov = dynamics_noise(bel_mu, action, time)
    obs_cov = observation_noise(_dynamics_mean(bel_mu, action, time), time)

    G = symmetrize(A @ bel_cov @ A.T + dyn_cov)
    S = H @ G @ H.T + obs_cov
    K = jnp.linalg.solve(S, H @ G.T).T

    next_bel_mu = _dynamics_mean(bel_mu, action, time) + K @ (
        measurement
        - _observation_mean(_dynamics_mean(bel_mu, action, time), time)
    )
    next_bel_cov = symmetrize(G - K @ S @ K.T)

    return next_bel_mu, next_bel_cov


def cubature_filter(
    dynamics_mean: Callable,
    dynamics_noise: Callable,
    state_space: Box,
    observation_mean: Callable,
    observation_noise: Callable,
    observation_space: Box,
    bel_mu: jnp.ndarray,
    bel_cov: jnp.ndarray,
    action: jnp.ndarray,
    measurement: jnp.ndarray,
    time: int,
):
    def cubature_weights(dim):
        w = jnp.ones(shape=(2 * dim,)) / (2 * dim)
        xi = jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0) * jnp.sqrt(
            dim
        )
        return w, xi

    def sigma_points(mu, cov):
        dim = mu.shape[0]
        chol = jax.scipy.linalg.cholesky(cov)

        w, xi = cubature_weights(dim)
        xs = mu[None, :] + jnp.dot(chol, xi.T).T
        return xs, w

    _dynamics_mean = vmap(state_space(dynamics_mean), in_axes=(0, None, None))
    dyn_cov = dynamics_noise(bel_mu, action, time)

    Xi, Wx = sigma_points(bel_mu, bel_cov)
    Xn = _dynamics_mean(Xi, action, time)

    mx = jnp.einsum("n,nd->d", Wx, Xn)
    Px = dyn_cov + jnp.einsum(
        "n,nd,nk->dk", Wx, Xn - mx[None], Xn - mx[None]
    )

    _observation_mean = vmap(
        observation_space(observation_mean), in_axes=(0, None)
    )
    obs_cov = observation_noise(mx, time)

    Xi, Wx = sigma_points(mx, Px)
    Yn = _observation_mean(Xi, time)

    my = jnp.einsum("n,nd->d", Wx, Yn)
    S = obs_cov + jnp.einsum(
        "n,nd,nk->dk", Wx, Yn - my[None], Yn - my[None]
    )
    C = jnp.einsum("n,nd,nk->dk", Wx, Xn - mx[None], Yn - my[None])

    # K = C @ jnp.linalg.inv(S)
    K = jnp.linalg.solve(S, C.T).T

    next_bel_mu = mx + K @ (measurement - my)
    next_bel_cov = symmetrize(Px - K @ S @ K.T)

    return next_bel_mu, next_bel_cov


def sqrt_kalman_filter_dynamics(
    dynamics: Callable,
    state_space: Box,
    observation: Callable,
    observation_space: Box,
    flatten_belief: Callable,
    unflatten_belief: Callable,
):

    dx = state_space.shape[0]
    dy = observation_space.shape[0]

    dynamics = state_space(dynamics)
    observation = observation_space(observation)

    def belief_dynamics_mean(belief, action, time):
        # state is a concatenation of mean and lower triangle cholesky of belief
        m, Pc = unflatten_belief(belief)

        delta = jnp.zeros((dx,))
        A = jacfwd(dynamics, 0)(m, action, delta, time)
        Mc = jacfwd(dynamics, 2)(m, action, delta, time)

        eta = jnp.zeros((dy,))
        H = jacfwd(observation, 0)(dynamics(m, action, delta, time), eta, time)
        Nc = jacfwd(observation, 1)(dynamics(m, action, delta, time), eta, time)

        Pc_ = tria(jnp.concatenate([A @ Pc, Mc], axis=1))

        zeros = jnp.zeros((dx, dy))
        Bc = tria(jnp.block([[H @ Pc_, Nc],
                             [Pc_, zeros]]))

        m = dynamics(m, action, delta, time)
        Pc = Bc[dy:, dy:]

        next_belief_mean = flatten_belief(m, Pc)
        return next_belief_mean

    def belief_dynamics_noise(belief, action, time):
        # state is a concatenation of mean and lower triangle cholesky of belief
        m, Pc = unflatten_belief(belief)

        delta = jnp.zeros((dx,))
        A = jacfwd(dynamics, 0)(m, action, delta, time)
        Mc = jacfwd(dynamics, 2)(m, action, delta, time)

        eta = jnp.zeros((dy,))
        H = jacfwd(observation, 0)(dynamics(m, action, delta, time), eta, time)
        Nc = jacfwd(observation, 1)(dynamics(m, action, delta, time), eta, time)

        Pc_ = tria(jnp.concatenate([A @ Pc, Mc], axis=1))

        zeros = jnp.zeros((dx, dy))
        Bc = tria(jnp.block([[H @ Pc_, Nc],
                             [Pc_, zeros]]))

        Q = Bc[dy:, :dy]
        db = int(dx + dx * (dx - 1) / 2)
        next_belief_chol = jsc.linalg.block_diag(tria(Q), jnp.zeros((db + dx - dy, db + dx - dy)))

        return next_belief_chol

    return belief_dynamics_mean, belief_dynamics_noise


def sqrt_kalman_filter(
    dynamics: Callable,
    state_space: Box,
    observation: Callable,
    observation_space: Box,
    flatten_belief: Callable,
    unflatten_belief: Callable,
):

    dx = state_space.shape[0]
    dy = observation_space.shape[0]

    dynamics = state_space(dynamics)
    observation = observation_space(observation)

    def bayes_filter(
        belief: jnp.ndarray,
        action: jnp.ndarray,
        measurement: jnp.ndarray,
        time: int
    ):

        m, Pc = unflatten_belief(belief)

        delta = jnp.zeros((dx,))
        A = jacfwd(dynamics, 0)(m, action, delta, time)
        Mc = jax.jacfwd(dynamics, 2)(m, action, delta, time)

        eta = jnp.zeros((dy,))
        H = jacfwd(observation, 0)(dynamics(m, action, delta, time), eta, time)
        Nc = jacfwd(observation, 1)(dynamics(m, action, delta, time), eta, time)

        Pc_ = tria(jnp.concatenate([A @ Pc, Mc], axis=1))

        zeros = jnp.zeros((dx, dy))
        Bc = tria(jnp.block([[H @ Pc_, Nc],
                             [Pc_, zeros]]))

        G = Bc[dy:, :dy]
        I = Bc[:dy, :dy]

        diff = measurement - observation(dynamics(m, action, delta, time), eta, time)
        m = dynamics(m, action, delta, time) + G @ solve_triangular(I, diff, lower=True)
        Pc = Bc[dy:, dy:]

        return flatten_belief(m, Pc)

    return bayes_filter
