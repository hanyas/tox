from typing import NamedTuple, Tuple, Callable

import jax.numpy as jnp


class Box(NamedTuple):
    low: jnp.ndarray
    high: jnp.ndarray
    shape: Tuple

    def __call__(self, func) -> Callable:
        return lambda *args: jnp.clip(func(*args), self.low, self.high)

    def contains(self, x: jnp.ndarray) -> bool:
        range_cond = jnp.logical_and(
            jnp.all(x >= self.low), jnp.all(x <= self.high)
        )
        return range_cond

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.low, self.high)
