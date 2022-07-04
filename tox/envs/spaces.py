from typing import Tuple

import jax.numpy as jnp


class Box:
    def __init__(
        self,
        low: jnp.ndarray,
        high: jnp.ndarray,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def contains(self, x: jnp.ndarray) -> bool:
        range_cond = jnp.logical_and(
            jnp.all(x >= self.low), jnp.all(x <= self.high)
        )
        return range_cond

    def clip(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.low, self.high)
