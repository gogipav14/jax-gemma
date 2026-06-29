"""Minimal JAX policy/value network for the YR agent: obs -> action-type logits + value.

This is the trunk + the primary action-type head (a full AlphaStar-style policy would add
auto-regressive heads for type_id / cell / target). Validated on CPU here; PPO self-play
training runs on a GPU box (CPU-only + non-headless YR can't train at scale — see docs).

The whole point: unlike the static 'Nightmare' script, THIS improves via self-play.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

N_ACTIONS = 11   # ActType count (NOOP..DEPLOY)


def init_params(key, obs_dim: int, hidden: int = 128):
    k1, k2, k3, k4 = random.split(key, 4)

    def lin(k, i, o):
        return (random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), jnp.zeros(o))

    return {"w1": lin(k1, obs_dim, hidden), "w2": lin(k2, hidden, hidden),
            "pi": lin(k3, hidden, N_ACTIONS), "v": lin(k4, hidden, 1)}


def forward(params, obs):
    def ap(layer, x):
        w, b = layer
        return x @ w + b
    h = jnp.tanh(ap(params["w1"], obs))
    h = jnp.tanh(ap(params["w2"], h))
    logits = ap(params["pi"], h)
    value = ap(params["v"], h)[..., 0]
    return logits, value


def act(params, obs, key):
    logits, value = forward(params, obs)
    action_type = random.categorical(key, logits)
    return action_type, value


if __name__ == "__main__":
    import numpy as np
    key = random.PRNGKey(0)
    obs_dim = 17  # matches yr_env/env.py OBS_DIM
    params = init_params(key, obs_dim)
    batch = jnp.asarray(np.random.randn(4, obs_dim), dtype=jnp.float32)
    logits, value = forward(params, batch)
    a, v = act(params, batch[0], random.PRNGKey(1))
    print("jax devices:", jax.devices())
    print("logits:", logits.shape, "value:", value.shape)
    print("sampled action_type:", int(a), " value:", round(float(v), 3))
    print("JAX policy forward pass OK -> the brain is wired (training = GPU/self-play)")
