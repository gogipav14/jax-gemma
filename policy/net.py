"""The brain with EYES: a fused policy/value network.

  spatial branch  : a small CNN over the 64x64x7 vision grid  -> WHERE things are
  scalar branch   : an MLP over the role-count + scalar obs    -> WHAT we have
  fuse -> heads   : concat the two -> MLP -> {policy over macros, value}  -> the DECISION

This is the AlphaStar-mini shape (spatial-CNN + scalar-MLP -> heads); an entity-attention branch
over the unit set is a later refinement. Pure JAX (no flax); runs on CPU.

    python policy/net.py     # forward-pass + decision self-test (shapes, runs on CPU)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def _lin(k, i, o):
    return (random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), jnp.zeros(o))


def _conv(k, o, i, kh, kw):
    return (random.normal(k, (o, i, kh, kw)) * jnp.sqrt(2.0 / (i * kh * kw)), jnp.zeros(o))


def init_params(key, grid_ch=7, grid_dim=64, scalar_dim=21, n_act=15, hidden=128):
    k = random.split(key, 8)
    flat = 32 * (grid_dim // 8) * (grid_dim // 8)        # after 3 stride-2 convs
    return {"c1": _conv(k[0], 16, grid_ch, 3, 3),
            "c2": _conv(k[1], 32, 16, 3, 3),
            "c3": _conv(k[2], 32, 32, 3, 3),
            "sp": _lin(k[3], flat, 64),                  # spatial feature
            "sc": _lin(k[4], scalar_dim, 64),            # scalar feature
            "f":  _lin(k[5], 128, hidden),               # fused
            "pi": _lin(k[6], hidden, n_act),
            "v":  _lin(k[7], hidden, 1)}


def _conv2d(x, layer, stride):
    w, b = layer
    y = jax.lax.conv_general_dilated(x, w, (stride, stride), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]


def _ap(layer, z):
    w, b = layer
    return z @ w + b


def forward(p, grid, scalar):
    """grid: (B, C, H, W) in [0,1]; scalar: (B, scalar_dim). Returns (logits[B,n_act], value[B])."""
    h = jnp.tanh(_conv2d(grid, p["c1"], 2))              # 64 -> 32
    h = jnp.tanh(_conv2d(h, p["c2"], 2))                 # 32 -> 16
    h = jnp.tanh(_conv2d(h, p["c3"], 2))                 # 16 -> 8
    h = h.reshape(h.shape[0], -1)
    sp = jnp.tanh(_ap(p["sp"], h))
    sc = jnp.tanh(_ap(p["sc"], scalar))
    f = jnp.tanh(_ap(p["f"], jnp.concatenate([sp, sc], -1)))
    return _ap(p["pi"], f), _ap(p["v"], f)[..., 0]


def decide(p, grid, scalar, key=None, greedy=True):
    """One decision from a single (grid, scalar) observation -> (macro_index, value)."""
    g = grid[None] if grid.ndim == 3 else grid
    s = scalar[None] if scalar.ndim == 1 else scalar
    logits, value = forward(p, jnp.asarray(g), jnp.asarray(s))
    a = int(jnp.argmax(logits[0])) if greedy else int(random.categorical(key, logits[0]))
    return a, float(value[0])


if __name__ == "__main__":
    import numpy as np
    key = random.PRNGKey(0)
    p = init_params(key)
    n = sum(int(np.prod(w.shape)) + int(np.prod(b.shape)) for w, b in p.values())
    print(f"params: {n:,}   devices: {jax.devices()}")

    # batched forward
    grid = jnp.asarray(np.random.rand(4, 7, 64, 64), jnp.float32)     # normalized vision grids
    scalar = jnp.asarray(np.random.rand(4, 21), jnp.float32)
    logits, value = forward(p, grid, scalar)
    print(f"forward: grid{tuple(grid.shape)} + scalar{tuple(scalar.shape)} -> logits{tuple(logits.shape)} value{tuple(value.shape)}")

    # single decision (eyes + state -> a macro)
    a, v = decide(p, np.random.rand(7, 64, 64).astype(np.float32), np.random.rand(21).astype(np.float32))
    print(f"decide: macro index = {a}  value = {v:+.2f}")
    print("fused brain (eyes + state -> decision) forward pass OK on CPU")
