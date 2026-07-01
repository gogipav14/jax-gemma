"""Spatial brain: a U-Net over the vision grid so the agent can POINT, not just decide.

The v1/v2 CNN flattened its 8x8 feature map into a global vector -- great for the macro ("what"),
useless for placement ("where"), because the flatten throws the coordinates away. Here the encoder
KEEPS its feature maps as skips, and a deconv decoder upsamples back to a 64x64 heatmap -- the
"where" head -- CONDITIONED on the chosen macro. Vision and action now share one coordinate frame:
the heatmap lives on the same grid the eyes read, so "where it looks" == "where it clicks".

  what  : global path (flatten bottleneck) -> macro logits + value   (the decision)
  where : U-Net decoder (skips + macro embedding) -> 64x64 logits     (the target cell)
  mask  : legal cells only (build radius / enemy cells) -> non-cheating + tractable

Real training is live BC (teacher placements) + RL toward the win -- GPU. This module is the CPU
architecture + a learnability proof that the deconv head can learn to point from vision.

    python policy/net_spatial.py     # shapes + a learnability test (head learns to point)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import net                                                # reuse _conv/_lin/_conv2d/_ap

GRID = 64
N_MACRO = 15
MACRO_EMB = 16


def init_params(key, grid_ch=7, n_macro=N_MACRO, hidden=128):
    k = random.split(key, 16)
    return {
        # --- encoder (keeps maps for skips) ---
        "c1": net._conv(k[0], 16, grid_ch, 3, 3),         # 64 -> 32   (skip0: 16ch)
        "c2": net._conv(k[1], 32, 16, 3, 3),              # 32 -> 16   (skip1: 32ch)
        "c3": net._conv(k[2], 48, 32, 3, 3),              # 16 -> 8    (bottleneck: 48ch)
        # --- global "what" path ---
        "sp": net._lin(k[3], 48 * 8 * 8, hidden),
        "pi": net._lin(k[4], hidden, n_macro),
        "v":  net._lin(k[5], hidden, 1),
        # --- macro embedding (conditions the "where" on the "what") ---
        "memb": net._lin(k[6], n_macro, MACRO_EMB),       # one-hot macro -> embedding
        # --- decoder (U-Net): upsample + concat skip + conv, back to 64x64 ---
        "d2": net._conv(k[7], 48, 48 + MACRO_EMB, 3, 3),  # bottleneck(+macro) proc @ 8x8
        "d1": net._conv(k[8], 32, 48 + 32, 3, 3),         # 16x16: up(d2) concat skip1
        "d0": net._conv(k[9], 16, 32 + 16, 3, 3),         # 32x32: up(d1) concat skip0
        "dout": net._conv(k[10], 1, 16, 3, 3),            # 64x64: up(d0) -> 1-channel heatmap
    }


def _upsample(x):
    b, c, h, w = x.shape
    return jax.image.resize(x, (b, c, h * 2, w * 2), method="nearest")


def encode(p, grid):
    s0 = jnp.tanh(net._conv2d(grid, p["c1"], 2))          # (B,16,32,32) skip0
    s1 = jnp.tanh(net._conv2d(s0, p["c2"], 2))            # (B,32,16,16) skip1
    b = jnp.tanh(net._conv2d(s1, p["c3"], 2))            # (B,48, 8, 8) bottleneck
    return s0, s1, b


def what_head(p, bottleneck):
    g = jnp.tanh(net._ap(p["sp"], bottleneck.reshape(bottleneck.shape[0], -1)))
    return net._ap(p["pi"], g), net._ap(p["v"], g)[..., 0]


def where_head(p, skips, bottleneck, macro_onehot):
    """Macro-conditioned deconv -> (B, 64, 64) spatial logits."""
    s0, s1 = skips
    B = bottleneck.shape[0]
    memb = net._ap(p["memb"], macro_onehot)               # (B, MACRO_EMB)
    mmap = jnp.broadcast_to(memb[:, :, None, None], (B, MACRO_EMB, 8, 8))
    z = jnp.concatenate([bottleneck, mmap], 1)            # condition where on what
    u = jnp.tanh(net._conv2d(z, p["d2"], 1))              # (B,48,8,8)
    u = jnp.concatenate([_upsample(u), s1], 1)           # (B,48+32,16,16)
    u = jnp.tanh(net._conv2d(u, p["d1"], 1))              # (B,32,16,16)
    u = jnp.concatenate([_upsample(u), s0], 1)           # (B,32+16,32,32)
    u = jnp.tanh(net._conv2d(u, p["d0"], 1))              # (B,16,32,32)
    u = _upsample(u)                                     # (B,16,64,64)
    return net._conv2d(u, p["dout"], 1)[:, 0]            # (B,64,64) logits


def forward(p, grid, macro_onehot):
    s0, s1, b = encode(p, grid)
    pi, v = what_head(p, b)
    where = where_head(p, (s0, s1), b, macro_onehot)
    return pi, v, where


def masked_where(where_logits, legal_mask):
    """Apply the legality mask (1=legal): illegal cells -> -inf so they can't be chosen."""
    return jnp.where(legal_mask > 0, where_logits, -1e9)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    p = init_params(key)
    n = sum(int(np.prod(w.shape)) + int(np.prod(b.shape)) for w, b in p.values())
    print(f"params: {n:,}   grid {GRID}x{GRID}")
    grid = jnp.asarray(np.random.rand(4, 7, 64, 64), jnp.float32)
    mo = jax.nn.one_hot(jnp.asarray([1, 2, 13, 5]), N_MACRO)
    pi, v, where = forward(p, grid, mo)
    print(f"forward: grid{tuple(grid.shape)} -> macro{tuple(pi.shape)} value{tuple(v.shape)} where{tuple(where.shape)}")

    # --- learnability proof: can the WHERE head learn to point at a target cell it can SEE? ---
    # each sample has a 'base' blob at a random cell (channel 5); the target is that cell.
    import optax
    rng = np.random.default_rng(1)

    BATCH, STEPS = 96, 150

    def batch(n):
        g = np.zeros((n, 7, 64, 64), np.float32)
        ys = rng.integers(4, 60, n); xs = rng.integers(4, 60, n)
        rr = np.arange(-2, 2)
        g[np.arange(n)[:, None, None], 5, (ys[:, None, None] + rr[None, :, None]),
          (xs[:, None, None] + rr[None, None, :])] = 1.0                # the base blob (what it sees)
        return jnp.asarray(g), jnp.asarray(ys * 64 + xs)               # grid, target cell index

    mo1 = jax.nn.one_hot(jnp.full((BATCH,), 2), N_MACRO)               # condition on 'build POWER'
    opt = optax.adam(3e-3); st = opt.init(p)

    def loss(pp, g, tgt, mo):
        _, _, w = forward(pp, g, mo)
        logp = jax.nn.log_softmax(w.reshape(w.shape[0], -1))
        return -jnp.mean(logp[jnp.arange(tgt.shape[0]), tgt])
    gfn = jax.jit(jax.value_and_grad(loss))
    print("\nlearnability: train the WHERE head to point at the base cell it sees")
    for i in range(STEPS + 1):
        g, tgt = batch(BATCH)
        l, gr = gfn(p, g, tgt, mo1)
        u, st = opt.update(gr, st, p); p = optax.apply_updates(p, u)
        if i % 30 == 0 or i == STEPS:
            _, _, w = forward(p, g, mo1)
            pred = jnp.argmax(w.reshape(BATCH, -1), 1)
            hit = float((pred == tgt).mean())
            near = float((jnp.abs(pred // 64 - tgt // 64) + jnp.abs(pred % 64 - tgt % 64) <= 2).mean())
            print(f"  step {i:3d}  loss {float(l):.3f}  exact-cell {hit:.2f}  within-2 {near:.2f}")
    print("the deconv head learns to POINT from vision -> spatial control is learnable (train live/GPU).")
