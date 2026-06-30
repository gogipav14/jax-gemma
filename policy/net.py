"""The brain with EYES + a ROSTER: a fused policy/value network (AlphaStar-mini shape).

  spatial branch  : a small CNN over the 64x64x7 vision grid     -> WHERE things are
  entity branch   : a Transformer over the unit/building tokens   -> WHAT each thing is (per-unit)
  scalar branch   : an MLP over the role-count + scalar obs        -> the aggregate WHAT
  fuse -> heads   : concat the three -> MLP -> {policy, value, aux} -> the DECISION

The entity branch is the modality AlphaStar credits for targeting / reading the enemy composition:
each visible technO is a token (role, side, position, at-base), self-attention reasons over the set,
masked-mean pool -> one entity feature. Role-counts (the scalar) throw per-unit identity away; this
keeps it. Vanilla softmax attention (the set is small, <= MAX_ENT) -- per the research, exotic
efficient-attention isn't needed at this token count. The attention block's projections (wo, ff) are
isolated so a fixed Walsh-Hadamard / butterfly transform (arXiv:2603.08343) can drop in later.

Backward-compatible: forward(grid, scalar) with no entities -> the entity feature is zero, so the
existing CNN+scalar trainers keep working unchanged. Pure JAX (no flax); runs on CPU.

    python policy/net.py     # forward-pass + decision self-test (shapes, runs on CPU)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

# aux heads (the game STRUCTURE the brain learns alongside copying the teacher's move) --
#   bld -> roles LEGAL to build now (prereqs) | cnt -> answer to the threat (counters)
#   thr -> base under attack (defense/info)   | ev  -> position eval V (not the RL goal-value)
N_BUILD_ROLE, N_COUNTER = 7, 4
# entity token = role one-hot (ENT_ROLES) + [is_enemy, x/64, y/64, at_base]; see rl_env.encode_entities
ENT_FEAT, ENT_DIM, MAX_ENT = 17, 64, 48


def _lin(k, i, o):
    return (random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), jnp.zeros(o))


def _conv(k, o, i, kh, kw):
    return (random.normal(k, (o, i, kh, kw)) * jnp.sqrt(2.0 / (i * kh * kw)), jnp.zeros(o))


def _norm(k, d):
    return (jnp.ones(d), jnp.zeros(d))           # LayerNorm (scale, bias)


def init_params(key, grid_ch=7, grid_dim=64, scalar_dim=21, n_act=15, hidden=128,
                ent_feat=ENT_FEAT, ent_dim=ENT_DIM):
    k = random.split(key, 24)
    flat = 32 * (grid_dim // 8) * (grid_dim // 8)        # after 3 stride-2 convs
    return {"c1": _conv(k[0], 16, grid_ch, 3, 3),
            "c2": _conv(k[1], 32, 16, 3, 3),
            "c3": _conv(k[2], 32, 32, 3, 3),
            "sp": _lin(k[3], flat, 64),                  # spatial feature (WHERE)
            "sc": _lin(k[4], scalar_dim, 64),            # scalar feature (aggregate WHAT)
            # --- entity transformer (per-unit WHAT) ---
            "et":  _lin(k[10], ent_feat, ent_dim),       # token embedding
            "ln1": _norm(k[11], ent_dim),
            "wq":  _lin(k[12], ent_dim, ent_dim), "wk": _lin(k[13], ent_dim, ent_dim),
            "wv":  _lin(k[14], ent_dim, ent_dim), "wo": _lin(k[15], ent_dim, ent_dim),
            "ln2": _norm(k[16], ent_dim),
            "ff1": _lin(k[17], ent_dim, 2 * ent_dim), "ff2": _lin(k[18], 2 * ent_dim, ent_dim),
            "en":  _lin(k[19], ent_dim, 64),             # pooled entity feature
            # --- fuse + heads ---
            "f":   _lin(k[5], 192, hidden),              # concat[spatial, scalar, entity] -> fused
            "pi":  _lin(k[6], hidden, n_act),            # policy over macros
            "v":   _lin(k[7], hidden, 1),                # RL value (goal-return; left for RL)
            "bld": _lin(k[8], hidden, N_BUILD_ROLE),     # aux: buildable / prereqs
            "cnt": _lin(k[9], hidden, N_COUNTER),        # aux: counter to the threat
            "thr": _lin(k[20], hidden, 1),               # aux: threat at base
            "ev":  _lin(k[21], hidden, 1),               # aux: position eval V
            "pwr": _lin(k[22], hidden, 1)}               # aux: power adequate (output >= drain -> defenses fire)


def _conv2d(x, layer, stride):
    w, b = layer
    y = jax.lax.conv_general_dilated(x, w, (stride, stride), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]


def _ap(layer, z):
    w, b = layer
    return z @ w + b


def _ln(x, layer, eps=1e-5):
    g, b = layer
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def _entity_feat(p, entities, mask):
    """entities: (B, N, ENT_FEAT); mask: (B, N) 1=real token. -> (B, 64) pooled entity feature.

    One pre-norm self-attention block + FF over the roster, then masked-mean pool. Attention is
    masked so padding tokens never contribute. The set is small so a single head/block suffices."""
    x = jnp.tanh(_ap(p["et"], entities))                  # (B, N, d)
    h = _ln(x, p["ln1"])
    q, k, v = _ap(p["wq"], h), _ap(p["wk"], h), _ap(p["wv"], h)
    d = q.shape[-1]
    scores = jnp.einsum("bnd,bmd->bnm", q, k) / jnp.sqrt(d)
    scores = scores + jnp.where(mask[:, None, :] > 0, 0.0, -1e9)   # mask padding keys
    a = jax.nn.softmax(scores, -1)
    x = x + _ap(p["wo"], jnp.einsum("bnm,bmd->bnd", a, v))         # attention residual
    x = x + _ap(p["ff2"], jax.nn.relu(_ap(p["ff1"], _ln(x, p["ln2"]))))   # FF residual
    m = mask[:, :, None]
    pooled = (x * m).sum(1) / jnp.clip(m.sum(1), 1.0)             # masked mean over real tokens
    return jnp.tanh(_ap(p["en"], pooled))


def _trunk(p, grid, scalar, entities=None, ent_mask=None):
    """Shared eyes + roster + state trunk -> the fused representation f (B, hidden)."""
    h = jnp.tanh(_conv2d(grid, p["c1"], 2))              # 64 -> 32
    h = jnp.tanh(_conv2d(h, p["c2"], 2))                 # 32 -> 16
    h = jnp.tanh(_conv2d(h, p["c3"], 2))                 # 16 -> 8
    sp = jnp.tanh(_ap(p["sp"], h.reshape(h.shape[0], -1)))
    sc = jnp.tanh(_ap(p["sc"], scalar))
    if entities is None:
        en = jnp.zeros((sp.shape[0], 64), sp.dtype)      # no roster -> entity feature is zero
    else:
        en = _entity_feat(p, entities, ent_mask)
    return jnp.tanh(_ap(p["f"], jnp.concatenate([sp, sc, en], -1)))


def forward(p, grid, scalar, entities=None, ent_mask=None):
    """grid:(B,C,H,W) in [0,1]; scalar:(B,scalar_dim); entities:(B,N,ENT_FEAT) opt. -> (logits,value)."""
    f = _trunk(p, grid, scalar, entities, ent_mask)
    return _ap(p["pi"], f), _ap(p["v"], f)[..., 0]


def heads(p, grid, scalar, entities=None, ent_mask=None):
    """Full multi-task read of the shared trunk: policy + value + the game-structure aux heads.

    Returns a dict: pi (logits), v (RL value), bld (buildable logits), cnt (counter logits),
    thr (threat-at-base logit), ev (position-eval scalar). Offline BC supervises pi/bld/cnt/thr/ev;
    RL trains v. The aux heads force the trunk to ENCODE prereqs / counters / threat / value."""
    f = _trunk(p, grid, scalar, entities, ent_mask)
    return {"pi": _ap(p["pi"], f), "v": _ap(p["v"], f)[..., 0],
            "bld": _ap(p["bld"], f), "cnt": _ap(p["cnt"], f),
            "thr": _ap(p["thr"], f)[..., 0], "ev": _ap(p["ev"], f)[..., 0],
            "pwr": _ap(p["pwr"], f)[..., 0]}


def capture_inputs(p, grid, scalar, entities, ent_mask):
    """Run the forward and return {dense_layer_key: (rows, in)} -- the input to each 2D layer, for
    activation-aware (influence) quantization. Mirrors _trunk/_entity_feat exactly. NumPy out."""
    import numpy as _np
    cap = {}
    h = jnp.tanh(_conv2d(grid, p["c1"], 2))
    h = jnp.tanh(_conv2d(h, p["c2"], 2))
    h = jnp.tanh(_conv2d(h, p["c3"], 2))
    hf = h.reshape(h.shape[0], -1)
    cap["sp"] = hf
    sp = jnp.tanh(_ap(p["sp"], hf))
    cap["sc"] = scalar
    sc = jnp.tanh(_ap(p["sc"], scalar))
    # entity transformer
    cap["et"] = entities.reshape(-1, entities.shape[-1])
    x = jnp.tanh(_ap(p["et"], entities))
    hn = _ln(x, p["ln1"])
    cap["wq"] = cap["wk"] = cap["wv"] = hn.reshape(-1, hn.shape[-1])
    q, k, v = _ap(p["wq"], hn), _ap(p["wk"], hn), _ap(p["wv"], hn)
    d = q.shape[-1]
    scores = jnp.einsum("bnd,bmd->bnm", q, k) / jnp.sqrt(d)
    scores = scores + jnp.where(ent_mask[:, None, :] > 0, 0.0, -1e9)
    a = jax.nn.softmax(scores, -1)
    o = jnp.einsum("bnm,bmd->bnd", a, v)
    cap["wo"] = o.reshape(-1, o.shape[-1])
    x = x + _ap(p["wo"], o)
    h2 = _ln(x, p["ln2"])
    cap["ff1"] = h2.reshape(-1, h2.shape[-1])
    r = jax.nn.relu(_ap(p["ff1"], h2))
    cap["ff2"] = r.reshape(-1, r.shape[-1])
    x = x + _ap(p["ff2"], r)
    m = ent_mask[:, :, None]
    pooled = (x * m).sum(1) / jnp.clip(m.sum(1), 1.0)
    cap["en"] = pooled
    en = jnp.tanh(_ap(p["en"], pooled))
    f_in = jnp.concatenate([sp, sc, en], -1)
    cap["f"] = f_in
    f = jnp.tanh(_ap(p["f"], f_in))
    for hk in ("pi", "v", "bld", "cnt", "thr", "ev", "pwr"):
        cap[hk] = f
    return {kk: _np.asarray(vv) for kk, vv in cap.items()}


def decide(p, grid, scalar, key=None, greedy=True, entities=None, ent_mask=None):
    """One decision from a single (grid, scalar[, roster]) observation -> (macro_index, value)."""
    g = grid[None] if grid.ndim == 3 else grid
    s = scalar[None] if scalar.ndim == 1 else scalar
    e = entities[None] if (entities is not None and entities.ndim == 2) else entities
    em = ent_mask[None] if (ent_mask is not None and ent_mask.ndim == 1) else ent_mask
    logits, value = forward(p, jnp.asarray(g), jnp.asarray(s),
                            None if e is None else jnp.asarray(e),
                            None if em is None else jnp.asarray(em))
    a = int(jnp.argmax(logits[0])) if greedy else int(random.categorical(key, logits[0]))
    return a, float(value[0])


if __name__ == "__main__":
    import numpy as np
    key = random.PRNGKey(0)
    p = init_params(key)
    n = sum(int(np.prod(w.shape)) + int(np.prod(b.shape)) for w, b in p.values())
    print(f"params: {n:,}   devices: {jax.devices()}")

    grid = jnp.asarray(np.random.rand(4, 7, 64, 64), jnp.float32)
    scalar = jnp.asarray(np.random.rand(4, 21), jnp.float32)
    ent = jnp.asarray(np.random.rand(4, MAX_ENT, ENT_FEAT), jnp.float32)
    mask = jnp.asarray((np.random.rand(4, MAX_ENT) > 0.5).astype(np.float32))

    # CNN + scalar only (backward-compatible path)
    logits, value = forward(p, grid, scalar)
    print(f"forward (no roster): grid{tuple(grid.shape)} + scalar{tuple(scalar.shape)} -> logits{tuple(logits.shape)} value{tuple(value.shape)}")

    # CNN + scalar + entity transformer (full path)
    logits, value = forward(p, grid, scalar, ent, mask)
    print(f"forward (+roster):   + entities{tuple(ent.shape)} mask{tuple(mask.shape)} -> logits{tuple(logits.shape)}")
    h = heads(p, grid, scalar, ent, mask)
    print(f"heads: { {k: tuple(v.shape) for k, v in h.items()} }")

    a, v = decide(p, np.random.rand(7, 64, 64).astype(np.float32), np.random.rand(21).astype(np.float32),
                  entities=np.random.rand(MAX_ENT, ENT_FEAT).astype(np.float32),
                  ent_mask=(np.random.rand(MAX_ENT) > 0.5).astype(np.float32))
    print(f"decide (+roster): macro index = {a}  value = {v:+.2f}")
    print("fused brain (eyes + roster + state -> decision) forward pass OK on CPU")
