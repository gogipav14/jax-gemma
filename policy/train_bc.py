"""Behavioral-cloning trainer (JAX): teach the policy to imitate (obs -> action_type) traces.

This is REAL learning — gradient descent updating the policy weights — and it runs on CPU,
because BC is supervised, not RL (the compute wall is only at self-play). Phase 4 of the plan:
a competent, non-cheating *warm start* before the league.

Here we prove the training loop converges on a synthetic 'expert' (a fixed rule mapping obs
features -> action), so the machinery is validated end-to-end. Swap in real commander /
scripted-agent traces (a record_traces.py over build_base) for the actual warm-start policy.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(__file__))
from policy import init_params, forward          # noqa: E402

try:
    import optax
    _HAS_OPTAX = True
except Exception:
    _HAS_OPTAX = False

# obs layout matches yr_env/env.py: idx5 = owned_buildings, idx10 = n_enemy
SCALE = jnp.array([1e3, 10, 10, 3, 8, 6, 4, 1, 1, 8, 3, 3, 3, 3, 1, 1, 1])


def expert_action(obs):
    """Synthetic 'expert' (stand-in for recorded traces): no buildings -> DEPLOY(10);
    few buildings -> PRODUCE(1); enemy visible -> GROUP_ATTACK(6); else GROUP_MOVE(5)."""
    buildings, n_enemy = obs[..., 5], obs[..., 10]
    a = jnp.where(buildings < 1, 10,
                  jnp.where(buildings < 5, 1,
                            jnp.where(n_enemy > 0, 6, 5)))
    return a.astype(jnp.int32)


def make_data(key, n, obs_dim=17):
    obs = random.normal(key, (n, obs_dim)) * SCALE
    obs = obs.at[:, 5].set(jnp.abs(obs[:, 5]) % 8)     # owned_buildings 0..7
    obs = obs.at[:, 10].set(jnp.abs(obs[:, 10]) % 4)   # n_enemy 0..3
    return obs, expert_action(obs)


def load_traces(paths, keep_failed=False):
    """Load real recorded traces (.npz from record_traces.py) -> (obs, action_type).
    By default keeps only legal/successful (result==OK) decisions for imitation."""
    import numpy as np
    O, Y = [], []
    for p in paths:
        d = np.load(p)
        mask = np.ones(len(d["result"]), bool) if keep_failed else (d["result"] == 0)
        O.append(np.asarray(d["obs"])[mask])
        Y.append(np.asarray(d["act"])[mask, 0])         # column 0 = action_type
    obs = jnp.asarray(np.concatenate(O), jnp.float32)
    y = jnp.asarray(np.concatenate(Y).astype("int32"))
    return obs, y


def loss_fn(params, obs, y):
    logits, _ = forward(params, obs)
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])


def accuracy(params, obs, y):
    logits, _ = forward(params, obs)
    return jnp.mean((jnp.argmax(logits, -1) == y).astype(jnp.float32))


def train(steps=400, lr=1e-2, n=512, data=None):
    k_d, k_p = random.split(random.PRNGKey(0))
    obs, y = data if data is not None else make_data(k_d, n)
    params = init_params(k_p, obs.shape[1])
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    if _HAS_OPTAX:
        opt = optax.adam(lr)
        opt_state = opt.init(params)
    print(f"BC training: {n} samples, optax={_HAS_OPTAX}, start acc={float(accuracy(params, obs, y)):.2f}")
    for step in range(steps):
        loss, grads = grad_fn(params, obs, y)
        if _HAS_OPTAX:
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss={float(loss):.4f}  acc={float(accuracy(params, obs, y)):.3f}")
    print(f"FINAL acc={float(accuracy(params, obs, y)):.3f}  -> policy trained (CPU, JAX)")
    return params


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    if paths:
        obs, y = load_traces(paths)
        print(f"loaded {obs.shape[0]} real transitions from {len(paths)} trace(s); "
              f"action_types present: {sorted(set(int(v) for v in y))}")
        train(data=(obs, y))
    else:
        print("(no trace .npz given -> training on the synthetic expert; "
              "record real traces with yr_env/record_traces.py)")
        train()
