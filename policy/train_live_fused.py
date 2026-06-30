"""Train the FUSED brain (eyes + state -> decision) on the LIVE game with the GOAL reward.

  initial condition = BC the teacher through the fused net (grid + scalar -> action)
  boundary          = the GOAL (raze the opponent + survive) -- rl_env's goal reward
  the learning      = self-play RL, KL-leashed to BC (annealed) + best checkpoint

Game by game on this machine (slow; the recipe is proven on the mock in policy/learner.py).

    PYTHONPATH=yr_env;commander  python policy/train_live_fused.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "yr_env"))
sys.path.insert(0, os.path.join(ROOT, "policy"))
import net                                               # noqa: E402  (the fused brain)
from rl_env import YRLearnEnv, MACROS                    # noqa: E402
from train_live import teacher_action                    # noqa: E402


def discounted(R, gamma=0.99):
    g, out = 0.0, []
    for r in reversed(R):
        g = r + gamma * g
        out.append(g)
    return np.asarray(out[::-1], np.float32)


def loss_fn(p, ref, grids, scalars, act, ret, beta):
    logits, val = net.forward(p, grids, scalars)
    logp = jax.nn.log_softmax(logits)
    adv = ret - val
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
    pg = -jnp.mean(logp[jnp.arange(act.shape[0]), act] * jax.lax.stop_gradient(adv_n))
    vloss = jnp.mean((val - ret) ** 2)
    pi = jax.nn.softmax(logits)
    ref_logp = jax.lax.stop_gradient(jax.nn.log_softmax(net.forward(ref, grids, scalars)[0]))
    kl = jnp.mean(jnp.sum(pi * (logp - ref_logp), -1))
    ent = -jnp.mean(jnp.sum(pi * logp, -1))
    return pg + 0.5 * vloss + beta * kl - 0.01 * ent


def play_fused(env, p, key, greedy=False, max_steps=55):
    obs = env.reset()
    grid = env.grid()
    G, S, A, R = [], [], [], []
    won = False
    for t in range(max_steps):
        logits, _ = net.forward(p, jnp.asarray(grid[None]), jnp.asarray(obs[None]))
        if greedy:
            a = int(jnp.argmax(logits[0]))
        else:
            key, sk = random.split(key)
            a = int(random.categorical(sk, logits[0]))
        G.append(grid); S.append(obs); A.append(a)
        obs, r, done, info = env.step(a)
        grid = env.grid()
        R.append(r)
        if done:
            won = r > 0
            break
    return np.asarray(G, np.float32), np.asarray(S, np.float32), np.asarray(A), R, won, key


def collect_bc(env, n_games, max_steps=55):
    G, S, A = [], [], []
    for g in range(n_games):
        obs = env.reset()
        grid = env.grid()
        print(f"\n-- BC teacher game {g + 1} (live) --")
        for t in range(max_steps):
            a = teacher_action(env.pos)
            G.append(grid); S.append(obs); A.append(a)
            obs, r, done, info = env.step(a)
            grid = env.grid()
            print(f"  t{t:2d} {MACROS[a][0]:14s} r={r:+.2f}")
            if done:
                break
    return np.asarray(G, np.float32), np.asarray(S, np.float32), np.asarray(A)


def bc_train(p, G, S, A, steps=300, lr=3e-3):
    opt = optax.adam(lr)
    st = opt.init(p)
    G, S, A = jnp.asarray(G), jnp.asarray(S), jnp.asarray(A)

    def bcl(pp, g, s, a):
        l, _ = net.forward(pp, g, s)
        return -jnp.mean(jax.nn.log_softmax(l)[jnp.arange(a.shape[0]), a])
    gfn = jax.jit(jax.value_and_grad(bcl))
    for _ in range(steps):
        _, gr = gfn(p, G, S, A)
        u, st = opt.update(gr, st, p)
        p = optax.apply_updates(p, u)
    return p


def main(bc_games=1, rl_games=3):
    env = YRLearnEnv(launch=True)
    print("=== BC: teach the FUSED brain the opening from the live teacher ===")
    G, S, A = collect_bc(env, bc_games)
    p = net.init_params(random.PRNGKey(0))
    p = bc_train(p, G, S, A)
    logits, _ = net.forward(p, jnp.asarray(G[:1]), jnp.asarray(S[:1]))
    print(f"  BC done ({len(A)} demos). first-state argmax = {MACROS[int(jnp.argmax(logits[0]))][0]}")

    print("\n=== RL: fused brain self-play toward the GOAL (raze the opponent + survive) ===")
    ref = p
    opt = optax.adam(1e-3)
    st = opt.init(p)
    gfn = jax.jit(jax.value_and_grad(loss_fn))
    key = random.PRNGKey(1)
    for g in range(rl_games):
        G, S, A, R, won, key = play_fused(env, p, key)
        ret = discounted(R)
        _, grads = gfn(p, ref, jnp.asarray(G), jnp.asarray(S), jnp.asarray(A), jnp.asarray(ret), 0.3)
        u, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, u)
        acts = [MACROS[a][0] for a in A]
        print(f"\n  RL game {g + 1}: return={sum(R):+6.1f}  won={won}  steps={len(A)}")
        print(f"    actions: {acts}")
    env.close()


if __name__ == "__main__":
    main(bc_games=1, rl_games=2)
