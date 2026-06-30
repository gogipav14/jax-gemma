"""The learner as a boundary-value problem:
  initial condition  = BC warm-start from the distilled script (a nudge off the cold start)
  boundary condition = the GOAL (destroy the opponent) + the rules (legal moves, fog)
  the learning       = the trajectory that connects them — RL integrates the policy from the
                       script-start toward the win, free to abandon the script once it stops winning.

No means are rewarded; the reward is the goal. Trains on any env with reset()/step() (MockYREnv now,
live rl_env later).

    python policy/learner.py     # script teacher -> BC warm-start -> RL surpasses it (mock)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yr_env"))


def init_params(key, obs_dim, n_act, hidden=64):
    k = random.split(key, 4)

    def lin(kk, i, o):
        return (random.normal(kk, (i, o)) * jnp.sqrt(2.0 / i), jnp.zeros(o))
    return {"w1": lin(k[0], obs_dim, hidden), "w2": lin(k[1], hidden, hidden),
            "pi": lin(k[2], hidden, n_act), "v": lin(k[3], hidden, 1)}


def forward(p, x):
    def ap(layer, z):
        w, b = layer
        return z @ w + b
    h = jnp.tanh(ap(p["w1"], x))
    h = jnp.tanh(ap(p["w2"], h))
    return ap(p["pi"], h), ap(p["v"], h)[..., 0]


def discounted(rews, gamma=0.99):
    g, out = 0.0, []
    for r in reversed(rews):
        g = r + gamma * g
        out.append(g)
    return np.asarray(out[::-1], np.float32)


def loss_fn(p, ref, obs, act, ret, beta):
    logits, val = forward(p, obs)
    logp = jax.nn.log_softmax(logits)
    adv = ret - val
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
    pg = -jnp.mean(logp[jnp.arange(act.shape[0]), act] * jax.lax.stop_gradient(adv_n))
    vloss = jnp.mean((val - ret) ** 2)
    pi = jax.nn.softmax(logits)
    ref_logp = jax.lax.stop_gradient(jax.nn.log_softmax(forward(ref, obs)[0]))
    kl = jnp.mean(jnp.sum(pi * (logp - ref_logp), -1))      # KL leash to the BC reference (not a cage)
    ent = -jnp.mean(jnp.sum(pi * logp, -1))
    return pg + 0.5 * vloss + beta * kl - 0.01 * ent


def play_episode(env, p, key, max_steps=70, greedy=False):
    obs = env.reset()
    O, A, R = [], [], []
    won = False
    for _ in range(max_steps):
        logits, _ = forward(p, jnp.asarray(obs))
        if greedy:
            a = int(jnp.argmax(logits))
        else:
            key, sk = random.split(key)
            a = int(random.categorical(sk, logits))
        O.append(obs)
        A.append(a)
        obs, r, done, info = env.step(a)
        R.append(r)
        if done:
            won = info.get("win", False)
            break
    return np.asarray(O, np.float32), np.asarray(A), R, won, key


# --- the GOAL + rules: maximize win reward via self-play RL (the trajectory) ---
def train(env, obs_dim, n_act, n_updates=80, games_per=10, lr=1e-3, seed=0, init=None, beta=0.5, log_every=8):
    key = random.PRNGKey(seed)
    key, ik = random.split(key)
    p = init if init is not None else init_params(ik, obs_dim, n_act)
    ref = p                                                 # frozen reference (the BC policy) for the KL leash
    b = beta if init is not None else 0.0
    opt = optax.adam(lr)
    st = opt.init(p)
    gfn = jax.jit(jax.value_and_grad(loss_fn))
    best_p, best_w = p, -1.0                                 # keep the best policy, not the drifting tail
    for upd in range(n_updates):
        b_now = b * max(0.05, 1.0 - upd / n_updates)         # ANNEAL the leash (let it hold the win it found)
        Os, As, Rs, rets, wins = [], [], [], [], []
        for _ in range(games_per):                           # batch many games per gradient step (low variance)
            O, A, R, won, key = play_episode(env, p, key)
            Os.append(O); As.append(A); Rs.append(discounted(R))
            rets.append(sum(R)); wins.append(int(won))
        wr = np.mean(wins)
        if wr >= best_w:
            best_w, best_p = wr, p
        O = jnp.asarray(np.concatenate(Os))
        A = jnp.asarray(np.concatenate(As))
        Ret = jnp.asarray(np.concatenate(Rs))
        _, grads = gfn(p, ref, O, A, Ret, b_now)
        gupd, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, gupd)
        if upd % log_every == 0 or upd == n_updates - 1:
            print(f"  update {upd:3d} ({(upd + 1) * games_per:4d} games)  "
                  f"winrate={wr:.0%}  avgret={np.mean(rets):+6.1f}")
    print(f"  [kept best checkpoint: winrate {best_w:.0%}]")
    return best_p


# --- the INITIAL CONDITION: behavioral-clone the distilled script (a nudge, not a cage) ---
def bc_collect(env, policy_fn, n_games=250, max_steps=70):
    O, A = [], []
    for _ in range(n_games):
        obs = env.reset()
        for _ in range(max_steps):
            a = policy_fn(env)
            O.append(obs)
            A.append(a)
            obs, r, done, _ = env.step(a)
            if done:
                break
    return np.asarray(O, np.float32), np.asarray(A)


def bc_train(params, O, A, steps=400, lr=3e-3):
    opt = optax.adam(lr)
    st = opt.init(params)
    O, A = jnp.asarray(O), jnp.asarray(A)

    def bcl(p, o, a):
        logits, _ = forward(p, o)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a])
    gfn = jax.jit(jax.value_and_grad(bcl))
    for _ in range(steps):
        _, g = gfn(params, O, A)
        upd, st = opt.update(g, st, params)
        params = optax.apply_updates(params, upd)
    return params


def evaluate(env, params, n=80, greedy=False):     # sampled rollouts = the honest eval of a stochastic policy
    rets, wins = [], 0
    key = random.PRNGKey(12345)
    for i in range(n):
        _, _, R, won, key = play_episode(env, params, random.fold_in(key, i), greedy=greedy)
        rets.append(sum(R))
        wins += int(won)
    return np.mean(rets), wins / n


def eval_policy_fn(env, policy_fn, n=60):
    rets, wins = [], 0
    for _ in range(n):
        env.reset()
        R, won = [], False
        for _ in range(70):
            _, r, done, info = env.step(policy_fn(env))
            R.append(r)
            if done:
                won = info.get("win", False)
                break
        rets.append(sum(R))
        wins += int(won)
    return np.mean(rets), wins / n


if __name__ == "__main__":
    from mock_env import MockYREnv, OBS_DIM, N_ACT, script_policy
    env = MockYREnv()

    print("=== TEACHER: the distilled script (the nudge) ===")
    sret, swin = eval_policy_fn(env, script_policy)
    print(f"  script: return {sret:+.1f}  winrate {swin:.0%}\n")

    print("=== from scratch (random net) — the cold start ===")
    p0 = init_params(random.PRNGKey(0), OBS_DIM, N_ACT)
    r0, w0 = evaluate(env, p0)
    print(f"  random: return {r0:+.1f}  winrate {w0:.0%}\n")

    print("=== INITIAL CONDITION: BC warm-start from the script ===")
    O, A = bc_collect(env, script_policy)
    bc = bc_train(p0, O, A)
    rb, wb = evaluate(env, bc)
    print(f"  after BC: return {rb:+.1f}  winrate {wb:.0%}\n")

    print("=== LEARN THE PATH: RL toward the win (boundary), starting from BC ===")
    rl = train(env, OBS_DIM, N_ACT, init=bc)
    rr, wr = evaluate(env, rl)
    print(f"\n  random {w0:.0%}  ->  BC(initial) {wb:.0%}  ->  RL(learned path) {wr:.0%} winrate "
          f"| return {r0:+.1f} -> {rb:+.1f} -> {rr:+.1f}")
    print("  (RL started from the script's opening and improved past it toward the goal — not locked in)")
