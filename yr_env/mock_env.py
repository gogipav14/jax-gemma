"""Fast synthetic YR MDP — GOAL-CENTERED. There is an opponent with health you must DESTROY while
staying alive. Reward = damage dealt to the enemy (+ win bonus / - loss), and NOTHING for the means:
building an economy and an army is an *unrewarded* prerequisite the agent must discover it needs.
Mirrors rl_env's interface (obs=21, 15 macros) so the same learner trains on it.

Used to validate the AlphaStar recipe: BC warm-start from the distilled script (a nudge past the
hard cold-start exploration), then self-play RL toward the win — NOT reward-shaping the means.
"""
from __future__ import annotations

import numpy as np

OBS_DIM, N_ACT = 21, 15
ACTION_NAME = ["noop", "deploy", "POWER", "ECONOMY", "BARRACKS", "WAR_FACTORY", "RADAR", "DEF_G",
               "DEF_AA", "train_MAIN", "train_ANTI", "train_AA", "scout", "ATTACK", "defend"]
ENEMY_HP0 = 30.0


class MockYREnv:
    def __init__(self, max_steps=70):
        self.max_steps = max_steps

    def reset(self):
        self.b = {"POWER": 0, "REFN": 0, "BARR": 0, "WEAP": 0, "RADR": 0, "DEFG": 0, "DEFA": 0}
        self.u = {"MAIN": 0, "ANTI": 0, "AA": 0}
        self.credits = 4000
        self.t = 0
        self.conyard = 1
        self.enemy_hp = ENEMY_HP0          # the opponent — destroy this to WIN
        self.enemy_army = 0.0              # enemy pressure (grows; overrun you if undefended)
        return self._obs()

    def _mil(self):
        return self.u["MAIN"] * 1.0 + self.u["ANTI"] * 1.2 + self.b["DEFG"] * 0.8

    def step(self, a):
        self.t += 1
        self.credits += self.b["REFN"] * 500 + 300
        self.enemy_army = max(0.0, (self.t - 10) * 0.45)     # harder pressure: mass-4 won't survive late
        c, b, u = self.credits, self.b, self.u
        defended = False

        # build / train — PREREQS ENFORCED (illegal => nothing; the means are not told, not rewarded)
        if a == 2 and c >= 800:
            b["POWER"] += 1; self.credits -= 800
        elif a == 3 and b["POWER"] >= 1 and c >= 2000:
            b["REFN"] += 1; self.credits -= 2000
        elif a == 4 and b["POWER"] >= 1 and c >= 500:
            b["BARR"] += 1; self.credits -= 500
        elif a == 5 and b["BARR"] >= 1 and b["POWER"] >= 1 and c >= 2000:   # WAR FACTORY needs BARRACKS
            b["WEAP"] += 1; self.credits -= 2000
        elif a == 6 and b["WEAP"] >= 1 and c >= 1000:
            b["RADR"] += 1; self.credits -= 1000
        elif a == 7 and b["POWER"] >= 1 and c >= 1000:
            b["DEFG"] += 1; self.credits -= 1000
        elif a == 9 and b["WEAP"] >= 1 and c >= 900:
            u["MAIN"] += 1; self.credits -= 900
        elif a == 10 and b["WEAP"] >= 1 and c >= 900:
            u["ANTI"] += 1; self.credits -= 900
        elif a == 14:
            defended = True

        # ATTACK the opponent (the GOAL) — damage scales with your army
        dmg = 0.0
        if a == 13 and self._mil() > 0:
            dmg = min(self.enemy_hp, self._mil() * 0.6)
            self.enemy_hp -= dmg

        # the enemy attacks you: if your defense can't cover its army, you lose a building (conyard last)
        defense = self._mil() + (1.5 if defended else 0.0)
        own_loss = 0.0
        if self.enemy_army > defense + 0.5:
            lost = next((k for k in ("RADR", "WEAP", "BARR", "REFN", "DEFG", "POWER") if b[k] > 0), None)
            if lost:
                b[lost] -= 1
                own_loss = 1.0
            else:
                self.conyard = 0

        # GOAL-CENTERED reward: damage the opponent, stay alive. NOTHING for building the means.
        r = dmg - 0.5 * own_loss
        done = False
        win = self.enemy_hp <= 0
        if win:
            r += 20.0; done = True
        elif self.conyard == 0:
            r -= 10.0; done = True
        elif self.t >= self.max_steps:
            done = True
        return self._obs(), float(r), done, {"win": win}

    def _obs(self):
        own = [self.b[k] for k in ("POWER", "REFN", "BARR", "WEAP", "RADR", "DEFG", "DEFA")] + \
              [self.u[k] for k in ("MAIN", "ANTI", "AA")]
        enemy = [0.0, 0.0, 0.0, self.enemy_army, self.enemy_hp / ENEMY_HP0, 0.0]
        scal = [self.credits / 20000.0, 0.0, 1.0 if self.b["RADR"] else 0.0,
                self.enemy_army, self.enemy_hp / ENEMY_HP0]
        return np.asarray(own + enemy + scal, np.float32)


def script_policy(env: MockYREnv) -> int:
    """The DISTILLED stock-script nudge (for the BC warm-start). Competent, not optimal: build an
    economy + army, lean on more army when pressured, then attack the opponent. RL improves past it."""
    b, u = env.b, env.u
    if b["POWER"] < 1:
        return 2
    if b["REFN"] < 2 and env.credits >= 2000:
        return 3
    if b["BARR"] < 1:
        return 4
    if b["WEAP"] < 1:
        return 5 if env.credits >= 2000 else 0
    if env.enemy_army > env._mil() + 0.5 and env.credits >= 900:
        return 14 if env._mil() == 0 else 9        # under pressure: defend if no army, else build army
    if env._mil() < 4 and env.credits >= 900:
        return 9                                    # mass to only ~4 (deliberately suboptimal -> RL can beat it)
    return 13                                       # attack early-ish
