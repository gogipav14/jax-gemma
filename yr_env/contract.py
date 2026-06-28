"""Single source of truth for the bridge IPC contract (Python side).

Mirrors docs/bridge-contract.md. The C++ bridge DLL must use the same magic/version,
shared-memory layout, and action-type ordering. Keep this in lockstep with the DLL header.
"""
from __future__ import annotations

import enum

# --- IPC header ---
MAGIC = 0x59524252  # 'YRBR'
VERSION = 1  # bump on any layout change; must match Bridge.h BridgeContract::VERSION

SHMEM_OBS_NAME = "Local\\yr_bridge_obs"
SHMEM_ACT_NAME = "Local\\yr_bridge_act"


class ActionType(enum.IntEnum):
    """v1 action space: macro + group commands. Per-unit micro is engine-scripted.

    Ordering is part of the contract — the DLL switch() relies on these integer values.
    """
    NOOP = 0
    PRODUCE = 1       # start producing type_id in its factory queue
    PLACE = 2         # place a completed building at (cell_x, cell_y)
    SET_PRIMARY = 3   # set primary factory for a category
    SELL = 4          # sell building of type_id (or at cell)
    GROUP_MOVE = 5    # move group_id to (cell_x, cell_y) [attack-move scripted]
    GROUP_ATTACK = 6  # group_id attacks target_entity (or cell)
    GROUP_FORM = 7    # (re)assign visible own units to group_id
    SUPERWEAPON = 8   # fire ready superweapon type_id at (cell_x, cell_y)
    STANCE = 9        # set group_id stance


# Fixed caps for padded NN tensors (tune later against real games).
N_OWN = 256        # max own technos surfaced per frame
N_ENEMY = 256      # max visible enemy technos per frame
H_MAX = 256        # max map cell height for the spatial grid
W_MAX = 256        # max map cell width

# Spatial grid channels (index = channel order in the OBS region).
SPATIAL_CHANNELS = (
    "passability",
    "ore",
    "fog",            # 0 shrouded, 1 discovered, 2 currently visible
    "own_units",
    "enemy_units",    # visible only (fog-honored)
    "own_buildings",
    "height",
)

# Factory categories (production state is reported per category).
FACTORY_CATEGORIES = ("building", "unit", "infantry", "aircraft", "navy")
