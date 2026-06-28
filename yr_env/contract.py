"""Single source of truth for the bridge IPC contract (Python side).

Mirrors docs/bridge-contract.md. The C++ bridge DLL must use the same magic/version,
shared-memory layout, and action-type ordering. Keep this in lockstep with the DLL header.
"""
from __future__ import annotations

import enum

# --- IPC header ---
MAGIC = 0x59524252  # 'YRBR'
VERSION = 2  # bump on any layout change; must match Bridge.h BridgeContract::VERSION

SHMEM_OBS_NAME = "Local\\yr_bridge_obs"
SHMEM_ACT_NAME = "Local\\yr_bridge_act"

# --- struct formats (little-endian, packed; mirror Bridge.h #pragma pack(1)) ---
HEADER_FMT = "<IIQiI"          # magic, version, frame_seq, house_index, status  (24)
GLOBALS_FMT = "9i"            # 9 * int32  (36)
ENTITY_FMT = "iihhBBBB"       # unique_id, type_id, x, y, category_rtti, hp_frac, state, group_id (16)
FACTORY_FMT = "iiBBBB"       # current_type_id, queue_head_type_id, category_rtti, progress, queue_count, flags (12)
# BridgeAction (24): type, category_rtti, is_naval, stance, type_id, cell_x, cell_y, target_unique, group_id, pad[3]
ACTION_FMT = "BBBBiiiiB3x"
N_FACTORY = 16

# AbstractType RTTI values (engine). OBS entity.category_rtti is the RUNTIME RTTI;
# PRODUCE/PLACE actions need the *Type* RTTI (the +Type variants below).
class RTTI:  # runtime (what OBS reports via WhatAmI)
    UNIT = 1
    AIRCRAFT = 2
    BUILDING = 6
    INFANTRY = 15

class RTTIType:  # type-class RTTI (what PRODUCE/PLACE require — wrong one CRASHES the game)
    AIRCRAFT_TYPE = 3
    BUILDING_TYPE = 7
    INFANTRY_TYPE = 16
    UNIT_TYPE = 40

# Map a runtime RTTI (from OBS) -> the *Type* RTTI needed to PRODUCE it.
RUNTIME_TO_TYPE_RTTI = {
    RTTI.UNIT: RTTIType.UNIT_TYPE,
    RTTI.AIRCRAFT: RTTIType.AIRCRAFT_TYPE,
    RTTI.BUILDING: RTTIType.BUILDING_TYPE,
    RTTI.INFANTRY: RTTIType.INFANTRY_TYPE,
}


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
    DEPLOY = 10       # deploy a deployable unit (MCV -> Construction Yard) by target_unique


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
