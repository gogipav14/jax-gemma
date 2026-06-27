"""yr_env — Gym-style environment wrapping the YR bridge DLL over shared memory.

Implemented in Phase 3. For now this package exposes the IPC `contract` so the bridge
DLL and (future) env stay in lockstep.
"""
from . import contract  # noqa: F401
