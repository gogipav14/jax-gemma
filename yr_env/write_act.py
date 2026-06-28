"""Writer for the YR bridge ACT shared-memory region (Python -> DLL).

Packs a BridgeAction, publishes it (writes action_seq LAST), and waits for the DLL
to ack (ack_seq == action_seq) + report a result code. The DLL injects the action
into EventClass::OutList — the non-cheating human command path.

Run while a skirmish with our bridge DLL is live. Example:
    w = ActWriter()
    print(w.produce(contract.RTTIType.UNIT_TYPE, type_id=<rhino idx>))  # build a vehicle
"""
from __future__ import annotations

import ctypes
import struct
import time
from ctypes import wintypes

import contract  # run with PYTHONPATH=yr_env

PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0x000F001F
ACT_SIZE = 24 + 8 + 8 + 4 + 24  # BridgeACT = 68

# offsets within BridgeACT
OFF_SEQ, OFF_ACK, OFF_RESULT, OFF_ACTION = 24, 32, 40, 44

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateFileMappingA.restype = wintypes.HANDLE
_k32.CreateFileMappingA.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.LPCSTR]
_k32.MapViewOfFile.restype = ctypes.c_void_p
_k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
_k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]

RESULT_NAMES = {
    0: "OK", 1: "NOOP", 2: "REJECTED_CANBUILD", 3: "REJECTED_NOFACTORY",
    4: "BAD_TARGET", 5: "BAD_RTTI", 6: "BAD_INDEX", 7: "PLACE_INVALID_CELL",
    8: "NO_AGENT", 9: "UNSUPPORTED",
}


class ActWriter:
    def __init__(self):
        # CreateFileMapping opens the DLL's existing mapping or creates it if Python ran first.
        self.h = _k32.CreateFileMappingA(ctypes.c_void_p(-1), None, PAGE_READWRITE,
                                         0, ACT_SIZE, contract.SHMEM_ACT_NAME.encode())
        if not self.h:
            raise OSError(f"CreateFileMapping failed: {ctypes.get_last_error()}")
        self.addr = _k32.MapViewOfFile(self.h, FILE_MAP_ALL_ACCESS, 0, 0, ACT_SIZE)
        if not self.addr:
            _k32.CloseHandle(self.h)
            raise OSError(f"MapViewOfFile failed: {ctypes.get_last_error()}")
        self.seq = self._r64(OFF_SEQ)  # continue from whatever counter is live

    def _w(self, off, data: bytes):
        ctypes.memmove(self.addr + off, data, len(data))

    def _r(self, off, n) -> bytes:
        return ctypes.string_at(self.addr + off, n)

    def _r64(self, off) -> int:
        return struct.unpack("<Q", self._r(off, 8))[0]

    def _pack(self, atype, category_rtti=0, is_naval=0, stance=0, type_id=0,
              cell_x=0, cell_y=0, target_unique=-1, group_id=0) -> bytes:
        return struct.pack("<" + contract.ACTION_FMT, int(atype), category_rtti, is_naval,
                           stance, type_id, cell_x, cell_y, target_unique, group_id)

    def send(self, action_bytes: bytes, wait_s: float = 2.0):
        """Publish an action and wait for the DLL ack. Returns (result_code, name) or None on timeout."""
        self._w(OFF_ACTION, action_bytes)           # body first
        self.seq += 1
        self._w(OFF_SEQ, struct.pack("<Q", self.seq))  # publish seq LAST
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self._r64(OFF_ACK) == self.seq:
                code = struct.unpack("<I", self._r(OFF_RESULT, 4))[0]
                return code, RESULT_NAMES.get(code, str(code))
            time.sleep(0.005)
        return None  # DLL never acked (not in a match / paused)

    # --- convenience actions ---
    def noop(self):
        return self.send(self._pack(contract.ActionType.NOOP))

    def produce(self, type_rtti: int, type_id: int, naval: bool = False):
        """type_rtti must be a *Type* RTTI (contract.RTTIType.UNIT_TYPE, etc.)."""
        return self.send(self._pack(contract.ActionType.PRODUCE, category_rtti=type_rtti,
                                    is_naval=1 if naval else 0, type_id=type_id))

    def place(self, building_type_id: int, x: int, y: int, naval: bool = False):
        return self.send(self._pack(contract.ActionType.PLACE, category_rtti=contract.RTTIType.BUILDING_TYPE,
                                    is_naval=1 if naval else 0, type_id=building_type_id, cell_x=x, cell_y=y))

    def deploy(self, unique_id: int):
        """Deploy a deployable unit (MCV -> Construction Yard)."""
        return self.send(self._pack(contract.ActionType.DEPLOY, target_unique=unique_id))

    def sell(self, unique_id: int):
        return self.send(self._pack(contract.ActionType.SELL, target_unique=unique_id))

    def set_primary(self, unique_id: int):
        return self.send(self._pack(contract.ActionType.SET_PRIMARY, target_unique=unique_id))

    def move(self, unique_id: int, x: int, y: int):
        return self.send(self._pack(contract.ActionType.GROUP_MOVE, target_unique=unique_id, cell_x=x, cell_y=y))

    def attack_move(self, unique_id: int, x: int, y: int):
        return self.send(self._pack(contract.ActionType.GROUP_ATTACK, target_unique=unique_id, cell_x=x, cell_y=y))

    def superweapon(self, supers_index: int, x: int, y: int):
        return self.send(self._pack(contract.ActionType.SUPERWEAPON, type_id=supers_index, cell_x=x, cell_y=y))

    def close(self):
        if self.addr:
            _k32.UnmapViewOfFile(self.addr); self.addr = None
        if self.h:
            _k32.CloseHandle(self.h); self.h = None


if __name__ == "__main__":
    try:
        w = ActWriter()
    except OSError as e:
        print("Could not open ACT mapping:", e); raise SystemExit(1)
    print("ACT mapping open. Sending a NOOP to test the ack handshake...")
    r = w.noop()
    print("result:", r if r else "no ack (is a skirmish with the bridge DLL running?)")
    w.close()
