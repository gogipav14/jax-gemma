"""Reader for the YR bridge OBS shared-memory region (DLL -> Python).

Reads header + globals + counts (seqlock), and the per-factory production state
used by the 2-war-factory serial-proof (at most one factory in-progress per category
on the human/OutList path; the AI cheat would show several at once).

Run while a skirmish with our bridge DLL is live:  python yr_env/read_obs.py
"""
from __future__ import annotations

import ctypes
import struct
import sys
import time
from ctypes import wintypes

from contract import (MAGIC, VERSION, SHMEM_OBS_NAME, HEADER_FMT, GLOBALS_FMT,
                      ENTITY_FMT, FACTORY_FMT, N_FACTORY)

FILE_MAP_READ = 0x0004
N_OWN = N_ENEMY = 256

HEADER_SIZE = struct.calcsize(HEADER_FMT)          # 24
GLOBALS_SIZE = struct.calcsize("<" + GLOBALS_FMT)  # 36
ENTITY_SIZE = struct.calcsize("<" + ENTITY_FMT)    # 16
FACTORY_SIZE = struct.calcsize("<" + FACTORY_FMT)  # 12

OFF_COUNTS = HEADER_SIZE + GLOBALS_SIZE            # 60
OFF_OWN = OFF_COUNTS + 8                            # 68  (n_own,n_enemy,n_factory,pad)
OFF_ENEMY = OFF_OWN + N_OWN * ENTITY_SIZE
OFF_FACTORY = OFF_ENEMY + N_ENEMY * ENTITY_SIZE    # 8260
OBS_SIZE = OFF_FACTORY + N_FACTORY * FACTORY_SIZE  # 8452

_HEAD_FMT = HEADER_FMT + GLOBALS_FMT + "HHHH"      # + n_own,n_enemy,n_factory,_pad
_HEAD_SIZE = struct.calcsize(_HEAD_FMT)            # 68
_GLOB = ("credits", "power_output", "power_drain", "side_index", "owned_units",
         "owned_buildings", "owned_infantry", "owned_aircraft", "owned_navy")

_RTTI = {1: "Unit", 2: "Aircraft", 6: "Building", 15: "Infantry", 0: "-"}

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenFileMappingA.restype = wintypes.HANDLE
_k32.OpenFileMappingA.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCSTR]
_k32.MapViewOfFile.restype = ctypes.c_void_p
_k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
_k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]


class ObsReader:
    def __init__(self):
        self.h = _k32.OpenFileMappingA(FILE_MAP_READ, False, SHMEM_OBS_NAME.encode())
        if not self.h:
            raise OSError("OBS mapping not found (is a skirmish with the bridge DLL running?)")
        self.addr = _k32.MapViewOfFile(self.h, FILE_MAP_READ, 0, 0, OBS_SIZE)
        if not self.addr:
            _k32.CloseHandle(self.h)
            raise OSError(f"MapViewOfFile failed: {ctypes.get_last_error()}")

    def _bytes(self, off, n):
        return ctypes.string_at(self.addr + off, n)

    def read_state(self):
        """Seqlock read of header + globals + counts. Returns dict or None on torn read."""
        head = struct.unpack(_HEAD_FMT, self._bytes(0, _HEAD_SIZE))
        seq1 = head[2]
        seq2 = struct.unpack("<IIQ", self._bytes(0, 16))[2]
        if seq1 != seq2:
            return None
        s = {"magic": head[0], "version": head[1], "frame_seq": seq1,
             "house_index": head[3], "status": head[4]}
        s.update({f: head[5 + i] for i, f in enumerate(_GLOB)})
        s["n_own"], s["n_enemy"], s["n_factory"] = head[14], head[15], head[16]
        return s

    def read_own(self, limit=None):
        """Return own entities: dicts with unique_id, type_id, category, x, y, hp."""
        n = struct.unpack("<H", self._bytes(OFF_COUNTS, 2))[0]  # n_own
        n = min(n, N_OWN, limit) if limit else min(n, N_OWN)
        out = []
        for i in range(n):
            uid, tid, x, y, cat, hp, st, gid = struct.unpack(
                "<" + ENTITY_FMT, self._bytes(OFF_OWN + i * ENTITY_SIZE, ENTITY_SIZE))
            out.append({"unique_id": uid, "type_id": tid, "category": _RTTI.get(cat, cat),
                        "category_rtti": cat, "x": x, "y": y, "hp": hp})
        return out

    def read_factories(self):
        """Return the list of own factories: dicts with category, current_type_id, progress, queue."""
        n = struct.unpack("<H", self._bytes(OFF_COUNTS + 4, 2))[0]  # n_factory
        out = []
        for i in range(min(n, N_FACTORY)):
            cur, qhead, cat, prog, qcount, flags = struct.unpack(
                "<" + FACTORY_FMT, self._bytes(OFF_FACTORY + i * FACTORY_SIZE, FACTORY_SIZE))
            out.append({"category": _RTTI.get(cat, cat), "current_type_id": cur,
                        "progress": prog, "queue_count": qcount,
                        "on_hold": bool(flags & 1), "suspended": bool(flags & 2),
                        "active": cur != -1})
        return out

    def close(self):
        if self.addr:
            _k32.UnmapViewOfFile(self.addr); self.addr = None
        if self.h:
            _k32.CloseHandle(self.h); self.h = None


def main():
    try:
        r = ObsReader()
    except OSError as e:
        print(e); return 1
    print(f"Opened OBS (expect version {VERSION}). 6 samples:\n")
    last, got = -1, 0
    for _ in range(400):
        s = r.read_state()
        if s and s["frame_seq"] != last:
            last = s["frame_seq"]
            ok = "OK" if s["magic"] == MAGIC and s["version"] == VERSION else "BAD"
            facs = [f for f in r.read_factories() if f["active"]]
            fac_str = ", ".join(f"{f['category']}#{f['current_type_id']}@{f['progress']}/54" for f in facs) or "none"
            print(f"[{ok}] f={s['frame_seq']} credits={s['credits']} "
                  f"U={s['owned_units']} B={s['owned_buildings']} I={s['owned_infantry']} "
                  f"nFac={s['n_factory']} active=[{fac_str}]")
            got += 1
            if got >= 6:
                break
        time.sleep(0.1)
    if got == 0:
        print("Mapping open but frame_seq never advanced (match not in progress?).")
    r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
