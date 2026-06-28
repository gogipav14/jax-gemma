"""Minimal reader for the YR bridge OBS shared-memory region.

Opens the named mapping the DLL creates (BridgeContract::OBS_NAME) and unpacks the
header + globals, using the seqlock pattern (read frame_seq, body, re-read frame_seq).
This validates the real DLL -> shared-memory -> Python data path end to end.

Run while a skirmish (with our bridge DLL) is live:  python yr_env/read_obs.py
"""
from __future__ import annotations

import ctypes
import struct
import sys
import time
from ctypes import wintypes

from contract import MAGIC, VERSION  # type: ignore  # run from yr_env/ dir

FILE_MAP_READ = 0x0004

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenFileMappingA.restype = wintypes.HANDLE
_k32.OpenFileMappingA.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCSTR]
_k32.MapViewOfFile.restype = ctypes.c_void_p
_k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
_k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]

# Header(24) = magic I, version I, frame_seq Q, house_index i, status I
# Globals(36) = 9 * int32
# n_own H, n_enemy H  -> first 64 bytes are all we need to validate.
_HEAD_FMT = "<IIQiI" + "9i" + "HH"
_HEAD_SIZE = struct.calcsize(_HEAD_FMT)  # 64
_GLOB_FIELDS = ("credits", "power_output", "power_drain", "side_index",
                "owned_units", "owned_buildings", "owned_infantry",
                "owned_aircraft", "owned_navy")

# Try the exact name first, then plausible fallbacks.
_NAMES = [b"Local\\yr_bridge_obs", b"yr_bridge_obs", b"Global\\yr_bridge_obs"]


def _open():
    for name in _NAMES:
        h = _k32.OpenFileMappingA(FILE_MAP_READ, False, name)
        if h:
            return h, name.decode()
    return None, None


def read_once(addr):
    """Seqlock read: returns dict or None if a torn read (caller retries)."""
    buf = ctypes.string_at(addr, _HEAD_SIZE)
    vals = struct.unpack(_HEAD_FMT, buf)
    seq1 = vals[2]
    # tiny window then re-read frame_seq to detect a write mid-read
    buf2 = ctypes.string_at(addr, 16)  # header up to frame_seq end
    seq2 = struct.unpack("<IIQ", buf2)[2]
    if seq1 != seq2:
        return None
    out = {"magic": vals[0], "version": vals[1], "frame_seq": seq1,
           "house_index": vals[3], "status": vals[4]}
    out.update({f: vals[5 + i] for i, f in enumerate(_GLOB_FIELDS)})
    out["n_own"], out["n_enemy"] = vals[14], vals[15]
    return out


def main():
    h, name = _open()
    if not h:
        print("Could not open the OBS mapping. Is a skirmish with the bridge DLL running?")
        return 1
    addr = _k32.MapViewOfFile(h, FILE_MAP_READ, 0, 0, _HEAD_SIZE)
    if not addr:
        print("MapViewOfFile failed:", ctypes.get_last_error())
        _k32.CloseHandle(h)
        return 1
    print(f"Opened mapping '{name}'. Reading 8 samples...\n")
    last = -1
    got = 0
    for _ in range(400):
        s = read_once(addr)
        if s and s["frame_seq"] != last:
            last = s["frame_seq"]
            ok = "OK" if s["magic"] == MAGIC and s["version"] == VERSION else "BAD-MAGIC"
            print(f"[{ok}] f={s['frame_seq']} house={s['house_index']} "
                  f"credits={s['credits']} pwr={s['power_output']}/{s['power_drain']} "
                  f"U={s['owned_units']} B={s['owned_buildings']} I={s['owned_infantry']} "
                  f"A={s['owned_aircraft']} navy={s['owned_navy']} "
                  f"nOwn={s['n_own']} nEnemyVis={s['n_enemy']}")
            got += 1
            if got >= 8:
                break
        time.sleep(0.1)
    if got == 0:
        print("Mapping opened but frame_seq never advanced (match not in progress?).")
    _k32.UnmapViewOfFile(addr)
    _k32.CloseHandle(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
