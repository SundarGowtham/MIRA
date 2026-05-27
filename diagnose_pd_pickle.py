"""
diagnose_pd_pickle.py
---------------------
Safely inspect the first N bytes of phase_diagrams.pkl to understand
how PhaseDiagram objects are constructed in the pickle stream.

Never loads the full file. Reads at most --max-mb megabytes.
Prints every REDUCE call with its callable's name/module so we can
see whether PhaseDiagram appears via REDUCE, BUILD, or something else.

Usage:
    python diagnose_pd_pickle.py              # reads first 5 MB
    python diagnose_pd_pickle.py --max-mb 20  # reads first 20 MB
"""

import argparse
import io
import os
import resource
import sys
import time
from pathlib import Path
from pickle import _Unpickler as PyUnpickler

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

DATA_CACHE = Path("data/cache")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def limit_ram_mb(mb: int):
    """Set a hard RAM ceiling via RLIMIT_AS. Only works on Unix/macOS."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = mb * 1024 * 1024
        # Only tighten — never raise above current hard limit
        if hard == resource.RLIM_INFINITY or new_limit < hard:
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
            log(f"RAM ceiling set to {mb} MB")
        else:
            log(f"Could not set RAM ceiling (hard limit is {hard // (1024**2)} MB)")
    except Exception as e:
        log(f"RLIMIT_AS not available on this platform ({e}) — no RAM ceiling set")


class DiagnosticUnpickler(PyUnpickler):
    """
    Intercepts every significant opcode and prints what it sees.
    Tracks PhaseDiagram specifically across all construction paths:
      REDUCE    — callable(*args)
      BUILD     — obj.__setstate__(state)
      NEWOBJ    — cls.__new__(cls, *args)
      STACK_GLOBAL — push module.name onto stack
    """

    def __init__(self, f, max_reduces=2000):
        super().__init__(f)
        self.max_reduces = max_reduces
        self.n_reduces = 0
        self.n_builds = 0
        self.n_newobjs = 0
        self.n_stack_globals = 0
        self.pd_reduces = 0
        self.pd_builds = 0
        self.pd_newobjs = 0
        self.last_string = None   # track most recent string (likely a key)
        self._stack_global_last = None  # last STACK_GLOBAL result

    # ── track strings (these are dict keys) ──────────────────────────────
    def load_short_binunicode(self):
        super().load_short_binunicode()
        if self.stack:
            s = self.stack[-1]
            if isinstance(s, str):
                self.last_string = s

    def load_binunicode(self):
        super().load_binunicode()
        if self.stack:
            s = self.stack[-1]
            if isinstance(s, str):
                self.last_string = s

    # ── STACK_GLOBAL: pushes a class/function from module + name ─────────
    def load_stack_global(self):
        super().load_stack_global()
        if self.stack:
            obj = self.stack[-1]
            name = getattr(obj, '__name__', '')
            module = getattr(obj, '__module__', '')
            self.n_stack_globals += 1
            if 'PhaseDiagram' in name or 'phase_diagram' in str(module):
                log(f"  STACK_GLOBAL → {module}.{name}  "
                    f"(last_key='{self.last_string}')")
            self._stack_global_last = obj

    # ── REDUCE: callable(*args_tuple) ────────────────────────────────────
    def load_reduce(self):
        # Stack before REDUCE: [..., callable, args_tuple]
        func = self.stack[-2] if len(self.stack) >= 2 else None
        fname = getattr(func, '__name__', '') if func else ''
        fmod = getattr(func, '__module__', '') if func else ''

        is_pd = 'PhaseDiagram' in fname or 'phase_diagram' in str(fmod)

        super().load_reduce()

        self.n_reduces += 1
        if is_pd:
            self.pd_reduces += 1
            log(f"  REDUCE: {fmod}.{fname}  "
                f"(last_key='{self.last_string}')  "
                f"→ object on stack: {type(self.stack[-1]).__name__ if self.stack else 'empty'}")

        # Replace result with None immediately to free memory
        if self.stack:
            self.stack[-1] = None

        if self.n_reduces >= self.max_reduces:
            raise StopIteration(f"Reached max_reduces={self.max_reduces}")

    # ── BUILD: calls obj.__setstate__(state) ─────────────────────────────
    def load_build(self):
        # Stack before BUILD: [..., obj, state]
        obj = self.stack[-2] if len(self.stack) >= 2 else None
        oname = type(obj).__name__ if obj is not None else ''
        omod = getattr(type(obj), '__module__', '') if obj is not None else ''

        is_pd = 'PhaseDiagram' in oname or 'phase_diagram' in str(omod)

        super().load_build()

        self.n_builds += 1
        if is_pd:
            self.pd_builds += 1
            log(f"  BUILD on {omod}.{oname}  "
                f"(last_key='{self.last_string}')")

    # ── NEWOBJ: cls.__new__(cls, *args) ───────────────────────────────────
    def load_newobj(self):
        # Stack before NEWOBJ: [..., cls, args]
        cls = self.stack[-2] if len(self.stack) >= 2 else None
        cname = getattr(cls, '__name__', '') if cls else ''
        cmod = getattr(cls, '__module__', '') if cls else ''

        is_pd = 'PhaseDiagram' in cname or 'phase_diagram' in str(cmod)

        super().load_newobj()

        self.n_newobjs += 1
        if is_pd:
            self.pd_newobjs += 1
            log(f"  NEWOBJ: {cmod}.{cname}  "
                f"(last_key='{self.last_string}')")
        if self.stack:
            self.stack[-1] = None  # free memory immediately


def diagnose(path: Path, max_mb: int, max_reduces: int):
    size = path.stat().st_size
    read_bytes = min(max_mb * 1024 * 1024, size)
    log(f"File: {path}  size={size / 1e9:.2f} GB")
    log(f"Reading first {read_bytes / 1e6:.1f} MB of {size / 1e6:.0f} MB")
    log(f"Max REDUCE calls before stopping: {max_reduces}")
    log("")

    with path.open("rb") as f:
        data = f.read(read_bytes)

    log(f"Bytes read into buffer: {len(data) / 1e6:.1f} MB")
    log("Starting diagnostic unpickle...")
    log("")

    buf = io.BytesIO(data)
    up = DiagnosticUnpickler(buf, max_reduces=max_reduces)

    try:
        up.load()
    except StopIteration as e:
        log(f"\nStopped early: {e}")
    except EOFError:
        log("\nReached end of buffer (expected — we only read a slice)")
    except Exception as e:
        log(f"\nStopped with exception: {type(e).__name__}: {e}")

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  STACK_GLOBAL calls: {up.n_stack_globals}")
    log(f"  REDUCE calls:       {up.n_reduces}  (PhaseDiagram: {up.pd_reduces})")
    log(f"  BUILD calls:        {up.n_builds}   (PhaseDiagram: {up.pd_builds})")
    log(f"  NEWOBJ calls:       {up.n_newobjs}  (PhaseDiagram: {up.pd_newobjs})")
    log("")
    if up.pd_reduces + up.pd_builds + up.pd_newobjs == 0:
        log("  !! PhaseDiagram NOT seen in any construction opcode.")
        log("     Either the file hasn't reached a PD yet, or they use")
        log("     a different construction path. Try --max-mb 50 or more.")
    else:
        log("  ✓ PhaseDiagram construction opcode(s) identified above.")
        log("    Use this information to fix convert_pd_cache_lazy.py.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=DATA_CACHE / "phase_diagrams.pkl")
    p.add_argument("--max-mb", type=int, default=5,
                   help="Max MB to read from the file (default 5). "
                        "Increase if PhaseDiagram not found.")
    p.add_argument("--ram-ceiling-mb", type=int, default=1024,
                   help="Hard RAM ceiling in MB (default 1024 = 1 GB). "
                        "Process will be killed if it tries to exceed this.")
    p.add_argument("--max-reduces", type=int, default=2000,
                   help="Stop after this many REDUCE calls (safety limit).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set RAM ceiling BEFORE doing anything else
    limit_ram_mb(args.ram_ceiling_mb)

    diagnose(args.cache, args.max_mb, args.max_reduces)