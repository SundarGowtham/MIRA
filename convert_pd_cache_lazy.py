"""
convert_pd_cache_lazy.py
Intercepts at load_reduce — fires for every REDUCE opcode regardless
of __init__ / __new__ / __setstate__ usage.
"""
import argparse, json, pickle, time
from pathlib import Path
from pickle import _Unpickler as PyUnpickler

INPUT  = Path("data/cache/phase_diagrams.pkl")
OUTDIR = Path("data/cache/pd_shards")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

_index = {}
_n_written = [0]
_outdir_ref = [None]
_last_key = [None]


def _flush_index():
    tmp = _outdir_ref[0] / "index.json.tmp"
    tmp.write_text(json.dumps(_index))
    tmp.replace(_outdir_ref[0] / "index.json")


class TrackingUnpickler(PyUnpickler):

    def find_class(self, module, name):
        cls = super().find_class(module, name)
        return cls

    def _maybe_track_key(self):
        if self.stack:
            s = self.stack[-1]
            if isinstance(s, str) and "-" in s and 2 <= len(s) <= 40:
                _last_key[0] = s

    def load_short_binunicode(self):
        super().load_short_binunicode()
        self._maybe_track_key()

    def load_binunicode(self):
        super().load_binunicode()
        self._maybe_track_key()

    def load_reduce(self):
        # Stack before REDUCE: [..., callable, args_tuple]
        # Peek at the callable to see if it's PhaseDiagram
        func = self.stack[-2] if len(self.stack) >= 2 else None
        is_pd = (func is not None and
                 getattr(func, '__name__', '') == 'PhaseDiagram')
        super().load_reduce()
        if is_pd:
            obj = self.stack[-1]
            key = _last_key[0]
            if key:
                safe = key.replace("/", "_") + ".pkl"
                out = _outdir_ref[0] / safe
                if not out.exists():
                    out.write_bytes(pickle.dumps(obj, protocol=4))
                    _index[key] = safe
                    _n_written[0] += 1
                    if _n_written[0] % 100 == 0:
                        _flush_index()
                        log(f"  {_n_written[0]} written")
                # Replace with None to free memory immediately
                self.stack[-1] = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, default=INPUT)
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _outdir_ref[0] = args.outdir

    if args.resume:
        idx = args.outdir / "index.json"
        if idx.exists():
            _index.update(json.loads(idx.read_text()))
            log(f"Resume: {len(_index)} already done")

    log(f"Opening {args.input} ({args.input.stat().st_size/1e9:.1f} GB)")
    log("Intercepting at REDUCE opcode — peak RAM should stay low")

    t0 = time.time()
    with args.input.open("rb") as f:
        up = TrackingUnpickler(f)
        up.load()

    _flush_index()
    log(f"Done in {time.time()-t0:.0f}s — {_n_written[0]} PDs written")
    log(f"Index: {len(_index)} entries at {args.outdir}/index.json")


if __name__ == "__main__":
    main()