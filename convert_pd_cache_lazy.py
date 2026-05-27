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

    # State shared across all hook calls
    current_key: str | None = None
    n_written: int = 0
    index: dict[str, str] = {}
    outdir: Path | None = None

    def __init__(self, f, outdir: Path, resume_index: dict):
        super().__init__(f)
        
        TrackingUnpickler.current_key = None
        TrackingUnpickler.n_written = 0
        TrackingUnpickler.index = resume_index
        TrackingUnpickler.outdir = outdir


        # dispatch table points to base class methods, not our overrides
        # must patch explicitly or subclass method overrides are ignored
        self.dispatch = self.dispatch.copy()
        self.dispatch[pickle.NEWOBJ[0]]            = TrackingUnpickler.load_newobj
        self.dispatch[pickle.BUILD[0]]             = TrackingUnpickler.load_build
        self.dispatch[pickle.SHORT_BINUNICODE[0]]  = TrackingUnpickler.load_short_binunicode
        self.dispatch[pickle.BINUNICODE[0]]        = TrackingUnpickler.load_binunicode

    def find_class(self, module, name):
        cls = super().find_class(module, name)
        return cls

    def _maybe_track_key(self):
        if self.stack:
            s = self.stack[-1]
            if isinstance(s, str) and "-" in s and 2 <= len(s) <= 40:
                TrackingUnpickler.current_key = s

    def load_short_binunicode(self):
        super().load_short_binunicode()
        self._maybe_track_key()

    def load_binunicode(self):
        super().load_binunicode()
        self._maybe_track_key()

    def load_newobj(self):
        # Stack: [..., cls, args_tuple]
        cls = self.stack[-2] if len(self.stack) >= 2 else None
        is_pd = getattr(cls, '__name__', '') == 'PhaseDiagram'
        super().load_newobj()
        if is_pd and self.stack:
            # object just constructed via __new__, __setstate__ will follow
            # don't write yet — wait for BUILD
            pass
        if self.stack:
            pass  # keep object alive until BUILD fires

    def load_build(self):
        # Stack: [..., obj, state_dict]
        obj = self.stack[-2] if len(self.stack) >= 2 else None
        is_pd = type(obj).__name__ == 'PhaseDiagram'
        super().load_build()
        # Now obj is fully constructed (__setstate__ has run)
        if is_pd and self.stack:
            key = TrackingUnpickler.current_key
            if key:
                safe = key.replace("/", "_") + ".pkl"
                out = TrackingUnpickler.outdir / safe
                if not out.exists():
                    out.write_bytes(pickle.dumps(self.stack[-1], protocol=4))
                    _index[key] = safe
                    _n_written[0] += 1
                    if _n_written[0] % 100 == 0:
                        _flush_index()
                        log(f"  {_n_written[0]} written")
                # Free memory immediately
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
        up = TrackingUnpickler(f, outdir=args.outdir, resume_index=_index)
        up.load()

    _flush_index()
    log(f"Done in {time.time()-t0:.0f}s — {_n_written[0]} PDs written")
    log(f"Index: {len(_index)} entries at {args.outdir}/index.json")


if __name__ == "__main__":
    main()