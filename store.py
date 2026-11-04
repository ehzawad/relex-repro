"""Delta store: one safetensors file per training step holding Delta_t = theta_t - theta_0 for every tracked tensor.

  <root>/meta.json          {"base": <hf id or path>, "dtype": "bfloat16"|"float32", "tied": bool, "tensors": {name: shape}}
  <root>/step_0001.safetensors ...

Tracked tensors: every parameter of the base model, minus lm_head.weight when the embeddings are tied (same rule as the
authors' precompute_deltas.py). Files are read lazily (safe_open), so a tensor can be streamed across all steps, in row
chunks for the embedding matrix.
"""
import json, math, os, re, threading
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_state(path, dtype=torch.float32):
    """All tensors of a HF checkpoint directory (safetensors shards) in `dtype`."""
    sd = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            sd[k] = v.to(dtype)
    return sd


def tracked(sd, tied):
    return [k for k in sd if not (tied and k == "lm_head.weight")]


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.steps = sorted(int(m.group(1)) for f in self.root.glob("step_*.safetensors")
                            for m in [re.match(r"step_(\d+)\.safetensors$", f.name)] if m)
        # Open safetensors handles (each mmaps a whole step file). RELEX_MAX_OPEN > 0 bounds how many stay open (LRU),
        # for nodes with strict overcommit where mapping every step file at once is refused (ENOMEM).
        self._h = OrderedDict(); self._lock = threading.Lock(); self._max = int(os.environ.get("RELEX_MAX_OPEN", "0"))
        self._hdr = {}; self._pread = os.environ.get("RELEX_PREAD") == "1"

    @staticmethod
    def create(root, base, dtype, tied, shapes):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        (root / "meta.json").write_text(json.dumps({"base": str(base), "dtype": dtype, "tied": tied, "tensors": shapes}, indent=1))

    def path(self, step):
        return self.root / f"step_{step:04d}.safetensors"

    def write(self, step, deltas):
        tmp = self.root / f".tmp_step_{step:04d}.safetensors"
        save_file({k: v.contiguous() for k, v in deltas.items()}, str(tmp))
        tmp.rename(self.path(step))

    def handle(self, step):
        with self._lock:
            h = self._h.get(step)
            if h is None:
                h = self._h[step] = safe_open(str(self.path(step)), framework="pt")
            self._h.move_to_end(step)
            while self._max and len(self._h) > self._max:
                self._h.popitem(last=False)   # unmapped once no caller still holds it
            return h

    @property
    def names(self):
        return list(self.meta["tensors"])

    def shape(self, name):
        return tuple(self.meta["tensors"][name])

    _DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}

    def header(self, step):
        """Parsed safetensors header of a step file (cached): name -> (dtype, shape, absolute byte offset)."""
        with self._lock:
            hd = self._hdr.get(step)
        if hd is None:
            with open(self.path(step), "rb") as f:
                n = int.from_bytes(f.read(8), "little"); raw = json.loads(f.read(n))
            hd = {k: (v["dtype"], v["shape"], 8 + n + v["data_offsets"][0]) for k, v in raw.items() if k != "__metadata__"}
            with self._lock:
                self._hdr[step] = hd
        return hd

    def rows(self, name, step, r0=None, r1=None):
        """Delta of `name` at `step` as float32, optionally rows r0:r1 only (dim 0)."""
        if self._pread:   # plain reads, no mmap (RELEX_PREAD=1): strict-overcommit nodes refuse many large mappings
            dt, shape, off = self.header(step)[name]; dtype = self._DT[dt]; esz = torch.tensor([], dtype=dtype).element_size()
            row = esz * (math.prod(shape[1:]) if len(shape) > 1 else 1)
            a, b = (0, shape[0]) if r0 is None or not shape else (r0, r1)
            n = row * (b - a) if shape else esz
            buf = bytearray(n)
            with open(self.path(step), "rb", buffering=0) as f:
                f.seek(off + row * a); got = f.readinto(buf)
            assert got == n, (name, step, got, n)
            x = torch.frombuffer(buf, dtype=dtype)
            return x.reshape([b - a] + list(shape[1:]) if shape else []).float()
        h = self.handle(step)
        x = h.get_tensor(name) if r0 is None else h.get_slice(name)[r0:r1]
        return x.float()
