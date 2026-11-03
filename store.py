"""Delta store: one safetensors file per training step holding Delta_t = theta_t - theta_0 for every tracked tensor.

  <root>/meta.json          {"base": <hf id or path>, "dtype": "bfloat16"|"float32", "tied": bool, "tensors": {name: shape}}
  <root>/step_0001.safetensors ...

Tracked tensors: every parameter of the base model, minus lm_head.weight when the embeddings are tied (same rule as the
authors' precompute_deltas.py). Files are read lazily (safe_open), so a tensor can be streamed across all steps, in row
chunks for the embedding matrix.
"""
import json, re
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
        self._h = {}

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
        if step not in self._h:
            self._h[step] = safe_open(str(self.path(step)), framework="pt")
        return self._h[step]

    @property
    def names(self):
        return list(self.meta["tensors"])

    def shape(self, name):
        return tuple(self.meta["tensors"][name])

    def rows(self, name, step, r0=None, r1=None):
        """Delta of `name` at `step` as float32, optionally rows r0:r1 only (dim 0)."""
        h = self.handle(step)
        x = h.get_tensor(name) if r0 is None else h.get_slice(name)[r0:r1]
        return x.float()
