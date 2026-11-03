"""GPU check: trimmed/sorted log-prob computation == TRL's original on a padded batch (left-padded prompts, right-padded
completions of very different lengths)."""
import sys, contextlib; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer
import train_grpo

def ref_impl(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, **kw):
    """Same contract as TRL's method, using plain logits (TRL's fused head only exists inside the trainer)."""
    bs = batch_size or input_ids.size(0); lps, ents = [], []
    for s in range(0, input_ids.size(0), bs):
        ids, am = input_ids[s:s + bs], attention_mask[s:s + bs]
        with self.accelerator.autocast():
            logits = model(input_ids=ids, attention_mask=am).logits[:, -logits_to_keep - 1:-1].float()
        logp = logits.log_softmax(-1)
        lps.append(logp.gather(-1, ids[:, -logits_to_keep:, None])[..., 0])
        ents.append(-(logp.exp() * logp).sum(-1))
    return torch.cat(lps), (torch.cat(ents) if compute_entropy else None), None
GRPOTrainer._get_per_token_logps_and_entropies = ref_impl
Trimmed = train_grpo._speed_patches()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", dtype=torch.float32).cuda().eval()
g = torch.Generator().manual_seed(0)
B, P, C = 6, 40, 600
plen = [12, 40, 25, 30, 8, 33]; clen = [5, 600, 50, 200, 1, 77]
ids = torch.full((B, P + C), tok.pad_token_id); am = torch.zeros(B, P + C, dtype=torch.long)
for i in range(B):
    ids[i, P - plen[i]:P] = torch.randint(100, 50000, (plen[i],), generator=g); am[i, P - plen[i]:P] = 1
    ids[i, P:P + clen[i]] = torch.randint(100, 50000, (clen[i],), generator=g); am[i, P:P + clen[i]] = 1
ids, am = ids.cuda(), am.cuda()
class A:  # minimal stand-in for the accelerator
    is_main_process = True
    def __init__(self, bf16=True): self.bf16 = bf16
    def autocast(self): return torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else contextlib.nullcontext()
cm = am[:, P:].bool()
for bf16 in (True, False):
    o = GRPOTrainer.__new__(GRPOTrainer); o.accelerator = A(bf16)
    t = Trimmed.__new__(Trimmed); t.accelerator = A(bf16)
    with torch.no_grad():
        lp0, _, _ = GRPOTrainer._get_per_token_logps_and_entropies(o, m, ids, am, C, batch_size=3)
        lpb1, _, _ = GRPOTrainer._get_per_token_logps_and_entropies(o, m, ids, am, C, batch_size=1)
        lp1, _, _ = t._get_per_token_logps_and_entropies(m, ids, am, C, batch_size=3)
    trim = (lp0 - lp1)[cm].abs(); floor = (lp0 - lpb1)[cm].abs()
    print(f"bf16={bf16}: trimmed vs original max {float(trim.max()):.2e} mean {float(trim.mean()):.2e} | "
          f"noise floor (orig bs3 vs bs1) max {float(floor.max()):.2e} mean {float(floor.mean()):.2e}")
    if not bf16:
        assert float(trim.max()) < 1e-3, "trimmed log-probs differ in fp32"
print("trim check passed")
