"""GPU check (council request): is the sort+trim patch gradient-equivalent under token-mean ("dapo") normalisation?
Fixed synthetic rollout batch on Qwen2.5-Math-1.5B (fp32 master, bf16 autocast, as in training). Accumulated gradient of
the on-policy GRPO policy loss  -sum A * exp(logp - logp.detach()) / N_tokens  over micro-batches:
  A  = original order, full batch width (what plain TRL does)      B = sorted by length, each micro-batch trimmed (patch)
  A2 = a different random order, full width (noise floor of reordering alone)
Reports cosine and relative L2 of B vs A next to A2 vs A, and the same for one AdamW step's parameter update."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", dtype=torch.float32).cuda()
m.gradient_checkpointing_enable(); m.config.use_cache = False
B, P, C, mb = 24, 48, 700, 4
g = torch.Generator().manual_seed(1)
plen = torch.randint(8, P + 1, (B,), generator=g); clen = torch.randint(1, C + 1, (B,), generator=g)
clen[:3] = C                                                    # a few truncated completions, as in real batches
ids = torch.full((B, P + C), 151643); am = torch.zeros(B, P + C, dtype=torch.long)
for i in range(B):
    ids[i, P - plen[i]:P] = torch.randint(100, 50000, (int(plen[i]),), generator=g); am[i, P - plen[i]:P] = 1
    ids[i, P:P + clen[i]] = torch.randint(100, 50000, (int(clen[i]),), generator=g); am[i, P:P + clen[i]] = 1
adv = torch.randn(B, generator=g)
ids, am, adv = ids.cuda(), am.cuda(), adv.cuda(); N = float(am[:, P:].sum())


AC = {"on": True}


def loss_chunk(rows, trim):
    i, a = ids[rows], am[rows]
    if trim:
        keep = int(a[:, P:].sum(1).max()); lead = int((a[:, :P].sum(0) == 0).sum())
        i, a = i[:, lead:P + keep], a[:, lead:P + keep]; k = keep
    else:
        k = C
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AC["on"]):
        logits = m(input_ids=i, attention_mask=a).logits[:, -k - 1:-1].float()
    lp = logits.log_softmax(-1).gather(-1, i[:, -k:, None])[..., 0]
    cm = a[:, -k:].float()
    return -(adv[rows, None] * torch.exp(lp - lp.detach()) * cm).sum() / N


def accumulate(order, trim):
    m.zero_grad(set_to_none=True)
    for s in range(0, B, mb):
        loss_chunk(order[s:s + mb], trim).backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()


def adam_update(grad):
    """One AdamW step from fresh state (lr 1e-6): the update is lr * g / (|g| + eps) elementwise at step 1."""
    return 1e-6 * grad / (grad.abs() + 1e-8)


def cmp(x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0)), float((x - y).norm() / y.norm())


ga = accumulate(torch.randperm(B, generator=torch.Generator().manual_seed(2)).cuda(), False)
ga2 = accumulate(torch.randperm(B, generator=torch.Generator().manual_seed(3)).cuda(), False)
gb = accumulate(torch.argsort(am[:, P:].sum(1), descending=True), True)
AC["on"] = False                                   # fp32 reference: no autocast, original order, full width
gref = accumulate(torch.randperm(B, generator=torch.Generator().manual_seed(2)).cuda(), False)
gref_t = accumulate(torch.argsort(am[:, P:].sum(1), descending=True), True)
c, r = cmp(gref_t, gref); print(f"fp32: sort+trim vs original: grad cos {c:.7f} relL2 {r:.2e}")
for name, x in [("plain TRL bf16 (A)", ga), ("plain TRL bf16, other order (A2)", ga2), ("sort+trim bf16 (B)", gb)]:
    c, r = cmp(x, gref); cu, ru = cmp(adam_update(x), adam_update(gref))
    print(f"{name} vs fp32 reference: grad cos {c:.6f} relL2 {r:.2e} | AdamW step-1 update cos {cu:.6f}")
