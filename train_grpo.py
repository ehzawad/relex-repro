"""GRPO on MATH (train split) with TRL + colocated vLLM, matching the paper's verl recipe, dumping fp32 weight deltas.

Paper (App. A): GRPO, AdamW lr 1e-6, KL coef 0.001, G=8 rollouts per prompt, mini-batch 256, 500 steps, checkpoint every step.
Mapping to TRL: 256 prompts x 8 = 2048 completions per step, ONE optimizer step per rollout batch (steps_per_generation =
gradient_accumulation_steps), constant lr, no warmup, weight decay 0.01 and grad-norm clip 1.0 (verl actor defaults),
k3 ("low_var_kl") KL loss with beta 0.001, clip eps 0.2, token-mean loss aggregation (TRL "dapo" == verl token-mean),
group-std advantage normalisation, temperature 1.0, 4096-token context (Qwen2.5-Math-1.5B's limit), the authors' prompt and
grader for the 0/1 reward, no truncation masking, no vLLM importance-sampling correction (verl at the paper's time had none).

The master weights are fp32 (model loaded in fp32, bf16 autocast). After every optimizer step t <= --dense, and every
--sparse steps after that, Delta_t = theta_t(fp32) - theta_0 is written to the delta store as float32, so the same run can be
analysed both at full precision and after rounding theta_t to bf16 (what a saved bf16 HF checkpoint would hold).
Resumable: TRL checkpoints (model + optimizer + RNG) every --save-every steps; rerunning the same command resumes.

  python relex/train_grpo.py --out $SCRATCH/.../relex/own/q15_run1 [--max-steps 500] [--smoke]
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party"))
from store import Store  # noqa: E402

PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, "
          "and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n")


from grade_worker import _score  # noqa: E402  (torch-free; see grade_worker.py)

_POOL = None


def _pool(procs):
    """Grading pool started from a small forkserver that preloads only grade_worker (no torch/CUDA/vLLM state).
    RELEX_POOL=fork restores the old behaviour (fork of the full trainer process)."""
    import multiprocessing
    if os.environ.get("RELEX_POOL", "forkserver") == "fork":
        return ProcessPoolExecutor(procs)
    import importlib.util
    # Workers re-run the parent's __main__ module before unpickling a task (multiprocessing "prepare"). Point
    # __main__.__spec__ at grade_worker so they import only that module, not this torch/TRL/vLLM script.
    sys.modules["__main__"].__spec__ = importlib.util.find_spec("grade_worker")
    ctx = multiprocessing.get_context("forkserver"); ctx.set_forkserver_preload(["grade_worker"])
    return ProcessPoolExecutor(procs, mp_context=ctx)


def scored(pairs, procs, timeout):
    """Grade (response, gold) pairs in a process pool with a hard deadline. The authors' grader runs some sympy steps
    without a timeout and can spin forever on a pathological answer (seen at step 61 of q15_run1); anything not graded
    by the deadline scores 0 (as a verl reward timeout would) and the pool is killed and rebuilt."""
    global _POOL
    import concurrent.futures as cf
    if _POOL is None:
        _POOL = _pool(procs)
    futs = [_POOL.submit(_score, p) for p in pairs]
    done, late = cf.wait(futs, timeout=timeout)
    out = [f.result() if f in done and f.exception() is None else 0.0 for f in futs]
    if late:
        print(f"[reward] {len(late)} of {len(futs)} gradings exceeded {timeout}s; scored 0, rebuilding pool", flush=True)
        for pr in list(getattr(_POOL, "_processes", {}).values()):
            pr.kill()
        _POOL.shutdown(wait=False, cancel_futures=True); _POOL = None
    return out


def correct(prompts, completions, gold, **kw):
    return scored(list(zip(completions, gold)), int(os.environ.get("REWARD_PROCS", "8")), float(os.environ.get("REWARD_TIMEOUT", "120")))


def _speed_patches():
    """Two exact speed-ups (the loss is normalised over the whole accumulated batch, so micro-batch order is irrelevant):
    (1) micro-batches are formed from the generation batch sorted by completion length instead of a random shuffle;
    (2) every forward chunk is trimmed to its own longest sequence (leading all-pad prompt columns, trailing all-pad
    completion columns) and the log-probs are padded back. Without this every row is processed at the batch-wide max
    width (prompt + 3072) while the mean completion is ~860 tokens. RoPE is relative, so dropping shared left padding
    does not change attention."""
    import trl.trainer.grpo_trainer as gt
    orig = gt.shuffle_sequence_dict

    def sorted_dict(d):
        cm = d.get("completion_mask")
        if cm is None:
            return orig(d)
        perm = torch.argsort(cm.sum(1).cpu(), descending=True, stable=True)
        old = torch.randperm
        torch.randperm = lambda *a, **k: perm
        try:
            return orig(d)
        finally:
            torch.randperm = old

    gt.shuffle_sequence_dict = sorted_dict

    class Trimmed(GRPOTrainer):
        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                               compute_entropy=False, **kw):
            if any(v is not None for k, v in kw.items() if k != "compute_aux_loss"):
                return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep,
                                                                  batch_size, compute_entropy, **kw)
            bs = batch_size or input_ids.size(0); P = input_ids.size(1) - logits_to_keep
            lps, ents = [], []
            for s in range(0, input_ids.size(0), bs):
                ids, am = input_ids[s:s + bs], attention_mask[s:s + bs]
                keep = max(int(am[:, P:].sum(1).max()), 1)
                lead = int((am[:, :P].sum(0) == 0).sum())
                lp, ent, _ = super()._get_per_token_logps_and_entropies(
                    model, ids[:, lead:P + keep], am[:, lead:P + keep], keep, None, compute_entropy, **kw)
                pad = logits_to_keep - keep
                lps.append(torch.nn.functional.pad(lp, (0, pad)))
                if compute_entropy:
                    ents.append(torch.nn.functional.pad(ent, (0, pad)))
            return torch.cat(lps), (torch.cat(ents) if compute_entropy else None), None

    return Trimmed


class Dump(TrainerCallback):
    """fp32 deltas after selected optimizer steps (async write, at most one pending) + a jsonl of the training logs."""

    def __init__(self, out, base_model, dense, sparse):
        self.root, self.dense, self.sparse = Path(out) / "deltas", dense, sparse
        self.logf, self.th, self.trainer, self.base = Path(out) / "train_log.jsonl", None, None, None
        self.base_model = base_model

    def on_train_begin(self, args, state, control, **kw):
        m = self.trainer.accelerator.unwrap_model(self.trainer.model)
        names = [n for n, _ in m.named_parameters()]
        if not (self.root / "meta.json").exists():
            Store.create(self.root, self.base_model, "float32", bool(m.config.tie_word_embeddings),
                         {n: list(p.shape) for n, p in m.named_parameters()})
        from store import load_state
        from huggingface_hub import snapshot_download
        sd = load_state(snapshot_download(self.base_model, allow_patterns=["*.safetensors"]), torch.bfloat16)
        dev = next(m.parameters()).device
        self.base = {n: sd[n].to(dev) for n in names}              # theta_0 is exactly bf16 (the released base weights)
        self.store = Store(self.root)

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if not (s <= self.dense or s % self.sparse == 0):
            return
        self._join()                     # finish (and surface errors of) the previous write BEFORE building the next delta,
        m = self.trainer.accelerator.unwrap_model(self.trainer.model)   # so at most one ~6 GB host copy exists at a time
        d = {n: (p.detach().float() - self.base[n].float()).to("cpu", non_blocking=True) for n, p in m.named_parameters()}
        torch.cuda.synchronize()
        self.err = None
        self.th = threading.Thread(target=self._write, args=(s, d)); self.th.start()

    def _write(self, s, d):
        try:
            self.store.write(s, d)
        except BaseException as e:      # re-raised on the training thread by _join()
            self.err = e

    def _join(self):
        if self.th is not None:
            self.th.join(); self.th = None
        if getattr(self, "err", None) is not None:
            raise RuntimeError("delta write failed") from self.err

    def on_log(self, args, state, control, logs=None, **kw):
        if state.is_world_process_zero and logs:
            with open(self.logf, "a") as f:
                f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

    def on_train_end(self, args, state, control, **kw):
        self._join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--max-steps", type=int, default=500); ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--max-completion", type=int, default=3072); ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.001); ap.add_argument("--dense", type=int, default=125)
    ap.add_argument("--sparse", type=int, default=25); ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--vllm-mem", type=float, default=0.35); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="3 steps, no TRL checkpoints")
    ap.add_argument("--no-patch", action="store_true", help="plain TRL (no length sorting / trimming)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    ds = load_dataset("TianHongZXY/MATH", split="train")
    from utils import extract_answer_math
    ds = ds.map(lambda x: {"prompt": PROMPT.replace("{input}", x["problem"]), "gold": extract_answer_math(x["solution"])},
                remove_columns=[c for c in ds.column_names])
    tok = AutoTokenizer.from_pretrained(a.model)
    per_step = a.prompts * a.group
    cfg = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=3 if a.smoke else a.max_steps, seed=a.seed,
        learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.01, max_grad_norm=1.0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=per_step // a.micro,
        steps_per_generation=per_step // a.micro, num_generations=a.group, num_iterations=1,
        max_completion_length=a.max_completion, temperature=1.0, top_p=1.0, top_k=0,
        beta=a.beta, epsilon=0.2, loss_type="dapo", scale_rewards="group", mask_truncated_completions=False,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_max_model_length=4096,
        vllm_importance_sampling_correction=False,
        model_init_kwargs={"dtype": torch.float32}, bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no" if a.smoke else "steps", save_steps=a.save_every, save_total_limit=2,
        report_to="none", log_completions=False, disable_dropout=True, dataloader_num_workers=0,
    )
    dump = Dump(out, a.model, a.dense, a.sparse)
    Trainer = _speed_patches() if not a.no_patch else GRPOTrainer
    tr = Trainer(model=a.model, reward_funcs=[correct], args=cfg, train_dataset=ds, processing_class=tok, callbacks=[dump])
    dump.trainer = tr
    ck = sorted((p for p in (out / "trainer").glob("checkpoint-*")                    # only fully written checkpoints
                 if (p / "trainer_state.json").exists() and any(p.glob("optimizer*"))),
                key=lambda p: int(p.name.split("-")[1])) if not a.smoke else []
    print("resuming from", ck[-1] if ck else None, flush=True)
    tr.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
    (out / "DONE").write_text(time.ctime())


if __name__ == "__main__":
    main()
