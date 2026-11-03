"""R3: multi-turn tool-use GRPO on the generated ERP tasks (TRL environment_factory + colocated vLLM), fp32 delta dumps.

Recipe (same optimiser settings as the math run, so RELEX is tested on the same kind of trajectory): AdamW lr 1e-6 constant,
weight decay 0.01, grad clip 1.0, KL k3 beta 0.001, clip 0.2, token-mean loss, group-std advantages, one optimizer step per
rollout batch of 32 tasks x 8 rollouts, temperature 1.0. Reward = 1 if the erpagent rubric passes (answer values / final DB
state / guard), else 0. Tool results are masked from the loss by TRL. Deltas: fp32 after every step <= --dense, then every
--sparse steps. Resumable from TRL checkpoints like relex/train_grpo.py.

  python relex/agent/train_agent.py --out $SCRATCH/.../relex/agent/run1 [--smoke]
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "third_party"))
from erp_env import SYSTEM, ErpEnv, grade, strip_think  # noqa: E402
from train_grpo import Dump, _speed_patches  # noqa: E402


def final_text(completion):
    if isinstance(completion, str):
        return completion
    asst = [m for m in completion if m.get("role") == "assistant"]
    return (asst[-1].get("content") or "") if asst else ""


def erp_reward(prompts, completions, environments=None, **kw):
    out = []
    for i, (comp, env) in enumerate(zip(completions, environments)):
        try:
            out.append(float(grade(env.task, final_text(comp), env.con)))
        except Exception:
            out.append(0.0)
    return out


def n_calls(prompts, completions, environments=None, **kw):
    """Logged only (weight 0): tool calls per rollout."""
    return [float(len(e.calls)) for e in environments]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tasks", default="$DATA/relex/agent/tasks_train.jsonl")
    ap.add_argument("--max-steps", type=int, default=100); ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=4096); ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--beta", type=float, default=0.001)
    ap.add_argument("--dense", type=int, default=30); ap.add_argument("--sparse", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=5); ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vllm-server", default="", help="host:port of a `trl vllm-serve` (server mode); default colocate")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    rows = [json.loads(l) for l in open(a.tasks)]
    ds = Dataset.from_list([{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["question"]}],
                             "id": t["id"], "db": t["db"], "cat": t["cat"], "kind": t["kind"], "gold_sql": t["gold_sql"] or "",
                             "check": json.dumps(t["check"]) if t["check"] else "", "question": t["question"]} for t in rows])
    tok = AutoTokenizer.from_pretrained(a.model)
    per_step = a.prompts * a.group
    cfg = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=2 if a.smoke else a.max_steps, seed=a.seed,
        learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.01, max_grad_norm=1.0,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=per_step // a.micro,
        steps_per_generation=per_step // a.micro, num_generations=a.group, num_iterations=1,
        max_completion_length=a.max_completion, max_tool_calling_iterations=a.max_turns, temperature=1.0, top_p=1.0, top_k=0,
        beta=a.beta, epsilon=0.2, loss_type="dapo", scale_rewards="group", mask_truncated_completions=False,
        reward_weights=[1.0, 0.0],
        use_vllm=True, vllm_mode="server" if a.vllm_server else "colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_max_model_length=a.max_completion + 2048,
        **({"vllm_server_host": a.vllm_server.split(":")[0], "vllm_server_port": int(a.vllm_server.split(":")[1])} if a.vllm_server else {}),
        vllm_importance_sampling_correction=False,
        model_init_kwargs={"dtype": torch.float32}, bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no" if a.smoke else "steps", save_steps=a.save_every, save_total_limit=2,
        report_to="none", log_completions=False, disable_dropout=True, dataloader_num_workers=0,
    )
    assert per_step % a.micro == 0
    Trainer = _speed_patches()
    dump = Dump(out, a.model, a.dense, a.sparse)
    tr = Trainer(model=a.model, reward_funcs=[erp_reward, n_calls], args=cfg, train_dataset=ds, processing_class=tok,
                 environment_factory=ErpEnv, callbacks=[dump])
    dump.trainer = tr
    ck = sorted((p for p in (out / "trainer").glob("checkpoint-*")
                 if (p / "trainer_state.json").exists() and any(p.glob("optimizer*"))),
                key=lambda p: int(p.name.split("-")[1])) if not a.smoke else []
    print("resuming from", ck[-1] if ck else None, flush=True)
    tr.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
    (out / "DONE").write_text(time.ctime())


if __name__ == "__main__":
    main()
