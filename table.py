"""Collect eval results: python relex/table.py [--evals $SCRATCH/.../relex/evals] [--prefix q15_] -> markdown on stdout."""
import argparse, json
from pathlib import Path

OOD = ["TianHongZXY/AIME2025", "MathArena/aime_2026", "MathArena/hmmt_feb_2025", "zwhe99/simplerl-OlympiadBench", "TianHongZXY/amc23"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="$DATA/relex/evals"); ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    rows = []
    for d in sorted(Path(a.evals).iterdir()):
        if not d.name.startswith(a.prefix):
            continue
        r = {"name": d.name}
        if (d / "math.json").exists():
            m = json.loads((d / "math.json").read_text())
            r["MATH"] = m["TianHongZXY/MATH"]; r["trunc%"] = m["TianHongZXY/MATH:truncated%"]; r["tok"] = m["TianHongZXY/MATH:mean_tokens"]
        if (d / "ood.json").exists():
            o = json.loads((d / "ood.json").read_text())
            for k in OOD:
                r[k.split("/")[-1][:8]] = o[k]
            r["OOD avg"] = round(sum(o[k] for k in OOD) / len(OOD), 2)
        rows.append(r)
    cols = ["name"] + sorted({k for r in rows for k in r if k != "name"}, key=lambda k: (k != "MATH", k))
    print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    main()
