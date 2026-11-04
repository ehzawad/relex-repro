"""Build relex/report/relex_findings.pdf: the findings report of the RELEX (arXiv 2605.21468) reproduction.

Every number comes from relex/PLAN.md (cited by line), relex/README.md, relex/results/ (read here directly from the JSON
files) or the data directory's logs/final_paired.txt, logs/ood_paired.txt and logs/normood.log (parsed here). Scores are read from the result files rather than
typed, so a table cannot drift from its source; intervals that exist only in PLAN.md are copied verbatim.

  python relex/report/build_report.py      (run on a compute node; needs reportlab + matplotlib)
"""
import json, re, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402,F401

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,  # noqa: E402
                                KeepTogether, ListFlowable, ListItem)

HERE = Path(__file__).resolve().parent
RELEX = HERE.parent
RES = RELEX / "results"
FIGS = HERE / "figs"
OUT = HERE / "relex_findings.pdf"
LOGS = Path("$DATA/relex/logs")
FINAL_PAIRED = LOGS / "final_paired.txt"
OOD_PAIRED = LOGS / "ood_paired.txt"
NORM_LOG = LOGS / "normood.log"
FIGS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------- fonts
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(_ttf / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(_ttf / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

# ----------------------------------------------------------------------------------------------------------- colours
# One meaning per colour across every figure (reference categorical palette, light mode).
C_REAL = "#2a78d6"      # real RLVR checkpoints
C_FP32 = "#eb6834"      # extrapolated from fp32 deltas
C_BF16 = "#1baf7a"      # extrapolated from bf16 deltas (own run rounded to bf16, or the released bf16 trajectory)
C_REL = "#6250d6"       # released bf16 trajectory (geometry) / power-law variant (agent)
C_NORM = "#008300"      # norm-matched control
C_TWO = "#e87ba4"       # two-point extrapolation
C_GREY = "#52514e"      # paper values, nulls, base model
INK = "#0b0b0b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.fontsize": 7.2,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5, "savefig.dpi": 220,
})

# ----------------------------------------------------------------------------------------------------------- data
MATH_KEY = "TianHongZXY/MATH"
OOD_KEYS = [("TianHongZXY/AIME2025", "AIME 2025"), ("MathArena/aime_2026", "AIME 2026"),
            ("MathArena/hmmt_feb_2025", "HMMT Feb 2025"), ("TianHongZXY/amc23", "AMC 2023"),
            ("zwhe99/simplerl-OlympiadBench", "OlympiadBench")]


def ev(group, name, file="math.json"):
    return json.load(open(RES / "evals" / group / name / file))


def math(group, name, file="math.json"):
    return ev(group, name, file)[MATH_KEY]


def trunc(group, name, file="math.json"):
    return ev(group, name, file)[MATH_KEY + ":truncated%"]


def ag(tag):
    return json.load(open(RES / "agent_evals" / f"opencode__{tag}.summary.json"))


def ws(file, which, window):
    d = json.load(open(RES / "weight_stats" / file))
    k = [k for k in d if k.split("/")[-1].startswith(which)]
    assert len(k) == 1, (file, which, list(d))
    w = d[k[0]]["windows"][str(window)]
    return w["ev1"]["median"], w["r2"]["median"], w["frac_r2_gt_0.98"]


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def f3(x):
    return f"{x:.3f}"


R = lambda n, f="math.json": math("q15_released", n, f)          # noqa: E731
O = lambda n, f="math.json": math("q15_own", n, f)               # noqa: E731
Q = lambda n, f="math16k.json": math("q4b_released", n, f)       # noqa: E731

# paired comparisons (parsed, not typed): MATH from final_paired.txt, OOD from ood_paired.txt (relex/ood_paired.py)
def parse_pairs(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"(\S+) - (\S+): (\S+)\s+95% CI \[(\S+), (\S+)\]", line)
        if m:
            out[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
    return out


FPP = parse_pairs(FINAL_PAIRED)
FP = {a: v for (a, b), v in FPP.items() if b == "own_rlvr_500"}
assert len(FP) == 13 and len(FPP) == 14, FPP
OPP = parse_pairs(OOD_PAIRED)
OP = {a: v for (a, b), v in OPP.items() if b == "own_rlvr_500"}
assert len(OP) == 9, OPP
OOD_N = sorted(set(re.findall(r"problems=(\d+)", OOD_PAIRED.read_text())))
assert len(OOD_N) == 1, OOD_N
OOD_N = OOD_N[0]
# global norm ratio ||RELEX c75->500 fp32|| / ||real step-500 delta|| (own/q15_run1/normratio.py, logged in normood.log)
NORM_RATIO = float(re.search(r"global \|\|RELEX c75->500\|\| / \|\|real 500\|\| = ([0-9.]+)",
                             NORM_LOG.read_text()).group(1))


def ci(v):
    d, lo, hi = v
    return f"{d} [{lo}, {hi}]"


def fpci(name, ref="own_rlvr_500"):
    return ci(FPP[(name, ref)])


def opci(name):
    return ci(OP[name])


def ood_mean(name):
    """Macro mean over the five OOD benchmarks of avg@8 (as relex/ood_paired.py)."""
    d = ev("q15_own", name, "ood.json")
    return sum(d[k] for k, _ in OOD_KEYS) / len(OOD_KEYS)


# Weight-match checks
CMP15 = json.load(open(RES / "weight_stats" / "released_q15_compare_released_relex.json"))
CMP4 = json.load(open(RES / "weight_stats" / "released_q4b_compare.json"))
c15 = {k.split("/")[-1]: v for k, v in CMP15.items()}
c4 = {k.split("/")[-1]: v for k, v in CMP4.items()}

# ----------------------------------------------------------------------------------------------------------- styles
ss = {}
ss["body"] = ParagraphStyle("body", fontName="DV", fontSize=8.9, leading=11.9, spaceAfter=4.5, alignment=TA_LEFT,
                            textColor=INK)
ss["small"] = ParagraphStyle("small", parent=ss["body"], fontSize=8.2, leading=10.6, spaceAfter=3)
ss["title"] = ParagraphStyle("title", parent=ss["body"], fontName="DVB", fontSize=14.5, leading=18, spaceAfter=3)
ss["sub"] = ParagraphStyle("sub", parent=ss["body"], fontSize=8.2, leading=10.5, textColor=colors.HexColor(C_GREY),
                           spaceAfter=6)
ss["h1"] = ParagraphStyle("h1", parent=ss["body"], fontName="DVB", fontSize=11.5, leading=14.5, spaceBefore=7,
                          spaceAfter=4, keepWithNext=1)
ss["h2"] = ParagraphStyle("h2", parent=ss["body"], fontName="DVB", fontSize=9.4, leading=12, spaceBefore=4,
                          spaceAfter=2.5, keepWithNext=1)
ss["cap"] = ParagraphStyle("cap", parent=ss["body"], fontSize=7.4, leading=9.2, textColor=colors.HexColor(C_GREY),
                           spaceBefore=1.5, spaceAfter=6)
ss["cell"] = ParagraphStyle("cell", parent=ss["body"], fontSize=7.5, leading=9.1, spaceAfter=0)
ss["cellb"] = ParagraphStyle("cellb", parent=ss["cell"], fontName="DVB")
ss["li"] = ParagraphStyle("li", parent=ss["body"], fontSize=8.6, leading=11.2, spaceAfter=2.2)


def P(t, s="body"):
    return Paragraph(t, ss[s])


def H1(t):
    return P(t, "h1")


def H2(t):
    return P(t, "h2")


def CAP(t):
    return P(t, "cap")


def table(rows, widths, header=1, zebra=True, align_right_from=1):
    data = [[c if not isinstance(c, str) else Paragraph(c, ss["cellb" if i < header else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=header, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, header - 1), (-1, header - 1), 0.6, colors.HexColor("#8a8984")),
          ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#8a8984")),
          ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#8a8984")),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(header, len(rows)):
            if (i - header) % 2 == 1:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f3f0")))
    t.setStyle(TableStyle(st))
    return t


def bullets(items, style="li"):
    return ListFlowable([ListItem(P(x, style), leftIndent=11, value="circle") for x in items],
                        bulletType="bullet", start="•", leftIndent=11, bulletFontName="DV", bulletFontSize=7)


def numbered(items, style="li"):
    return ListFlowable([ListItem(P(x, style), leftIndent=13) for x in items], bulletType="1", leftIndent=13,
                        bulletFontName="DVB", bulletFontSize=8.4, bulletFormat="%s.")


def fig(path, width_in, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_in * inch, height=width_in * inch * ih / iw)
    img.hAlign = "LEFT"
    return KeepTogether([img, CAP(caption)])


# ----------------------------------------------------------------------------------------------------------- figures
def fig_coordinate():
    """Released Qwen2.5-Math-1.5B: MATH vs fit coordinate; real branches, RELEX c75, paper Table 3 at 2x label."""
    real_x = [75, 100, 200, 300, 400, 500]
    real_y = [R(f"rlvr_{s}") for s in real_x]
    rx = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000]
    ry = [R(f"relex_c75_t{s}") for s in rx]
    # Paper Table 3, T_obs=75 row (labels 100/200/500/750/1000), placed at 2x the label (PLAN.md L58)
    px = [200, 400, 1000, 1500, 2000]
    py = [67.8, 70.0, 71.6, 65.7, 68.4]
    fig_, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.axvspan(0, 75, color="#e4e3df", alpha=0.6, lw=0)
    ax.text(80, 73.0, "fit window (1-75)", ha="left", va="top", fontsize=6.8, color=C_GREY)
    ax.plot(real_x, real_y, "-o", color=C_REAL, label="real RLVR (released branches step_75..step_500)")
    ax.plot(rx, ry, "-o", color=C_BF16, label="RELEX c75 from released bf16 trajectory (this reproduction)")
    ax.plot(px, py, "s", mfc="white", mec=C_GREY, mew=1.3, ms=6.5, ls="none",
            label="paper Table 3 (c75 row), plotted at 2 x its step label")
    ax.axvline(500, color=C_REAL, lw=0.8, ls=":")
    ax.axvline(1000, color=C_GREY, lw=0.8, ls=":")
    ax.annotate("comparator:\nbranch step_500", xy=(500, 66.2), xytext=(560, 65.0), fontsize=6.8, color=C_GREY)
    ax.annotate("paper's 'RELEX step 500'\n= coordinate 1000", xy=(1000, 66.2), xytext=(1040, 65.0), fontsize=6.8,
                color=C_GREY)
    ax.set_xlim(0, 2080)
    ax.set_ylim(64.2, 73.2)
    ax.set_xlabel("fit coordinate (released branch-label index; RELEX target T)")
    ax.set_ylabel("MATH-5000 accuracy (%)")
    fig_.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig_.tight_layout(rect=(0, 0.12, 1, 1))
    p = FIGS / "fig1_coordinate.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def fig_geometry():
    wins = [25, 50, 75, 100, 125]
    own32 = [ws("own_q15_run1_gram_stats_s125.json", "grams_fp32", w) for w in wins]
    own16 = [ws("own_q15_run1_gram_stats_s125.json", "grams_bf16", w) for w in wins]
    rel = [ws("released_q15_gram_stats.json", "grams", w) for w in wins]
    fig_, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for k, (ax, lab) in enumerate(zip(axs, ["median rank-1 EV (fraction)", "median R² of linear fit to c₁(t)"])):
        ax.plot(wins, [x[k] for x in own32], "-o", color=C_FP32, label="own run, fp32 deltas")
        ax.plot(wins, [x[k] for x in own16], "-o", color=C_BF16, label="own run, same deltas rounded to bf16")
        ax.plot(wins, [x[k] for x in rel], "--s", color=C_REL, label="released 1.5B trajectory (bf16)")
        # random-walk null at T=75 (PLAN.md L61): driftless 0.811 / 0.958; drift 5% of step noise 0.830 / 0.971
        ax.plot([75], [(0.811, 0.958)[k]], "X", color=C_GREY, ms=7.5, ls="none", label="random-walk null, T=75 (no drift)")
        ax.plot([75], [(0.830, 0.971)[k]], "P", color=C_GREY, ms=7, mfc="white", ls="none",
                label="random-walk null, T=75 (drift = 5% of step noise)")
        ax.set_xlabel("fit window (steps 1..T_obs)")
        ax.set_ylabel(lab)
        ax.set_xticks(wins)
    axs[0].set_ylim(0.75, 1.0)
    axs[1].set_ylim(0.84, 1.0)
    axs[1].axhline(0.95, color=C_GREY, lw=0.8, ls=":", label="P5 threshold R² = 0.95")
    h, l_ = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig_.legend(h + h2[-1:], l_ + l2[-1:], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig_.tight_layout(rect=(0, 0.25, 1, 1))
    p = FIGS / "fig2_geometry.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def fig_horizon():
    real_x = [25, 50, 75, 100, 125, 150, 175, 200, 250, 275, 300, 400, 450, 475, 500]
    real_y = [O(f"own_rlvr_{s}") for s in real_x]
    tx = [100, 200, 300, 400, 500, 1000]
    y32 = [O(f"own_relex_c75_t{s}_fp32") for s in tx]
    y16 = [O(f"own_relex_c75_t{s}_bf16") for s in tx]
    fig_, ax = plt.subplots(figsize=(6.6, 2.75))
    ax.axvspan(0, 75, color="#e4e3df", alpha=0.6, lw=0)
    ax.text(80, 73.4, "fit window (1-75)", ha="left", va="top", fontsize=6.8, color=C_GREY)
    ax.plot(real_x, real_y, "-o", color=C_REAL, label="real GRPO checkpoints (own run, seed 0)")
    ax.plot(tx[:5], y32[:5], "-o", color=C_FP32, label="RELEX c75 from fp32 deltas")
    ax.plot(tx[:5], y16[:5], "-o", color=C_BF16, label="RELEX c75 from the same deltas rounded to bf16")
    ax.plot(tx[4:], y32[4:], ":", color=C_FP32, lw=1.2)
    ax.plot(tx[4:], y16[4:], ":", color=C_BF16, lw=1.2)
    ax.plot([1000], [y32[5]], "o", color=C_FP32)
    ax.plot([1000], [y16[5]], "o", color=C_BF16)
    ax.text(985, y32[5] + 0.25, "target 1000\n(no real step)", fontsize=6.6, color=C_GREY, ha="right", va="bottom")
    ax.set_xlim(0, 1030)
    ax.set_ylim(65.5, 73.5)
    ax.set_xlabel("optimizer step (real) / extrapolation target T (RELEX)")
    ax.set_ylabel("MATH-5000 accuracy (%)")
    ax.legend(loc="lower right")
    fig_.tight_layout()
    p = FIGS / "fig3_horizon.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def fig_math_ood():
    """Own run: MATH (left) and OOD macro mean (right) along the RELEX c75 fp32 line vs real checkpoints."""
    tx = [100, 200, 300, 400, 500, 1000]
    real_m = [25, 50, 75, 100, 125, 150, 175, 200, 250, 275, 300, 400, 450, 475, 500]
    real_o = [s for s in real_m if (RES / "evals" / "q15_own" / f"own_rlvr_{s}" / "ood.json").exists()]
    panels = [("MATH-5000 accuracy (%)", lambda n: O(n), real_m),
              ("OOD macro mean, avg@8 (%)", ood_mean, real_o)]
    fig_, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, (lab, f, rx) in zip(axs, panels):
        ry = [f(f"own_rlvr_{s}") for s in rx]
        y = [f(f"own_relex_c75_t{s}_fp32") for s in tx]
        ax.axvspan(0, 75, color="#e4e3df", alpha=0.6, lw=0)
        ax.plot(rx, ry, "-o", color=C_REAL, ms=4, label="real GRPO checkpoint (own run, seed 0)")
        ax.plot(tx[:5], y[:5], "-o", color=C_FP32, ms=4, label="RELEX c75, fp32, target T")
        ax.plot(tx[4:], y[4:], ":", color=C_FP32, lw=1.2)
        ax.plot([1000], [y[5]], "o", color=C_FP32, ms=4)
        ax.plot([500], [f("own_relexnorm_c75_t500_fp32")], "*", color=C_NORM, ms=9, ls="none",
                label="RELEX c75 direction at real step-500 norm (T = 500)")
        ax.set_xlim(0, 1030)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlabel("optimizer step (real) / target T (RELEX)")
        ax.set_ylabel(lab)
    axs[0].set_ylim(65.5, 73.5)
    axs[1].set_ylim(15.0, 23.0)
    fig_.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig_.tight_layout(rect=(0, 0.14, 1, 1))
    p = FIGS / "fig5_math_ood.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def fig_agent():
    """Seed-1 agent run, test split pass rate vs target step. Filled = full protocol (300 x 2), hollow = quick (300 x 1)."""
    fig_, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axvspan(0, 25, color="#e4e3df", alpha=0.6, lw=0)
    ax.text(12.5, 2, "fit window\n(1-25)", ha="center", va="bottom", fontsize=6.8, color=C_GREY)
    base = ag("qwen35_2b_base")["pass"]
    ax.axhline(base, color=C_GREY, ls="--", lw=1)
    ax.text(101, base + 1.2, f"base Qwen3.5-2B {f1(base)}", fontsize=6.8, color=C_GREY, ha="right", va="bottom")
    # real: full where available, quick for 50
    rf = {25: "agent_rlvr_25", 30: "agent_rlvr_30", 40: "agent_rlvr_40", 100: "agent_rlvr_100"}
    rq = {25: "agent_rlvr_25_quick", 30: "agent_rlvr_30_quick", 40: "agent_rlvr_40_quick", 50: "agent_rlvr_50_quick"}
    lf = {30: "agent_relex_c25_t30_fp32", 40: "agent_relex_c25_t40_fp32", 100: "agent_relex_c25_t100_fp32"}
    lq = {30: "agent_relex_c25_t30_fp32_quick", 40: "agent_relex_c25_t40_fp32_quick", 50: "agent_relex_c25_t50_fp32_quick"}
    line_real = sorted({**{k: ag(v)["pass"] for k, v in rq.items()}, **{k: ag(v)["pass"] for k, v in rf.items()}}.items())
    line_lin = sorted({**{k: ag(v)["pass"] for k, v in lq.items()}, **{k: ag(v)["pass"] for k, v in lf.items()}}.items())
    ax.plot(*zip(*line_real), "-", color=C_REAL, lw=1.4)
    ax.plot(*zip(*line_lin), "-", color=C_FP32, lw=1.4)
    ax.plot(list(rf), [ag(v)["pass"] for v in rf.values()], "o", color=C_REAL, ms=6, label="real GRPO checkpoint")
    ax.plot(list(rq), [ag(v)["pass"] for v in rq.values()], "o", mfc="white", mec=C_REAL, mew=1.3, ms=6)
    ax.plot(list(lf), [ag(v)["pass"] for v in lf.values()], "o", color=C_FP32, ms=6, label="RELEX c25, linear, fp32")
    ax.plot(list(lq), [ag(v)["pass"] for v in lq.values()], "o", mfc="white", mec=C_FP32, mew=1.3, ms=6)
    ax.plot([50, 98.2], [ag("agent_relexpow_c25_t50_fp32_quick")["pass"], ag("agent_relexpow_c25_t100_fp32_quick")["pass"]],
            "D", mfc="white", mec=C_REL, mew=1.3, ms=5.5, ls="none", label="RELEX c25, power-law coefficient")
    ax.plot([100], [ag("agent_relex_c25_t100_bf16")["pass"]], "o", color=C_BF16, ms=6, ls="none",
            label="RELEX c25, linear, bf16-rounded")
    ax.plot([40], [ag("agent_twopoint_10_25_t40_quick")["pass"]], "^", mfc="white", mec=C_TWO, mew=1.3, ms=6.5, ls="none")
    ax.plot([101.8], [ag("agent_twopoint_10_25_t100")["pass"]], "^", color=C_TWO, ms=6.5, ls="none",
            label="two-point (10, 25)")
    ax.plot([100], [ag("agent_relexnorm_c25_t100")["pass"]], "*", color=C_NORM, ms=10, ls="none",
            label="RELEX c25 direction, real step-100 norm (oracle control)")
    ax.plot([], [], "o", mfc="white", mec=C_GREY, ls="none", label="hollow = quick probe (300 tasks x 1)")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 40, 50, 75, 100])
    ax.set_xticks([30], minor=True)
    ax.set_xlabel("optimizer step (real) / extrapolation target T")
    ax.set_ylabel("test pass rate (%), 300 unseen-DB tasks")
    fig_.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig_.tight_layout(rect=(0, 0.22, 1, 1))
    p = FIGS / "fig4_agent.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


# ----------------------------------------------------------------------------------------------------------- page deco
def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(colors.HexColor(C_GREY))
    c.drawString(0.7 * inch, 0.45 * inch, "RELEX (arXiv 2605.21468): critical reproduction, findings report")
    c.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"page {doc.page}")
    c.restoreState()


# ----------------------------------------------------------------------------------------------------------- content
def build():
    F1, F2, F3, F4, F5 = fig_coordinate(), fig_geometry(), fig_horizon(), fig_agent(), fig_math_ood()
    g = {
        "null": ("0.811", "0.958", "-"), "null5": ("0.830", "0.971", "-"),
        "rel75": ws("released_q15_gram_stats.json", "grams", 75), "q4b75": ws("released_q4b_gram_stats.json", "grams", 75),
        "own32_25": ws("own_q15_run1_gram_stats_s125.json", "grams_fp32", 25),
        "own32_75": ws("own_q15_run1_gram_stats_s125.json", "grams_fp32", 75),
        "own32_125": ws("own_q15_run1_gram_stats_s125.json", "grams_fp32", 125),
        "own16_75": ws("own_q15_run1_gram_stats_s125.json", "grams_bf16", 75),
        "own16_125": ws("own_q15_run1_gram_stats_s125.json", "grams_bf16", 125),
        "ag32": ws("agent_run1_gram_stats.json", "grams_fp32", 25), "ag16": ws("agent_run1_gram_stats.json", "grams_bf16", 25),
    }
    G3 = {k: (f3(v[0]), f3(v[1])) for k, v in g.items() if not isinstance(v[0], str)}
    S = []
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------------------------------------------------------- 1 summary
    S.append(P("RELEX, a critical reproduction: findings", "title"))
    S.append(P("Reproduction of Wei, Zhu, Chen, Huang, Huang, Meng, “You Only Need Minimal RLVR Training: Extrapolating "
               "LLMs via Rank-1 Trajectories” (arXiv 2605.21468, May 2026). Results as logged in relex/PLAN.md up to the "
               f"2026-11-04 10:23 EST entry. One own math seed; a second seed is running. Built {today}.", "sub"))
    S.append(H1("1. Summary"))
    S.append(P("<b>What was reproduced.</b> (a) The authors' released Qwen2.5-Math-1.5B trajectory (500 bf16 branch "
               "checkpoints) and Qwen3-4B-Base trajectory (steps 1-75 and the main branch), re-scored with the authors' "
               "prompt, decoding and grader. (b) RELEX and its baselines, re-implemented independently and checked "
               "weight-for-weight against the released RELEX checkpoints. (c) One own 500-step GRPO run of "
               "Qwen2.5-Math-1.5B that stores fp32 deltas, analysed in fp32 and after rounding to bf16. (d) A new "
               "agentic test: 100 steps of multi-turn tool-use GRPO of Qwen3.5-2B on a synthetic ERP database."))
    S.append(P("<b>Headline findings</b> (95% intervals are problem-paired bootstrap, unadjusted for multiple comparisons "
               "except pre-registered P5 and P10):", "body"))
    a100, a100s2, abase = ag("agent_rlvr_100")["pass"], ag("agent2_rlvr_100")["pass"], ag("qwen35_2b_base")["pass"]
    anorm_frac = 100 * (ag("agent_relexnorm_c25_t100")["pass"] - abase) / (a100 - abase)
    S.append(numbered([
        f"The independent implementation matches the released RELEX weights: relative L2 difference 0.18% for 1.5B "
        f"(per-tensor cosine 1.0000) and 0.075% for Qwen3-4B. [PLAN L56, L106]",
        f"The paper's MATH-5000 numbers reproduce on the released checkpoints: 1.5B base / RLVR-500 / RELEX "
        f"{f2(R('base'))} / {f2(R('rlvr_500'))} / {f2(R('relex_c75_t1000'))} (paper 48.2 / 71.5 / 71.6); RELEX minus "
        f"RLVR-500 = -0.04 [-0.96, +0.90]. [PLAN L57, L60]",
        f"The 1.5B “RELEX step 500” is the rank-1 line evaluated at fit coordinate 1000, twice the released "
        f"branch-label coordinate of the RLVR checkpoint it is compared with. At coordinate 500 RELEX scores "
        f"{f2(R('relex_c75_t500'))}, -2.04 [-3.00, -1.10] versus RLVR-500, and it trails rank-10 (-1.02 [-1.78, -0.28]) "
        f"and two-point (25,75) extrapolation (-1.08 [-1.92, -0.20]). [PLAN L56, L58, L60]",
        "The paper's weight-space summaries (released 1.5B, 75 steps: median rank-1 explained variance 0.816, median "
        "R² 0.963) are compatible with a driftless random walk (0.811 / 0.958); on their own they do not show "
        "structure. [PLAN L61]",
        f"In the own fp32 run the trajectory is strongly rank-1 (median EV {f3(ws('own_q15_run1_gram_stats_s75.json','grams_fp32',75)[0])} "
        f"over 75 steps) but its coefficient is curved (median R² {f3(ws('own_q15_run1_gram_stats_s75.json','grams_fp32',75)[1])}, "
        f"below the pre-registered 0.95). Rounding the same run to bf16 gives {f3(ws('own_q15_run1_gram_stats_s75.json','grams_bf16',75)[0])} / "
        f"{f3(ws('own_q15_run1_gram_stats_s75.json','grams_bf16',75)[1])}. [weight_stats/own_q15_run1_gram_stats_s75.json; PLAN L110]",
        f"Own run at the matched horizon (real step 500 = {f2(O('own_rlvr_500'))}): RELEX c75→500 from fp32 deltas "
        f"{fpci('own_relex_c75_t500_fp32')}; from bf16-rounded deltas {fpci('own_relex_c75_t500_bf16')}. Simpler "
        f"extrapolations do as well: raw per-weight least squares {fpci('own_rawls_c75_t500_fp32')}, two-point (25,75) "
        f"{fpci('own_twopoint25_c75_t500_fp32')}. [final_paired.txt; PLAN L116]",
        f"The own-run MATH gain comes with a larger step, not a better direction. The plain RELEX c75→500 delta has "
        f"{NORM_RATIO:.2f}x the norm of the real step-500 delta; rescaling its direction to the real per-tensor norm "
        f"gives {f2(O('own_relexnorm_c75_t500_fp32'))}, {fpci('own_relexnorm_c75_t500_fp32')} versus real 500 "
        f"(pre-registered P10 fails). On five OOD benchmarks (macro mean of avg@8) RELEX fp32 minus real 500 is "
        f"{opci('own_relex_c75_t500_fp32')}, but real step 400 minus real 500 is also {opci('own_rlvr_400')}, so the "
        f"gap is within checkpoint-to-checkpoint variation: no robust OOD advantage or deficit. "
        f"[final_paired.txt, ood_paired.txt; PLAN L117-121]",
        f"Qwen3-4B-Base (no coordinate scaling): released RELEX {f2(Q('relex-rlvr__RELEX-Qwen3-4B-Base'))} versus the "
        f"RLVR main branch {f2(Q('relex-rlvr__RLVR-Qwen3-4B-Base'))} (a late checkpoint, consistent with step 500): "
        f"-0.30 [-1.14, +0.56]. Here rank-1 helps: raw least squares is -3.46 [-4.40, -2.54] versus RLVR main. "
        f"[PLAN L110-112]",
        f"Agentic test: GRPO lifts Qwen3.5-2B from {f1(abase)}% to {f1(a100)}% on unseen-database tasks, but RELEX "
        f"c25→100 falls to {f1(ag('agent_relex_c25_t100_fp32')['pass'])}% (seed 2: {f1(ag('agent2_relex_c25_t100_fp32')['pass'])}% "
        f"vs real {f1(a100s2)}%). Rescaling the RELEX direction to the real step-100 per-tensor norm gives "
        f"{f1(ag('agent_relexnorm_c25_t100')['pass'])}% ({anorm_frac:.0f}% of the base→step-100 gain): the collapse is "
        f"mostly magnitude overshoot. [PLAN L90, L113; agent_evals]",
    ]))
    S.append(P("<b>Bottom line.</b> <i>Holds:</i> the released artefacts and the paper's reported scores reproduce, and "
               "the method is what the authors' code computes. On Qwen3-4B, where the code applies no coordinate "
               "scaling, RELEX from 75 steps is statistically indistinguishable from the late RLVR checkpoint at this "
               "sample size (not proof of equivalence) and clearly beats raw least squares. On the own 1.5B fp32 run, "
               "RELEX from 75 steps lands at or above real step 500 on MATH. "
               "<i>Does not hold as stated:</i> the 1.5B headline comparison sits at twice the branch-label "
               "coordinate, and at the matched coordinate RELEX trails real RLVR by 2.0 points and does not beat "
               "simple baselines; the rank-1/linearity statistics are what a random walk produces in bf16 and the "
               "fp32 coefficient is not linear; on 1.5B, rank-1 is not consistently better than raw least squares or "
               f"two-point at the matched horizon, and the own-run gain over real 500 depends on a step {NORM_RATIO:.1f}x "
               f"larger than real training's (at matched norm the rank-1 direction is {FP['own_relexnorm_c75_t500_fp32'][0]} "
               "points from real 500); extrapolation does not transfer to the agentic run beyond about 1.6x the fit window. <i>Open:</i> the second own math seed (running); how released branches map to "
               "optimizer steps; multiplicity-adjusted inference; agentic results beyond one synthetic database family.",
               "body"))
    S.append(PageBreak())

    # ---------------------------------------------------------------- 2 setup
    S.append(H1("2. Setup"))
    S.append(H2("Released trajectories"))
    S.append(P("<b>Qwen2.5-Math-1.5B</b> (hf relex-rlvr/RLVR-Qwen2.5-Math-1.5B): branches step_1..step_500 stored in "
               "bf16. Converted to exact bf16 deltas for steps 1..125 (every step) and 150..500 (every 25). Because "
               "one Adam step at lr 1e-6 is far below the bf16 spacing (~1e-4 at |w|~0.02), the released step_1 "
               "delta is exactly zero everywhere and steps 2/3/4 are 99.5/99.1/98.7% zero. <b>Qwen3-4B-Base</b> "
               "(RLVR-Qwen3-4B-Base): steps 1..75 plus the main branch; the main branch is a late checkpoint "
               "(||Δ_main|| = 2.41 x ||Δ_75||, cosine 0.33), consistent with step 500 but not labelled. "
               "[PLAN L5-6, L26-28, L33, L110]"))
    S.append(H2("Own GRPO run (seed 0)"))
    S.append(P("TRL 1.16 with colocated vLLM 0.32 (relex/train_grpo.py), Qwen2.5-Math-1.5B, 500 steps, mapped from the "
               "paper's verl recipe: 256 prompts x 8 rollouts per step with one optimizer step per rollout batch, AdamW "
               "lr 1e-6 constant, weight decay 0.01, gradient clip 1.0, k3 KL loss with β 0.001, clip ε 0.2, "
               "token-mean aggregation. Master weights are fp32 (bf16 autocast); fp32 deltas Δ_t = θ_t - θ_0 are "
               "stored every step to 125, then every 25, so the same run can be analysed in fp32 and after rounding "
               "θ_t to bf16 (what a saved bf16 checkpoint holds). A length-sort and trim speed patch is "
               "mathematically equivalent to TRL and was gradient-checked (Appendix). A hang in the authors' grader "
               "at step 61 led to a 120 s grading deadline per step and a restart from checkpoint-60. "
               "[PLAN L36-37, L69, L81, L85; train_grpo.py docstring]"))
    S.append(H2("Evaluation"))
    S.append(P("MATH = TianHongZXY/MATH test, 5000 problems (not MATH-500). Qwen2.5-Math-1.5B: greedy, 4096 tokens. "
               "Qwen3-4B: the authors' Qwen3 decoding, sampled (T 0.7, top_p 0.8), 16K tokens, one sample per problem. "
               "OOD (AIME 2025, AIME 2026, HMMT Feb 2025, AMC 2023, OlympiadBench): avg@8, T 0.7, top_p 0.8 (16K for "
               "Qwen3). Scoring uses the authors' grader.py/utils.py, vendored unchanged (MIT). Paired comparisons "
               "resample the 5000 problems with replacement (relex/paired.py, 4000 resamples, percentile 95% "
               "interval). For the own run's OOD results, the paired difference of the macro mean over the five "
               "benchmarks (each avg@8) uses a problem-clustered bootstrap stratified by benchmark: the 8 samples of "
               "a problem stay together and each benchmark's problems are resampled within it (relex/ood_paired.py). "
               "Intervals are exploratory and unadjusted except P5 and P10; no "
               "intervals are computed for the released-model and Qwen3-4B OOD tables. "
               "[PLAN L4, L30, L104, L110, L116, L121]"))
    S.append(H2("Independent implementation check"))
    S.append(P("Every predictor is written as a weighted sum of observed deltas, computed from one Gram matrix per "
               "tensor (Appendix A). My RELEX c75 evaluated at target 1000 against the released "
               f"RELEX-Qwen2.5-Math-1.5B: median per-tensor cosine {c15['relex_c75_t1000']['cos_median']:.4f}, relative "
               f"L2 {100*c15['relex_c75_t1000']['rel_l2']:.2f}%, norm ratio {c15['relex_c75_t1000']['norm_ratio']:.5f}; at "
               f"target 500 the norm ratio is {c15['relex_c75_t500']['norm_ratio']:.2f}. My Qwen3-4B RELEX c75→500 against "
               f"the released RELEX-Qwen3-4B-Base: relative L2 {100*c4['q4_relex_c75_t500']['rel_l2']:.3f}% (x1 "
               "scaling confirmed for Qwen3). [PLAN L56, L106; weight_stats/released_q15_compare_released_relex.json, "
               "released_q4b_compare.json]"))

    # ---------------------------------------------------------------- 3 paper vs reproduction
    S.append(H1("3. Paper versus reproduction"))
    rows = [["Model / checkpoint", "Paper MATH", "Reproduced MATH", "Truncated %", "Note"],
            ["Qwen2.5-Math-1.5B base", "48.2", f2(R("base")), f2(trunc("q15_released", "base")), "greedy 4K"],
            ["RLVR step_500 (released)", "71.5", f2(R("rlvr_500")), f2(trunc("q15_released", "rlvr_500")), ""],
            ["RELEX c75 “step 500” = coordinate 1000", "71.6", f2(R("relex_c75_t1000")),
             f2(trunc("q15_released", "relex_c75_t1000")), "weights match released RELEX"],
            ["RELEX c75 at coordinate 500", "-", f2(R("relex_c75_t500")), f2(trunc("q15_released", "relex_c75_t500")),
             "horizon matched to RLVR-500"],
            ["Qwen3-4B-Base base", "64.0", f2(Q("Qwen__Qwen3-4B-Base")), f2(trunc("q4b_released", "Qwen__Qwen3-4B-Base", "math16k.json")),
             "sampled 16K"],
            ["RLVR-Qwen3-4B-Base main branch", "85.5", f2(Q("relex-rlvr__RLVR-Qwen3-4B-Base")),
             f2(trunc("q4b_released", "relex-rlvr__RLVR-Qwen3-4B-Base", "math16k.json")), "late ckpt, ~step 500"],
            ["RELEX-Qwen3-4B-Base (released)", "85.6", f2(Q("relex-rlvr__RELEX-Qwen3-4B-Base")),
             f2(trunc("q4b_released", "relex-rlvr__RELEX-Qwen3-4B-Base", "math16k.json")), "x1 coordinate"]]
    S.append(table(rows, [2.35, 0.75, 0.95, 0.8, 2.05]))
    S.append(CAP("Table 1a. MATH-5000. Paper values from PLAN L57, L104, L111; reproduced values from "
                 "results/evals/q15_released/*/math.json and q4b_released/*/math16k.json."))
    ood_models = [("q15_released", "base", "1.5B base"), ("q15_released", "rlvr_500", "1.5B RLVR-500"),
                  ("q15_released", "relex_c75_t1000", "1.5B RELEX c75@1000"),
                  ("q15_released", "relex_c75_t500", "1.5B RELEX c75@500")]
    rows = [["Benchmark (avg@8)"] + [m[2] for m in ood_models]]
    for k, lab in OOD_KEYS:
        rows.append([lab] + [f"{f2(ev(g, n, 'ood.json')[k])} ({f1(ev(g, n, 'ood.json')[k + ':truncated%'])})"
                             for g, n, _ in ood_models])
    S.append(table(rows, [1.45, 1.35, 1.35, 1.5, 1.45]))
    S.append(CAP("Table 1b. OOD accuracy (%) with truncation % in parentheses, Qwen2.5-Math-1.5B, 4K tokens. The paper's "
                 "OOD values are not transcribed in the project sources, so no paper column is shown; the Qwen3-4B OOD "
                 "rows are in Section 7. Source: results/evals/q15_released/*/ood.json."))
    abl = [("RELEX rank-1 linear", "71.6", "relex_c75_t1000", "relex_c75_t500", "reference", "reference"),
           ("raw per-weight LS", "70.7", "rawls_c75_t1000", "rawls_c75_t500", "+2.14 [+1.16, +3.18]", "-0.58 [-1.30, +0.12]"),
           ("rank-5", "70.6", "rank5_c75_t1000", "rank5_c75_t500", "+2.04 [+1.10, +2.96]", "-"),
           ("rank-10", "70.5", "rank10_c75_t1000", "rank10_c75_t500", "-", "-1.02 [-1.78, -0.28]"),
           ("two-point (0, 75)", "-", "twopoint0_c75_t1000", "twopoint0_c75_t500", "-", "-"),
           ("two-point (25, 75)", "-", "twopoint25_c75_t1000", "twopoint25_c75_t500", "-0.16 [-1.18, +0.84]",
            "-1.08 [-1.92, -0.20]")]
    rows = [["Method (T_obs = 75)", "Paper", "Repro @1000", "Repro @500", "RELEX minus method @1000",
             "RELEX minus method @500"]]
    for lab, pap, a, b, c1, c2 in abl:
        rows.append([lab, pap, f2(R(a)), f2(R(b)), c1, c2])
    rows.append(["polynomial-3 fit", "0.1", f2(R("poly3_c75_t1000")), f"{f2(R('poly3_c75_t400'))} (@400)", "-", "-"])
    rows.append(["ExPO (1+α)Δ_75, α=1 / α=3", "-", f"{f2(R('expo1_c75'))} / {f2(R('expo3_c75'))}", "(no target)", "-", "-"])
    S.append(table(rows, [1.55, 0.5, 0.85, 0.85, 1.6, 1.6]))
    S.append(CAP("Table 2. Ablations on the released 1.5B trajectory, MATH-5000. Paper values (RELEX / raw / rank-5 / "
                 "rank-10 / poly-3 = 71.6 / 70.7 / 70.6 / 70.5 / 0.1) are at the paper's “step 500”, i.e. coordinate "
                 "1000. Paired differences from PLAN L60; scores from results/evals/q15_released/; mapping PLAN L59. "
                 "Ordering at 1000 matches the paper; at the matched coordinate 500 it reverses."))
    lab_ = [100, 200, 500, 750, 1000]
    pap_ = ["67.8", "70.0", "71.6", "65.7", "68.4"]
    rows = [["Paper step label", *[str(x) for x in lab_]],
            ["Paper Table 3, T_obs = 75", *pap_],
            ["Reproduced at 2 x label (100..2000)", *[f2(R(f"relex_c75_t{2*x}")) for x in lab_]],
            ["Reproduced at 1 x label", *[f2(R(f"relex_c75_t{x}")) for x in lab_]],
            ["Real RLVR branch at the label", f2(R("rlvr_100")), f2(R("rlvr_200")), f2(R("rlvr_500")), "-", "-"]]
    S.append(table(rows, [2.25, 0.8, 0.8, 0.8, 0.8, 0.8]))
    S.append(CAP("Table 3. Horizon row (T_obs = 75). Every paper value is within 0.5 of the reproduction at twice the "
                 "label and the 1x row does not fit (e.g. label 750: paper 65.7, 2x 65.58, 1x 70.68). Paper values "
                 "and mapping from PLAN L58; scores from results/evals/q15_released/relex_c75_t*/math.json."))

    # ---------------------------------------------------------------- 4 coordinate
    S.append(H1("4. The 2x fit coordinate (Qwen2.5-Math-1.5B only)"))
    S.append(P("The authors' svd_extrapolation.py multiplies the target step by 2 for Qwen2.5-Math-1.5B only (comment: "
               "“one saved ckpt per 2 optimizer steps”), while the fit window is not scaled and the released trajectory "
               "has exactly 500 checkpoints. The paper's 1.5B “RELEX step 500” is therefore the rank-1 line evaluated "
               "at fit coordinate 1000, twice the released branch-label coordinate of the RLVR checkpoint it is "
               "compared with (branch step_500, the last of the 500). How branches map to optimizer steps is not "
               "documented, so this report does not claim the comparison is at twice the optimizer steps; it claims "
               "only what the coordinates are. My evaluation of branch step_500 (71.78) matches the paper's RLVR value "
               "(71.5), and the released RELEX weights match my line at 1000 (relative L2 0.18%) and not at 500 (norm "
               "ratio 0.50). The Qwen3 models use x1. [PLAN L23-25, L56, L61, L110]"))
    S.append(fig(F1, 6.6, "Figure 1. Released Qwen2.5-Math-1.5B, MATH-5000 versus fit coordinate. The paper's Table 3 "
                 "values (hollow squares) sit on the reproduced RELEX curve when placed at twice their step label. At "
                 "matched coordinates 100..500 RELEX trails the real branch by 0.1 / 1.4 / 1.4 / 1.7 / 2.0 points and "
                 "needs the 2x overshoot to reach RLVR-500; it drops at 1500 (65.58, 8.9% truncated). Sources: PLAN "
                 "L58; results/evals/q15_released/."))
    S.append(P("At the paper's coordinate the RELEX delta is 5.5x the norm of the real step-500 delta, with median "
               "per-tensor cosine 0.37 [PLAN L56], so “matches RLVR-500 in accuracy” is not “approximates the "
               "step-500 weights”. The paper's Appendix B trend reproduces on the index axis: extrapolation c75 "
               "cosine 0.78→0.41 and norm ratio 1.19→2.48 from index 100 to 500 (paper 0.72→0.35, 1.26→2.70); "
               "rank-1 reconstruction from a uniform every-25 fit, cosine 0.64→0.94 peaking at 400 (paper "
               "0.50→0.91, peak 400). Rank-1 reconstruction keeps the gain (step 500: 71.28 vs real 71.78; "
               "-0.50 [-1.24, +0.24]), so finding F1 holds. [PLAN L56, L59, L60]"))

    # ---------------------------------------------------------------- 5 geometry
    S.append(H1("5. Weight geometry: random-walk null, released, own fp32 and bf16"))
    fm = lambda t: [f3(t[0]), f3(t[1]), f3(t[2])] if not isinstance(t[0], str) else list(t)  # noqa: E731
    rows = [["Trajectory", "Window", "Median rank-1 EV", "Median R² of c₁ line", "Fraction R² > 0.98"],
            ["Random walk, no drift (d = 20000, 40 draws)", "75", *fm(g["null"])],
            ["Random walk, drift = 5% of step noise", "75", *fm(g["null5"])],
            ["Released 1.5B (bf16 checkpoints)", "1-75", *fm(g["rel75"])],
            ["Released Qwen3-4B (bf16 checkpoints)", "1-75", *fm(g["q4b75"])],
            ["Own run, fp32 deltas", "1-25", *fm(g["own32_25"])],
            ["Own run, fp32 deltas", "1-75", *fm(g["own32_75"])],
            ["Own run, fp32 deltas", "1-125", *fm(g["own32_125"])],
            ["Own run, rounded to bf16", "1-75", *fm(g["own16_75"])],
            ["Own run, rounded to bf16", "1-125", *fm(g["own16_125"])],
            ["Agent run 1, fp32 deltas", "1-25", *fm(g["ag32"])],
            ["Agent run 1, rounded to bf16", "1-25", *fm(g["ag16"])]]
    S.append(table(rows, [2.65, 0.65, 1.15, 1.25, 1.15]))
    S.append(CAP("Table 4. Per-tensor rank-1 statistics. Null rows: tests/null_random_walk.py as logged in PLAN L61. "
                 "Other rows: results/weight_stats/released_q15_gram_stats.json, released_q4b_gram_stats.json, "
                 "own_q15_run1_gram_stats_s125.json, agent_run1_gram_stats.json (medians over tensors)."))
    S.append(P(f"Three things follow. (1) The released bf16 statistics (EV {G3['rel75'][0]}, R² {G3['rel75'][1]} over "
               "75 steps) are what a driftless random walk gives (0.811 / 0.958); they are compatible with structure "
               "but do not establish it. This does not refute the paper's separate reconstruction evidence (Section 4). "
               f"(2) In fp32 the own run is far more concentrated in one direction (EV {G3['own32_75'][0]} over 75 steps) "
               f"but its coefficient is visibly curved: median R² falls from {G3['own32_25'][1]} (25 steps) to "
               f"{G3['own32_75'][1]} (75) and {G3['own32_125'][1]} (125). Pre-registered P5 required R² > 0.95 and fails. "
               f"(3) Rounding the fp32 run to bf16 moves it onto the random-walk values ({G3['own16_75'][0]} / "
               f"{G3['own16_75'][1]} at 75): bf16 storage hides sub-ulp updates, lowering EV and raising the apparent "
               f"linearity. The agent run behaves the same way (fp32 {G3['ag32'][0]} / {G3['ag32'][1]}; bf16 "
               f"{G3['ag16'][0]} / {G3['ag16'][1]}), so A4 fails in fp32. [PLAN L61, L86, L90-93, L110]"))
    S.append(fig(F2, 6.6, "Figure 2. Median rank-1 explained variance (left) and median R² of the linear fit to the "
                 "rank-1 coefficient (right) as the fit window grows. Grey markers: random-walk null at T = 75. "
                 "Sources as Table 4."))

    # ---------------------------------------------------------------- 6 own run
    S.append(KeepTogether([H1("6. Own GRPO run (Qwen2.5-Math-1.5B, seed 0)"), fig(F3, 6.6, "Figure 3. Own run, MATH-5000. Real checkpoints at optimizer steps 25..500 versus RELEX "
                 "fitted on steps 1-75 and evaluated at T = 100..500 and 1000, from fp32 deltas and from the same "
                 "deltas rounded to bf16. Sources: results/evals/q15_own/own_rlvr_*/ and own_relex_c75_t*_{fp32,bf16}/"
                 "math.json; PLAN L88, L103, L107, L114, L116.")]))
    rows = [["Predictor (target 500 unless noted)", "MATH", "Minus real step 500 [95% CI]"]]
    fin = [("own_relex_c75_t500_fp32", "RELEX c75, fp32"), ("own_relex_c75_t500_bf16", "RELEX c75, bf16-rounded"),
           ("own_relex_c50_t500_fp32", "RELEX c50, fp32"), ("own_relex_c100_t500_fp32", "RELEX c100, fp32"),
           ("own_relex_c125_t500_fp32", "RELEX c125, fp32"), ("own_rawls_c75_t500_fp32", "raw per-weight LS c75, fp32"),
           ("own_rawls_c125_t500_fp32", "raw per-weight LS c125, fp32"),
           ("own_twopoint25_c75_t500_fp32", "two-point (25, 75), fp32"),
           ("own_relexpow_c75_t500_fp32", "RELEX-pow c75 (power-law coefficient), fp32"),
           ("own_relex_c75_t1000_fp32", "RELEX c75 → 1000, fp32"),
           ("own_relexnorm_c75_t500_fp32", "RELEX c75 direction at real step-500 per-tensor norm, fp32"),
           ("own_rlvr_75", "real step 75 (stop early)"), ("own_rlvr_475", "real step 475")]
    for n, lab in fin:
        rows.append([lab, f2(O(n)), fpci(n)])
    S.append(table(rows, [3.0, 0.8, 2.0]))
    S.append(CAP(f"Table 5. Final paired comparisons against real step 500 ({f2(O('own_rlvr_500'))}), n = 5000 problems. "
                 "Source: logs/final_paired.txt in the data directory (differences and intervals), "
                 "results/evals/q15_own/*/math.json (scores); PLAN L116, L119. Only the first row (P5) and the "
                 "norm-matched row (P10) are pre-registered."))
    S.append(P(f"From 75 fp32 steps RELEX lands {FP['own_relex_c75_t500_fp32'][0].lstrip('+')} points above real step 500 on MATH, "
               "so the extrapolation is useful on this run. It is not specific to rank-1: raw per-weight least squares and two-point extrapolation from the "
               "same window are at least as good, and the power-law coefficient (P8) changes little. Real step 475 is "
               "+0.34 [-0.34, +1.00] from step 500, a reminder of checkpoint-to-checkpoint noise of the same order as "
               "several of these gaps. At moderate horizons RELEX tracks real training closely (T = 200: 69.64 vs real "
               "69.78; T = 300: 70.68 vs 70.42) and runs ahead of it at 400 (72.18 vs 71.12). [PLAN L103, L114, L116]"))
    prec = [("RELEX", 50, 200, "own_relex_c50_t200"), ("raw LS", 50, 200, "own_rawls_c50_t200"),
            ("RELEX", 50, 500, "own_relex_c50_t500"), ("raw LS", 50, 500, "own_rawls_c50_t500"),
            ("RELEX", 75, 500, "own_relex_c75_t500"), ("raw LS", 75, 500, "own_rawls_c75_t500"),
            ("rank-5", 75, 500, "own_rank5_c75_t500"), ("two-point (25,75)", 75, 500, "own_twopoint25_c75_t500"),
            ("RELEX", 75, 1000, "own_relex_c75_t1000"), ("raw LS", 75, 1000, "own_rawls_c75_t1000"),
            ("rank-5", 75, 1000, "own_rank5_c75_t1000"), ("two-point (25,75)", 75, 1000, "own_twopoint25_c75_t1000"),
            ("RELEX", 100, 500, "own_relex_c100_t500"), ("RELEX", 100, 1000, "own_relex_c100_t1000"),
            ("RELEX", 125, 500, "own_relex_c125_t500"), ("RELEX", 125, 1000, "own_relex_c125_t1000"),
            ("raw LS", 125, 500, "own_rawls_c125_t500"), ("raw LS", 125, 1000, "own_rawls_c125_t1000")]
    cis = {"own_relex_c50_t500": "+3.04 [+2.04, +4.02]", "own_rawls_c50_t500": "+1.96 [+0.96, +2.98]"}
    rows = [["Method", "Window", "Target", "fp32", "bf16-rounded", "fp32 - bf16"]]
    for lab, c, t, n in prec:
        a, b = O(n + "_fp32"), O(n + "_bf16")
        rows.append([lab, f"1-{c}", str(t), f2(a), f2(b), cis.get(n, f"{a - b:+.2f}")])
    S.append(table(rows, [1.5, 0.65, 0.6, 0.75, 1.0, 1.6]))
    S.append(CAP("Table 6. Precision effect: the same run's deltas used in fp32 or rounded to bf16. Last column: exact "
                 "difference of the two scores; the two bracketed intervals are the paired CIs logged in PLAN L87. "
                 "Scores: results/evals/q15_own/*_{fp32,bf16}/math.json; PLAN L87, L88, L101."))
    S.append(P("Storage precision matters, but by method and horizon. At target 500 fp32 deltas help every method "
               "(RELEX +3.04 [+2.04, +4.02] from 50 steps, +1.70 from 75). At target 1000 the full-rank methods "
               "over-extrapolate in fp32 and their bf16 versions score higher (raw LS 72.70 vs 70.32), while RELEX is "
               "nearly unchanged (+0.36): rank-1's clearest advantage on this run is robustness at the far target in "
               "fp32 (RELEX 72.16 vs raw LS 70.32). The pre-registered P6 claim that fp32 would shrink the "
               "RELEX-minus-raw gap by at least 1 point fails (gap at 500: fp32 -0.18, bf16 -0.52). "
               "[PLAN L87, L88, L110; README finding 6]"))
    # ------------------------------------------------ 6b norm-matched control and OOD
    nm, rl = "own_relexnorm_c75_t500_fp32", "own_relex_c75_t500_fp32"
    S.append(H2("Norm-matched control (P10) and OOD"))
    S.append(P("<b>Is the MATH gain a step-size effect?</b> Pre-registered before the build (PLAN L117): rescale the "
               "RELEX c75→500 fp32 direction, tensor by tensor, to the norm of the real step-500 delta; P10 predicted "
               "the result would land within 1.0 of real 500 and below plain RELEX. It does not land within 1.0: "
               f"the norm-matched model scores {f2(O(nm))}, {fpci(nm)} versus real 500 and {fpci(nm, rl)} versus "
               f"plain RELEX, so <b>P10 fails</b>. Over all tensors the plain RELEX delta has {NORM_RATIO:.2f}x the norm "
               "of the real step-500 delta. On MATH the 75-step rank-1 direction is therefore less effective per unit "
               "norm than real training; plain RELEX reaches or exceeds real 500 by taking a much larger step, not "
               "because its direction is better. In the agentic run the same overshoot caused the collapse and "
               "norm-matching largely rescued it (Section 8); on MATH the overshoot is what produces the gain. "
               "[PLAN L117, L119, L120]"))
    own_ood = [("own_rlvr_100", "real 100"), ("own_rlvr_200", "real 200"), ("own_rlvr_300", "real 300"),
               ("own_rlvr_400", "real 400"), ("own_rlvr_475", "real 475"), ("own_rlvr_500", "real 500"),
               ("own_relex_c75_t300_fp32", "RELEX c75→300 fp32"),
               ("own_relex_c75_t500_fp32", "RELEX c75→500 fp32"), ("own_relex_c75_t500_bf16", "RELEX c75→500 bf16"),
               ("own_relex_c75_t1000_fp32", "RELEX c75→1000 fp32"), (nm, "norm-matched RELEX→500 fp32"),
               ("own_rawls_c75_t500_fp32", "raw LS c75→500 fp32"), ("own_rawls_c125_t500_fp32", "raw LS c125→500 fp32"),
               ("own_twopoint25_c75_t500_fp32", "two-point→500 fp32")]
    rows = [["Checkpoint"] + [lab for _, lab in OOD_KEYS] + ["Macro mean"]]
    for n, lab in own_ood:
        d = ev("q15_own", n, "ood.json")
        rows.append([lab] + [f2(d[k]) for k, _ in OOD_KEYS] + [f2(ood_mean(n))])
    S.append(table(rows, [1.8, 0.75, 0.75, 0.9, 0.8, 1.05, 0.85]))
    S.append(CAP("Table 7. Own run, OOD avg@8 accuracy (%), 4K tokens; macro mean = unweighted mean of the five "
                 "benchmarks. AIME and HMMT scores move in steps of 0.42 points and AMC 2023 has 40 problems, so "
                 "per-benchmark differences between rows are noisy; paired intervals for the macro mean are in "
                 "Table 7b. Source: results/evals/q15_own/*/ood.json; PLAN L118, L120, L121."))
    olab = {"own_relex_c75_t500_fp32": "RELEX c75→500, fp32", "own_relex_c75_t500_bf16": "RELEX c75→500, bf16-rounded",
            nm: "norm-matched RELEX c75→500, fp32", "own_rawls_c75_t500_fp32": "raw per-weight LS c75→500, fp32",
            "own_twopoint25_c75_t500_fp32": "two-point (25, 75)→500, fp32", "own_relex_c75_t300_fp32": "RELEX c75→300, fp32",
            "own_rlvr_475": "real step 475", "own_rlvr_400": "real step 400", "own_rlvr_300": "real step 300"}
    rows = [["Checkpoint", "MATH minus real 500, 95% CI", "OOD macro mean", "OOD minus real 500, 95% CI"]]
    for n in OP:
        rows.append([olab[n], fpci(n) if n in FP else "not computed", f2(ood_mean(n)), opci(n)])
    t7b = table(rows, [2.2, 1.8, 1.0, 1.8])
    S.append(KeepTogether([t7b, CAP(f"Table 7b. Paired differences against real step 500 (MATH {f2(O('own_rlvr_500'))}, OOD macro mean "
                 f"{f2(ood_mean('own_rlvr_500'))}). MATH: problem-paired bootstrap over 5000 problems "
                 "(logs/final_paired.txt). OOD: macro mean over five benchmarks, problem-clustered bootstrap stratified "
                 "by benchmark (relex/ood_paired.py; logs/ood_paired.txt). All intervals unadjusted. PLAN L116, L119, L121.")]))
    S.append(fig(F5, 6.6, "Figure 5. Own run: MATH-5000 (left) and OOD macro mean (right) along the RELEX c75 fp32 "
                 "line (T = 100..500, and 1000) against the real checkpoints that have each evaluation; the star is "
                 "the norm-matched control at T = 500. Sources: results/evals/q15_own/*/{math,ood}.json; PLAN L121."))
    S.append(P("<b>OOD.</b> On the five OOD benchmarks the MATH ordering does not carry over: RELEX c75→500 fp32 is "
               f"{opci(rl)} versus real 500 (interval excludes zero) while it is above real 500 on MATH, and every "
               "75-step extrapolation to 500 has a negative point estimate. But real step 400 is also "
               f"{opci('own_rlvr_400')} versus real 500, and real 300 is {opci('own_rlvr_300')}, so OOD gaps of about "
               "2 points are within the variation between nearby real checkpoints. Along the RELEX line the OOD mean "
               "is non-monotone and highest near T = 300 "
               f"({f2(ood_mean('own_relex_c75_t300_fp32'))}; {opci('own_relex_c75_t300_fp32')} versus real 500), "
               "while MATH keeps rising to T = 500. The norm-matched model, worse on MATH, is in between on OOD "
               f"({opci(nm)}). Conclusion: no robust OOD advantage or deficit for RELEX on this run; intervals are "
               "unadjusted and from one seed. [PLAN L118, L120, L121]"))

    # ---------------------------------------------------------------- 7 qwen3-4b
    S.append(H1("7. Qwen3-4B-Base (released trajectory, no coordinate scaling)"))
    q4 = [("Qwen__Qwen3-4B-Base", "base", "64.0", "-"),
          ("q4_rlvr_75", "real step 75 (released)", "-", "-4.50 [-5.44, -3.54]"),
          ("q4_relex_c75_t250", "RELEX c75 → 250 (own build)", "-", "-2.12 [-3.04, -1.22]"),
          ("relex-rlvr__RELEX-Qwen3-4B-Base", "RELEX c75 → 500 (released; = own build)", "85.6", "-0.30 [-1.14, +0.56]"),
          ("q4_rawls_c75_t500", "raw per-weight LS c75 → 500", "-", "-3.46 [-4.40, -2.54]"),
          ("q4_twopoint25_c75_t500", "two-point (25, 75) → 500", "-", "not computed"),
          ("relex-rlvr__RLVR-Qwen3-4B-Base", "RLVR main branch (late, ~500)", "85.5", "reference")]
    rows = [["Checkpoint", "Paper", "MATH (16K, sampled)", "Truncated %", "Minus RLVR main [95% CI]"]]
    for n, lab, pap, ci in q4:
        rows.append([lab, pap, f2(Q(n)), f2(trunc("q4b_released", n, "math16k.json")), ci])
    S.append(table(rows, [2.45, 0.5, 1.05, 0.8, 1.75]))
    S.append(CAP("Table 8. Qwen3-4B-Base, MATH-5000, one sample per problem. Paired intervals from PLAN L112; scores "
                 "from results/evals/q4b_released/*/math16k.json; paper values PLAN L104, L111."))
    S.append(P("At the matched coordinate RELEX from 75 steps recovers (4.50 - 0.30)/4.50 = 93% of the step-75 → "
               "main-branch gain and is statistically indistinguishable from RLVR main at this sample size; rank-1 "
               "beats raw least squares by 3.2 points, and two-point extrapolation collapses (68.30, 40.02% truncated). "
               "This is the paper's strongest case and it holds. Two qualifications: the comparator branch is not "
               "labelled with a step (its norm, 2.41 x ||Δ_75||, is consistent with step 500), and the RELEX delta is "
               "6.9 x ||Δ_75||, about 2.9x the real late-checkpoint delta, so again accuracy matches without the "
               "weights matching. [PLAN L106, L110-112; q4_twopoint25_c75_t500/math16k.json]"))
    q4o = [("Qwen__Qwen3-4B-Base", "base"), ("q4_rlvr_75", "real 75"), ("relex-rlvr__RELEX-Qwen3-4B-Base", "RELEX (released)"),
           ("relex-rlvr__RLVR-Qwen3-4B-Base", "RLVR main"), ("q4_rawls_c75_t500", "raw LS → 500")]
    rows = [["Benchmark (avg@8, 16K)"] + [lab for _, lab in q4o]]
    for k, lab in OOD_KEYS:
        rows.append([lab] + [f"{f2(ev('q4b_released', n, 'ood16k.json')[k])} ({f1(ev('q4b_released', n, 'ood16k.json')[k + ':truncated%'])})"
                             for n, _ in q4o])
    S.append(table(rows, [1.45, 1.1, 1.1, 1.3, 1.1, 1.2]))
    S.append(CAP("Table 9. Qwen3-4B OOD accuracy (%) with truncation % in parentheses. Source: "
                 "results/evals/q4b_released/*/ood16k.json."))
    S.append(P("<b>Truncation note.</b> At 16K tokens a large share of OOD responses never finish (RLVR main: 54.17-61.67% on "
               "AIME/HMMT; released RELEX 40.83-45.83%; raw LS 64.58-79.58%), and an unfinished response scores zero. The OOD "
               "numbers therefore mix accuracy with response length; RELEX is above RLVR main on HMMT and AMC and below "
               "it on both AIMEs, while also truncating less. On MATH, raw least squares truncates 15.42% of answers "
               "versus 5.50% for RELEX, which accounts for part of its 3.2-point deficit."))

    # ---------------------------------------------------------------- 8 agent
    S.append(H1("8. Agentic ERP test (new, not in the paper)"))
    S.append(P("<b>Setup.</b> Qwen3.5-2B with thinking on, trained with TRL 1.16 environment_factory on six SQLite tools "
               "over a private in-memory database copy per rollout; reward is the erpagent rubric (0/1). Tasks are "
               "parameterised versions of 37 ERP templates: train 2400 on database seeds 101-104, test 300 on an unseen "
               "database (seed 301), plus the original 37. Recipe as the math run (AdamW 1e-6 constant, wd 0.01, clip "
               "1.0, β 0.001 k3, ε 0.2, token-mean), 32 tasks x 8 rollouts per step, at most 10 tool turns, 4096 "
               "completion tokens, 100 steps, fp32 deltas every step to 30 then every 10. Evaluation uses the opencode "
               "harness through a vLLM server: <i>full</i> protocol = test 300 x 2 trials + orig37 x 3 trials; "
               "<i>quick</i> probe = test 300 x 1 trial. Seed 2 (run 2) repeats training with a different seed. "
               "[PLAN L63-74, L82]"))
    arow = lambda lab, tag, tag37, prot: [lab, prot, f1(ag(tag)["pass"]), f1(ag(tag37)["pass"]) if tag37 else "-",  # noqa: E731
                                         f"{ag(tag)['mcp_calls_mean']:.2f}", str(ag(tag)["completion_tokens_mean"])]
    rows = [["Checkpoint (seed 1 unless noted)", "Protocol", "Test %", "Orig37 %", "Tool calls / task", "Completion tokens"],
            arow("base Qwen3.5-2B", "qwen35_2b_base", "qwen35_2b_base_orig37", "full"),
            arow("real step 25", "agent_rlvr_25", "agent_rlvr_25_orig37", "full"),
            arow("real step 30", "agent_rlvr_30", "agent_rlvr_30_orig37", "full"),
            arow("real step 40", "agent_rlvr_40", "agent_rlvr_40_orig37", "full"),
            arow("real step 50", "agent_rlvr_50_quick", None, "quick"),
            arow("real step 100", "agent_rlvr_100", "agent_rlvr_100_orig37", "full"),
            arow("RELEX c25 → 30, fp32", "agent_relex_c25_t30_fp32", "agent_relex_c25_t30_fp32_orig37", "full"),
            arow("RELEX c25 → 40, fp32", "agent_relex_c25_t40_fp32", "agent_relex_c25_t40_fp32_orig37", "full"),
            arow("RELEX c25 → 50, fp32", "agent_relex_c25_t50_fp32_quick", None, "quick"),
            arow("RELEX c25 → 100, fp32", "agent_relex_c25_t100_fp32", "agent_relex_c25_t100_fp32_orig37", "full"),
            arow("RELEX c25 → 100, bf16-rounded", "agent_relex_c25_t100_bf16", "agent_relex_c25_t100_bf16_orig37", "full"),
            arow("two-point (10, 25) → 40", "agent_twopoint_10_25_t40_quick", None, "quick"),
            arow("two-point (10, 25) → 100", "agent_twopoint_10_25_t100", "agent_twopoint_10_25_t100_orig37", "full"),
            arow("RELEX-pow c25 → 50", "agent_relexpow_c25_t50_fp32_quick", None, "quick"),
            arow("RELEX-pow c25 → 100", "agent_relexpow_c25_t100_fp32_quick", None, "quick"),
            arow("RELEX steps {1-30, 40, 50} → 70", "agent_relex_c50_t70_fp32_quick", None, "quick"),
            arow("RELEX steps {1-30, 40, 50} → 100", "agent_relex_c50_t100_fp32_quick", None, "quick"),
            arow("norm-matched RELEX c25 → 100 (oracle norm)", "agent_relexnorm_c25_t100", "agent_relexnorm_c25_t100_orig37", "full"),
            arow("seed 2: real step 25", "agent2_rlvr_25_quick", None, "quick"),
            arow("seed 2: real step 30", "agent2_rlvr_30_quick", None, "quick"),
            arow("seed 2: real step 100", "agent2_rlvr_100", "agent2_rlvr_100_orig37", "full"),
            arow("seed 2: RELEX c25 → 30", "agent2_relex_c25_t30_fp32_quick", None, "quick"),
            arow("seed 2: RELEX c25 → 100", "agent2_relex_c25_t100_fp32", "agent2_relex_c25_t100_fp32_orig37", "full"),
            arow("seed 2: two-point (10, 25) → 100", "agent2_twopoint_10_25_t100", "agent2_twopoint_10_25_t100_orig37", "full")]
    S.append(table(rows, [2.6, 0.65, 0.6, 0.7, 0.95, 1.05]))
    S.append(CAP("Table 10. Agentic ERP results (opencode harness). Tool calls and tokens are per test task. Source: "
                 "results/agent_evals/opencode__*.summary.json; PLAN L89-90, L99-101, L106, L113-115."))
    S.append(fig(F4, 6.6, "Figure 4. Seed-1 agent run, test pass rate versus target step. Filled markers: full protocol "
                 "(300 tasks x 2); hollow: quick probe (300 x 1). Lines join the real checkpoints and the linear RELEX "
                 "predictions (full where available). At T = 100 the power-law and two-point markers are shifted "
                 "1.8 steps left and right so they do not hide the linear-RELEX marker. Source as Table 10."))
    S.append(P("<b>Results.</b> GRPO helps a lot (base 59.5 → real step 100 94.5 on test, +35 points; A1 holds). "
               "Extrapolation does not: linear RELEX from steps 1-25 roughly ties training at 30 (91.7 vs 92.8 test; "
               "92.8 vs 88.3 orig37), falls behind at 40 under the full protocol (78.3 vs 92.7), and collapses "
               "at 100 (11.2; 4.5 on seed 2), where the agent stops calling tools (0.01 calls per task) and runs to "
               "the token limit. A2 and A3 fail. In the quick probes, neither a power-law coefficient (P7) nor a "
               "longer window prevents the collapse. Two-point extrapolation to 100 differs widely between seeds "
               "(10.8 vs 69.2), so single-seed agent comparisons of predictors are fragile. [PLAN L90-92, L99-101, "
               "L113-115]"))
    S.append(P("<b>Norm-matched control.</b> Keeping the RELEX c25 direction but rescaling each tensor's delta to the "
               "norm of the real step-100 delta gives 88.2 test / 81.1 orig37, against 11.2 / 18.9 for plain RELEX and "
               f"94.5 / 93.7 for real step 100; on test that is {anorm_frac:.0f}% of the base→step-100 gain "
               f"(({f1(ag('agent_relexnorm_c25_t100')['pass'])} - {f1(abase)}) / ({f1(a100)} - {f1(abase)})). "
               "The collapse is therefore mostly magnitude overshoot: real deltas grow "
               "only 1.5x from step 25 to 100 (median per tensor) while a line predicts 4x, and the RELEX c25→100 delta "
               "has norm ratio 2.44 and cosine 0.72 to the real one. The control uses the real step-100 norm, which "
               "is not available in practice; it diagnoses the failure but is not a usable predictor, and neither "
               "fitted magnitude model (linear, power law) recovered that norm from steps 1-25. [PLAN L90-94, L99, L113]"))

    # ---------------------------------------------------------------- 9 scorecard
    S.append(H1("9. Pre-registered predictions"))
    sc = [["ID", "Prediction (short)", "Verdict", "Number", "Source"],
          ["P1", "My RELEX c75@1000 matches released weights (cos > 0.99 for > 95% of tensors, rel L2 < 5%); @500 worse",
           "holds", "cos 1.0000, rel L2 0.18%; @500 norm ratio 0.50", "L56"],
          ["P2", "MATH within 1.5 of paper: base 48.2, RLVR-500 71.5, RELEX 71.6", "holds", "46.9 / 71.78 / 71.74", "L57"],
          ["P3", "RELEX c75 at index 500 lower than at 1000 but ≥ 68", "holds", "69.74 (2.0 below RLVR-500)", "L57"],
          ["P4", "At the far target rank-1 beats raw LS and rank-5 by ≥ 0.5; poly-3 < 20 by 400", "holds (at 1000)",
           "+2.14 / +2.04; poly-3 0.06 @400. Reverses at 500", "L59-60"],
          ["P5", "Own fp32: median R² (1-125) > 0.95, and RELEX c75→500 within 2 of real 500", "fails (R² part); score part holds",
           f"R² {G3['own32_75'][1]} @75, {G3['own32_125'][1]} @125; {fpci('own_relex_c75_t500_fp32')}", "L110, L116"],
          ["P6", "bf16 lowers EV, and fp32 shrinks the RELEX-minus-raw gap by ≥ 1 point", "fails (EV part holds)",
           f"EV {G3['own32_75'][0]} vs {G3['own16_75'][0]}; gap -0.18 vs -0.52", "L86, L88, L110"],
          ["P7", "Agent RELEX-pow c25→100 ≥ 90 on test", "fails (quick probe; full pending)", "10.7", "L99, L110"],
          ["P8", "Math RELEX-pow c75→500 within 1 of linear RELEX", "holds (one run)", "71.58 vs 72.48", "L99, L110"],
          ["P9", "Agent linear RELEX degrades with horizon: ≥ 85 at 30, collapse by 100", "holds",
           "93.7 / 81.0 / 43.3 / 11.2 at 30/40/50/100", "L99"],
          ["P10", "Own math: RELEX c75→500 direction at real step-500 per-tensor norm lands within 1.0 of real 500 "
           "and below plain RELEX", "fails (below plain RELEX, but not within 1.0)",
           f"{f2(O('own_relexnorm_c75_t500_fp32'))}; {fpci('own_relexnorm_c75_t500_fp32')} vs real 500", "L117, L119"],
          ["A1", "Real agent step 100 beats base on test by ≥ 5", "holds", "94.5 vs 59.5 (+35)", "L90-92"],
          ["A2", "RELEX c25→100 gets ≥ half the gain and is within 3 of step 100", "fails", "11.2 vs 94.5", "L90-92"],
          ["A3", "RELEX c25→100 beats real step 25", "fails", "11.2 vs 90.8", "L90-92"],
          ["A4", "Agent rank-1 statistics (1-25) match the random-walk null", "fails in fp32 (holds in bf16)",
           f"fp32 {G3['ag32'][0]} / {G3['ag32'][1]}; bf16 {G3['ag16'][0]} / {G3['ag16'][1]}", "L92-93, L110"]]
    S.append(table(sc, [0.35, 2.75, 1.25, 1.75, 0.75]))
    S.append(CAP("Table 11. Pre-registered predictions P1-P6 (2026-11-03 07:40 EST), A1-A4 (before any R3 run) and "
                 "P7-P9 (before their evaluations), scored as in the Codex Council review (PLAN L110); P10 registered "
                 "2026-11-04 09:11 EST before its build and evaluation (PLAN L117). Source column: relex/PLAN.md line "
                 "numbers."))

    # ---------------------------------------------------------------- 10 limitations
    S.append(H1("10. Limitations and open questions"))
    S.append(bullets([
        "<b>One own math seed.</b> All own-run conclusions (Sections 5-6) rest on seed 0; a second seed (q15_run2, "
        "--seed 1) is running. Real step 475 vs 500 already differs by 0.34 points.",
        "<b>Exploratory, unadjusted intervals.</b> Dozens of paired comparisons were run; only the P5 and P10 "
        f"comparisons were pre-registered. Single intervals that barely exclude zero (e.g. "
        f"{fpci('own_relex_c75_t500_fp32')} on MATH, {opci('own_relex_c75_t500_fp32')} on OOD) should not be read as "
        "firm. Problem-clustered resampling is done for the own run's OOD evaluations (relex/ood_paired.py) but not "
        "for the released-model or Qwen3-4B OOD tables (PLAN L110, L121).",
        "<b>Branch-to-step accounting.</b> The 2x finding is about fit coordinates. Whether a released 1.5B branch is one "
        "or two optimizer steps is undocumented; the Qwen3-4B main branch is unlabelled and only consistent with "
        "step 500 by its norm.",
        "<b>fp32 vs bf16 storage on one run only.</b> The precision effect was measured by rounding one fp32 trajectory; "
        "it is method- and horizon-dependent and may differ for other learning rates and models.",
        "<b>Agentic test.</b> One synthetic ERP database family and task generator; seed-to-seed variance of some "
        "predictors is large (two-point 10.8 vs 69.2); several horizon points are 1-trial quick probes; P7 under the "
        "full protocol is pending; the norm-matched control uses oracle information.",
        f"<b>OOD sets are small</b> ({OOD_N} problems over five benchmarks; AMC 2023 has 40) and, for Qwen3-4B at 16K, "
        "dominated by truncation. Own-run OOD differences of about 2 points are of the same size as the gaps between "
        "nearby real checkpoints (real 400 vs 500). Intervals are reported only for the own run, and the paper's OOD "
        "values are not in the project sources, so OOD is not scored against the paper.",
        "<b>Unchecked numerics.</b> Rank-5/10 predictions from the Gram eigendecomposition have not yet been compared "
        "against an fp64 SVD (PLAN L61).",
        "<b>Recipe mapping.</b> The own run uses TRL, not the authors' verl, with a 120 s grading deadline (ungraded "
        "answers score 0) and resumes across GPU leases.",
    ]))

    # ---------------------------------------------------------------- appendix
    S.append(H1("Appendix A. Gram formulation"))
    S.append(P("For one weight tensor, stack the stored deltas Δ_s = θ_s - θ_0 as rows of M and form G = M Mᵀ (float64; "
               "size = number of stored steps). Every predictor in this report outputs a predicted Δ_T = Σ_s w_s Δ_s, so it is "
               "fully described by a weight vector w. With G_obs = U diag(S²) Uᵀ over the fit window, the rank-1 "
               "coefficients are c_t = U_t1 S_1 (= ⟨Δ_t, v_1⟩, v_1 = Mᵀ u_1 / S_1); RELEX fits c(t) linearly (or as a "
               "power law, or poly-3) and sets w = ĉ(T) u_1 / S_1 on the window. Raw per-weight least squares uses "
               "w_s = 1/n + (t_s - t̄)(T - t̄)/Σ(t - t̄)²; two-point (t0, t1) puts r = (T - t0)/(t1 - t0) on Δ_t1 and "
               "1 - r on Δ_t0; ExPO is (1 + α) Δ_t1. Explained variance is S_1²/ΣS², R² is that of the linear fit to "
               "c_t, and cosine and norm ratio against any stored step s follow from G alone: cos = wᵀG[:,s] / "
               "sqrt(wᵀGw · G[s,s]). Checkpoints are then built in one streaming pass. Bf16-rounded variants replace "
               "Δ_s with bf16(θ_0 + Δ_s) - θ_0. [README Implementation; relex.py]"))
    S.append(H1("Appendix B. Speed-patch gradient check"))
    S.append(P("The own run sorts rollouts by length and trims padding before the loss. On Qwen2.5-Math-1.5B with 24 "
               "synthetic rollouts (tests/test_grad_equiv.py): in fp32 the patched and original gradients agree to "
               "cosine 1.0000000, relative L2 1.2e-4 (equivalent); max |Δ log p| versus TRL in fp32 is 4e-5. Under bf16 "
               "autocast against the fp32 reference, plain TRL has relative L2 10.4% (cosine 0.99467), reordered plain "
               "TRL 10.4%, sort+trim 7.5% (cosine 0.99730): the patch is mathematically equivalent and numerically "
               "closer to fp32 than plain TRL, whose full-width padding carries more bf16 error. [PLAN L56, L81]"))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.7 * inch, title="RELEX reproduction: findings",
                            author="relex reproduction", subject="arXiv 2605.21468")
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
