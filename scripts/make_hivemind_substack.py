#!/usr/bin/env python3
"""Build the Substack version of the hivemind write-up as one self-contained HTML file.

Reads docs/substack_hivemind.md, renders minimalist reader aids from the committed
artifacts (so the figures cannot drift from the paper's numbers), rasterises them to
docs/substack_assets/hivemind_*.png, and embeds every image as a base64 data URI.
Open the result in a browser, select all, copy, paste into the Substack editor.

Figure markers in the markdown are HTML comments of the form <!-- fig:name -->.

    python scripts/make_hivemind_substack.py
    -> docs/substack_hivemind.html
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import markdown

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "artifacts" / "nla" / "atlas"
MD = ROOT / "docs" / "substack_hivemind.md"
OUT = ROOT / "docs" / "substack_hivemind.html"
ASSETS = ROOT / "docs" / "substack_assets"
ASSETS.mkdir(exist_ok=True)

# sunstonenorth.com theme tokens (OKLCH in the site's stylesheet, converted to hex)
INK = "#43332c"        # neutral / base-content
MUTED = "#8a7a6e"      # secondary
GRID = "#e9e0d2"       # base-300
TERRA = "#b96a45"      # primary, terracotta
TERRA_LT = "#e6c4ae"   # terracotta tint for secondary bars
OCHRE = "#d89a5f"      # accent
RED = "#a4472f"        # error, for floors
SAND = "#d9cdbb"       # neutral bars, between base-300 and secondary
PAPER = "#fbf8f3"      # base-100, ivory

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})

load = lambda p: json.loads(Path(p).read_text())
iso = load(ATLAS / "hivemind_posttraining_isolation.json")
dec = load(ATLAS / "hivemind_template_decomp.json")
sup = load(ATLAS / "hivemind_suppression.json")
atlas = load(ATLAS / "openweight_transport_atlas.json")
ladder = load(ROOT / "artifacts" / "nla" / "coder_matrix1024" / "scaling_curve.json")

THEIRS = (0.71, 0.82)   # Jiang et al., inter-model band
THEIR_FLOOR = (0.10, 0.20)


def title(ax, text, sub=None):
    """Left-aligned title in figure coordinates so it lines up with the axes edge, subtitle beneath it."""
    ax.set_title(text, loc="left", fontsize=12.5, color=INK, pad=26 if sub else 12, fontweight="medium")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def note(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8.2, color=MUTED, ha="left", va="top")


# ------------------------------------------------------------------ figures

def fig_suspects(out):
    """Three candidate explanations and the verdict on each. Typographic, no data."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02)
    ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.axis("off")
    cols = [
        ("Born alike", "shared internal\ngeometry from\npretraining", "True. Does not\nproduce the writing.", SAND),
        ("Raised alike", "alignment and\ninstruction tuning", "Small.\nAbout a fifth of it.", TERRA_LT),
        ("Dressed alike", "the chat template:\nsystem, user,\nassistant turns", "Most of it.\nAnd reversible.", TERRA),
    ]
    for i, (h, what, verdict, c) in enumerate(cols):
        x = i + 0.08
        ax.add_patch(FancyBboxPatch((x, 0.06), 0.84, 0.88, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    fc=PAPER, ec=GRID, lw=1.2))
        ax.add_patch(FancyBboxPatch((x + 0.06, 0.80), 0.72, 0.045, boxstyle="round,pad=0.0,rounding_size=0.02",
                                    fc=c, ec="none"))
        ax.text(x + 0.42, 0.68, h, ha="center", va="center", fontsize=13, color=INK, fontweight="medium")
        ax.text(x + 0.42, 0.47, what, ha="center", va="center", fontsize=9.2, color=MUTED, linespacing=1.35)
        ax.text(x + 0.42, 0.20, verdict, ha="center", va="center", fontsize=9.6, color=INK, linespacing=1.35)
    fig.text(0.035, 0.97, "Three places the sameness could come from", fontsize=12.5, color=INK,
             va="top", fontweight="medium")
    fig.savefig(out / "hivemind_suspects.png"); plt.close(fig)


def fig_scale(out):
    """A number line so the reader can place every similarity figure in the piece."""
    b = iso["base"]; r = iso["instruct_raw"]; c = iso["instruct_chat"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 1); ax.set_ylim(-1.1, 2.1)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
    ax.axhline(0, color=INK, lw=1.2, zorder=1)
    ax.axvspan(*THEIR_FLOOR, ymin=0.22, ymax=0.33, color=GRID, zorder=0)
    ax.axvspan(*THEIRS, ymin=0.22, ymax=0.33, color=TERRA_LT, zorder=0)
    ax.text((THEIR_FLOOR[0] + THEIR_FLOOR[1]) / 2, -0.55, "their floor:\nunrelated answers", ha="center",
            va="top", fontsize=8.4, color=MUTED, linespacing=1.3)
    ax.text((THEIRS[0] + THEIRS[1]) / 2, -0.55, "their models,\nbetween labs", ha="center", va="top",
            fontsize=8.4, color=TERRA, linespacing=1.3)
    pts = [(b["floor"], "our floor", MUTED, 0.5), (b["intra_mean"], "base models", INK, 0.5),
           (r["intra_mean"], "tuned weights,\nbare prompt", INK, 1.05), (c["intra_mean"], "same weights,\nchat template", TERRA, 0.5)]
    for x, lab, col, dy in pts:
        ax.plot([x], [0], "o", ms=8, color=col, zorder=3)
        ax.annotate(f"{lab}\n{x:.2f}", (x, 0), (x, dy), ha="center", va="bottom", fontsize=8.6, color=col,
                    linespacing=1.3, arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
    ax.text(0, 2.05, "How to read the numbers: similarity of two answers to the same question, 0 to 1",
            fontsize=11, color=INK, va="bottom", fontweight="medium")
    fig.savefig(out / "hivemind_scale.png"); plt.close(fig)


def fig_steps(out):
    """Base -> tuned weights -> chat template, six matched pairs, with their band and the floor."""
    b = iso["base"]; r = iso["instruct_raw"]; c = iso["instruct_chat"]
    labels = ["base model,\nbare prompt", "tuned weights,\nbare prompt", "tuned weights,\nchat template"]
    vals = [b["intra_mean"], r["intra_mean"], c["intra_mean"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(*THEIRS, color=TERRA_LT, alpha=0.45, lw=0, zorder=0)
    ax.text(2.42, THEIRS[0] + 0.01, "band reported\nfor frontier\nmodels", fontsize=8.4, color=TERRA,
            va="bottom", ha="left", linespacing=1.3)
    bars = ax.bar(labels, vals, width=0.56, color=[SAND, TERRA_LT, TERRA], edgecolor="none", zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    floor = c["floor"]
    ax.axhline(floor, color=RED, lw=1, ls="--", zorder=1)
    ax.text(2.42, floor + 0.012, f"floor {floor:.2f}", color=RED, fontsize=8.4, va="bottom")
    # steps, drawn well clear of the bar-top labels
    d1 = r["intra_mean"] - b["intra_mean"]; d2 = c["intra_mean"] - r["intra_mean"]; dt = c["intra_mean"] - b["intra_mean"]
    y1 = r["intra_mean"] + 0.10; y2 = c["intra_mean"] + 0.10
    ax.annotate("", (0.72, y1), (0.0, y1), arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.text(0.36, y1 + 0.015, f"swap in the tuned weights  +{d1:.2f}", ha="center", fontsize=9, color=INK)
    ax.annotate("", (1.72, y2), (1.0, y2), arrowprops=dict(arrowstyle="->", color=TERRA, lw=0.9))
    ax.text(1.36, y2 + 0.015, f"add the chat template  +{d2:.2f} more\n(+{dt:.2f} from base)", ha="center", fontsize=9,
            color=TERRA, fontweight="medium", linespacing=1.25)
    ax.set_ylim(0, 0.98); ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("within-model similarity", fontsize=9.5, color=MUTED)
    ax.tick_params(axis="x", length=0, labelsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ratio = (c["intra_mean"] - b["intra_mean"]) / d1
    title(ax, f"Same weights, one change: the chat template does {ratio:.1f}x the work of the tuning",
          "six matched base/instruct pairs from four labs, identical prompts, decoding and scorer")
    fig.savefig(out / "hivemind_steps.png"); plt.close(fig)


def fig_decomp(out):
    """What inside the template carries the effect. Persona-alone is its own arm, not a term of the sum;
    the three chained bars add to the full chat-minus-raw effect."""
    d = dec["decomposition"]
    chain = [("turn structure\n(generic markers)", d["role_structure_alone"], TERRA),
             ("persona inside\nthe structure", d["persona_added_within_structure"], TERRA_LT),
             ("the model's own\ntuned tokens", d["native_token_premium"], TERRA_LT)]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    # the persona-alone arm, on its own
    pa = d["persona_alone"]
    ax.bar(0, pa, width=0.56, color=SAND, edgecolor="none", zorder=2)
    ax.text(0, 0.008, f"{pa:+.3f}", ha="center", fontsize=10, color=INK)
    ax.axvline(0.85, color=GRID, lw=1)
    ax.text(0, -0.075, "on its own,\nno turn structure", ha="center", fontsize=8, color=MUTED, linespacing=1.25)
    run = 0.0
    for k, (lab, v, col) in enumerate(chain):
        i = k + 1.5
        ax.bar(i, v, bottom=run, width=0.56, color=col, edgecolor="none", zorder=2)
        ax.text(i, run + v + 0.008, f"{v:+.3f}", ha="center", fontsize=10, color=INK)
        run += v
        ax.plot([i + 0.28, i + 0.72], [run, run], color=MUTED, lw=0.8, ls=":")
    ax.bar(len(chain) + 1.5, run, width=0.56, color=INK, edgecolor="none", zorder=2)
    ax.text(len(chain) + 1.5, run + 0.008, f"{run:+.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks([0] + [k + 1.5 for k in range(len(chain) + 1)])
    ax.set_xticklabels(["persona text\nalone"] + [s[0] for s in chain] + ["full template\neffect"], fontsize=9.2)
    ax.tick_params(axis="x", length=0, pad=22)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.09, 0.34)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_ylabel("gain in within-model similarity", fontsize=9.5, color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    share = d["share_reachable_without_native_tokens"]
    title(ax, "Inside the template: the turn structure carries it, the persona line does nothing",
          f"generic markers no model was trained on reproduce {share:.0%} of the full effect")
    fig.savefig(out / "hivemind_decomp.png"); plt.close(fig)


def fig_heatmap(out):
    """12x12 transport matrix: how well a linear map moves one model's states into another's."""
    tags = [t for t in atlas["models"] if t != "qwen25_7b_f32"]
    short = {"qwen25_05b": "Qwen2.5-0.5B", "qwen3_06b": "Qwen3-0.6B", "gemma2_2b": "gemma-2-2b",
             "llama32_1b": "Llama-3.2-1B", "llama32_3b": "Llama-3.2-3B", "tinyllama": "TinyLlama-1.1B",
             "olmo2_1b": "OLMo-2-1B", "smollm2_360m": "SmolLM2-360M", "pythia_410m": "pythia-410m",
             "qwen25_7b": "Qwen2.5-7B", "gptoss_20b": "gpt-oss-20b"}
    labs = {t: atlas["models"][t]["lab"] for t in tags}
    M = [[atlas["matrix"][a][b] for b in tags] for a in tags]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#f4eee4", TERRA_LT, TERRA, "#7e3f22"]),
                   vmin=0.7, vmax=1.0)
    for i in range(len(tags)):
        for j in range(len(tags)):
            v = M[i][j]
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 0.995 else "1.0", ha="center", va="center", fontsize=7.6,
                    color=PAPER if v > 0.9 else INK)
    names = [f"{short[t]}  ({labs[t]})" for t in tags]
    ax.set_xticks(range(len(tags))); ax.set_yticks(range(len(tags)))
    ax.set_xticklabels([short[t] for t in tags], rotation=45, ha="right", fontsize=8.2)
    ax.set_yticklabels(names, fontsize=8.2)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    s = atlas["summary"]
    title(ax, "One model's internal states, read through another's",
          f"share of 1,000 held-out captions recovered. Across labs {s['mean_cross_lab']:.3f}, same lab {s['mean_within_lab']:.3f}, chance {s['mean_shuffled_floor']:.5f}")
    ax.set_xlabel("mapped into", fontsize=9, color=MUTED); ax.set_ylabel("from", fontsize=9, color=MUTED)
    fig.savefig(out / "hivemind_transport.png"); plt.close(fig)


def fig_ladder(out):
    """Format effect against parameter count, one family, HumanEval."""
    rows = sorted(ladder["curve"].values(), key=lambda r: r["params_b"])
    xs = [r["params_b"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(xs, [r["format_gain"] for r in rows], "o-", color=TERRA, lw=1.8, ms=6, label="format (chat template)")
    ax.plot(xs, [r["tuning_gain"] for r in rows], "s-", color=SAND, lw=1.4, ms=5, label="tuning (weights alone)")
    for r in rows:
        ax.text(r["params_b"], r["format_gain"] + 0.012, f"{r['format_gain']:.2f}", ha="center", fontsize=8.4, color=TERRA)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}B" for x in xs])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_ylim(0, 0.27)
    ax.set_ylabel("gain in within-model similarity", fontsize=9.5, color=MUTED)
    ax.set_xlabel("parameters, Qwen2.5-Coder, one lab and one recipe", fontsize=9.5, color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    t = ladder["trend"]
    title(ax, "The format effect does not shrink with scale; the tuning effect does",
          f"HumanEval: format {t['format_gain_slope_per_decade']:+.3f} per decade of parameters, tuning {t['tuning_gain_slope_per_decade']:+.3f}. On MBPP the format effect is a fifth the size and flat.")
    fig.savefig(out / "hivemind_ladder.png"); plt.close(fig)


def fig_correctness(out):
    """Text similarity is not correctness: three numbers from the same 7B arm (paper 5.4)."""
    sim, single, pool = 0.8765, 0.7904, 0.9573
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 2.6)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
    rows = [(2, sim, "how alike its eight answers read", TERRA),
            (1, single, "how often one answer is correct", INK),
            (0, pool, "how often at least one of the eight is correct", INK)]
    for y, v, lab, col in rows:
        ax.plot([0, v], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.plot([v], [y], "o", ms=9, color=col, zorder=3)
        ax.text(0.0, y + 0.28, lab, fontsize=9.2, color=MUTED, va="bottom")
        ax.text(v + 0.015, y, f"{v:.0%}" if y < 2 else f"{v:.2f}", fontsize=10, color=col, va="center")
    title(ax, "Sounding alike is not being right alike",
          "one 7B coding model, eight answers per problem, 164 HumanEval problems, every candidate executed in a sandbox")
    fig.savefig(out / "hivemind_correctness.png"); plt.close(fig)


def fig_suppression(out):
    """Between-model similarity under three prompt regimes, small models and three labs at 14B-31B."""
    a = sup["arms"]
    small = [("no system prompt\n(chat baseline)", a["chat"]["inter_mean"]),
             ("eight shipped personas,\none per model", a["deployed_model"]["inter_mean"]),
             ("eight distinct voices,\none per model", a["persona_model"]["inter_mean"])]
    # paper section 5.5, cross-lab arm at 14B-31B (Qwen2.5-Coder-14B, Ministral-3-14B, gemma-4-31B)
    big = [("no system prompt\n(chat baseline)", 0.7792), ("eight shipped personas,\none per model", 0.7903),
           ("eight distinct voices,\none per model", 0.5156)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, rows, head in ((axes[0], small, "six models, 0.36B to 2B"), (axes[1], big, "three labs, 14B to 31B")):
        cols = [SAND, TERRA_LT, TERRA]
        ys = range(len(rows))[::-1]
        ax.barh(list(ys), [r[1] for r in rows], height=0.55, color=cols, edgecolor="none")
        for y, (lab, v) in zip(ys, rows):
            ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=9.6, color=INK)
        ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
        ax.tick_params(length=0); ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(head, loc="left", fontsize=9.5, color=MUTED, pad=6)
    fig.suptitle("Between-model similarity: the lever works, and the shipped prompts do not pull it",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="medium", y=1.04)
    fig.savefig(out / "hivemind_suppression.png"); plt.close(fig)


FIGS = {
    "suspects": (fig_suspects, "hivemind_suspects.png",
                 "The three candidate explanations, and the verdict the experiments returned on each."),
    "scale": (fig_scale, "hivemind_scale.png",
              "Every similarity figure in this piece on one line. The floor is what two unrelated answers score."),
    "transport": (fig_heatmap, "hivemind_transport.png",
                  "A linear map from one model's hidden states into another's recovers the right caption among a thousand almost every time, regardless of lab."),
    "steps": (fig_steps, "hivemind_steps.png",
              "Six matched base/instruct pairs. Swapping in the tuned weights buys a little; routing the same weights through the chat template buys most of the distance to the reported band."),
    "decomp": (fig_decomp, "hivemind_decomp.png",
               "The template taken apart. Persona text alone moves the number the wrong way; user/assistant turn markers that no model was trained on carry the largest share."),
    "ladder": (fig_ladder, "hivemind_ladder.png",
               "Qwen2.5-Coder from 0.5B to 32B on HumanEval. The format effect grows with scale while the tuning effect declines."),
    "correctness": (fig_correctness, "hivemind_correctness.png",
                    "Answers an encoder scores as 0.88 alike disagree about correctness on roughly a sixth of problems."),
    "suppression": (fig_suppression, "hivemind_suppression.png",
                    "Eight genuinely distinct voices cut between-model similarity sharply. Eight rewordings of 'helpful assistant' do nothing, and across three labs make the models slightly more alike."),
}


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure_html(src: str, caption: str, alt: str) -> str:
    return (f'<figure style="margin:28px 0;text-align:center;">'
            f'<img src="{src}" alt="{alt}" style="width:100%;max-width:100%;height:auto;" />'
            f'<figcaption style="font-size:0.86em;color:#8a7a6e;margin-top:8px;line-height:1.45;">{caption}</figcaption>'
            f"</figure>")


def main() -> None:
    for key, (fn, name, _) in FIGS.items():
        fn(ASSETS)
        print(f"  {name}")
    md = MD.read_text()
    # The screenshot: embedded when present, a visible placeholder when not.
    shot = ASSETS / "yejin_choi_reply.png"
    md_img = re.compile(r"!\[([^\]]*)\]\(substack_assets/yejin_choi_reply\.png\)")
    if shot.exists():
        md = md_img.sub(lambda m: f'<figure style="margin:20px 0;text-align:center;"><img src="{b64(shot)}" alt="{m.group(1)}" style="max-width:100%;height:auto;border:1px solid #e9e0d2;border-radius:6px;" /></figure>', md)
    else:
        md = md_img.sub('<div style="border:1px dashed #a4472f;color:#a4472f;padding:14px;margin:20px 0;font-size:0.9em;">'
                        "Screenshot missing: save the reply as docs/substack_assets/yejin_choi_reply.png and rebuild.</div>", md)
    html = markdown.markdown(md, extensions=["extra", "smarty"])
    for key, (_, name, caption) in FIGS.items():
        marker = f"<!-- fig:{key} -->"
        if marker not in html:
            raise SystemExit(f"marker {marker} not found in {MD.name}")
        html = html.replace(marker, figure_html(b64(ASSETS / name), caption, caption))
    page = ("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
            "<title>Why Every Chatbot Sounds the Same</title></head>\n"
            "<body style=\"max-width:720px;margin:0 auto;font-family:Georgia,serif;line-height:1.6;"
            "color:#2b2420;padding:24px;\">\n" + html + "\n</body></html>\n")
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
