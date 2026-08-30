"""ChestX-ray14 probe demo: the reading is what ships.

Most medical imaging demos show a heatmap and ask you to trust it. This one
shows the floor and the shortcut next to the result, because a bare AUROC on
this dataset is not interpretable and the controls are the interesting part.

The 339 KB probe runs live on precomputed frozen states, so the 31B backbone
never runs here. That is the claim being demonstrated rather than described.
"""
import json

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = "RiverRider/srt-cxr14-linear-probe"
POOLED_REPO = "RiverRider/srt-cxr14-pooled-probe"
DATA_REPO = "RiverRider/srt-cxr14-frozen-probe"

ck = torch.load(hf_hub_download(MODEL_REPO, "cxr14_probe.pt"), weights_only=True)
FINDINGS = ck["findings"]
sub = np.load(hf_hub_download(DATA_REPO, "demo/test_subset.npz",
                              repo_type="dataset"))
X = torch.tensor(sub["states"])
Y = sub["labels"]
KEYS = sub["keys"]
VIEW = sub["view"]
EVAL = json.load(open(hf_hub_download(MODEL_REPO, "eval/cxr14_probe_full112k.json")))

Xn = (X - ck["mu"]) / ck["sd"]
P = torch.sigmoid(Xn @ ck["W"] + ck["b"]).numpy()

# Each backbone is standardised with its own training statistics, which is why
# the pooled checkpoint ships mu and sd alongside every weight matrix.
pck = torch.load(hf_hub_download(POOLED_REPO, "pooled_probe.pt"),
                 weights_only=False)
VENDORS = pck["vendors"]
sub3 = np.load(hf_hub_download(DATA_REPO, "demo/test_subset_3vendor.npz",
                               repo_type="dataset"))
LOGIT, PV = {}, {}
for _v in VENDORS:
    _p = pck["probes"][_v]
    _x = (torch.tensor(sub3[f"states_{_v}"]) - _p["mu"]) / _p["sd"]
    LOGIT[_v] = _x @ _p["weight"] + _p["bias"]
    PV[_v] = torch.sigmoid(LOGIT[_v]).numpy()
POOLED = torch.sigmoid(torch.stack([LOGIT[v] for v in VENDORS]).mean(0)).numpy()
POOLED_EVAL = json.load(open(hf_hub_download(POOLED_REPO,
                                             "results/cxr14_ensemble3.json")))


def auroc(scores, labels):
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def read_film(i):
    i = int(i) % len(KEYS)
    truth = {FINDINGS[j] for j in range(14) if Y[i, j] == 1}
    rows = sorted(zip(FINDINGS, P[i]), key=lambda t: -t[1])
    table = [[f, round(float(p), 3), "yes" if f in truth else ""] for f, p in rows]
    head = (f"**{KEYS[i]}**, view {VIEW[i]}, held out from training.\n\n"
            f"Reported findings: {', '.join(sorted(truth)) or 'none'}")
    # Hand gradio the decoded image rather than the cache path, which keeps
    # this independent of whatever the SDK currently allows to be served.
    p = hf_hub_download(DATA_REPO, f"demo/images/{str(KEYS[i])[:-4]}.jpg",
                        repo_type="dataset")
    return Image.open(p).convert("L"), head, table


def controls():
    """The probe against its own floor and its own shortcut, per finding."""
    out = []
    for j, f in enumerate(FINDINGS):
        pf = EVAL["per_finding"][f]
        out.append([f, pf["auroc"], pf["shuffled_floor"],
                    pf["view_only_baseline"],
                    "yes" if pf["beats_view"] else "NO"])
    out.sort(key=lambda r: -r[1])
    return out


def read_pooled(i):
    """One film, scored by each backbone separately and by their average."""
    i = int(i) % len(KEYS)
    truth = {FINDINGS[j] for j in range(14) if Y[i, j] == 1}
    rows = []
    for j, f in enumerate(FINDINGS):
        rows.append([f] + [round(float(PV[v][i, j]), 3) for v in VENDORS]
                    + [round(float(POOLED[i, j]), 3),
                       "yes" if f in truth else ""])
    rows.sort(key=lambda r: -r[-2])
    head = (f"**{KEYS[i]}**, view {VIEW[i]}, held out from training.\n\n"
            f"Reported findings: {', '.join(sorted(truth)) or 'none'}")
    p = hf_hub_download(DATA_REPO, f"demo/images/{str(KEYS[i])[:-4]}.jpg",
                        repo_type="dataset")
    return Image.open(p).convert("L"), head, rows


def pooled_table():
    """Per-backbone and pooled AUROC on these 300 films only.

    Deliberately recomputed on the demo cut rather than quoting the full-test
    numbers, so a visitor can check the arithmetic against what is on screen.
    The full-test figures are in the card and are the ones to cite.
    """
    out = []
    for v in VENDORS:
        m = np.nanmean([auroc(PV[v][:, j], Y[:, j]) for j in range(14)])
        out.append([v, round(float(m), 4),
                    POOLED_EVAL["single"].get(v, float("nan"))])
    m = np.nanmean([auroc(POOLED[:, j], Y[:, j]) for j in range(14)])
    out.append(["pooled (logit mean)", round(float(m), 4),
                POOLED_EVAL["summary"]["best_ensemble"]])
    return out


with gr.Blocks(title="ChestX-ray14: frozen features, linear probe") as demo:
    gr.Markdown(
        "# A 339 KB linear probe reads chest radiographs\n"
        "`Linear(5376, 14)` on frozen `google/gemma-4-31B-it` states. No "
        "fine-tuning, no radiology pretraining. On the **official** "
        "ChestX-ray14 split it scores **0.7590** mean AUROC against **0.7451** "
        "for the dataset authors' fine-tuned ResNet-50, ahead on 12 of 14.\n\n"
        "CheXNet's 0.8414 is a **different test set** (their own random "
        "70/10/20 partition) and is not comparable to either number above.\n\n"
        "The backbone does not run here. The probe runs live on precomputed "
        "states, which is the whole point: what ships is the reading.")

    with gr.Tab("Read a held-out film"):
        gr.Markdown("Every film below is from a patient the probe never saw.")
        idx = gr.Slider(0, len(KEYS) - 1, value=0, step=1, label="film")
        shuffle = gr.Button("Random film")
        with gr.Row():
            film = gr.Image(label="radiograph", height=460)
            with gr.Column():
                info = gr.Markdown()
                out = gr.Dataframe(
                    headers=["finding", "probability", "reported"],
                    label="probe output, most confident first")
        idx.change(read_film, idx, [film, info, out])
        shuffle.click(lambda: np.random.randint(len(KEYS)), None, idx)
        demo.load(read_film, idx, [film, info, out])

    with gr.Tab("Three backbones at once"):
        gr.Markdown(
            "The same film read by three frozen backbones from three different "
            "companies, and by the **average of their logits**, which adds no "
            "parameters at all.\n\n"
            "On the full official test split these score "
            f"**{POOLED_EVAL['single']['qwen3omni']}** (Qwen3-Omni), "
            f"**{POOLED_EVAL['single']['gemma4']}** (gemma-4) and "
            f"**{POOLED_EVAL['single']['aria']}** (Aria). Averaging them reaches "
            f"**{POOLED_EVAL['summary']['best_ensemble']}**, against "
            f"**{POOLED_EVAL['split_matched_baseline']}** for the dataset "
            "authors' fine-tuned ResNet-50.\n\n"
            "Aria is **behind** that baseline on its own and still improves the "
            "average, which is what makes this information the others lack "
            "rather than extra capacity.")
        pidx = gr.Slider(0, len(KEYS) - 1, value=0, step=1, label="film")
        pshuffle = gr.Button("Random film")
        with gr.Row():
            pfilm = gr.Image(label="radiograph", height=460)
            with gr.Column():
                pinfo = gr.Markdown()
                pout = gr.Dataframe(
                    headers=["finding"] + list(VENDORS) + ["pooled", "reported"],
                    label="each backbone, then their average, pooled first")
        gr.Markdown(
            "Mean AUROC over **these 300 films only**, recomputed live so it can "
            "be checked against what is on screen. The full-test figures above "
            "are the ones to cite.")
        gr.Dataframe(pooled_table(),
                     headers=["backbone", "these 300 films", "full test split"],
                     label="demo cut versus the real evaluation")
        pidx.change(read_pooled, pidx, [pfilm, pinfo, pout])
        pshuffle.click(lambda: np.random.randint(len(KEYS)), None, pidx)
        demo.load(read_pooled, pidx, [pfilm, pinfo, pout])

    with gr.Tab("The controls"):
        gr.Markdown(
            "A bare AUROC on this dataset is not interpretable, so here is what "
            "it is measured against.\n\n"
            "**Shuffled** permutes the training labels and refits. Anything "
            "above 0.5 on held-out data would be leakage.\n\n"
            "**View-only** uses the AP/PA marker alone. Portable AP films are "
            "taken of sicker, bedbound patients, so view is a real route to a "
            "high score involving no pathology. A finding that fails to clear "
            "it has not been detected. The baseline is folded, because a value "
            "far below 0.5 is strongly predictive once flipped: Hernia's raw "
            "view-only is 0.3436, which is 0.6564 of shortcut.")
        gr.Dataframe(controls(),
                     headers=["finding", "probe", "shuffled floor",
                              "view-only", "beats view"],
                     label="all 14 findings clear the view baseline")

    with gr.Tab("Scope"):
        gr.Markdown(
            "**Detection, not early detection.** These labels describe what is "
            "visible in the image in front of you. Nothing here speaks to "
            "catching disease before it is apparent, which needs longitudinal "
            "data with outcomes.\n\n"
            "**Not a diagnostic device.** Research artifact. No clinical "
            "validation, no prospective evaluation, no regulatory clearance. "
            "Do not use it for anything that touches a patient.\n\n"
            "**Labels are NLP-mined** from radiology reports by the dataset "
            "authors. Every model on this benchmark inherits that ceiling.\n\n"
            "**Confidence intervals resample patients, not images.** The test "
            "split is 25,596 films from 2,797 patients, and those films are "
            "not independent. Resampling images gives intervals about 1.5x too "
            "narrow.\n\n"
            "**Banked negatives.** Max-pooling and top-16 pooling were "
            "predicted to help focal findings and did the opposite, costing "
            "0.0537 and 0.0225. Readout depth barely matters: 0.7600 to 0.7605 "
            "across three depths.\n\n"
            f"Model: `{MODEL_REPO}` · Data: `{DATA_REPO}` · "
            "Code: https://github.com/space-bacon/SRT")

demo.launch()
