import json
import pathlib
import re

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

STATES = "RiverRider/srt-omni-crossvendor-states"
VENDORS = {
    "qwen3omni": ("omni_states_s", "omni_manifest_s"),
    "gemma4": ("gemma4_states_s", "xv_manifest_s"),
    "mistral": ("mistral_states_s", "img_s"),
    "aria": ("aria_states_s", "img_s"),
}
LABEL = {"qwen3omni": "Qwen3-Omni-30B", "gemma4": "gemma-4-31B",
         "mistral": "Mistral-Small-3.1-24B", "aria": "Aria"}
_cache = {}


def shard_no(f):
    return int(re.search(r"_s(\d+)\.", f).group(1))


def load(vendor):
    """Rows for one vendor, keyed by manifest key so vendors can be aligned."""
    if vendor in _cache:
        return _cache[vendor]
    spre, mpre = VENDORS[vendor]
    files = list_repo_files(STATES, repo_type="dataset")
    sn = sorted([f for f in files if f.startswith(f"states/{spre}")], key=shard_no)
    mn = sorted([f for f in files if f.startswith(f"manifests/{mpre}")], key=shard_no)
    item, text, keys, caps = [], [], [], []
    for s, m in zip(sn, mn):
        z = np.load(hf_hub_download(STATES, s, repo_type="dataset"))
        rows = json.load(open(hf_hub_download(STATES, m, repo_type="dataset")))["rows"]
        ok = z["ok"]
        item.append(z["item"][ok])
        text.append(z["text"][ok])
        keys += [r["key"] for r, k in zip(rows, ok) if k]
        caps += [r["caption"] for r, k in zip(rows, ok) if k]
    out = (np.concatenate(item).astype(np.float32),
           np.concatenate(text).astype(np.float32),
           np.array(keys), np.array(caps))
    _cache[vendor] = out
    return out


def fit(Xi, Ta, Tb, dim=256, epochs=400, lr=0.5):
    """One tower per input, all pairs trained together into a shared space."""
    mus = [M.mean(0, keepdims=True) for M in (Xi, Ta, Tb)]
    A, Pa, Pb = Xi - mus[0], Ta - mus[1], Tb - mus[2]
    rng = np.random.default_rng(0)
    W = [rng.normal(0, .02, (M.shape[1], dim)).astype(np.float32) for M in (A, Pa, Pb)]
    n = len(A)
    for _ in range(epochs):
        Z = []
        for M, w in zip((A, Pa, Pb), W):
            Y = M @ w
            Z.append(Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8))
        g = [np.zeros_like(w) for w in W]
        for ti, P in ((1, Pa), (2, Pb)):
            S = Z[0] @ Z[ti].T / .05
            S -= S.max(1, keepdims=True)
            G = np.exp(S)
            G /= G.sum(1, keepdims=True)
            G[np.arange(n), np.arange(n)] -= 1.
            G /= n
            g[0] += A.T @ (G @ Z[ti])
            g[ti] += P.T @ (G.T @ Z[0])
        for i in range(3):
            W[i] -= lr * g[i]
    return W, mus


def proj(X, W, mu):
    Y = (X - mu) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)


def run(gallery, querier, query, progress=gr.Progress()):
    if gallery == querier:
        return "Pick two different models. The point is searching across them.", ""
    progress(0.1, desc=f"loading {LABEL[gallery]} states")
    Ig, Tg, Kg, Cg = load(gallery)
    progress(0.4, desc=f"loading {LABEL[querier]} states")
    Iq, Tq, Kq, _ = load(querier)

    pos = {k: i for i, k in enumerate(Kq)}
    shared = [k for k in Kg if k in pos]
    ig = np.array([i for i, k in enumerate(Kg) if k in pos])
    iq = np.array([pos[k] for k in shared])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(shared))
    te, tr = perm[:800], perm[800:]

    progress(0.6, desc="fitting towers (CPU)")
    W, mus = fit(Ig[ig][tr], Tg[ig][tr], Tq[iq][tr])

    G = proj(Ig[ig][te], W[0], mus[0])
    caps = Cg[ig][te]

    def r1(Q):
        S = Q @ G.T
        d = np.arange(len(Q))
        return float(((S > S[d, d][:, None]).sum(1) + 1 == 1).mean())

    within = r1(proj(Tg[ig][te], W[1], mus[1]))
    cross = r1(proj(Tq[iq][te], W[2], mus[2]))

    summary = (
        f"### Gallery built by **{LABEL[gallery]}**\n\n"
        f"| query encoder | r@1 |\n|---|---|\n"
        f"| {LABEL[gallery]} (within-vendor) | {within:.4f} |\n"
        f"| **{LABEL[querier]} (cross-vendor)** | **{cross:.4f}** |\n\n"
        f"retention (cross / within) = **{cross / within:.3f}**\n\n"
        f"800 held-out items. A plain numpy fit, weaker in absolute r@1 than the "
        f"published torch fit, so read the comparison rather than the score. "
        f"Published 4-vendor figure: retention 0.988, 95% CI [0.955, 1.023]."
    )

    progress(0.95, desc="retrieving")
    sims = [len(set(query.lower().split()) & set(c.lower().split())) for c in caps]
    hit = int(np.argmax(sims))
    q = proj(Tq[iq][te][hit:hit + 1], W[2], mus[2])
    top = np.argsort(-(q @ G.T)[0])[:5]
    lines = [f"Closest held-out caption to your query:\n\n> {caps[hit]}\n",
             f"\nTop 5 from the **{LABEL[gallery]}** gallery, "
             f"queried by **{LABEL[querier]}**:\n"]
    for j, t in enumerate(top, 1):
        mark = " **<- the correct item**" if t == hit else ""
        lines.append(f"{j}. {caps[t]}{mark}")
    return summary, "\n".join(lines)


with gr.Blocks(title="SRT omni cross-vendor retrieval") as demo:
    gr.Markdown(
        "# One gallery, any encoder\n\n"
        "Four multimodal models from four vendors encoded the same 5,000 images. "
        "This searches **one model's gallery using a different model's text "
        "encoder**, fitting the towers live on CPU.\n\n"
        "Across four vendors the cross-vendor rate is statistically "
        "indistinguishable from the within-vendor rate: retention **0.988**, "
        "95% CI **[0.955, 1.023]**. The interval contains 1.0, so the claim is "
        "indistinguishability, not a retained fraction.\n\n"
        "*First run downloads a few hundred MB of states and takes a minute.*"
    )
    with gr.Row():
        g = gr.Dropdown(list(VENDORS), value="gemma4", label="Gallery encoded by")
        q = gr.Dropdown(list(VENDORS), value="aria", label="Searched with")
    txt = gr.Textbox(value="a man riding a horse on the beach", label="Your query")
    btn = gr.Button("Search across models", variant="primary")
    out_a = gr.Markdown()
    out_b = gr.Markdown()
    btn.click(run, [g, q, txt], [out_a, out_b])
    gr.Markdown(
        "---\n"
        "**Centering is not optional.** Raw cosine between unrelated items on "
        "these states is +0.869 and raw retrieval sits exactly at chance. Every "
        "number here is per-modality centered.\n\n"
        "**Scope.** Images only: two of the four hosts have no audio or video "
        "tower. All four are trained on overlapping web-scale corpora, so this "
        "shows the structure survives a change of vendor, architecture and "
        "training run. It does not show that independent minds would converge "
        "on it, and that reading is not asserted.\n\n"
        f"States: [`{STATES}`](https://huggingface.co/datasets/{STATES}) · "
        "Towers: [`srt-omni-xvendor-towers`](https://huggingface.co/RiverRider/srt-omni-xvendor-towers) · "
        "Add your own model: [`srt-omni-manifest`](https://huggingface.co/datasets/RiverRider/srt-omni-manifest)"
    )

demo.queue().launch()
