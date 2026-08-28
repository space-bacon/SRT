import json
import re
import time

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
HOLDOUT = 800
EPOCHS = 30
_cache = {}


def shard_no(f):
    return int(re.search(r"_s(\d+)\.", f).group(1))


def load_iter(vendor, log):
    """Fill _cache[vendor], yielding after each shard.

    A generator rather than a plain call because the caller has to emit
    something to the browser between shards: a long silence drops the SSE
    stream and the page then shows nothing at all.
    """
    if vendor in _cache:
        log(f"- {LABEL[vendor]}: already in memory")
        yield
        return
    spre, mpre = VENDORS[vendor]
    files = list_repo_files(STATES, repo_type="dataset")
    sn = sorted([f for f in files if f.startswith(f"states/{spre}")], key=shard_no)
    mn = sorted([f for f in files if f.startswith(f"manifests/{mpre}")], key=shard_no)
    item, text, keys, caps = [], [], [], []
    for i, (s, m) in enumerate(zip(sn, mn), 1):
        log(f"- {LABEL[vendor]}: downloading shard {i} of {len(sn)}")
        yield
        z = np.load(hf_hub_download(STATES, s, repo_type="dataset"))
        rows = json.load(open(hf_hub_download(STATES, m, repo_type="dataset")))["rows"]
        ok = z["ok"]
        item.append(z["item"][ok])
        text.append(z["text"][ok])
        keys += [r["key"] for r, k in zip(rows, ok) if k]
        caps += [r["caption"] for r, k in zip(rows, ok) if k]
        yield
    _cache[vendor] = (np.concatenate(item).astype(np.float32),
                      np.concatenate(text).astype(np.float32),
                      np.array(keys), np.array(caps))
    log(f"- {LABEL[vendor]}: {len(keys)} rows, hidden dim "
        f"{_cache[vendor][0].shape[1]}")
    yield


def fit_iter(mats, out, log, dim=256, epochs=EPOCHS, batch=512, lr=5.0):
    """One tower per input, every item/text pair trained together.

    Minibatched because the full-batch version multiplies n-by-n similarity
    matrices every step, which a free CPU box cannot do in reasonable time.
    Each input is centred and scaled to unit RMS first: hidden sizes run from
    2048 to 5376 and a shared lr would otherwise train the towers at different
    speeds, which shows up as a fake vendor difference. Yields once per epoch
    so the page keeps receiving data.
    """
    mus = [M.mean(0, keepdims=True) for M in mats]
    C = [M - mu for M, mu in zip(mats, mus)]
    scales = [float(np.sqrt((M ** 2).sum(1).mean())) + 1e-8 for M in C]
    C = [M / s for M, s in zip(C, scales)]
    rng = np.random.default_rng(0)
    W = [rng.normal(0, .02, (M.shape[1], dim)).astype(np.float32) for M in C]
    items, texts = (0, 2), (1, 3)  # [itemA, textA, itemB, textB]
    n = len(C[0])
    t0 = time.time()
    for ep in range(epochs):
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            B = [M[sl] for M in C]
            b = len(B[0])
            if b < 8:
                continue
            Z = []
            for M, w in zip(B, W):
                Y = M @ w
                Z.append(Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8))
            g = [np.zeros_like(w) for w in W]
            for i in items:
                for t in texts:
                    S = Z[i] @ Z[t].T / .05
                    S -= S.max(1, keepdims=True)
                    G = np.exp(S)
                    G /= G.sum(1, keepdims=True)
                    G[np.arange(b), np.arange(b)] -= 1.
                    G /= b
                    g[i] += B[i].T @ (G @ Z[t])
                    g[t] += B[t].T @ (G.T @ Z[i])
            for k in range(4):
                W[k] -= lr * g[k]
        log(f"- fitting towers: epoch {ep + 1} of {epochs} "
            f"({time.time() - t0:.0f}s)")
        yield
    out["W"], out["mus"], out["scales"] = W, mus, scales


def proj(X, W, mu, scale):
    Y = ((X - mu) / scale) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)


def prepare(gallery, querier, progress=gr.Progress()):
    lines = []

    def log(msg):
        lines.append(msg)

    def emit(state=None, summary=""):
        # Keep only the last few fit lines so the panel does not grow forever.
        shown = lines if len(lines) <= 14 else lines[:6] + ["- ..."] + lines[-7:]
        return "\n".join(shown), state, summary

    if gallery == querier:
        yield ("Pick two different models. The point is searching across them.",
               None, "")
        return

    t0 = time.time()
    log(f"**{LABEL[gallery]} gallery, queried by {LABEL[querier]}**\n")
    yield emit()

    progress(0.05, desc="downloading gallery states")
    for _ in load_iter(gallery, log):
        yield emit()
    progress(0.3, desc="downloading query states")
    for _ in load_iter(querier, log):
        yield emit()

    Ig, Tg, Kg, Cg = _cache[gallery]
    Iq, Tq, Kq, _cq = _cache[querier]
    pos = {k: i for i, k in enumerate(Kq)}
    ig = np.array([i for i, k in enumerate(Kg) if k in pos])
    iq = np.array([pos[k] for k in Kg if k in pos])
    log(f"- aligned on {len(ig)} keys present in both")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ig))
    te, tr = perm[:HOLDOUT], perm[HOLDOUT:]
    log(f"- train {len(tr)}, holdout {len(te)}")
    yield emit()

    # [item_A, text_A, item_B, text_B], so both cross directions are measured.
    mats = [Ig[ig][tr], Tg[ig][tr], Iq[iq][tr], Tq[iq][tr]]
    res = {}
    for k, _ in enumerate(fit_iter(mats, res, log), 1):
        progress(0.45 + 0.45 * k / EPOCHS, desc=f"fitting epoch {k}/{EPOCHS}")
        yield emit()

    W, mus, sc = res["W"], res["mus"], res["scales"]
    hold = [Ig[ig][te], Tg[ig][te], Iq[iq][te], Tq[iq][te]]
    Z = [proj(M, w, mu, s) for M, w, mu, s in zip(hold, W, mus, sc)]
    caps = Cg[ig][te]

    def r1(item, text):
        S = Z[text] @ Z[item].T
        d = np.arange(len(S))
        return float(((S > S[d, d][:, None]).sum(1) + 1 == 1).mean())

    names = [gallery, querier]
    within = [r1(0, 1), r1(2, 3)]
    cross = [r1(0, 3), r1(2, 1)]
    ret = float(np.mean(cross) / np.mean(within)) if np.mean(within) else 0.0
    log(f"\n**Ready in {time.time() - t0:.0f}s. Enter a query in step 2.**")

    state = {"G": Z[0], "caps": caps, "Q": Z[3],
             "gallery": gallery, "querier": querier}

    rows = "\n".join(
        f"| {LABEL[names[i]]} gallery | {LABEL[names[j]]} | "
        f"{'within' if i == j else '**cross**'} | {r1(i * 2, j * 2 + 1):.4f} |"
        for i in (0, 1) for j in (0, 1))
    summary = (
        f"### All four directions for this pair\n\n"
        f"| gallery | queried by | | r@1 |\n|---|---|---|---|\n{rows}\n\n"
        f"mean cross **{np.mean(cross):.4f}** against mean within "
        f"**{np.mean(within):.4f}**, so retention **{ret:.3f}** on this run.\n\n"
        f"{HOLDOUT} held-out items, towers fitted here with a plain numpy loop so "
        f"this runs on a free CPU box. That fit is much weaker than the torch fit "
        f"behind the published numbers, and any single direction is noisy enough "
        f"to land well above or below 1.0 on its own. Averaging both cross "
        f"directions against both within directions, as here, is what makes the "
        f"quantity stable. **The measured figure is retention 0.988, 95% CI "
        f"[0.955, 1.023]** over four vendors and twelve cross directions. Treat "
        f"that interval as the result and this panel as an illustration."
    )
    yield emit(state, summary)


def search(state, query):
    if not state:
        return "Run step 1 first, so there is a gallery to search."
    caps, G = state["caps"], state["G"]
    words = set(query.lower().split())
    sims = [len(words & set(c.lower().split())) for c in caps]
    hit = int(np.argmax(sims))
    if sims[hit] == 0:
        return ("No held-out caption shares a word with that query. Try wording "
                "it more like a photo caption.")
    top = np.argsort(-(state["Q"][hit:hit + 1] @ G.T)[0])[:5]
    out = [f"The gallery holds {len(caps)} held-out items. Closest caption to "
           f"your query:\n\n> {caps[hit]}\n",
           f"\nTop 5 from the **{LABEL[state['gallery']]}** gallery, queried by "
           f"**{LABEL[state['querier']]}**:\n"]
    for j, t in enumerate(top, 1):
        mark = "  **<- the correct item**" if t == hit else ""
        out.append(f"{j}. {caps[t]}{mark}")
    return "\n".join(out)


with gr.Blocks(title="SRT omni cross-vendor retrieval") as demo:
    fitted = gr.State(None)
    gr.Markdown(
        "# One gallery, any encoder\n\n"
        "Four multimodal models from four vendors encoded the same 5,000 images. "
        "This searches **one model's gallery using a different model's text "
        "encoder**, fitting the towers live on CPU.\n\n"
        "Across four vendors the cross-vendor rate is statistically "
        "indistinguishable from the within-vendor rate: retention **0.988**, "
        "95% CI **[0.955, 1.023]**. The interval contains 1.0, so the claim is "
        "indistinguishability, not a retained fraction."
    )

    gr.Markdown("## Step 1 · load states and fit the towers")
    with gr.Row():
        g = gr.Dropdown(list(VENDORS), value="gemma4", label="Gallery encoded by")
        q = gr.Dropdown(list(VENDORS), value="aria", label="Searched with")
    load_btn = gr.Button("Load and fit", variant="primary")
    gr.Markdown(
        "*First load of a model downloads its states (20 to 110 MB) and then fits "
        "on CPU, which takes roughly a minute in total. Progress prints below and "
        "updates every epoch. A pair already loaded reuses what is in memory.*"
    )
    status = gr.Markdown()
    summary = gr.Markdown()

    gr.Markdown("## Step 2 · search across the two models")
    txt = gr.Textbox(value="a man riding a horse on the beach",
                     label="Your query", info="Press enter or click Search.")
    search_btn = gr.Button("Search")
    result = gr.Markdown()

    load_btn.click(prepare, [g, q], [status, fitted, summary],
                   show_progress="minimal")
    # The browser logs "Too many arguments provided for the endpoint" here,
    # because it counts gr.State as an argument while the generated endpoint
    # signature does not. Cosmetic; api_name=False silences nothing and only
    # costs the /search endpoint, so the handlers stay exposed.
    search_btn.click(search, [fitted, txt], result)
    txt.submit(search, [fitted, txt], result)

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

demo.queue(default_concurrency_limit=2).launch()
