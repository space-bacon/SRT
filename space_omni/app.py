import base64
import io
import json
import pathlib
import re
import time

import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

STATES = "RiverRider/srt-omni-crossvendor-states"
VENDORS = {
    "qwen3omni": ("omni_states_s", "omni_manifest_s"),
    "gemma4": ("gemma4_states_s", "xv_manifest_s"),
    "mistral": ("mistral_states_s", "img_s"),
    "aria": ("aria_states_s", "img_s"),
}
LABEL = {"qwen3omni": "Qwen3-Omni 30B", "gemma4": "Gemma-4 31B",
         "mistral": "Mistral Small 3.1 24B", "aria": "Aria"}
MAKER = {"qwen3omni": "Alibaba", "gemma4": "Google",
         "mistral": "Mistral AI", "aria": "Rhymes AI"}
HOLDOUT = 800
OMNI_HOLDOUT = 700
EPOCHS = 30
TOPK = 5
EXAMPLES = [
    "a man riding a surfboard on a wave",
    "a plate of food on a wooden table",
    "two dogs playing in the grass",
    "a red double decker bus on a city street",
    "a child holding an umbrella in the rain",
    "a giraffe standing next to a tree",
]
OMNI_EXAMPLES = [
    "a dog barking outdoors",
    "a woman cooking in a kitchen",
    "an engine revving loudly",
    "a man playing a guitar",
    "rain falling and thunder",
    "a person riding a motorcycle",
]
_cache = {}
_thumbs = {}
_media = {}

CSS = """
.hero {background:linear-gradient(135deg,#131233,#1c1a47);color:#eae7fb;
 padding:26px 28px;border-radius:14px;border:1px solid #332e66}
.hero h1 {margin:0 0 6px;font-size:30px;color:#fff}
.hero p {margin:6px 0 0;color:#c9c3ee;font-size:15px;line-height:1.55}
.verdict {font-size:19px;font-weight:600;padding:16px 18px;border-radius:12px;
 border:1px solid #332e66;background:#1c1a47;color:#eae7fb;margin-top:4px}
.vendorbar {font-weight:600;padding:9px 13px;border-radius:9px;margin-bottom:2px;
 font-size:14px}
.bar-a {background:#23205a;color:#cfc7ff;border-left:4px solid #a48dff}
.bar-b {background:#1d3340;color:#bfe6f5;border-left:4px solid #4fc3e8}
.cards {display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}
.card {width:200px;background:#1c1a47;border:1px solid #332e66;border-radius:10px;
 padding:8px}
.card .thumb {width:100%;height:130px;object-fit:cover;border-radius:7px}
.cap {font-size:12px;color:#c9c3ee;margin-top:6px;line-height:1.35}
.tag {font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:6px}
.t-image {color:#a48dff}
.t-video {color:#4fc3e8}
.t-audio {color:#ffb86b}
footer {display:none !important}
"""


def hero_image():
    """Inlined: gr.Image serves from http://0.0.0.0:7860 here, which an https
    Space blocks as mixed content."""
    p = pathlib.Path(__file__).with_name("four_models.jpg")
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<img src="data:image/jpeg;base64,{b64}" '
            'style="display:block;margin:14px auto 4px;max-width:620px;'
            'width:100%;border-radius:12px;border:1px solid #332e66">')


STOP = {"a", "an", "the", "of", "on", "in", "at", "to", "and", "or", "is",
        "are", "was", "were", "with", "for", "by", "from", "as", "that",
        "this", "it", "its", "into", "over", "near", "next", "some", "then",
        "while", "there", "their", "his", "her", "be", "been", "has", "have"}


def seed_index(query, caps):
    """Best-matching caption, scoring content words by rarity.

    Plain set overlap lets stopwords carry the match, which picks an arbitrary
    caption for any short query.
    """
    words = {w.strip(".,!?") for w in query.lower().split()} - STOP
    if not words:
        return -1
    df = {}
    for c in caps:
        for w in set(c.lower().split()) & words:
            df[w] = df.get(w, 0) + 1
    n = len(caps)
    best, best_score = -1, 0.0
    for i, c in enumerate(caps):
        toks = set(c.lower().split())
        score = sum(np.log(n / (1 + df.get(w, 0))) for w in words & toks)
        if score > best_score:
            best, best_score = i, score
    return best


def media(stem):
    """Packed audio mp3s or video frames, same layout as the image thumbs."""
    if stem not in _media:
        blob = open(hf_hub_download(STATES, f"media/{stem}.bin",
                                    repo_type="dataset"), "rb").read()
        idx = json.load(open(hf_hub_download(STATES, f"media/{stem}_index.json",
                                             repo_type="dataset")))
        _media[stem] = (blob, idx)
    return _media[stem]


def card(modality, key, caption):
    """One result, rendered inline. Audio needs a player, so a Gallery cannot
    hold a mixed-modality result set."""
    name = pathlib.PurePosixPath(key).name
    if modality == "image":
        im = image_of(key)
        if im is None:
            return ""
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=78)
        src = base64.b64encode(buf.getvalue()).decode()
        body = f'<img src="data:image/jpeg;base64,{src}" class="thumb">'
    elif modality == "video":
        blob, idx = media("video")
        if name not in idx:
            return ""
        off, n = idx[name]
        src = base64.b64encode(blob[off:off + n]).decode()
        body = f'<img src="data:image/jpeg;base64,{src}" class="thumb">'
    else:
        blob, idx = media("audio")
        if name not in idx:
            return ""
        off, n = idx[name]
        src = base64.b64encode(blob[off:off + n]).decode()
        body = ('<audio controls preload="none" style="width:100%" '
                f'src="data:audio/mpeg;base64,{src}"></audio>')
    tag = {"image": "PHOTO", "video": "VIDEO", "audio": "SOUND"}[modality]
    return (f'<div class="card"><div class="tag t-{modality}">{tag}</div>'
            f'{body}<div class="cap">{caption[:80]}</div></div>')


def fit_omni_iter(I, T, mods, out, log, dim=256, epochs=EPOCHS, batch=512, lr=5.0):
    """One item tower for every modality, against one text tower.

    Centring is per modality: the anisotropy differs by modality, so a single
    global mean would leave each modality's own offset in place.
    """
    mu_i = {m: I[mods == m].mean(0, keepdims=True) for m in set(mods.tolist())}
    A = np.concatenate([(I[i:i + 1] - mu_i[mods[i]]) for i in range(len(I))])
    mu_t = T.mean(0, keepdims=True)
    B = T - mu_t
    sa = float(np.sqrt((A ** 2).sum(1).mean())) + 1e-8
    sb = float(np.sqrt((B ** 2).sum(1).mean())) + 1e-8
    A, B = A / sa, B / sb
    rng = np.random.default_rng(0)
    Wi = rng.normal(0, .02, (A.shape[1], dim)).astype(np.float32)
    Wt = rng.normal(0, .02, (B.shape[1], dim)).astype(np.float32)
    n = len(A)
    for ep in range(epochs):
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            a, b = A[sl], B[sl]
            k = len(a)
            if k < 8:
                continue
            P = a @ Wi
            P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
            Q = b @ Wt
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
            S = P @ Q.T / .05
            S -= S.max(1, keepdims=True)
            G = np.exp(S)
            G /= G.sum(1, keepdims=True)
            G[np.arange(k), np.arange(k)] -= 1.
            G /= k
            Wi -= lr * (a.T @ (G @ Q))
            Wt -= lr * (b.T @ (G.T @ P))
        log(f"learning one space for three media: {int(100*(ep+1)/epochs)}%")
        yield
    out.update(Wi=Wi, Wt=Wt, mu_i=mu_i, mu_t=mu_t, sa=sa, sb=sb)


def shard_no(f):
    return int(re.search(r"_s(\d+)\.", f).group(1))


def thumbs():
    """Packed jpegs plus an offset index; COCO itself is http-only."""
    if not _thumbs:
        blob = open(hf_hub_download(STATES, "thumbs/thumbs.bin",
                                    repo_type="dataset"), "rb").read()
        idx = json.load(open(hf_hub_download(STATES, "thumbs/thumbs_index.json",
                                             repo_type="dataset")))
        _thumbs["blob"], _thumbs["idx"] = blob, idx
    return _thumbs


def image_of(key):
    t = thumbs()
    if key not in t["idx"]:
        return None
    off, n = t["idx"][key]
    return Image.open(io.BytesIO(t["blob"][off:off + n]))


def load_iter(vendor, log):
    """Fill _cache[vendor], yielding after each shard so the page keeps data."""
    if vendor in _cache:
        log(f"{LABEL[vendor]} ready")
        yield
        return
    spre, mpre = VENDORS[vendor]
    files = list_repo_files(STATES, repo_type="dataset")
    sn = sorted([f for f in files if f.startswith(f"states/{spre}")], key=shard_no)
    mn = sorted([f for f in files if f.startswith(f"manifests/{mpre}")], key=shard_no)
    item, text, keys, caps = [], [], [], []
    for i, (s, m) in enumerate(zip(sn, mn), 1):
        log(f"{LABEL[vendor]}: fetching part {i} of {len(sn)}")
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
    yield


def fit_iter(mats, out, log, dim=256, epochs=EPOCHS, batch=512, lr=5.0):
    """One tower per input, every item/text pair trained together.

    Inputs are centred and scaled to unit RMS first: hidden sizes run 2048 to
    5376 and a shared lr would otherwise train the towers at different speeds,
    which shows up as a fake vendor difference.
    """
    mus = [M.mean(0, keepdims=True) for M in mats]
    C = [M - mu for M, mu in zip(mats, mus)]
    scales = [float(np.sqrt((M ** 2).sum(1).mean())) + 1e-8 for M in C]
    C = [M / s for M, s in zip(C, scales)]
    rng = np.random.default_rng(0)
    W = [rng.normal(0, .02, (M.shape[1], dim)).astype(np.float32) for M in C]
    items, texts = (0, 2), (1, 3)
    n = len(C[0])
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
        log(f"learning the shared space: {int(100 * (ep + 1) / epochs)}%")
        yield
    out["W"], out["mus"], out["scales"] = W, mus, scales


def proj(X, W, mu, scale):
    Y = ((X - mu) / scale) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)


def prepare(gallery, querier, progress=gr.Progress()):
    line = [""]

    def log(m):
        line[0] = m

    def panel(msg):
        return f'<div class="verdict">{msg}</div>'

    if gallery == querier:
        yield panel("Pick two different companies. The point is crossing between them."), None
        return

    t0 = time.time()
    yield panel("Fetching the picture index…"), None
    thumbs()

    for _ in load_iter(gallery, log):
        progress(0.2, desc=line[0])
        yield panel(line[0]), None
    for _ in load_iter(querier, log):
        progress(0.4, desc=line[0])
        yield panel(line[0]), None

    Ig, Tg, Kg, Cg = _cache[gallery]
    Iq, Tq, Kq, _c = _cache[querier]
    pos = {k: i for i, k in enumerate(Kq)}
    ig = np.array([i for i, k in enumerate(Kg) if k in pos])
    iq = np.array([pos[k] for k in Kg if k in pos])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ig))
    te, tr = perm[:HOLDOUT], perm[HOLDOUT:]

    mats = [Ig[ig][tr], Tg[ig][tr], Iq[iq][tr], Tq[iq][tr]]
    res = {}
    for k, _ in enumerate(fit_iter(mats, res, log), 1):
        progress(0.5 + 0.45 * k / EPOCHS, desc=line[0])
        yield panel(line[0]), None

    W, mus, sc = res["W"], res["mus"], res["scales"]
    held = [Ig[ig][te], Tg[ig][te], Iq[iq][te], Tq[iq][te]]
    Z = [proj(M, w, mu, s) for M, w, mu, s in zip(held, W, mus, sc)]

    def r1(i, t):
        S = Z[t] @ Z[i].T
        d = np.arange(len(S))
        return float(((S > S[d, d][:, None]).sum(1) + 1 == 1).mean())

    within = np.mean([r1(0, 1), r1(2, 3)])
    cross = np.mean([r1(0, 3), r1(2, 1)])
    state = {"G": Z[0], "own": Z[1], "other": Z[3], "caps": Cg[ig][te],
             "keys": Kg[ig][te], "gallery": gallery, "querier": querier,
             "within": within, "cross": cross}
    yield panel(
        f"Ready in {time.time() - t0:.0f}s. <b>{MAKER[gallery]}</b> built an "
        f"index of {HOLDOUT} photos. Ask for something below and "
        f"<b>{MAKER[querier]}</b> will search it."), state


def omni_prepare(progress=gr.Progress()):
    """One backbone, one tower, all three modalities in a single index."""
    line = [""]

    def log(m):
        line[0] = m

    def panel(msg):
        return f'<div class="verdict">{msg}</div>'

    t0 = time.time()
    yield panel("Fetching media (photos, sounds and video frames)&hellip;"), None
    thumbs()
    media("audio")
    media("video")

    for _ in load_iter("qwen3omni", log):
        progress(0.3, desc=line[0])
        yield panel(line[0]), None

    I, T, K, C = _cache["qwen3omni"]
    # Modality per row, read from the shard manifests in the same order.
    mods = []
    files = list_repo_files(STATES, repo_type="dataset")
    mn = sorted([f for f in files if f.startswith("manifests/omni_manifest_s")],
                key=shard_no)
    sn = sorted([f for f in files if f.startswith("states/omni_states_s")],
                key=shard_no)
    for s, m in zip(sn, mn):
        z = np.load(hf_hub_download(STATES, s, repo_type="dataset"))
        mr = json.load(open(hf_hub_download(STATES, m, repo_type="dataset")))["rows"]
        mods += [r["modality"] for r, o in zip(mr, z["ok"]) if o]
    mods = np.array(mods)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(I))
    te, tr = perm[:OMNI_HOLDOUT], perm[OMNI_HOLDOUT:]
    res = {}
    for k, _ in enumerate(fit_omni_iter(I[tr], T[tr], mods[tr], res, log), 1):
        progress(0.4 + 0.55 * k / EPOCHS, desc=line[0])
        yield panel(line[0]), None

    def proj_i(X, m):
        Y = ((X - np.concatenate([res["mu_i"][mm] for mm in m])) / res["sa"]) @ res["Wi"]
        return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)

    G = proj_i(I[te], mods[te])
    Q = ((T[te] - res["mu_t"]) / res["sb"]) @ res["Wt"]
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8

    S = Q @ G.T
    d = np.arange(len(S))
    r = (S > S[d, d][:, None]).sum(1) + 1
    counts = {m: int((mods[te] == m).sum()) for m in ("image", "audio", "video")}
    state = {"G": G, "Q": Q, "keys": K[te], "caps": C[te], "mods": mods[te]}
    yield panel(
        f"Ready in {time.time()-t0:.0f}s. One index of {len(te)} items: "
        f"{counts['image']} photos, {counts['audio']} sounds, "
        f"{counts['video']} video clips. Mixed r@1 {float((r==1).mean()):.4f} "
        f"against a chance rate of {1/len(te):.4f}."), state


def omni_search(state, query):
    if not state:
        return '<div class="verdict">Press <b>Build one shared index</b> first.</div>', ""
    caps, mods = state["caps"], state["mods"]
    seed = seed_index(query, caps)
    if seed < 0:
        return ('<div class="verdict">Nothing in this index matches that. Try '
                'one of the examples.</div>', "")
    order = np.argsort(-(state["Q"][seed:seed + 1] @ state["G"].T)[0])[:6]
    kinds = [mods[i] for i in order]
    mix = ", ".join(f"{kinds.count(k)} {k}" for k in ("image", "audio", "video")
                    if kinds.count(k))
    head = (f'<div class="verdict">One text query, one index, three kinds of '
            f'media. Top 6 came back as {mix}.<br>'
            f'<span style="font-size:15px;font-weight:400;color:#a7a1cc">'
            f'Matched to the held-out caption “{caps[seed][:90]}”. The same '
            f'linear map placed all three kinds. Nothing was trained per '
            f'modality.</span></div>')
    cards = "".join(card(mods[i], state["keys"][i], caps[i]) for i in order)
    return head, f'<div class="cards">{cards}</div>'


def search(state, query):
    blank = ["", None, "", None]
    if not state:
        return ('<div class="verdict">Press <b>Build the shared space</b> '
                'first.</div>', *blank)
    if not query.strip():
        return '<div class="verdict">Type something to look for.</div>', *blank

    caps, keys = state["caps"], state["keys"]
    seed = seed_index(query, caps)
    if seed < 0:
        return ('<div class="verdict">Nothing in these 800 photos matches that. '
                'Try one of the examples.</div>', *blank)

    G = state["G"]
    own = np.argsort(-(state["own"][seed:seed + 1] @ G.T)[0])[:TOPK]
    other = np.argsort(-(state["other"][seed:seed + 1] @ G.T)[0])[:TOPK]
    shared = len(set(own.tolist()) & set(other.tolist()))

    ga, gb = state["gallery"], state["querier"]
    verdict = (
        f'<div class="verdict">Both encoders were given the same request. '
        f'<b>{shared} of {TOPK}</b> results are the same photograph, and '
        f'{"the top hit is identical" if own[0] == other[0] else "the order differs slightly"}.'
        f'<br><span style="font-size:15px;font-weight:400;color:#a7a1cc">'
        f'{MAKER[gb]}\'s model never saw {MAKER[ga]}\'s index while training. '
        f'It is reading another company\'s memory.</span></div>')

    def strip(order, cls, who, what):
        head = (f'<div class="vendorbar {cls}">{what}: <b>{LABEL[who]}</b> '
                f'&nbsp;·&nbsp; {MAKER[who]}</div>')
        return head, [(image_of(keys[i]), caps[i][:70]) for i in order]

    ha, ia = strip(own, "bar-a", ga, "Index built and searched by")
    hb, ib = strip(other, "bar-b", gb, "Same index, searched by")
    return verdict, ha, ia, hb, ib


with gr.Blocks(title="Two rival models, one memory", css=CSS,
               theme=gr.themes.Soft(primary_hue="violet")) as demo:
    fitted = gr.State(None)
    omni_fitted = gr.State(None)
    gr.HTML(
        '<div class="hero"><h1>Two rival models, one memory</h1>'
        '<p>Four multimodal models, four different companies. One of them '
        'indexes 800 photographs. Then a <b>different company\'s model</b> '
        'searches that index, having never been trained to work with it.</p>'
        '<p>Across four vendors the cross-company hit rate is statistically '
        'indistinguishable from a model searching its own index: retention '
        '<b>0.988</b>, 95% CI <b>[0.955, 1.023]</b>.</p></div>')
    gr.HTML(hero_image() +
            '<p style="text-align:center;color:#7a7398;margin:2px 0 10px;'
            'font-size:14px">Four labs, four architectures, four training runs. '
            'Underneath, one shared arrangement of meaning.</p>')

    with gr.Tabs():
        with gr.Tab("Two rival models"):
            with gr.Row():
                g = gr.Dropdown(list(VENDORS), value="gemma4",
                                label="Who builds the index")
                q = gr.Dropdown(list(VENDORS), value="aria",
                                label="Who searches it")
            build = gr.Button("Build the shared space", variant="primary",
                              size="lg")
            verdict = gr.HTML()
            query = gr.Textbox(label="Ask for a photo", value=EXAMPLES[0])
            gr.Examples(EXAMPLES, inputs=query, label="Try one")
            go = gr.Button("Search across companies", variant="primary",
                           size="lg")
            head_a = gr.HTML()
            gal_a = gr.Gallery(columns=5, height=190, show_label=False,
                               object_fit="cover", preview=False)
            head_b = gr.HTML()
            gal_b = gr.Gallery(columns=5, height=190, show_label=False,
                               object_fit="cover", preview=False)

        with gr.Tab("One index, three media"):
            gr.Markdown(
                "Qwen3-Omni encodes photographs, sounds and video clips. "
                "A **single linear tower** places all three beside text in one "
                "index, and it beats a separate tower per modality on every "
                "modality: mixed gallery 0.2885 against 0.2667, and video "
                "0.2486 against 0.1730.\n\n"
                "Ask for something and see what comes back. The results are "
                "not filtered by kind."
            )
            obuild = gr.Button("Build one shared index", variant="primary",
                               size="lg")
            overdict = gr.HTML()
            oquery = gr.Textbox(label="Ask for anything", value=OMNI_EXAMPLES[0])
            gr.Examples(OMNI_EXAMPLES, inputs=oquery, label="Try one")
            ogo = gr.Button("Search all three", variant="primary", size="lg")
            ohead = gr.HTML()
            ocards = gr.HTML()

    pair_out = [verdict, head_a, gal_a, head_b, gal_b]
    # api_name=False everywhere: gr.State carrying a dict makes Gradio's schema
    # generator raise in get_type, /info then 500s and the page reports
    # "No API found".
    build.click(prepare, [g, q], [verdict, fitted], show_progress="minimal",
                api_name=False)
    go.click(search, [fitted, query], pair_out, api_name=False)
    query.submit(search, [fitted, query], pair_out, api_name=False)

    obuild.click(omni_prepare, None, [overdict, omni_fitted],
                 show_progress="minimal", api_name=False)
    ogo.click(omni_search, [omni_fitted, oquery], [ohead, ocards], api_name=False)
    oquery.submit(omni_search, [omni_fitted, oquery], [ohead, ocards],
                  api_name=False)

    gr.Markdown(
        "---\n"
        "**How this works.** Each model's frozen hidden states are read at 60% "
        "depth, centred per modality, and mapped into one 256-dimensional space "
        "by a small linear tower fitted live on CPU. Nothing about the models "
        "is fine-tuned.\n\n"
        "**Centering is not optional.** Raw cosine between unrelated items on "
        "these states is +0.869 and raw retrieval sits at chance.\n\n"
        "**Scope.** Images only here: two of the four hosts have no audio or "
        "video tower. The towers fitted in this Space use a plain numpy loop and "
        "score below the torch fit behind the published numbers, so read the "
        "comparison rather than the absolute rate. All four hosts train on "
        "overlapping web-scale corpora, so this shows the readable structure "
        "survives a change of vendor, architecture and training run. It does "
        "not show that independent minds would converge on it, and that reading "
        "is not asserted.\n\n"
        f"[States and code](https://huggingface.co/datasets/{STATES}) · "
        "[Towers](https://huggingface.co/RiverRider/srt-omni-xvendor-towers) · "
        "[Add your own model](https://huggingface.co/datasets/RiverRider/srt-omni-manifest)"
    )

demo.queue(default_concurrency_limit=2).launch()
