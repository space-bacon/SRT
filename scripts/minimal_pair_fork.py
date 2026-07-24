#!/usr/bin/env python3
"""Bat x community minimal-pair "fork" test (Erik Otarola-Castillo design).

Hold a polysemous token fixed and vary ONLY whether its two preceding agents come
from ONE discourse community (congruent: one interpretant basin) or TWO different
communities (fork: two basins compete). The MAH divergence is defined as the pull
of context against a token's default reading, so it should be HIGHER in the fork.

Design controls: both sentences are "A and B <verb> the <TOKEN>." with identical
length, structure, and token position; only the community membership of A and B
differs. We also read the base-LM next-token entropy at the token, to show the
divergence registers a fork that the fluency signal does not.

    /venv/main/bin/python scripts/minimal_pair_fork.py --repo RiverRider/srt-adapter-v1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import (
    BENConfig,
    CommunityConfig,
    LossConfig,
    MAHConfig,
    RRMConfig,
    SRTConfig,
)

# token, fork (two rival community agents) vs congruent (same-community agents); identical frame.
# No agent name contains the target substring, so the target is the last token in every sentence.
PAIRS = [
    dict(token="bat", fork="The wildlife biologist and the baseball coach argued about the bat.",
         cong="The wildlife biologist and the cave researcher argued about the bat."),
    dict(token="bank", fork="The loan officer and the fly fisher waited beside the bank.",
         cong="The loan officer and the mortgage broker waited beside the bank."),
    dict(token="pitch", fork="The baseball scout and the choir director discussed the pitch.",
         cong="The choir director and the music teacher discussed the pitch."),
    dict(token="current", fork="The electrician and the kayaker warned about the current.",
         cong="The kayaker and the river guide warned about the current."),
    dict(token="mole", fork="The spy handler and the gardener mentioned the mole.",
         cong="The gardener and the pest controller mentioned the mole."),
    dict(token="star", fork="The astronomer and the film producer pointed to the star.",
         cong="The astronomer and the astrophysicist pointed to the star."),
    dict(token="mouse", fork="The zoologist and the programmer grabbed the mouse.",
         cong="The programmer and the software developer grabbed the mouse."),
    dict(token="crane", fork="The birdwatcher and the site foreman watched the crane.",
         cong="The site foreman and the tower rigger watched the crane."),
    dict(token="seal", fork="The marine biologist and the notary examined the seal.",
         cong="The notary and the office clerk examined the seal."),
    dict(token="virus", fork="The immunologist and the IT technician traced the virus.",
         cong="The IT technician and the software engineer traced the virus."),
    dict(token="organ", fork="The surgeon and the church musician admired the organ.",
         cong="The church musician and the choir director admired the organ."),
    dict(token="tank", fork="The army general and the plumber inspected the tank.",
         cong="The plumber and the maintenance worker inspected the tank."),
    dict(token="cell", fork="The microbiologist and the prison warden described the cell.",
         cong="The prison warden and the correctional officer described the cell."),
    dict(token="plant", fork="The botanist and the factory manager toured the plant.",
         cong="The factory manager and the site engineer toured the plant."),
    dict(token="bark", fork="The dog trainer and the arborist noticed the bark.",
         cong="The arborist and the tree surgeon noticed the bark."),
    dict(token="bass", fork="The angler and the guitarist listened for the bass.",
         cong="The guitarist and the music producer listened for the bass."),
    dict(token="bow", fork="The archer and the violinist adjusted the bow.",
         cong="The archer and the target shooter adjusted the bow."),
    dict(token="draft", fork="The brewer and the army recruiter mentioned the draft.",
         cong="The brewer and the bartender mentioned the draft."),
    dict(token="fan", fork="The HVAC technician and the pop singer greeted the fan.",
         cong="The HVAC technician and the cooling engineer greeted the fan."),
    dict(token="field", fork="The physicist and the farmer surveyed the field.",
         cong="The farmer and the crop scientist surveyed the field."),
    dict(token="jam", fork="The traffic reporter and the jazz musician got into the jam.",
         cong="The jazz musician and the session drummer got into the jam."),
    dict(token="key", fork="The locksmith and the pianist searched for the key.",
         cong="The locksmith and the security installer searched for the key."),
    dict(token="match", fork="The boxing referee and the dating coach set up the match.",
         cong="The boxing referee and the wrestling coach set up the match."),
    dict(token="note", fork="The pianist and the banker studied the note.",
         cong="The banker and the bank cashier studied the note."),
    dict(token="pool", fork="The lifeguard and the gambler eyed the pool.",
         cong="The lifeguard and the swim coach eyed the pool."),
    dict(token="port", fork="The harbor master and the system administrator checked the port.",
         cong="The harbor master and the dockworker checked the port."),
    dict(token="ring", fork="The jeweler and the boxing promoter inspected the ring.",
         cong="The jeweler and the goldsmith inspected the ring."),
    dict(token="rock", fork="The geologist and the guitarist talked about the rock.",
         cong="The geologist and the mineralogist talked about the rock."),
    dict(token="scale", fork="The fisherman and the pianist studied the scale.",
         cong="The pianist and the music teacher studied the scale."),
    dict(token="spring", fork="The hydrologist and the watchmaker examined the spring.",
         cong="The watchmaker and the clock repairer examined the spring."),
    dict(token="trunk", fork="The forester and the car mechanic inspected the trunk.",
         cong="The forester and the arborist inspected the trunk."),
    dict(token="wave", fork="The surfer and the physicist described the wave.",
         cong="The physicist and the acoustics engineer described the wave."),
    dict(token="chip", fork="The poker dealer and the hardware engineer examined the chip.",
         cong="The hardware engineer and the circuit designer examined the chip."),
    dict(token="club", fork="The party planner and the golf pro talked about the club.",
         cong="The golf pro and the golf caddie talked about the club."),
    dict(token="court", fork="The trial judge and the tennis coach walked onto the court.",
         cong="The tennis coach and the tennis umpire walked onto the court."),
    dict(token="deck", fork="The ship captain and the card dealer cleared the deck.",
         cong="The ship captain and the ship's mate cleared the deck."),
    dict(token="file", fork="The paralegal and the system administrator opened the file.",
         cong="The system administrator and the IT technician opened the file."),
    dict(token="mint", fork="The herbalist and the coin engraver visited the mint.",
         cong="The coin engraver and the treasury official visited the mint."),
    dict(token="pupil", fork="The schoolteacher and the eye doctor watched the pupil.",
         cong="The schoolteacher and the school principal watched the pupil."),
    dict(token="root", fork="The botanist and the mathematician explained the root.",
         cong="The botanist and the plant nursery owner explained the root."),
    dict(token="stamp", fork="The philatelist and the metal worker pressed the stamp.",
         cong="The philatelist and the postal clerk pressed the stamp."),
    dict(token="vault", fork="The gymnast and the bank guard approached the vault.",
         cong="The bank guard and the security teller approached the vault."),
]


def build_config(config_path) -> SRTConfig:
    raw = json.loads(Path(config_path).read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(**{k: v for k, v in raw["loss"].items() if k in LossConfig.__dataclass_fields__}),
    )


@torch.no_grad()
def measure(model, tok, text, token, device):
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=64)
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    D = out.divergences[-1][0].norm(dim=-1).float().cpu()
    lo = text.lower().rfind(token.lower())
    hi = lo + len(token)
    idxs = [i for i, (a, b) in enumerate(offs) if b > a and a < hi and b > lo]
    d_tok = float(D[idxs].mean())
    pos = int(idxs[-1])
    logits = out.logits[0, idxs[-1]].float()
    p = torch.softmax(logits, dim=-1)
    ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    return d_tok, ent, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/srt-adapter-v1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="artifacts/nla/minimal_pair_fork_powered.json")
    args = ap.parse_args()

    cfg = build_config(hf_hub_download(args.repo, "config.json"))
    model = SRTAdapter(cfg).to(args.device)
    model.load_state_dict(load_file(hf_hub_download(args.repo, "adapter.safetensors"), device=args.device), strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)

    print(f"repo={args.repo}   n_pairs={len(PAIRS)}\n")
    print(f"{'token':10s} {'D_fork':>7s} {'D_cong':>7s} {'dD':>8s}   {'H_fork':>7s} {'H_cong':>7s} {'dH':>7s}")
    print("-" * 66)
    rows, dD, dH, dP = [], [], [], []
    for it in PAIRS:
        df, hf, pf = measure(model, tok, it["fork"], it["token"], args.device)
        dc, hc, pc = measure(model, tok, it["cong"], it["token"], args.device)
        dD.append(df - dc)
        dH.append(hf - hc)
        dP.append(pf - pc)
        rows.append(dict(token=it["token"], d_fork=df, d_cong=dc, h_fork=hf, h_cong=hc,
                         pos_fork=pf, pos_cong=pc))
        print(f"{it['token']:10s} {df:7.3f} {dc:7.3f} {df - dc:+8.3f}   {hf:7.3f} {hc:7.3f} {hf - hc:+7.3f}")

    dD = np.asarray(dD); dH = np.asarray(dH); dP = np.asarray(dP, dtype=float)
    n = len(dD)
    mD, mH, mP = float(dD.mean()), float(dH.mean()), float(dP.mean())
    wins = int((dD > 0).sum())
    print(f"\nmean dD (fork - congruent) = {mD:+.3f}   ({wins}/{n} tokens fork > congruent)")
    print(f"mean dH (next-token entropy) = {mH:+.3f}")
    print(f"mean dPos (target token index) = {mP:+.3f}")

    rng = np.random.default_rng(0)
    P = 20000
    S = rng.integers(0, 2, size=(P, n)) * 2 - 1  # +/-1 paired sign flips

    # ---- raw (unadjusted) paired sign-flip permutation on dD ----
    raw_null = (S * dD).mean(axis=1)
    p_raw = (1 + int((np.abs(raw_null) >= abs(mD)).sum())) / (P + 1)
    print(f"\n[raw]       mean dD = {mD:+.4f}   sign-flip p = {p_raw:.4f}")

    # ---- decoupled: ANCOVA intercept of dD ~ dH + dPos (equivalent to a pair-fixed-effects
    #      model of D on a fork dummy with entropy and token-position as covariates). The
    #      intercept is the fork effect after partialling out the entropy and position nuisances.
    X = np.column_stack([np.ones(n), dH, dP])
    beta, *_ = np.linalg.lstsq(X, dD, rcond=None)
    b0, bH, bP = float(beta[0]), float(beta[1]), float(beta[2])
    rH = float(np.corrcoef(dD, dH)[0, 1])
    rP = float(np.corrcoef(dD, dP)[0, 1])

    # permutation null for the adjusted intercept: flipping a pair negates dD, dH, dPos together.
    b0_null = np.empty(P)
    for i in range(P):
        s = S[i]
        Xi = np.column_stack([np.ones(n), s * dH, s * dP])
        bi, *_ = np.linalg.lstsq(Xi, s * dD, rcond=None)
        b0_null[i] = bi[0]
    p_adj = (1 + int((np.abs(b0_null) >= abs(b0)).sum())) / (P + 1)

    # bootstrap 95% CI for the adjusted intercept (resample pairs with replacement)
    B = 5000
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        Xi = np.column_stack([np.ones(n), dH[idx], dP[idx]])
        bi, *_ = np.linalg.lstsq(Xi, dD[idx], rcond=None)
        boot[i] = bi[0]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    print(f"[decoupled] intercept b_fork = {b0:+.4f}   (entropy + position adjusted)")
    print(f"            slope b_H = {bH:+.4f}   slope b_Pos = {bP:+.4f}")
    print(f"            corr(dD,dH) = {rH:+.3f}   corr(dD,dPos) = {rP:+.3f}")
    print(f"            permutation p (b_fork) = {p_adj:.4f}")
    print(f"            bootstrap 95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # ---- entropy-matched subset: pairs whose |dH| <= median(|dH|) ----
    med = float(np.median(np.abs(dH)))
    mask = np.abs(dH) <= med
    dDm = dD[mask]
    nm = int(len(dDm))
    mDm = float(dDm.mean())
    Sm = rng.integers(0, 2, size=(P, nm)) * 2 - 1
    null_m = (Sm * dDm).mean(axis=1)
    p_match = (1 + int((np.abs(null_m) >= abs(mDm)).sum())) / (P + 1)
    print(f"\n[matched]   |dH| <= {med:.3f}: n={nm}  mean dD = {mDm:+.4f}  sign-flip p = {p_match:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(repo=args.repo, n=n, mean_dD=mD, mean_dH=mH, mean_dPos=mP, wins=wins,
                       p_raw=p_raw, b_fork=b0, b_H=bH, b_Pos=bP,
                       corr_dD_dH=rH, corr_dD_dPos=rP, p_adj=p_adj, boot_ci=ci,
                       matched=dict(median_absdH=med, n=nm, mean_dD=mDm, p=p_match),
                       rows=rows), f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
