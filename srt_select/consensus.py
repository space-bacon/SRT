"""Pick one reply out of K by what the candidates compute, not what they say.

Run every candidate on the same synthesised inputs and group them by the outputs
they produce. Candidates computing the same function land in the same cluster,
and the largest cluster is the pool's majority opinion. No ground truth, no
stated examples, no training, no scoring model.

Measured on the generations in `artifacts/nla/`, against a random-single-draw
floor and an any-candidate-passes oracle, every problem scored, unresolved pools
falling back to the first reply as `select()` does:

                       floor    selected   oracle
  HumanEval, 36 arms   0.4679   0.5854     0.7346
  MBPP, 10 arms        0.7185   0.8094     0.8887

Those two rows are given the benchmark's entry point and signature. Recovering
both from the chat turn alone costs coverage, not accuracy: HumanEval resolves
on 62.3% of problems and MBPP on 98.6%, and scoring the unresolved ones as an
arbitrary pick still leaves 0.5476 and 0.7991.

The gap between selected and oracle is what a better selector could still win.
The gap between floor and selected is what this one is worth.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .recover import code_of, infer_kinds, pick_entry, synth_cases
from .sandbox import probe_program, run_json


@dataclass
class Selection:
    """index is None when the pool could not be run; `reason` says why."""
    index: int | None
    entry: str | None = None
    kinds: list | None = None
    ran: int = 0
    cluster_size: int = 0
    clusters: int = 0
    reason: str | None = None

    @property
    def agreed(self) -> bool:
        """Did more than one candidate vote for the winner."""
        return self.cluster_size > 1


def choose(user_text: str, replies, n_cases: int = 2, timeout: float = 8.0,
           seed: int = 0) -> Selection:
    """Index of the reply the pool agrees with. Executes untrusted code.

    n_cases is 2 because the sweep from 2 to 48 moves accuracy at no K, while
    every extra input is another chance for a candidate to crash and be dropped.

    See `srt_select.sandbox` for what confinement this does and does not give.
    """
    codes = [code_of(r) for r in replies]

    entry = pick_entry(codes, user_text)
    if entry is None:
        return Selection(None, reason="no candidate defines a function")

    kinds = infer_kinds(codes, entry, user_text)
    if not kinds:
        return Selection(None, entry=entry, reason="could not infer arguments")

    cases = synth_cases(kinds, n_cases, seed)
    sigs = [run_json(probe_program(c, entry, cases), timeout) for c in codes]
    ran = [i for i, s in enumerate(sigs) if s is not None]
    if not ran:
        return Selection(None, entry=entry, kinds=kinds, reason="no candidate ran")

    top, votes = Counter(sigs[i] for i in ran).most_common(1)[0]
    return Selection(
        index=next(i for i in ran if sigs[i] == top),
        entry=entry, kinds=kinds, ran=len(ran), cluster_size=votes,
        clusters=len({sigs[i] for i in ran}),
    )


def select(user_text: str, replies, fallback: int = 0, **kw):
    """The chosen reply. Falls back to `replies[fallback]` when unresolved."""
    pick = choose(user_text, replies, **kw)
    return replies[pick.index if pick.index is not None else fallback]
