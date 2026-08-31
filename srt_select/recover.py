"""Recover what the benchmark would have told you, from the candidates alone.

A benchmark harness is handed an entry point, a signature and a test suite. A
chat assistant is handed a user message and K replies, and has to pick one
before anyone knows which is right. Everything the selector needs is therefore
recovered by agreement among the candidates:

  which function is the answer   the name most candidates define
  what arguments it takes        the majority signature, or a call in the user's text
  what to run it on              values synthesised to match those argument kinds

None of this reads a test suite or a reference solution.
"""
from __future__ import annotations

import ast
import random
import re
from collections import Counter

FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|\Z)", re.S)

POOL = {
    "int": [0, 1, 2, 3, 5, 7, 10, -1, -4, 100],
    "float": [0.0, 0.5, 1.0, 2.5, -1.5, 3.14],
    "str": ["", "a", "abc", "Hello", "abcdef", "a b c", "AaBb", "xyzxyz"],
    "bool": [True, False],
}

CONTAINERS = ("list", "tuple", "dict", "set")
SCALARS = ("int", "float", "str", "bool")


def code_of(reply: str) -> str:
    """The runnable part of a chat reply: its longest fenced block, else all of it."""
    blocks = FENCE.findall(reply)
    return max(blocks, key=len) if blocks else reply


def defs(code: str) -> list:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def pick_entry(codes, user_text: str = "") -> str | None:
    """The function name the pool agrees it is writing.

    A name the user themselves defined wins over a more popular helper, since a
    pool can define `_helper` more often than the function actually asked for.
    """
    asked = [n.name for n in defs(user_text)]
    names = Counter()
    for c in codes:
        for n in defs(c):
            names[n.name] += 1
    if not names:
        return None
    for a in asked:
        if a in names:
            return a
    return names.most_common(1)[0][0]


def annotation_kind(node) -> str | None:
    if node is None:
        return None
    try:
        text = ast.unparse(node).lower()
    except Exception:
        return None
    for k in CONTAINERS:
        if text.startswith(k):
            inner = next((t for t in ("str", "float", "int") if t in text[len(k):]), "int")
            return f"{k}[{inner}]"
    return next((t for t in SCALARS if t in text), None)


def kind_of(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple, set)):
        inner = kind_of(next(iter(value))) if value else "int"
        return f"{type(value).__name__}[{inner}]"
    if isinstance(value, dict):
        return "dict[str]"
    return "any"


def sample(kind: str, rng: random.Random):
    if kind.startswith(("list", "tuple", "set")):
        inner = kind[kind.index("[") + 1:-1]
        vals = [rng.choice(POOL.get(inner, POOL["int"])) for _ in range(rng.randint(0, 5))]
        return tuple(vals) if kind.startswith("tuple") else vals
    if kind.startswith("dict"):
        return {rng.choice(POOL["str"]): rng.choice(POOL["int"])
                for _ in range(rng.randint(0, 3))}
    if kind in POOL:
        return rng.choice(POOL[kind])
    return rng.choice(POOL["int"] + POOL["str"])


def infer_kinds(codes, entry: str, user_text: str = "") -> list[str]:
    """Argument kinds, from an example call in the user's text if there is one.

    An example the user wrote is better evidence than any annotation, because it
    is the one input we know the asked-for function must accept.
    """
    for m in re.finditer(rf"{re.escape(entry)}\s*\(([^\n)]*)\)", user_text):
        try:
            args = ast.literal_eval("(" + m.group(1) + ",)")
        except Exception:
            continue
        if args:
            return [kind_of(v) for v in args]

    arity_votes, annotated = Counter(), {}
    for c in codes:
        for n in defs(c):
            if n.name != entry:
                continue
            arity_votes[len(n.args.args)] += 1
            for a in n.args.args:
                k = annotation_kind(a.annotation)
                if k:
                    annotated.setdefault(a.arg, k)
    if not arity_votes:
        return []

    arity = arity_votes.most_common(1)[0][0]
    for c in codes:
        for n in defs(c):
            if n.name == entry and len(n.args.args) == arity:
                return [annotated.get(a.arg) or annotation_kind(a.annotation) or "any"
                        for a in n.args.args]
    return []


def synth_cases(kinds, n_cases: int, seed: int = 0) -> list[tuple]:
    rng = random.Random(seed)
    return [tuple(sample(k, rng) for k in kinds) for _ in range(n_cases)]
