"""Extract the tests a caller can see from a prompt alone, across HumanEval's formats.

The first version of this only read doctest blocks whose expected value parsed with
ast.literal_eval, which covered 54 of 164 problems. HumanEval states its examples in
at least four notations, and 131 problems carry a machine-readable one. Under-reading
them silently handicaps any execution-guided baseline, which biases a comparison in
favour of whatever it is being compared against.

Validity check is not optional: every extracted test must pass on the canonical
solution. A test that fails there is a parser bug, not a model failure.
"""
import ast
import doctest
import re

# Injected into the candidate program. doctest reports repr(result), prose examples
# usually give a literal, and some give neither cleanly, so accept any of the three.
HELPER = """
def _chk(got, want_lit, want_str, has_lit):
    if has_lit:
        try:
            if got == want_lit:
                return True
        except Exception:
            pass
        if isinstance(got, float) and isinstance(want_lit, (int, float)):
            return abs(got - want_lit) < 1e-6
    g = repr(got).strip()
    s = want_str.strip()
    return g == s or str(got).strip() == s or g.replace("'", '"') == s.replace("'", '"')
"""

# Longest first: `==` would otherwise match inside `==>` and leave `> 2` as the value.
ARROWS = r"==>|=>|==|\u279e|\u27a1|->"
WORDS = {"true": "True", "false": "False", "none": "None", "null": "None",
         "nil": "None"}


def _is_literal(text):
    try:
        ast.literal_eval(text)
        return True
    except Exception:
        return False


def _concrete(call):
    """True if every argument is a literal, so `fib4(n) == fib4(n-1)+...` is rejected."""
    try:
        tree = ast.parse(call, mode="eval")
    except SyntaxError:
        return False
    called = {id(n.func) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return not any(isinstance(n, ast.Name) and id(n) not in called
                   for n in ast.walk(tree))


def _strip_quotes(seg):
    seg = re.sub(r"^[rRbBuUfF]{0,2}", "", seg.strip())
    for q in ('"""', "'''", '"', "'"):
        if seg.startswith(q) and seg.endswith(q) and len(seg) >= 2 * len(q):
            return seg[len(q):-len(q)]
    return seg


def _docstring(prompt, entry):
    """Raw docstring source.

    ast.get_docstring returns the *evaluated* string, which turns a literal \\n
    inside an example into a real newline and destroys the doctest block. The raw
    source segment is what doctest needs to see.
    """
    best = None
    for suffix in ("", "\n    pass\n", "    pass\n"):
        try:
            tree = ast.parse(prompt + suffix)
        except SyntaxError:
            continue
        src = prompt + suffix
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if not (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                continue
            seg = ast.get_source_segment(src, body[0].value)
            if not seg:
                continue
            doc = _strip_quotes(seg)
            if node.name == entry:
                return doc
            if best is None:
                best = doc
        if best is not None:
            return best
    return prompt


def _pairs(prompt, entry):
    """(call_source, expected_text) visible in the prompt."""
    body = _docstring(prompt, entry)
    out = []
    try:
        examples = doctest.DocTestParser().get_examples(body)
    except ValueError:
        # Some docstrings have inconsistent indentation; regex still works on them.
        examples = []
    for ex in examples:
        call, want = ex.source.strip(), ex.want.strip()
        if want and entry in call:
            out.append((call, want))
    for m in re.finditer(
            rf"({re.escape(entry)}\s*\([^\n]*?\))\s*(?:#\s*)?(?:{ARROWS})\s*([^\n]+)", body):
        want = m.group(2).strip().rstrip(".,;").strip()
        if want:
            out.append((m.group(1).strip(), want))
    seen, uniq = set(), []
    for c, w in out:
        w = re.sub(r'\s*(?:"""|\'\'\')\s*$', "", w).strip()
        w = WORDS.get(w.lower(), w)
        if "=" in w and not _is_literal(w):
            # Derivation style: "19 - 5 - 6 = 8" states the value after the last =.
            tail = w.rsplit("=", 1)[-1].strip()
            if tail:
                w = tail
        if w and _concrete(c) and (c, w) not in seen:
            seen.add((c, w))
            uniq.append((c, w))
    return uniq


def visible_tests(prompt, entry):
    """Assert lines derivable from the prompt, using _chk from HELPER."""
    lines = []
    for call, want in _pairs(prompt, entry):
        try:
            ast.parse(call, mode="eval")
        except SyntaxError:
            continue
        try:
            lit = ast.literal_eval(want)
            has_lit = True
        except Exception:
            lit, has_lit = None, False
        lit_src = repr(lit) if has_lit else "None"
        # Parenthesised on its own line so a trailing # comment cannot swallow the rest.
        lines.append(f"_v = (\n{call}\n)\nassert _chk(_v, {lit_src}, {want!r}, {has_lit})")
    return lines


def validated(prompt, entry, canonical, runner, timeout=8.0):
    """Keep only tests the canonical solution passes.

    HumanEval docstrings contain genuine errors (problem 47 claims a median of 15.0
    where it is 8.0). A deployer could not know that, so filtering here is generous
    to any execution-guided baseline. That is the safe direction: it strengthens the
    competitor rather than the method being tested.
    """
    keep = []
    for t in visible_tests(prompt, entry):
        if runner(prompt + canonical + "\n" + HELPER + "\n" + t, timeout):
            keep.append(t)
    return keep
