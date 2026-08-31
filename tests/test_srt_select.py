"""The package is the deployable extraction of the harness in `scripts/`.

Two failures these pin against. The guard must not be the thing that fails: an
rlimit the platform refuses used to kill every child, so the selector reported
"no candidate ran" on a pool it should have resolved. And the package must not
drift from `scripts/chat_consensus.py`, which is the code the published numbers
came from.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from srt_select import choose  # noqa: E402
from srt_select.sandbox import run  # noqa: E402

PROMPT = "write is_even(n) returning True when n is even. is_even(4) is True."
MAJORITY_A = "def is_even(n):\n    return n % 2 == 0\n"
MAJORITY_B = "```python\ndef is_even(n):\n    return not (n & 1)\n```"
WRONG = "def is_even(n):\n    return n % 2 == 1\n"
UNPARSEABLE = "def is_even(n):\n    return n /// 2\n"


def test_guard_allows_ordinary_code():
    assert run("print(1)") is not None


def test_guard_stops_a_cpu_burn():
    assert run("while True: pass", timeout=3.0) is None


def test_guard_writes_no_bytes(tmp_path):
    target = tmp_path / "written"
    run(f"open({str(target)!r}, 'w').write('x' * 4096)")
    assert not target.exists() or target.stat().st_size == 0


def test_majority_wins_over_a_lone_wrong_answer():
    pick = choose(PROMPT, [WRONG, MAJORITY_A, UNPARSEABLE, MAJORITY_B])
    assert pick.index in (1, 3)
    assert pick.cluster_size == 2
    assert pick.agreed


def test_candidates_that_do_not_run_are_excluded_not_chosen():
    pick = choose(PROMPT, [UNPARSEABLE, MAJORITY_A, MAJORITY_B])
    assert pick.ran == 2
    assert pick.index != 0


def test_unresolvable_pool_reports_why():
    pick = choose("say hello", ["hello", "hi there"])
    assert pick.index is None
    assert pick.reason


def test_entry_point_from_the_user_beats_a_more_common_helper():
    asked = "def solve(xs):\n    ..."
    body = ("def helper(x):\n    return x\n\n"
            "def solve(xs):\n    return sorted(xs)\n")
    pick = choose(asked, [body, body])
    assert pick.entry == "solve"


def test_package_agrees_with_the_harness_it_was_extracted_from():
    chat_consensus = pytest.importorskip("chat_consensus")
    replies = [WRONG, MAJORITY_A, UNPARSEABLE, MAJORITY_B]
    harness_index, _ = chat_consensus.choose(PROMPT, replies)
    assert choose(PROMPT, replies).index == harness_index
