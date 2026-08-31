"""Run model-written code without trusting it.

Every candidate this package executes was written by a language model in
response to a stranger's prompt. Treat it as hostile input, because a selector
that runs it is a remote code execution primitive if the confinement is wrong.

Confinement is four things, and all four matter:

  a fresh interpreter    -I, so no site-packages, no PYTHON* inheritance
  a resource ceiling     4 GB address space, 10 s CPU, zero bytes written
  a stripped environment PATH, HOME and nothing else
  a wall clock           the rlimits do not bound a sleeping process

Two things that ceiling does not do. `RLIMIT_FSIZE` of zero stops bytes reaching
a file, but an empty file can still be created, so it bounds damage rather than
preventing side effects. And a write that trips the limit at interpreter
shutdown is reported as `Exception ignored`, leaving the exit code at zero, so
a clean return does not prove the child behaved.

This is confinement, not a security boundary. It stops accidents and resource
exhaustion. It does not stop a determined attacker, and nothing that runs
untrusted code in-process can. Run it somewhere you are willing to lose.
"""
from __future__ import annotations

import json
import subprocess
import sys

# Applied inside the child before any generated code is parsed.
#
# Each limit is clamped to the inherited hard limit. Asking for more than the
# hard limit raises, and an unguarded fallback would be worse than a loose one,
# so never let the guard itself be the thing that fails. macOS in particular
# refuses a 4 GB RLIMIT_AS that Linux grants.
GUARD = (
    "import resource,os\n"
    "def _cap(res, want):\n"
    "    try:\n"
    "        soft, hard = resource.getrlimit(res)\n"
    "    except (ValueError, OSError):\n"
    "        return\n"
    "    if hard != resource.RLIM_INFINITY:\n"
    "        want = min(want, hard)\n"
    "    try:\n"
    "        resource.setrlimit(res, (want, hard))\n"
    "    except (ValueError, OSError):\n"
    "        pass\n"
    "_cap(resource.RLIMIT_AS, 4 << 30)\n"
    "_cap(resource.RLIMIT_CPU, 10)\n"
    "_cap(resource.RLIMIT_FSIZE, 0)\n"
    "os.environ['OMP_NUM_THREADS']='1'\n"
    "del _cap\n"
)

ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
}


def run(program: str, timeout: float = 8.0) -> subprocess.CompletedProcess | None:
    """Execute `program` under the guard. None if it did not finish cleanly."""
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", GUARD + program],
            capture_output=True, text=True, timeout=timeout, cwd="/tmp", env=ENV,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return p if p.returncode == 0 else None


def run_json(program: str, timeout: float = 8.0) -> tuple | None:
    """Last stdout line parsed as a JSON list. None if the program failed."""
    p = run(program, timeout)
    if p is None:
        return None
    lines = p.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return tuple(json.loads(lines[-1]))
    except (ValueError, TypeError):
        return None


def probe_program(body: str, entry: str, cases) -> str:
    """A program that prints `entry`'s output on each case.

    Exceptions are recorded rather than raised: two candidates agreeing on which
    inputs raise is evidence they compute the same function.
    """
    return (
        body + "\n\nimport json as _j\n_o = []\n"
        f"for _a in {cases!r}:\n"
        "    try:\n"
        f"        _o.append(repr({entry}(*_a)))\n"
        "    except Exception as _e:\n"
        "        _o.append('ERR:' + type(_e).__name__)\n"
        "print(_j.dumps(_o))\n"
    )
