"""Choose among K model replies to a coding request, using only the replies.

    from srt_select import select
    best = select(user_message, replies)

Measured gains, sandbox caveats and the coverage limit are documented in
`srt_select.consensus` and `srt_select.sandbox`. Read the sandbox note before
pointing this at anything you care about: selecting requires executing every
candidate.
"""
from .consensus import Selection, choose, select
from .sandbox import GUARD, probe_program, run, run_json

__all__ = ["Selection", "choose", "select", "GUARD", "probe_program", "run", "run_json"]
__version__ = "0.1.0"
