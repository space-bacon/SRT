"""Semiotic-Reflexive Transformer (SRT) — Adapter Architecture."""

__version__ = "0.1.0"


def _nla_factory():
    """Lazy entry point for the SRT-NLA verbalizer/reconstructor pair.

    Imported on demand so callers that only want the v3 adapter don't
    pay the cost of pulling in ``srt.nla`` (which lazily loads
    ``transformers`` even further).
    """
    from srt.nla import (  # noqa: F401  (re-export)
        ActivationReconstructor,
        ActivationVerbalizer,
        NLAConfig,
    )

    return {
        "ActivationReconstructor": ActivationReconstructor,
        "ActivationVerbalizer": ActivationVerbalizer,
        "NLAConfig": NLAConfig,
    }


HEAD_FACTORIES = {
    "nla": _nla_factory,
}
