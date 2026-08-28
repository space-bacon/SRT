#!/usr/bin/env python3
"""Put the strongest work in collections so the profile leads with it.

Hugging Face surfaces collections on the profile, which is the closest thing to
pinning. Order matters: the first item is what a visitor sees.
"""
from huggingface_hub import HfApi, get_token

api = HfApi(token=get_token())
NS = "RiverRider"

RELEASE = [
    ("RiverRider/srt-omni-demo", "space",
     "Pick two companies. One indexes photographs, the other searches that "
     "index. Second tab puts photos, sounds and video clips in one index."),
    ("RiverRider/srt-omni-crossvendor-states", "dataset",
     "Frozen states for the same 7000 items through four multimodal hosts, "
     "plus the fitted towers, results and scripts."),
    ("RiverRider/srt-omni-xvendor-towers", "model",
     "The 2 and 4 vendor towers. Retention 0.988, 95% CI [0.955, 1.023]."),
    ("RiverRider/srt-omni-shared-tower", "model",
     "One linear tower placing image, audio and video beside text. Beats one "
     "tower per modality on every modality."),
    ("RiverRider/srt-omni-manifest", "dataset",
     "The item list on its own, so a fifth vendor can be added without "
     "re-deriving the setup."),
]

DEMOS = [
    ("RiverRider/srt-omni-demo", "space",
     "Four companies, one searchable memory. The newest and the strongest."),
    ("RiverRider/0.6b-reads-27b", "space",
     "A 0.6B model puts words to a 31B's raw internal state, scored against an "
     "index it did not build."),
    ("RiverRider/srt-browser-demo", "space",
     "The read-out running on the visitor's own hardware."),
    ("RiverRider/srt-introspect", "space",
     "Read the state of a frozen model while it runs."),
    ("RiverRider/srt-sunstone", "space",
     "Gemma-4 31B with an SRT read-out attached."),
    ("RiverRider/srt-showcase", "space",
     "The SRT program end to end."),
    ("RiverRider/srt-nla-demo", "space",
     "Natural language access to a frozen model's activations."),
    ("RiverRider/srt-adapter-v8a-demo", "space",
     "The adapter that started the line."),
]


def build(title, description, items):
    col = api.create_collection(title=title, namespace=NS,
                                description=description, exists_ok=True)
    have = {i.item_id for i in (col.items or [])}
    for item_id, kind, note in items:
        if item_id in have:
            print(f"    already in: {item_id}")
            continue
        try:
            api.add_collection_item(col.slug, item_id=item_id, item_type=kind,
                                    note=note, exists_ok=True)
            print(f"    added {kind:8s} {item_id}")
        except Exception as e:
            print(f"    FAILED {item_id}: {type(e).__name__} {str(e)[:90]}")
    return col


if __name__ == "__main__":
    print("Cross-vendor omni:")
    a = build(
        "Cross-vendor omni retrieval",
        "One company's model searches another's index at a rate "
        "indistinguishable from native: retention 0.988, 95% CI [0.955, 1.023].",
        RELEASE)
    print(f"  https://huggingface.co/collections/{a.slug}")

    print("Demos:")
    b = build(
        "SRT demos",
        "Read-outs on frozen models you can actually run. Nothing here "
        "fine-tunes the host.",
        DEMOS)
    print(f"  https://huggingface.co/collections/{b.slug}")
