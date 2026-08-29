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
    ("RiverRider/srt-cxr14-probe", "space",
     "A 339 KB linear probe on frozen features reads chest radiographs, ahead "
     "of the fine-tuned baseline on the official split."),
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

MEDICAL = [
    ("RiverRider/srt-cxr14-probe", "space",
     "Read a held-out radiograph and see the probe scored against its own "
     "floor and its own shortcut. The 31B backbone never runs."),
    ("RiverRider/srt-cxr14-linear-probe", "model",
     "Linear(5376, 14) on frozen gemma-4-31B-it states. 0.7590 mean AUROC on "
     "the official split against 0.7451 for a fine-tuned ResNet-50."),
    ("RiverRider/srt-cxr14-frozen-probe", "dataset",
     "Frozen states for all 112,120 images, the official split, and the "
     "controls: shuffled floor, folded view-only baseline, patient bootstrap."),
]


def order(col, first_item_ids):
    """Put named items at the head. A visitor reads the first card, not the set."""
    by_id = {i.item_id: i for i in (col.items or [])}
    for pos, item_id in enumerate(first_item_ids):
        it = by_id.get(item_id)
        if it is None:
            print(f"    cannot order, not in collection: {item_id}")
            continue
        try:
            api.update_collection_item(col.slug, item_object_id=it.item_object_id,
                                       position=pos)
            print(f"    position {pos}: {item_id}")
        except Exception as e:
            print(f"    FAILED ordering {item_id}: {type(e).__name__} {str(e)[:90]}")


def build(title, description, items):
    # The Hub rejects descriptions at 150 characters, after the collection has
    # already been created on a retry. Catch it here instead.
    if len(description) > 150:
        raise SystemExit(f"description is {len(description)} chars, limit 150:\n"
                         f"  {description}")
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
    print("Medical imaging:")
    med = build(
        "Frozen features read chest radiographs",
        "A 339 KB linear probe on a frozen general-purpose backbone: 0.7590 on "
        "the official ChestX-ray14 split, against 0.7451 fine-tuned.",
        MEDICAL)
    order(api.get_collection(med.slug), [i[0] for i in MEDICAL])
    # Collection position 0 is what the profile leads with.
    api.update_collection_metadata(med.slug, position=0)
    print(f"  https://huggingface.co/collections/{med.slug}")

    print("Cross-vendor omni:")
    a = build(
        "Cross-vendor omni retrieval",
        "One company's model searches another's index at a rate "
        "indistinguishable from native: retention 0.988, 95% CI [0.955, 1.023].",
        RELEASE)
    api.update_collection_metadata(a.slug, position=1)
    print(f"  https://huggingface.co/collections/{a.slug}")

    print("Demos:")
    b = build(
        "SRT demos",
        "Read-outs on frozen models you can actually run. Nothing here "
        "fine-tunes the host.",
        DEMOS)
    order(api.get_collection(b.slug), ["RiverRider/srt-cxr14-probe"])
    api.update_collection_metadata(b.slug, position=2)
    print(f"  https://huggingface.co/collections/{b.slug}")
