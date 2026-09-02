"""Write docs/paper_hivemind_hf.md from paper_hivemind.md.

The HF article editor needs absolute image URLs and single-line captions. Nothing
else differs from the source, so the source stays the only thing anyone edits.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper_hivemind.md"
DST = ROOT / "docs" / "paper_hivemind_hf.md"
RAW = "https://raw.githubusercontent.com/space-bacon/SRT/main/"


def convert(text: str) -> str:
    def fix(m):
        alt = " ".join(m.group(1).split())
        target = m.group(2).strip()
        if not target.startswith(("http://", "https://")):
            target = RAW + target.lstrip("./")
        return f"![{alt}]({target})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]*)\)", fix, text, flags=re.S)


if __name__ == "__main__":
    out = convert(SRC.read_text())
    DST.write_text(out)
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]*)\)", out)
    print(f"wrote {DST.relative_to(ROOT)}: {len(imgs)} images")
    for u in imgs:
        print("  ", u)
