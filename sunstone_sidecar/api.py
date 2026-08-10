"""The Sidecar: three verbs on top of a tap, a head, and an index."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from .heads import load_head, project
from .index import Index
from .taps import TransformersTap, read_image_vector, read_text_vector

if TYPE_CHECKING:
    from .mlx_tap import MLXTap
    AnyTap = Union[TransformersTap, "MLXTap"]

# Default tag vocabulary: the 80 COCO categories, the set on which the
# whole-scene inventory reading was measured (detection AUC 0.883,
# per-image R-precision 14x chance; artifacts/nla/q4/inventory_A_multilabel.json)
COCO_VOCAB = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class Sidecar:
    """Search, tag, and similarity from a VLM's hidden states.

    Two ways in:
      Sidecar.from_pretrained("google/gemma-4-31B-it")   loads a backbone
      Sidecar.attach(model, processor, backbone_name)    reuses yours

    And the zero-marginal path, no tap needed:
      side.read(out.hidden_states, input_ids=...)        image vector from
                                                         a pass you already ran
    """

    def __init__(self, head: dict, tap: "AnyTap | None",
                 backbone: str):
        self.head = head
        self.tap = tap
        self.backbone = backbone
        self.index = Index(head["proj_dim"])
        self._vocab_z: np.ndarray | None = None
        self._vocab: list[str] | None = None

    # -------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, backbone: str, quant4: bool = False,
                        **kw) -> "Sidecar":
        head = load_head(backbone, quant4=quant4)
        tap = TransformersTap.from_pretrained(backbone, head["layer"],
                                              quant4=quant4, **kw)
        return cls(head, tap, backbone)

    @classmethod
    def attach(cls, model, processor, backbone: str) -> "Sidecar":
        """Attach to the backbone your service already has resident."""
        head = load_head(backbone)
        tap = TransformersTap.attach(model, processor, head["layer"])
        return cls(head, tap, backbone)

    @classmethod
    def from_mlx(cls, model_id: str, backbone: str | None = None) -> "Sidecar":
        """Apple Silicon tier via mlx-vlm (e.g.
        'mlx-community/gemma-4-31b-it-4bit'). `backbone` names the head
        to load when the mlx repo id differs from the registry key;
        run calibrate() after construction for full cross-runtime
        accuracy (the 42KB fix)."""
        from .mlx_tap import MLXTap
        head = load_head(backbone or model_id)
        tap = MLXTap.from_pretrained(model_id, head["layer"])
        return cls(head, tap, backbone or model_id)

    @classmethod
    def headless(cls, backbone: str) -> "Sidecar":
        """No backbone at all: for read()-only integration or working
        over vectors produced elsewhere."""
        return cls(load_head(backbone), None, backbone)

    # ------------------------------------------------------------ encoding
    def read(self, hidden_states, input_ids=None,
             attention_mask=None) -> np.ndarray:
        """Zero-marginal read from a forward pass you already ran with
        output_hidden_states=True. Give input_ids for an image pass,
        attention_mask for a text pass. Returns projected vector(s)."""
        if input_ids is not None:
            img_tok = self.tap.image_token_id if self.tap else None
            if img_tok is None:
                raise ValueError("read() for images without a tap needs "
                                 "Sidecar.attach or from_pretrained")
            v = read_image_vector(hidden_states, input_ids,
                                  self.head["layer"], img_tok)
            return project(v, self.head["W_img"], self.head["b_img"],
                           self.head["mu_img"])
        if attention_mask is not None:
            v = read_text_vector(hidden_states, attention_mask,
                                 self.head["layer"])
            return project(v, self.head["W_txt"], self.head["b_txt"],
                           self.head["mu_txt"])
        raise ValueError("pass input_ids (image pass) or attention_mask "
                         "(text pass)")

    def encode_image(self, img) -> np.ndarray:
        tap = self._need_tap()
        v = tap.image_vector(img)
        return project(v, self.head["W_img"], self.head["b_img"],
                       self.head["mu_img"])

    def encode_text(self, texts: str | list[str]) -> np.ndarray:
        tap = self._need_tap()
        single = isinstance(texts, str)
        batch: list[str] = [texts] if isinstance(texts, str) else texts
        v = tap.text_vectors(batch)
        z = project(v, self.head["W_txt"], self.head["b_txt"],
                    self.head["mu_txt"])
        return z[0] if single else z

    # --------------------------------------------------------- three verbs
    def index_images(self, items) -> int:
        """items: iterable of (key, PIL.Image) or file paths."""
        from PIL import Image
        n = 0
        for it in items:
            if isinstance(it, tuple):
                key, img = it
            else:
                key, img = str(it), Image.open(it).convert("RGB")
            self.index.add(key, self.encode_image(img))
            n += 1
        return n

    def search(self, query: str, k: int = 8,
               shape_query: bool = True) -> list[tuple[str, float]]:
        """Text-to-image search over the index. LLM-grade queries.

        The head speaks caption: short keyword queries ("bear") land
        outside its text distribution and score near zero, while
        caption-shaped queries score decisively. With shape_query=True
        (default), queries of three words or fewer are wrapped as
        "a photo of {query}". Pass shape_query=False to send your text
        verbatim."""
        if shape_query and len(query.split()) <= 3:
            query = f"a photo of {query}"
        return self.index.search(self.encode_text(query), k=k)

    def similar(self, img, k: int = 8) -> list[tuple[str, float]]:
        """Image-to-image similarity over the index (dedup, clustering)."""
        return self.index.search(self.encode_image(img), k=k)

    def tag(self, img, vocab: list[str] | None = None, k: int = 10,
            template: str = "a photo of a {}") -> list[tuple[str, float]]:
        """Whole-scene inventory: rank a tag vocabulary against the image.

        The measured regime: object identity (what is present) at
        detection AUC 0.883 over this default vocabulary. Small
        background objects rank lowest; arrangement is out of scope."""
        vocab = vocab or COCO_VOCAB
        if self._vocab is not vocab or self._vocab_z is None:
            self._vocab_z = self.encode_text([template.format(t)
                                              for t in vocab])
            self._vocab = vocab
        zi = self.encode_image(img)
        sims = self._vocab_z @ zi
        order = np.argsort(-sims)[:k]
        return [(vocab[i], float(sims[i])) for i in order]

    # --------------------------------------------------------- calibration
    def calibrate(self, images=None, texts=None,
                  save_to: str | None = None) -> dict:
        """Measure this runtime's modality means on your own data and
        swap them into the head (the 42KB cross-runtime fix; ~256
        samples recommended). Returns the calibration dict."""
        from .calibrate import apply_calibration, measure_means, \
            save_calibration
        cal = measure_means(self._need_tap(), images=images, texts=texts)
        self.head = apply_calibration(self.head, cal)
        self._vocab_z = None      # vocab cache used old means
        if save_to:
            save_calibration(cal, save_to)
        return cal

    def load_calibration(self, path: str) -> None:
        from .calibrate import apply_calibration, load_calibration
        self.head = apply_calibration(self.head, load_calibration(path))
        self._vocab_z = None

    # ------------------------------------------------------------ plumbing
    def _need_tap(self) -> "AnyTap":
        if self.tap is None:
            raise RuntimeError("this Sidecar is headless; use read(), or "
                               "construct with from_pretrained/attach")
        return self.tap
