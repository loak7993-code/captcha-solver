"""captcha_ai.py — one entry point for both CAPTCHA types.

Instead of two tools you have to choose between, this exposes a single
`solve()` call (and a single CLI). It inspects the image, decides which
specialist applies, runs it, and returns one uniform result.

  text CAPTCHAs  ("type the letters")   -> CRNN+CTC        (solve_pro)
  image grids    ("click all the X")    -> CLIP + head     (solve_grid)

What this is NOT: a single neural network. There is no one model that both
reads text and selects tiles here — see README/NOTES for why (the local
vision-language-model route needs torchvision's image processor, which is
broken in this environment). This is one *interface* over two specialists,
chosen automatically.

    from captcha_ai import solve
    r = solve("captcha.png")                        # auto-detects: text
    r = solve("grid.png", target="traffic light")   # grid
    print(r.text, r.selected)

    python3 captcha_ai.py captcha.png
    python3 captcha_ai.py grid.png --target "traffic light"
"""
import os
import sys
import json
import tempfile
from dataclasses import dataclass, field, asdict

from PIL import Image


# --------------------------------------------------------------------------- result
@dataclass
class Result:
    kind: str                       # "text" | "grid"
    text: str = ""                  # text CAPTCHAs
    selected: list = field(default_factory=list)   # grid CAPTCHAs (tile indices)
    confidence: float = 0.0
    target: str = ""                # grid challenge label
    detail: dict = field(default_factory=dict)

    def __str__(self):
        if self.kind == "text":
            return f"[text] {self.text!r}  (score {self.confidence:.2f})"
        cells = ", ".join(f"({i//self.detail.get('cols', 3)},{i%self.detail.get('cols', 3)})"
                          for i in self.selected)
        return (f"[grid] target={self.target!r} -> tiles {self.selected} "
                f"(cells {cells}), {self.detail.get('rows', '?')}x{self.detail.get('cols', '?')}")

    def to_json(self):
        return json.dumps(asdict(self))


# --------------------------------------------------------------------------- routing
def detect_kind(img, target=None):
    """Decide which specialist applies.

    A target prompt only makes sense for a grid, so it forces grid mode.
    Otherwise: grids are near-square and carry long straight near-white
    separator bands between tiles; text CAPTCHAs are wide (aspect > 2) and have
    no such separators."""
    if target:
        return "grid"
    from solve_grid import separators
    w, h = img.size
    ratio = w / max(1, h)
    row_b, col_b = separators(img)
    looks_gridded = len(row_b) >= 1 and len(col_b) >= 1 and 0.55 <= ratio <= 1.8
    return "grid" if looks_gridded else "text"


# --------------------------------------------------------------------------- helpers
def _as_path(image):
    """solve_pro works from a file path; accept a PIL image too."""
    if isinstance(image, (str, bytes, os.PathLike)):
        return str(image), None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    return tmp.name, tmp.name


# --------------------------------------------------------------------------- presets
# speed presets: what each one turns on/off
PRESETS = {
    # ~8 ms text / ~160 ms grid, accuracy within ~0.5 pt of balanced on our tests
    "fast":     {"text": dict(tta=False, beam=1),
                 "grid": dict(tta=False), "quantized": True},
    # default: TTA + beam, fp32 vision
    "balanced": {"text": dict(tta=True, beam=12),
                 "grid": dict(tta=True), "quantized": False},
    # model soup + wider beam
    "accurate": {"text": dict(tta=True, beam=16, ensemble="soup"),
                 "grid": dict(tta=True), "quantized": False},
}


# --------------------------------------------------------------------------- main API
def solve(image, target=None, kind=None, candidates=None, threshold=0.3,
          speed="balanced", **kw):
    """Solve one CAPTCHA. Returns a `Result`.

    image       path or PIL.Image
    target      challenge label for grids ("traffic light"); ignored for text
    kind        force "text" or "grid" instead of auto-detecting
    candidates  candidate class list for discriminative grid scoring
    speed       "fast" | "balanced" (default) | "accurate"
    """
    img = image if isinstance(image, Image.Image) else Image.open(image)
    kind = kind or detect_kind(img, target)
    preset = PRESETS.get(speed, PRESETS["balanced"])

    if kind == "text":
        import solve_pro
        path, tmp = _as_path(img)
        opts = {**preset["text"], **kw}
        try:
            text, score = solve_pro.solve_pro(path, **opts)
        finally:
            if tmp:
                os.unlink(tmp)
        return Result(kind="text", text=text, confidence=float(score),
                      detail={"chars": len(text), "speed": speed})

    if not target:
        raise ValueError("grid CAPTCHAs need a target label, e.g. target='traffic light'")

    import solve_grid
    solve_grid.set_clip_options(quantized=preset["quantized"])
    # without a trained head and without candidate classes, zero-shot "open" mode
    # is permissive and over-selects; raise the bar in that case
    if not solve_grid.head_available(target) and not candidates:
        threshold = max(threshold, 0.5)
    opts = {**preset["grid"], **kw}
    sel, scores, (rows, cols) = solve_grid.solve_grid(
        img, target, threshold=threshold, candidates=candidates, **opts)
    conf = float(max([scores[i] for i in sel], default=0.0))
    return Result(kind="grid", selected=list(sel), confidence=conf, target=target,
                  detail={"rows": rows, "cols": cols, "speed": speed,
                          "scores": [round(float(s), 3) for s in scores]})


# --------------------------------------------------------------------------- CLI
def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Unified CAPTCHA solver (text + image grid)")
    ap.add_argument("image", help="image file, or a directory with --batch")
    ap.add_argument("--target", help="grid challenge label, e.g. 'traffic light'")
    ap.add_argument("--kind", choices=["text", "grid"], help="force the solver")
    ap.add_argument("--classes", nargs="*", help="candidate classes for grid scoring")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--speed", choices=["fast", "balanced", "accurate"], default="balanced")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    r = solve(a.image, target=a.target, kind=a.kind,
              candidates=a.classes, threshold=a.threshold, speed=a.speed)
    print(r.to_json() if a.json else str(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
