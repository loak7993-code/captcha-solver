"""solve_grid.py — image-selection ("click all the X") CAPTCHA solver.

The text-CAPTCHA work does not transfer: this is zero-shot visual classification
of grid tiles. Pipeline:

  grid image -> locate tile boundaries -> split into tiles
             -> CLIP zero-shot score each tile against the challenge label
             -> select tiles above threshold

Works for arbitrary challenge labels with no training (prompt-driven), and
exposes per-tile scores so a caller can implement the standard retry strategy.

Scope: automating services you own or are authorised to test, and measuring the
strength of a CAPTCHA you deploy.
"""
import os
import numpy as np
from PIL import Image
from clip_onnx import CLIP

_CLIP = None


def _clip():
    global _CLIP
    if _CLIP is None:
        _CLIP = CLIP()
    return _CLIP


# ------------------------------------------------------------------ grid split
def separators(img, white=235, min_band=2):
    """Internal near-white separator bands: (row_bands, col_bands).

    A row is a separator when almost every pixel in it is near-white, and a
    column likewise. Only bands strictly inside the image count (the outer
    border is not a separator)."""
    a = np.asarray(img.convert("L"))
    h, w = a.shape

    def bands(profile):
        near = profile > white
        idx = np.where(near)[0]
        if len(idx) == 0:
            return []
        groups, start = [], idx[0]
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                if idx[i - 1] - start + 1 >= min_band:
                    groups.append((start, idx[i - 1]))
                start = idx[i]
        if idx[-1] - start + 1 >= min_band:
            groups.append((start, idx[-1]))
        return groups

    row_sep = [b for b in bands((a > white).mean(axis=1) * 255) if 0 < b[0] and b[1] < h - 1]
    col_sep = [b for b in bands((a > white).mean(axis=0) * 255) if 0 < b[0] and b[1] < w - 1]
    return row_sep, col_sep


def detect_grid(img, white=235, min_band=2):
    """Find internal separator lines and return (rows, cols).
    Falls back to (3, 3) when none are found."""
    a = np.asarray(img.convert("L"))
    h, w = a.shape
    row_sep, col_sep = separators(img, white, min_band)
    rows = len(row_sep) + 1
    cols = len(col_sep) + 1
    if rows < 2 or cols < 2:
        return 3, 3
    return int(rows), int(cols)


def split_grid(img, rows, cols, inset=3):
    """Equal split into rows*cols tiles, with an inset to skip separator borders."""
    w, h = img.size
    tw, th = w / cols, h / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (int(c * tw) + inset, int(r * th) + inset,
                   int((c + 1) * tw) - inset, int((r + 1) * th) - inset)
            tiles.append(img.crop(box))
    return tiles


# ------------------------------------------------------------------ scoring
DEFAULT_NEGATIVES = [
    "a photo of an empty background", "a photo of a plain surface",
    "a photo of a random unrelated object", "a photo of a blank wall",
]


def _hflip(im):
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def _zoom(im, frac=0.88):
    w, h = im.size
    dw, dh = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return im.crop((dw, dh, w - dw, h - dh)).resize((w, h), Image.BICUBIC)


def _prompts_for(c):
    return [f"a photo of a {c}", f"a {c}", f"a close-up photo of a {c}",
            f"a picture of a {c}", f"a cropped photo of a {c}"]


_HEAD = None
HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_head.pt")


def _head(path=None):
    """Load the trained MLP head on CLIP features (grid_head.pt).

    Loaded with weights_only=True: the file holds only tensors and plain
    Python lists, so no unpickling of arbitrary objects is needed."""
    global _HEAD
    if _HEAD is None:
        import torch
        import torch.nn as nn
        d = torch.load(path or HEAD_PATH, map_location="cpu", weights_only=True)
        C = len(d["classes"])
        net = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(512, C))
        net.load_state_dict(d["state_dict"]); net.eval()
        mean = np.asarray(d["mean"], dtype=np.float32)
        std = np.asarray(d["std"], dtype=np.float32)
        _HEAD = (net, d["classes"], mean, std)
    return _HEAD


def score_tiles_head(tiles, target, clip=None, tta=True, head_path=None):
    """Trained-head scorer: MLP on frozen CLIP embeddings. Beats zero-shot once
    enough labelled tiles exist (measured recall 93.8% vs 90.6%)."""
    import torch
    clip = clip or _clip()
    net, classes, mean, std = _head(head_path)
    views = [tiles]
    if tta:
        views += [[_hflip(t) for t in tiles], [_zoom(t) for t in tiles]]
    P = None
    with torch.no_grad():
        for vt in views:
            X = np.array([clip.image_embeds(t)[0] for t in vt])
            X = (X - mean) / std
            p = torch.softmax(net(torch.tensor(X, dtype=torch.float32)), 1).numpy()
            P = p if P is None else P + p
    P = P / len(views)
    ti = classes.index(target)
    return P[:, ti], P.argmax(1) == ti


def score_tiles(tiles, target, candidates=None, negatives=None, clip=None, tta=True):
    """Zero-shot score per tile for `target`.

    candidates given -> discriminative mode: softmax over the target plus every
    candidate class (and any generic negatives). The target has to actually beat
    the other plausible classes, which is far stronger than comparing against
    vague "something else" prompts.
    Otherwise        -> target prompt variants vs generic negatives only.

    tta: average the class probabilities over the original, horizontally
    flipped and slightly zoomed views - cheap, and it measurably steadies the
    per-tile decision.

    Returns (target_scores, wins) where `wins[i]` is True when the target is the
    highest-scoring class for that tile (always True in non-discriminative mode).
    """
    clip = clip or _clip()
    if candidates:
        classes = [target] + [c for c in candidates if c != target]
        prompts, spans = [], []
        for c in classes:
            vs = _prompts_for(c)
            spans.append((len(prompts), len(prompts) + len(vs)))
            prompts += vs
        prompts += list(negatives or [])
    else:
        prompts = _prompts_for(target) + list(negatives or DEFAULT_NEGATIVES)
        spans = [(0, len(_prompts_for(target)))]

    views = [tiles]
    if tta:
        views.append([_hflip(t) for t in tiles])
        views.append([_zoom(t) for t in tiles])
    acc = None
    for vt in views:
        prob, _ = clip.classify(vt, prompts)
        acc = prob if acc is None else acc + prob
    prob = acc / len(views)

    tscore = prob[:, spans[0][0]:spans[0][1]].sum(axis=1)
    if candidates:
        best_other = np.zeros(len(tiles))
        for s, e in spans[1:]:
            best_other = np.maximum(best_other, prob[:, s:e].sum(axis=1))
        return tscore, tscore >= best_other
    return tscore, np.ones(len(tiles), dtype=bool)


def solve_grid(path_or_img, target, rows=None, cols=None,
               threshold=0.3, inset=3, clip=None, candidates=None, use_head=None):
    """Returns (selected_indices, scores, (rows, cols)).

    selected_indices: 0-based tile indices chosen as containing `target`.
    scores: per-tile target probability (row-major).

    use_head: None -> auto (use the trained head when grid_head.pt exists and the
    target is in its class list, else zero-shot); True/False to force."""
    img = Image.open(path_or_img) if isinstance(path_or_img, (str, bytes)) else path_or_img
    if rows is None or cols is None:
        rows, cols = detect_grid(img)
    tiles = split_grid(img, rows, cols, inset=inset)
    if use_head is None:
        use_head = False
        if os.path.exists(HEAD_PATH):
            try:
                _, hclasses, _, _ = _head()
                use_head = target in hclasses
            except Exception:
                use_head = False
    if use_head:
        scores, wins = score_tiles_head(tiles, target, clip=clip)
    else:
        scores, wins = score_tiles(tiles, target, candidates=candidates, clip=clip)
    sel = [i for i, s in enumerate(scores) if s >= threshold and wins[i]]
    return sel, scores, (rows, cols)


def solve_grid_with_attempts(fetch, target, attempts=3, candidates=None, **kw):
    """Realistic CAPTCHA criterion: a failed read just means a fresh grid.

    `fetch()` returns the next grid (path or PIL image). Stops early on a
    confident all-positives read. Returns (selected, scores, attempts_used)."""
    best = None
    for n in range(1, attempts + 1):
        img = fetch()
        sel, scores, dims = solve_grid(img, target, candidates=candidates, **kw)
        if scores is not None and len(sel) > 0:
            # confident if every selected tile is well above threshold
            if min(scores[i] for i in sel) >= 0.6:
                return sel, scores, n
        best = (sel, scores, n)
    return best


if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser(description="image-selection CAPTCHA solver")
    ap.add_argument("image")
    ap.add_argument("target")
    ap.add_argument("--rows", type=int); ap.add_argument("--cols", type=int)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="candidate class list for discriminative scoring")
    a = ap.parse_args()
    sel, sc, (r, c) = solve_grid(a.image, a.target, rows=a.rows, cols=a.cols,
                                 threshold=a.threshold, candidates=a.classes)
    print(f"grid {r}x{c}  target='{a.target}'")
    for i, s in enumerate(sc):
        print(f"  tile {i:2d} ({i//c},{i%c}): {s*100:5.1f}%"
              + ("  <-- SELECT" if i in sel else ""))
    print("selected indices:", sel)
