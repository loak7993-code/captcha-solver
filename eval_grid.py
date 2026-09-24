"""eval_grid.py — measure the grid solver on labelled 3x3 grids of real photos.

Builds grids by compositing Wikimedia photos: k positive tiles of the target
class plus (9-k) negatives from other classes, with white separators, then asks
solve_grid to pick the positives. Reports tile precision/recall and the grid
solve rate (selected mask == ground truth), i.e. the CAPTCHA success criterion.
"""
import os, glob, random, json
import numpy as np
from PIL import Image
import solve_grid
from clip_onnx import CLIP

DATA = os.environ.get("GRID_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridtest", "data"))
TILE = 100
GAP = 6
SIDE = 3 * TILE + 4 * GAP


def load_pool():
    pool = {}
    for p in glob.glob(os.path.join(DATA, "*.jpg")):
        cls = os.path.basename(p).rsplit("_", 1)[0].replace("_", " ")
        pool.setdefault(cls, []).append(p)
    return {k: v for k, v in pool.items() if len(v) >= 4}


def make_grid(pool, target, k, rng):
    others = [c for c in pool if c != target]
    canvas = Image.new("RGB", (SIDE, SIDE), (255, 255, 255))
    truth = [False] * 9
    slots = rng.sample(range(9), k)
    for i in range(9):
        cls = target if i in slots else rng.choice(others)
        img = Image.open(rng.choice(pool[cls])).convert("RGB").resize((TILE, TILE))
        r, c = divmod(i, 3)
        canvas.paste(img, (GAP + c * (TILE + GAP), GAP + r * (TILE + GAP)))
        truth[i] = (i in slots)
    return canvas, truth


def run(n=240, classes=None, thresholds=(0.3, 0.4, 0.5, 0.6), discriminative=True):
    pool = load_pool()
    classes = classes or list(pool)
    rng = random.Random(7)
    clip = CLIP()
    grids = []
    for _ in range(n):
        t = rng.choice(classes); k = rng.randint(1, 4)
        grids.append((*make_grid(pool, t, k, rng), t, k))
    print(f"built {len(grids)} labelled grids across {len(classes)} classes "
          f"(discriminative={discriminative})\n")
    tp = np.zeros(len(thresholds)); fp = np.zeros(len(thresholds))
    fn = np.zeros(len(thresholds)); exact = np.zeros(len(thresholds))
    recall_all = np.zeros(len(thresholds))
    for img, truth, t, k in grids:
        _, scores, _ = solve_grid.solve_grid(
            img, t, rows=3, cols=3, threshold=1.0, clip=clip,
            candidates=classes if discriminative else None)
        for j, th in enumerate(thresholds):
            sel = scores >= th
            truth_a = np.array(truth)
            tp[j] += (sel & truth_a).sum(); fp[j] += (sel & ~truth_a).sum()
            fn[j] += (~sel & truth_a).sum(); exact[j] += (sel == truth_a).all()
            recall_all[j] += ((sel & truth_a).sum() == truth_a.sum())
    print(f"{'thresh':>7} {'prec':>7} {'recall':>7} {'all-pos':>8} {'solve(exact)':>13}")
    for j, th in enumerate(thresholds):
        prec = tp[j] / max(1, tp[j] + fp[j]); rec = tp[j] / max(1, tp[j] + fn[j])
        print(f"{th:>7.2f} {prec*100:>6.1f}% {rec*100:>6.1f}% "
              f"{100*recall_all[j]/len(grids):>7.1f}% "
              f"{exact[j]}/{len(grids)} ({100*exact[j]/len(grids):.1f}%)")


if __name__ == "__main__":
    import sys
    run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 240)
