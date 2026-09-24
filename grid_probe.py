"""grid_probe.py — train a linear probe on CLIP features and compare to zero-shot.

Zero-shot CLIP reads single tiles at ~91% recall, which bottlenecks the whole
grid (9 tiles must all be right). A multinomial logistic regression on the same
frozen CLIP embeddings is cheap, trains in seconds, and sharpens the decision.

Strict split: per class, 10 images train the probe and 4 are held out. Test
grids are built ONLY from held-out images, so nothing leaks into the score.

Also reports the multi-attempt solve rate, which is the realistic CAPTCHA
criterion: a failure just means refetching a fresh grid.
"""
import os, glob, random
import numpy as np
from PIL import Image
import solve_grid
from clip_onnx import CLIP

DATA = os.environ.get("GRID_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridtest", "data"))
TILE, GAP = 100, 6
SIDE = 3 * TILE + 4 * GAP
N_TRAIN, N_TEST = 24, 10


def load_split():
    by_cls = {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.jpg"))):
        cls = os.path.basename(p).rsplit("_", 1)[0].replace("_", " ")
        by_cls.setdefault(cls, []).append(p)
    tr, te = {}, {}
    for c, ps in by_cls.items():
        if len(ps) >= N_TRAIN + N_TEST:
            tr[c], te[c] = ps[:N_TRAIN], ps[N_TRAIN:N_TRAIN + N_TEST]
    return tr, te


def softmax_reg_fit(X, y, C, iters=400, lr=0.5, l2=1e-3):
    N, D = X.shape
    W = np.zeros((D, C)); b = np.zeros(C)
    Y = np.eye(C)[y]
    for _ in range(iters):
        z = X @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p - Y) / N
        W -= lr * (X.T @ g + l2 * W)
        b -= lr * g.sum(0)
    return W, b


def make_grid(te, target, k, rng):
    others = [c for c in te if c != target]
    canvas = Image.new("RGB", (SIDE, SIDE), (255, 255, 255))
    truth = [False] * 9
    slots = set(rng.sample(range(9), k))
    for i in range(9):
        cls = target if i in slots else rng.choice(others)
        img = Image.open(rng.choice(te[cls])).convert("RGB").resize((TILE, TILE))
        r, c = divmod(i, 3)
        canvas.paste(img, (GAP + c * (TILE + GAP), GAP + r * (TILE + GAP)))
        truth[i] = i in slots
    return canvas, truth


def eval_mode(grids, classes, scorer, thresholds):
    out = {th: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "allpos": 0} for th in thresholds}
    for img, truth, t in grids:
        tiles = solve_grid.split_grid(img, 3, 3)
        scores, wins = scorer(tiles, t)
        ta = np.array(truth)
        for th in thresholds:
            sel = (scores >= th) & wins
            o = out[th]
            o["tp"] += (sel & ta).sum(); o["fp"] += (sel & ~ta).sum(); o["fn"] += (~sel & ta).sum()
            o["exact"] += (sel == ta).all(); o["allpos"] += ((sel & ta).sum() == ta.sum())
    return out


if __name__ == "__main__":
    tr, te = load_split()
    classes = sorted(te)
    clip = CLIP()
    print(f"classes: {len(classes)}  train/cls={N_TRAIN} test/cls={N_TEST}", flush=True)

    # --- train probe (with horizontal-flip augmentation)
    X, y = [], []
    for ci, c in enumerate(classes):
        for p in tr[c]:
            im = Image.open(p).convert("RGB")
            X.append(clip.image_embeds(im)[0]); y.append(ci)
            X.append(clip.image_embeds(solve_grid._hflip(im))[0]); y.append(ci)
    X = np.array(X); y = np.array(y)
    W, b = softmax_reg_fit(X, y, len(classes))
    print(f"probe trained on {len(X)} embeddings", flush=True)

    # --- test grids from held-out images only
    rng = random.Random(21)
    grids = []
    for _ in range(240):
        t = rng.choice(classes); k = rng.randint(1, 4)
        img, truth = make_grid(te, t, k, rng)
        grids.append((img, truth, t))

    def zeroshot(tiles, t):
        return solve_grid.score_tiles(tiles, t, candidates=classes, clip=clip)

    def probe(tiles, t):
        Xt = np.array([clip.image_embeds(x)[0] for x in tiles])
        z = Xt @ W + b; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        ti = classes.index(t)
        return p[:, ti], p.argmax(1) == ti

    ths = (0.2, 0.3, 0.4, 0.5)
    for name, sc in [("zero-shot CLIP", zeroshot), ("linear probe", probe)]:
        res = eval_mode(grids, classes, sc, ths)
        print(f"\n{name}")
        print(f"{'thresh':>7} {'prec':>7} {'recall':>8} {'all-pos':>8} {'solve':>7} {'3-try':>7}")
        for th in ths:
            o = res[th]
            prec = o["tp"] / max(1, o["tp"] + o["fp"]); rec = o["tp"] / max(1, o["tp"] + o["fn"])
            ex = o["exact"] / len(grids)
            apr = 1 - (1 - ex) ** 3
            print(f"{th:>7.2f} {prec*100:>6.1f}% {rec*100:>7.1f}% "
                  f"{100*o['allpos']/len(grids):>7.1f}% {100*ex:>6.1f}% {100*apr:>6.1f}%")
