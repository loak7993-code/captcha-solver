"""grid_head.py — train a stronger head on frozen CLIP features.

Zero-shot CLIP caps tile recall around 88%; with more labelled tiles a trained
head should beat it. This trains a small MLP (512->512->C) on CLIP embeddings
and compares head-to-head with zero-shot on the same held-out grids.

Strict split: per class, the first N_TRAIN images train the head, the rest are
used only to build test grids. Nothing leaks.

    python3 grid_head.py [n_grades]      # default 240 test grids
"""
import os, sys, glob, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import solve_grid
from clip_onnx import CLIP

DATA = os.environ.get("GRID_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridtest", "data"))
TILE, GAP = 100, 6
SIDE = 3 * TILE + 4 * GAP


def load_split(n_train):
    by = {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.jpg"))):
        cls = os.path.basename(p).rsplit("_", 1)[0].replace("_", " ")
        by.setdefault(cls, []).append(p)
    tr, te = {}, {}
    for c, ps in by.items():
        if len(ps) >= n_train + 6:
            tr[c], te[c] = ps[:n_train], ps[n_train:]
    return tr, te


def emb(clip, imgs):
    return np.array([clip.image_embeds(i)[0] for i in imgs])


def train_head(X, y, C, epochs=400, lr=1e-3):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(512, C))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    net.eval()
    return net


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


def score(grids, classes, fn, ths):
    res = {t: dict(tp=0, fp=0, fn=0, ex=0, ap=0) for t in ths}
    for img, truth, t in grids:
        tiles = solve_grid.split_grid(img, 3, 3)
        scores, wins = fn(tiles, t)
        ta = np.array(truth)
        for th in ths:
            sel = (scores >= th) & wins
            o = res[th]
            o["tp"] += (sel & ta).sum(); o["fp"] += (sel & ~ta).sum(); o["fn"] += (~sel & ta).sum()
            o["ex"] += (sel == ta).all(); o["ap"] += ((sel & ta).sum() == ta.sum())
    return res


def show(name, res, n, ths):
    print(f"\n{name}")
    print(f"{'thresh':>7} {'prec':>7} {'recall':>8} {'exact':>7} {'3-try':>7}")
    for th in ths:
        o = res[th]
        prec = o["tp"] / max(1, o["tp"] + o["fp"]); rec = o["tp"] / max(1, o["tp"] + o["fn"])
        ex = o["ex"] / n
        print(f"{th:>7.2f} {prec*100:>6.1f}% {rec*100:>7.1f}% {100*ex:>6.1f}% "
              f"{100*(1-(1-ex)**3):>6.1f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    clip = CLIP()
    # use the largest split the data supports
    by = {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.jpg"))):
        by.setdefault(os.path.basename(p).rsplit("_", 1)[0].replace("_", " "), []).append(p)
    n_train = min(50, min(len(v) for v in by.values()) - 6)
    tr, te = load_split(n_train)
    classes = sorted(te)
    print(f"classes={len(classes)} train/cls={n_train} test/cls~{min(len(v) for v in te.values())}")

    X, y = [], []
    for ci, c in enumerate(classes):
        for p in tr[c]:
            im = Image.open(p).convert("RGB")
            X.append(clip.image_embeds(im)[0]); y.append(ci)
            X.append(clip.image_embeds(solve_grid._hflip(im))[0]); y.append(ci)
    X = np.array(X); y = np.array(y)
    print(f"head trained on {len(X)} embeddings", flush=True)
    head = train_head(X, y, len(classes))

    rng = random.Random(21)
    grids = []
    for _ in range(n):
        t = rng.choice(classes); k = rng.randint(1, 4)
        img, truth = make_grid(te, t, k, rng)
        grids.append((img, truth, t))

    ths = (0.2, 0.3, 0.4, 0.5)
    show("zero-shot CLIP",
         score(grids, classes,
               lambda tl, t: solve_grid.score_tiles(tl, t, candidates=classes, clip=clip), ths),
         len(grids), ths)

    def head_fn(tl, t):
        Xt = torch.tensor(emb(clip, tl), dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(head(Xt), 1).numpy()
        ti = classes.index(t)
        return p[:, ti], p.argmax(1) == ti

    show("MLP head", score(grids, classes, head_fn, ths), len(grids), ths)

    torch.save({"state_dict": head.state_dict(), "classes": classes,
                "mean": X.mean(0), "std": X.std(0) + 1e-6}, "grid_head.pt")
    print("\nsaved grid_head.pt (head weights + class list)", flush=True)
