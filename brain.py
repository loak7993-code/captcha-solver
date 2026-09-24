"""brain.py — ONE neural network for both CAPTCHA types.

A single multi-task model: a shared convolutional trunk with two heads.

    image -> shared trunk -> --+--> CTC sequence head   (text CAPTCHAs)
                               |
                               +--> tile class head     (grid CAPTCHAs)

One checkpoint, one set of weights, one forward pass. There is no router and no
second model: the same trunk learns both representations. Which head you read
out depends on the task, not on which network you loaded.

Trade-off (measured, see README): the single brain is a little behind the
two specialists, because the trunk must split its capacity across both tasks and
the tile task is data-starved compared with the text task. The specialists remain
available; this exists because "one brain" is a real, different design.
"""
import os, json, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import crnn
from crnn import CH2IDX, IDX2CH, ALPHABET, H

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_ROOT = os.environ.get("GRID_DATA", os.path.join(HERE, "gridtest", "data"))
BRAIN_PATH = os.path.join(HERE, "brain.pt")
TILE_H = 48
N_TEXT = len(ALPHABET) + 1        # + CTC blank


# ------------------------------------------------------------------ data
def tile_classes():
    cls = set()
    for p in glob.glob(os.path.join(TILE_ROOT, "*.jpg")):
        cls.add(os.path.basename(p).rsplit("_", 1)[0].replace("_", " "))
    return sorted(cls)


class TileDS(Dataset):
    """Real photos, one class per image. Per class: first `n_train` for training,
    the rest held out."""
    def __init__(self, classes, n_train=None, train=True, augment=True):
        self.classes = classes
        self.aug = augment
        by = {}
        for p in sorted(glob.glob(os.path.join(TILE_ROOT, "*.jpg"))):
            c = os.path.basename(p).rsplit("_", 1)[0].replace("_", " ")
            by.setdefault(c, []).append(p)
        self.items = []
        for ci, c in enumerate(classes):
            ps = by.get(c, [])
            cut = n_train if n_train is not None else max(1, len(ps) - 6)
            sel = ps[:cut] if train else ps[cut:]
            self.items += [(p, ci) for p in sel]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.aug:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # random crop window then resize -> scale/translation jitter
            w, h = img.size
            s = random.uniform(0.75, 1.0)
            cw, ch = int(w * s), int(h * s)
            x0 = random.randint(0, w - cw); y0 = random.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
        img = img.convert("L").resize((TILE_H, TILE_H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        return torch.from_numpy(a)[None], y


def tile_collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


# ------------------------------------------------------------------ model
class Brain(nn.Module):
    """Shared trunk + two heads."""

    def __init__(self, n_tile, rnn=192):
        super().__init__()
        def blk(i, o, pool=None):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers
        # the trunk: shared by both tasks
        self.trunk = nn.Sequential(
            *blk(1, 32, (2, 2)),
            *blk(32, 64, (2, 2)),
            *blk(64, 128, (2, 1)),
            *blk(128, 128, (2, 1)),
            *blk(128, 256, (2, 1)),
        )
        # head A: sequence (CTC) — over the width axis
        self.rnn = nn.LSTM(256, rnn, num_layers=2, bidirectional=True, batch_first=True)
        self.fc_text = nn.Linear(rnn * 2, N_TEXT)
        # head B: per-image tile class
        self.fc_tile = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True),
                                     nn.Dropout(0.2), nn.Linear(128, n_tile))

    def forward(self, x, task):
        f = self.trunk(x)                      # (B,256,H',W')
        if task == "text":
            f = f.mean(2).permute(0, 2, 1)     # collapse height -> (B,W',256)
            o, _ = self.rnn(f)
            return F.log_softmax(self.fc_text(o), dim=2)
        g = f.mean(dim=(2, 3))                 # global average pool
        return self.fc_tile(g)                 # raw logits


def greedy(logp):
    ids = logp.argmax(2)[0].tolist()
    out, prev = [], 0
    for i in ids:
        if i != 0 and i != prev:
            out.append(IDX2CH[i])
        prev = i
    return "".join(out)


# ------------------------------------------------------------------ train
def warm_start(model, crnn_ckpt, brain_ckpt=None):
    """Transplant a converged text CRNN into the brain.

    Brain.trunk/rnn/fc_text are structurally identical to crnn.CRNN's
    cnn/rnn/fc, and both use the same alphabet, so the weights map 1:1. This
    gives the text head a working starting point instead of cold-starting CTC
    (which collapses to the uniform distribution often).

    fc_tile is left to whatever a previous brain checkpoint had, else random.
    Returns (missing, unexpected) so the mapping can be asserted."""
    d = torch.load(crnn_ckpt, map_location="cpu")
    sd = {}
    for k, v in d.items():
        if k.startswith("cnn."):
            sd["trunk." + k[4:]] = v
        elif k.startswith("rnn."):
            sd[k] = v
        elif k.startswith("fc."):
            sd["fc_text." + k[3:]] = v
    if brain_ckpt and os.path.exists(brain_ckpt):
        blob = torch.load(brain_ckpt, map_location="cpu")
        prev = blob["state_dict"] if "state_dict" in blob else blob
        for k, v in prev.items():
            if k.startswith("fc_tile.") and v.shape == model.state_dict()[k].shape:
                sd[k] = v
    return model.load_state_dict(sd, strict=False)


def train(text_root="data/trainmix3k", epochs=18, bs=64, lr=1.5e-3,
          n_train_tiles=40, text_limit=0, out=BRAIN_PATH, init=None, warm_from=None):
    classes = tile_classes()
    labels = json.load(open(os.path.join(text_root, "labels.json")))
    items = list(labels.items()); random.seed(0); random.shuffle(items)
    if text_limit:
        items = items[:text_limit]
    n_val = max(40, len(items) // 10)
    tval, ttr = dict(items[:n_val]), dict(items[n_val:])
    dtext = DataLoader(crnn.CaptchaDS(text_root, ttr, True), bs, shuffle=True,
                       collate_fn=crnn.collate)
    dtile = DataLoader(TileDS(classes, n_train_tiles, True), bs, shuffle=True,
                       collate_fn=tile_collate)
    ltext = DataLoader(crnn.CaptchaDS(text_root, tval, False), bs, collate_fn=crnn.collate)

    model = Brain(len(classes))
    if warm_from:
        missing, unexpected = warm_start(model, warm_from, brain_ckpt=out)
        print(f"warm-started from {warm_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    elif init and os.path.exists(init):
        # resume instead of cold-starting: cold-start CTC collapses often.
        # brain.pt is a dict {"state_dict", "classes"} — unwrap it.
        blob = torch.load(init, map_location="cpu")
        model.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
        print(f"resumed from {init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    tile_val = TileDS(classes, n_train_tiles, train=False, augment=False)
    dtval = DataLoader(tile_val, 64, collate_fn=tile_collate)

    for ep in range(1, epochs + 1):
        model.train()
        tl = tt = 0.0; nt = nti = 0
        tile_iter = iter(dtile)
        for xt, yt, yl in dtext:
            # one shared-trunk step per task, per text batch
            lp = model(xt, "text"); T = lp.shape[1]
            loss_t = ctc(lp.permute(1, 0, 2), yt,
                         torch.full((xt.shape[0],), T, dtype=torch.long), yl)
            try:
                xti, yti = next(tile_iter)
            except StopIteration:
                tile_iter = iter(dtile); xti, yti = next(tile_iter)
            loss_i = F.cross_entropy(model(xti, "tile"), yti)
            loss = loss_t + loss_i
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            tl += loss_t.item(); tt += loss_i.item(); nt += 1; nti += 1
        sched.step()
        if ep % 3 == 0 or ep == 1:
            ta = eval_text(model, text_root, tval)
            ia = eval_tiles(model, dtval)
            torch.save({"state_dict": model.state_dict(), "classes": classes},
                       out)
            print(f"ep {ep:3d} text-ctc {tl/nt:.3f} tile-ce {tt/nti:.3f} "
                  f"| text-exact {ta:.3f} tile-acc {ia:.3f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out)
    print("saved", out, flush=True)
    return model


@torch.no_grad()
def eval_text(model, root, labels):
    model.eval()
    ok = 0
    for fn, truth in labels.items():
        img = Image.open(os.path.join(root, fn)).convert("L")
        w = max(8, int(H * img.width / img.height))
        img = img.resize((w, H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        lp = model(torch.from_numpy(a)[None, None], "text")
        ok += greedy(lp) == truth
    return ok / max(1, len(labels))


@torch.no_grad()
def eval_tiles(model, loader):
    model.eval()
    ok = n = 0
    for x, y in loader:
        p = model(x, "tile").argmax(1)
        ok += (p == y).sum().item(); n += len(y)
    return ok / max(1, n)


def load_brain(path=BRAIN_PATH):
    d = torch.load(path, map_location="cpu")
    m = Brain(len(d["classes"]))
    m.load_state_dict(d["state_dict"]); m.eval()
    return m, d["classes"]


# ------------------------------------------------------------------ inference
@torch.no_grad()
def read_text(model, image):
    """Text CAPTCHA -> string, using the shared brain's sequence head."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("L")
    w = max(8, int(H * img.width / img.height))
    img = img.resize((w, H), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-5)
    return greedy(model(torch.from_numpy(a)[None, None], "text"))


@torch.no_grad()
def classify_tile(model, classes, image, transform=None):
    """One tile -> class name, using the shared brain's tile head."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    im = img.convert("L").resize((TILE_H, TILE_H), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-5)
    p = F.softmax(model(torch.from_numpy(a)[None, None], "tile"), 1)[0]
    j = int(p.argmax())
    return classes[j], float(p[j])


@torch.no_grad()
def solve_grid_brain(model, classes, grid_img, target, rows=3, cols=3,
                     inset=3, threshold=0.35):
    """Grid CAPTCHA with the same brain: classify every tile, select the target."""
    from solve_grid import split_grid
    ti = classes.index(target) if target in classes else None
    tiles = split_grid(grid_img, rows, cols, inset=inset)
    x = []
    for t in tiles:
        im = t.convert("L").resize((TILE_H, TILE_H), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        x.append(a)
    logits = model(torch.from_numpy(np.stack(x))[:, None], "tile")
    p = F.softmax(logits, 1).numpy()
    scores = p[:, ti] if ti is not None else p.max(1)
    wins = p.argmax(1) == ti if ti is not None else np.ones(len(tiles), bool)
    sel = [i for i, s in enumerate(scores) if s >= threshold and wins[i]]
    return sel, scores, (rows, cols)


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    init = sys.argv[2] if len(sys.argv) > 2 else None
    warm = sys.argv[3] if len(sys.argv) > 3 else None
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5e-3
    train(epochs=ep, init=init, warm_from=warm, lr=lr)
