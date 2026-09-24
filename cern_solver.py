"""cern_solver.py — CRNN+CTC reader for the CERN captcha-api style.

Own 61-character alphabet (1-9, A-Z, a-z) because case is significant and the
live service validates it case-sensitively. Shares the CRNN architecture from
crnn.py; only the alphabet and the data pipeline differ.

    python3 cern_solver.py train [epochs]
    python3 cern_solver.py eval                 # held-out synthetic set
    python3 cern_solver.py live [n]             # against the live CERN API
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from crnn import CRNN, collate

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHABET = ([chr(i) for i in range(49, 58)] +
            [chr(i) for i in range(65, 91)] +
            [chr(i) for i in range(97, 123)])          # 61 chars, case-sensitive
IDX2CH = {i + 1: c for i, c in enumerate(ALPHABET)}
IDX2CH[0] = ""
CH2IDX = {c: i for i, c in IDX2CH.items()}
NCLS = len(ALPHABET) + 1
H = 48
CKPT = os.path.join(HERE, "cern_crnn.pt")


class CernDS(Dataset):
    def __init__(self, root, labels, augment=True):
        self.root = root
        self.items = list(labels.items())
        self.aug = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, text = self.items[i]
        img = Image.open(os.path.join(self.root, fn)).convert("L")
        if self.aug:
            # the renderer already JPEGs and rotates; keep augmentation light so
            # we stay inside the real distribution
            img = img.rotate(random.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=255)
        w = max(8, int(H * img.width / img.height))
        img = img.resize((w, H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        y = torch.tensor([CH2IDX[c] for c in text], dtype=torch.long)
        return torch.from_numpy(a)[None], y, w


def greedy(logp):
    ids = logp.argmax(-1)[0].tolist()
    out, prev = [], 0
    for i in ids:
        if i != 0 and i != prev:
            out.append(IDX2CH[i])
        prev = i
    return "".join(out)


@torch.no_grad()
def predict(model, path):
    img = Image.open(path).convert("L")
    w = max(8, int(H * img.width / img.height))
    img = img.resize((w, H), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-5)
    return greedy(model(torch.from_numpy(a)[None, None]))


def train(epochs=20, bs=64, lr=5e-3, root="data/cern_train", out=CKPT, limit=0):
    labels = json.load(open(os.path.join(root, "labels.json")))
    items = list(labels.items()); random.seed(0); random.shuffle(items)
    if limit:
        items = items[:limit]
    n_val = max(40, len(items) // 15)
    val, tr = dict(items[:n_val]), dict(items[n_val:])
    dtr = DataLoader(CernDS(root, tr, True), bs, shuffle=True, collate_fn=collate)
    model = CRNN(ncls=NCLS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    for ep in range(1, epochs + 1):
        model.train(); tl = 0
        for x, y, yl in dtr:
            logp = model(x); T = logp.shape[1]
            loss = ctc(logp.permute(1, 0, 2), y,
                       torch.full((x.shape[0],), T, dtype=torch.long), yl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); tl += loss.item()
        sched.step()
        if True:
            acc = evaluate(model, root, val)
            torch.save(model.state_dict(), out)
            print(f"ep {ep:3d} loss {tl/len(dtr):.3f} val-exact {acc:.3f}", flush=True)
    torch.save(model.state_dict(), out)
    print("saved", out)


def evaluate(model, root, labels):
    model.eval(); ok = 0
    for fn, truth in labels.items():
        ok += predict(model, os.path.join(root, fn)) == truth
    return ok / max(1, len(labels))


def load(path=CKPT):
    m = CRNN(ncls=NCLS)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def live(n=50):
    """Solve real CAPTCHAs from the live CERN API and validate the answers."""
    import base64, urllib.request, urllib.error, http.cookiejar
    model = load()
    ok = 0; t0 = time.time()
    for i in range(n):
        jar = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        with op.open("https://captcha.web.cern.ch/api/v1.0/captcha/", timeout=20) as r:
            d = json.load(r)
        p = f"/tmp/cern_live_{i}.jpg"
        open(p, "wb").write(base64.b64decode(d["img"].split(",", 1)[1]))
        ans = predict(model, p)
        body = json.dumps({"id": d["id"], "answer": ans}).encode()
        req = urllib.request.Request("https://captcha.web.cern.ch/api/v1.0/captcha/",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            with op.open(req, timeout=20) as resp:
                good = resp.status == 200
        except urllib.error.HTTPError:
            good = False
        ok += good
        print(f"  {i+1:3d}/{n}  read={ans:8s} {'OK' if good else '--'}", flush=True)
    print(f"live CERN API: {ok}/{n} validated correct "
          f"({(time.time()-t0)/n*1000:.0f} ms/img)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 22,
              limit=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif cmd == "eval":
        m = load()
        for d in ["data/cern_val", "data/cern_train"]:
            lab = json.load(open(os.path.join(d, "labels.json")))
            sub = dict(list(lab.items())[:400]) if "train" in d else lab
            print(f"{d}: {evaluate(m, d, sub)*100:.1f}% exact ({len(sub)})")
    elif cmd == "live":
        live(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
