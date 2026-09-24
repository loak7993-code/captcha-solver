"""Stronger local training: mixed-difficulty data + heavier augmentation.

Same CRNN+CTC architecture as crnn.py, but the dataset augments far harder
(rotation, shear, blur, contrast, erasing) so the model is robust to the
distortions it will see at inference, and trains across all difficulty tiers.
"""
import os, json, random, sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import crnn
from crnn import CRNN, collate, CH2IDX, H

class HardDS(crnn.CaptchaDS):
    """Light-but-effective augmentation. The earlier heavy version (affine +
    per-pixel erasing) cost ~4x the CPU for no measured gain, so it is gone.
    Purposely cheap ops here: rotate, blur, brightness, contrast."""
    def __getitem__(self, i):
        fn, text = self.items[i]
        img = Image.open(os.path.join(self.root, fn)).convert("L")
        if self.aug:
            img = img.rotate(random.uniform(-6, 6), resample=Image.BICUBIC, fillcolor=255)
            if random.random() < 0.6:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.6))
            if random.random() < 0.6:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        w = max(8, int(H * img.width / img.height))
        img = img.resize((w, H), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-5)
        x = torch.from_numpy(arr)[None]
        y = torch.tensor([CH2IDX[c] for c in text], dtype=torch.long)
        return x, y, w


def train(root, epochs=40, bs=64, lr=2e-3, out="crnn_hard.pt", limit=0):
    labels = json.load(open(os.path.join(root, "labels.json")))
    items = list(labels.items())
    random.seed(0); random.shuffle(items)
    if limit:
        items = items[:limit]
    n_val = max(40, len(items) // 10)
    val, tr = dict(items[:n_val]), dict(items[n_val:])
    dtr = DataLoader(HardDS(root, tr, True), bs, shuffle=True, collate_fn=collate,
                     num_workers=4, persistent_workers=True)
    model = CRNN()
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
        if ep % 5 == 0 or ep == 1:
            acc = crnn.evaluate(model, root, val)
            torch.save(model.state_dict(), out)          # checkpoint every report
            print(f"ep {ep:3d} loss {tl/len(dtr):.3f}  val-exact {acc:.3f}", flush=True)
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/trainmix")
    ap.add_argument("epochs", nargs="?", type=int, default=40)
    ap.add_argument("out", nargs="?", default="crnn_hard.pt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    train(a.root, epochs=a.epochs, out=a.out, limit=a.limit)
