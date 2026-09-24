"""Train one model into a pool, with collapse detection.

Two changes over the earlier runs:
  1. LR warmup (3 epochs ramp) + cosine decay. The cold-start high LR is what
     sends CTC into the blank plateau, so ramping in stabilises convergence.
  2. Early collapse detection: a healthy run has val-exact > 0 by epoch 10
     (the converged probe was 0 at ep5, 0.707 at ep10). A run still at 0.0 by
     ep10 is dead - abort it instead of burning 20 more epochs.

Usage: python3 train_pool.py <seed> <out.pt> [epochs] [root]
Exit code 0 = saved a usable model, 2 = collapsed (discarded).
"""
import sys, os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import crnn
from crnn import CRNN, collate


def train_pool(root, out, seed, epochs=35, bs=64, lr=1.5e-3, warmup=3, abort_at=10):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    labels = json.load(open(os.path.join(root, "labels.json")))
    items = list(labels.items())
    random.seed(seed); random.shuffle(items)
    n_val = max(40, len(items) // 10)
    val, tr = dict(items[:n_val]), dict(items[n_val:])
    dtr = DataLoader(crnn.CaptchaDS(root, tr, True), bs, shuffle=True, collate_fn=collate)
    model = CRNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def schedule(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
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
        tr_loss = tl / len(dtr)
        if ep in (10, 20) or ep == epochs:
            acc = crnn.evaluate(model, root, val)
            print(f"seed {seed} ep {ep:3d} loss {tr_loss:.3f} val {acc:.3f}", flush=True)
            if ep >= abort_at and acc == 0.0:
                print(f"seed {seed} COLLAPSED at ep{ep} - discard", flush=True)
                return False
            if acc > 0:
                torch.save(model.state_dict(), out)     # keep best-so-far
        elif ep % 5 == 0:
            print(f"seed {seed} ep {ep:3d} loss {tr_loss:.3f}", flush=True)
    print(f"seed {seed} saved -> {out}", flush=True)
    return True


if __name__ == "__main__":
    seed = int(sys.argv[1]); out = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 35
    root = sys.argv[4] if len(sys.argv) > 4 else "data/trainmix3k"
    ok = train_pool(root, out, seed, epochs=epochs)
    sys.exit(0 if ok else 2)
