"""Fine-tune an already-converged model instead of cold-starting.

Cold-start CTC on the mixed corpus collapses ~60-100% of the time (blank
plateau, val 0.000). Starting from a model that already learned to emit
characters removes that failure mode entirely: the optimizer only has to
sharpen an existing solution, not find one.

Fine-tuning also gives diverse, comparably-strong members for the soup.

Usage: python3 train_from.py <init.pt> <out.pt> <seed> [epochs] [lr] [root]
"""
import sys, os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import crnn
from crnn import CRNN, collate


def finetune(root, init, out, seed, epochs=20, lr=5e-4, bs=64):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    labels = json.load(open(os.path.join(root, "labels.json")))
    items = list(labels.items())
    random.seed(seed); random.shuffle(items)
    n_val = max(40, len(items) // 10)
    val, tr = dict(items[:n_val]), dict(items[n_val:])
    dtr = DataLoader(crnn.CaptchaDS(root, tr, True), bs, shuffle=True, collate_fn=collate)

    model = CRNN()
    model.load_state_dict(torch.load(init, map_location="cpu"))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best = 0.0
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
            print(f"init {os.path.basename(init)} seed {seed} ep {ep:3d} "
                  f"loss {tl/len(dtr):.3f} val {acc:.3f}", flush=True)
            if acc >= best:
                best = acc
                torch.save(model.state_dict(), out)
    print(f"saved {out} (val {best:.3f})", flush=True)
    return best


if __name__ == "__main__":
    init, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-4
    root = sys.argv[6] if len(sys.argv) > 6 else "data/trainmix3k"
    finetune(root, init, out, seed, epochs=epochs, lr=lr)
