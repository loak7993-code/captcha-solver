"""Evaluate a pool of local checkpoints and pick the soup.

Selects on the hard tier (greedy, cheap), then reports the full TTA+beam
result for the best single model and for the soup of selected members.
Collapsed runs (0% on hard) are excluded automatically.
"""
import os, sys, json, glob
import torch
from collections import Counter
import crnn
import solve_pro

HERE = os.path.dirname(os.path.abspath(__file__))


def hard_acc(ckpt, n=200):
    m = crnn.CRNN()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    labels = json.load(open(os.path.join(HERE, "data/hard/labels.json")))
    items = list(labels.items())[:n]
    ex = sum(1 for fn, t in items if crnn.predict(m, os.path.join(HERE, "data/hard", fn)) == t)
    return ex / len(items)


def full_eval(ensemble, tag, beam=16, tta=True):
    r = []
    for d in ("data/easy", "data/medium", "data/hard"):
        labels = json.load(open(os.path.join(HERE, d, "labels.json")))
        ex = co = ct = 0
        for fn, t in labels.items():
            got, _ = solve_pro.solve_pro(os.path.join(HERE, d, fn),
                                         ensemble=ensemble, beam=beam, tta=tta)
            ex += (got == t); co += sum((Counter(t) & Counter(got)).values()); ct += len(t)
        r.append(f"{d.split('/')[1]}={100*ex/len(labels):.1f}% (char {100*co/ct:.1f}%)")
    print(f"{tag}: " + "  ".join(r), flush=True)


if __name__ == "__main__":
    cands = [os.path.join(HERE, "crnn_best.pt"), os.path.join(HERE, "crnn_hard.pt")]
    cands += sorted(glob.glob(os.path.join(HERE, "crnn_pool", "*.pt")))
    scored = []
    for p in cands:
        if not os.path.exists(p):
            continue
        a = hard_acc(p)
        scored.append((a, p))
        print(f"{os.path.relpath(p, HERE):26s} hard(greedy)={100*a:.1f}%", flush=True)
    good = [p for a, p in scored if a >= 0.75]
    good.sort(key=lambda p: -dict((p, a) for a, p in scored)[p])
    print(f"\nselected {len(good)} non-collapsed models", flush=True)
    if good:
        best = max(scored)[1]
        full_eval(("local:" + best,), f"best single [{os.path.basename(best)}] + beam+TTA")
        full_eval(tuple("local:" + p for p in good[:8]), f"soup[{len(good[:8])}] + beam+TTA")
