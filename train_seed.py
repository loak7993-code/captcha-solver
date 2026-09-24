"""Train one local CRNN for a given seed and save it.

Several independently-seeded models are worthwhile for two reasons:
  1. CTC sometimes collapses to the blank plateau (seen in the failed run).
     Seeding + a lower LR makes each run reproducible and stable.
  2. Independent models can be log-prob-averaged ("soup") - they share a
     vocabulary, so the average is well-defined and strictly better than any
     single member, and it recovers runs that would otherwise be lost.
"""
import sys, random
import numpy as np
import torch
import crnn

if __name__ == "__main__":
    seed = int(sys.argv[1]); out = sys.argv[2]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    crnn.train("data/trainmix3k", "data/trainmix3k/labels.json",
               epochs=20, lr=1.5e-3, out=out)
