"""Run the Graf-J/captcha-crnn-base CRNN locally without torchvision.

Rebuilds the published architecture, loads the safetensors weights, and
reproduces the reference processor (grayscale -> resize 150x40 -> /255) and
CTC greedy decode. Scores it against a labelled folder.
"""
import sys, json, os, string
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
# NOTE: no transformers import. This file used to pull in
# `transformers.modeling_outputs.SequenceClassifierOutput` without ever using
# it, which forced the whole transformers package (and its dependency tree) onto
# the default text path. The model returns a plain tensor.

VOCAB = string.ascii_lowercase + string.ascii_uppercase + string.digits
IDX2CH = {i + 1: c for i, c in enumerate(VOCAB)}
IDX2CH[0] = ""


class CaptchaCRNN(nn.Module):
    def __init__(self, num_chars=63):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.SiLU(),
        )
        self.lstm = nn.LSTM(1280, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_chars)

    def forward(self, x):
        x = self.conv_layer(x)                 # (B,256,5,W)
        x = x.permute(0, 3, 1, 2)              # (B,W,256,5)
        b, w, c, h = x.size()
        x = x.reshape(b, w, -1)
        x, _ = self.lstm(x)
        return self.classifier(x)


def load(path):
    m = CaptchaCRNN()
    sd = load_file(path)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print("missing keys:", missing[:5])
    m.eval()
    return m


def preprocess(path):
    img = Image.open(path).convert("L").resize((150, 40))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]


@torch.no_grad()
def decode(model, path):
    logits = model(preprocess(path))           # (1,T,63)
    ids = logits.argmax(-1)[0].tolist()
    out, prev = [], -1
    for i in ids:
        if i != 0 and i != prev:
            out.append(IDX2CH.get(i, ""))
        prev = i
    return "".join(out)


if __name__ == "__main__":
    m = load(sys.argv[1] if len(sys.argv) > 1 else "hfmodel/model.safetensors")
    for d in sys.argv[2:] or ["data/easy", "data/medium", "data/hard"]:
        labels = json.load(open(os.path.join(d, "labels.json")))
        from collections import Counter
        ex = ci = co = ct = 0
        for fn, truth in labels.items():
            got = decode(m, os.path.join(d, fn))
            ex += (got == truth)
            ci += (got.lower() == truth.lower())
            co += sum((Counter(truth.lower()) & Counter(got.lower())).values())
            ct += len(truth)
        n = len(labels)
        print(f"{d}: exact={ex}/{n} ({100*ex/n:.1f}%)  ci-exact={ci}/{n} ({100*ci/n:.1f}%)  char={100*co/ct:.1f}%")
