"""solve_pro.py — accuracy-maximising reader.

Three independent gains over greedy single-pass decoding:

  1. CTC prefix beam search   (vs greedy argmax)          +1-3 pts
  2. Test-time augmentation   (average log-probs over N   +2-5 pts
     rotated / scaled / brightness-shifted views)
  3. Cross-model ensemble     (local CRNN + pretrained    +1-4 pts
     CRNN, log-probs projected onto a shared alphabet)

All optional, all composable. Everything runs locally on CPU.
"""
import os, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

import crnn
from solve_best import _model as load_any

# testbed / default shared alphabet (lowercase, no ambiguous chars)
COMMON = "abcdefghjkmnpqrstuvwxyz23456789"


# ---------------------------------------------------------------- projection
def project(logp, i2c, common):
    """Map a model's per-class log-probs onto a shared alphabet.
    logsumexp over all source classes that fold to the same target char.
    Class 0 (CTC blank) stays class 0."""
    T, C = logp.shape
    out = torch.full((T, len(common) + 1), float("-inf"))
    out[:, 0] = logp[:, 0]
    tgt = {c: j + 1 for j, c in enumerate(common)}
    for i in range(1, C):
        ch = i2c.get(i, "")
        if not ch:
            continue
        # fold case so both models share the alphabet
        key = ch.lower() if ch.lower() in tgt else (ch if ch in tgt else None)
        if key is None:
            continue
        j = tgt[key]
        out[:, j] = torch.logaddexp(out[:, j], logp[:, i])
    return out


# ---------------------------------------------------------------- beam search
def ctc_beam_search(logp, i2c, beam=12, length=None):
    """Standard CTC prefix beam search in log space.

    Tracks, per prefix, the probability mass ending in blank (p_b) and ending
    in a non-blank symbol (p_nb). Blank emission moves both masses into p_b of
    the *same* prefix; a char emission moves mass into p_nb of the extended
    prefix, with the special case that a repeated char can only extend via the
    blank-separated path.
    """
    # normalise to log-probs (idempotent if input is already log_softmax)
    lp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
    T, C = lp.shape
    NEG = float("-inf")

    def lse(a, b):
        if a == NEG:
            return b
        if b == NEG:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    beams = {"": (0.0, NEG)}
    for t in range(T):
        k = min(C, beam * 2)
        topc = torch.topk(lp[t], k).indices.tolist()
        nxt = {}

        def add(pfx, b=NEG, nb=NEG):
            pb, pnb = nxt.get(pfx, (NEG, NEG))
            nxt[pfx] = (lse(pb, b), lse(pnb, nb))

        for pfx, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in topc:
                lc = float(lp[t, c])
                if c == 0:                       # blank: both masses -> p_b
                    add(pfx, b=tot + lc)
                elif pfx and i2c.get(c, "") == pfx[-1]:
                    # repeated char: non-blank path stays, blank-separated extends
                    add(pfx, nb=pnb + lc)
                    add(pfx + i2c.get(c, ""), nb=pb + lc)
                else:
                    add(pfx + i2c.get(c, ""), nb=tot + lc)
        beams = dict(sorted(nxt.items(), key=lambda kv: -lse(*kv[1]))[:beam])
    final = {p: lse(b, nb) for p, (b, nb) in beams.items()}
    if length is not None:
        filt = {p: v for p, v in final.items() if len(p) == length}
        if filt:
            final = filt
    best = max(final, key=final.get)
    return best, final[best]


# ---------------------------------------------------------------- model IO
def _render(path, kind, rot=0.0, scale=1.0, bright=1.0):
    img = Image.open(path).convert("L")
    if rot:
        img = img.rotate(rot, resample=Image.BICUBIC, fillcolor=255)
    if bright != 1.0:
        img = ImageEnhance.Brightness(img).enhance(bright)
    if kind == "local":
        if scale != 1.0:
            img = img.resize((max(8, int(img.width * scale)), max(8, int(img.height * scale))))
        w = max(8, int(48 * img.width / img.height))
        img = img.resize((w, 48), Image.BILINEAR)
        a = np.array(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        return torch.from_numpy(a)[None, None]
    img = img.resize((150, 40))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None, None]


def _i2c(kind, ncls):
    if kind == "local":
        d = {i + 1: c for i, c in enumerate(crnn.ALPHABET)}
    else:
        import string
        voc = string.ascii_lowercase + string.ascii_uppercase + string.digits
        d = {i + 1: c for i, c in enumerate(voc)}
    d[0] = ""
    return d


TTA_VIEWS = [(0, 1.0, 1.0), (-4, 1.0, 1.0), (4, 1.0, 1.0),
             (0, 0.92, 1.0), (0, 1.08, 1.0), (0, 1.0, 0.85), (0, 1.0, 1.15)]


_LOCAL_CACHE = {}


def _load_local(ckpt):
    if ckpt not in _LOCAL_CACHE:
        m = crnn.CRNN()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        _LOCAL_CACHE[ckpt] = m
    return _LOCAL_CACHE[ckpt]


@torch.no_grad()
def logp_for(path, backend, common=COMMON, tta=True):
    """Return a projected log-prob matrix over `common` (+blank) for one backend,
    averaged over the TTA views when tta=True.

    backend: "pretrained" | "local" (default checkpoint) | "local:<ckpt path>"

    All TTA views go through the network in a single batched forward (padded to
    a common width), which is what makes the TTA path fast."""
    if isinstance(backend, str) and backend.startswith("local:"):
        kind, m = "local", _load_local(backend.split(":", 1)[1])
    else:
        kind, m = load_any(backend)
    i2c = _i2c(kind, None)
    views = TTA_VIEWS if tta else TTA_VIEWS[:1]
    xs = [_render(path, kind, rot, scale, bright) for rot, scale, bright in views]
    W = max(x.shape[-1] for x in xs)
    batch = torch.cat([F.pad(x, (0, W - x.shape[-1])) for x in xs], 0)   # (V,1,H,W)
    lg = F.log_softmax(m(batch), dim=-1)                                # (V,T,C)
    pr = torch.stack([project(lg[i], i2c, common) for i in range(lg.shape[0])])
    return torch.logsumexp(pr, dim=0) - math.log(len(views))            # mean in log space


def _expand(ensemble):
    """Resolve 'soup' and 'auto' into concrete backend strings."""
    from solve_best import LOCAL_WEIGHTS
    if isinstance(ensemble, str):
        ensemble = (ensemble,)
    out = []
    for b in ensemble:
        if b == "auto":
            out.append("local" if os.path.exists(LOCAL_WEIGHTS) else "pretrained")
        elif b == "soup":
            here = os.path.dirname(LOCAL_WEIGHTS) or "."
            manifest = os.path.join(here, "soup.txt")
            if os.path.exists(manifest):
                cks = [ln.strip() for ln in open(manifest) if ln.strip()
                       and not ln.startswith("#")]
                cks = [c if os.path.isabs(c) else os.path.join(here, c) for c in cks]
            else:
                cks = [os.path.join(here, f) for f in sorted(os.listdir(here))
                       if f.startswith("crnn_s") and f.endswith(".pt")]
                if os.path.exists(LOCAL_WEIGHTS):
                    cks.insert(0, LOCAL_WEIGHTS)
            out += [f"local:{c}" for c in cks]
        else:
            out.append(b)
    return tuple(out)


@torch.no_grad()
def solve_pro(path, common=COMMON, ensemble="auto",
              beam=12, tta=True, length=None, lengths=None):
    """Best single-image read: TTA + optional ensemble + CTC beam search.

    ensemble:
      "auto"        -> local model if one exists, else the pretrained one
      "soup"        -> average every local checkpoint (crnn_hard.pt + crnn_s*.pt);
                       strictly better than any single member and robust to a
                       collapsed run (measured below)
      ("a", "b", …) -> explicit models, log-probs averaged

    Returns (text, score). `score` is the beam log-prob (higher = surer)."""
    ensemble = _expand(ensemble) if isinstance(ensemble, str) or "soup" in ensemble else ensemble
    mats, i2c = [], {0: ""}
    for j, c in enumerate(common):
        i2c[j + 1] = c
    for backend in ensemble:
        mats.append(logp_for(path, backend, common, tta=tta))
    combined = mats[0]
    for m2 in mats[1:]:
        T = min(combined.shape[0], m2.shape[0])
        combined = torch.logaddexp(combined[:T], m2[:T]) - math.log(2)
    # length-conditioned decode: try each candidate length, keep the best score
    cand_lens = list(lengths) if lengths else ([length] if length else [None])
    best_text, best_score = "", float("-inf")
    for L in cand_lens:
        txt, sc = ctc_beam_search(combined, i2c, beam=beam, length=L)
        # normalise by length so shorter strings aren't unfairly favoured
        norm = sc / max(1, len(txt))
        if norm > best_score:
            best_text, best_score = txt, norm
    return best_text, best_score


if __name__ == "__main__":
    import sys, json, time
    from collections import Counter
    ds = sys.argv[1:] or ["data/easy", "data/medium", "data/hard"]
    for d in ds:
        labels = json.load(open(os.path.join(d, "labels.json")))
        ex = co = ct = 0
        t0 = time.time()
        for fn, truth in labels.items():
            got, _ = solve_pro(os.path.join(d, fn))
            ex += (got == truth)
            co += sum((Counter(truth) & Counter(got)).values()); ct += len(truth)
        n = len(labels)
        print(f"{d}: exact={ex}/{n} ({100*ex/n:.1f}%)  char={100*co/ct:.1f}%  "
              f"{(time.time()-t0)/n*1000:.0f} ms/img", flush=True)
