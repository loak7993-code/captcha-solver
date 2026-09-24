"""solve_best.py — the winning reader for text CAPTCHAs.

Pipeline: pretrained CRNN+CTC (Graf-J/captcha-crnn-base, 3.5M params) with
decode-time charset masking, so one model serves any target alphabet. Falls
back to the Tesseract path in solve.py when the CRNN confidence is low.

    from solve_best import solve_best
    solve_best("captcha.png")                     # free decode
    solve_best("captcha.png", mask="abcdefghjkmnpqrstuvwxyz23456789")
    solve_best("captcha.png", length=6)           # force output length

Scope: for automating services you own or are authorized to test, and for
measuring the strength of a CAPTCHA you deploy. It is a local model — no
network calls, no third-party solving service.
"""
import os, string
import numpy as np
import torch
from PIL import Image

from hf_local import CaptchaCRNN, load, IDX2CH

WEIGHTS = os.path.join(os.path.dirname(__file__), "hfmodel", "model.safetensors")
# prefer the best local model, then the mixed one, then the first one
def _pick_local():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("crnn_round5/c.pt", "crnn_pool2/r2_c.pt", "crnn_best.pt",
                 "crnn_s303.pt", "crnn_hard.pt", "crnn.pt"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    return os.path.join(here, "crnn.pt")


LOCAL_WEIGHTS = _pick_local()
_MODELS = {}


def _model(backend="pretrained"):
    """pretrained = the general HF model (out-of-distribution safe).
    local = the CRNN trained on your target's distribution (in-distribution best)."""
    if backend in _MODELS:
        return _MODELS[backend]
    if backend == "local" and os.path.exists(LOCAL_WEIGHTS):
        import crnn
        m = crnn.CRNN()
        m.load_state_dict(torch.load(LOCAL_WEIGHTS, map_location="cpu"))
        m.eval()
        _MODELS[backend] = ("local", m)
    else:
        _MODELS[backend] = ("pretrained", load(WEIGHTS))
    return _MODELS[backend]


@torch.no_grad()
def _logits(path, backend="pretrained"):
    kind, m = _model(backend)
    if kind == "local":
        img = Image.open(path).convert("L")
        w = max(8, int(48 * img.width / img.height))
        img = img.resize((w, 48), Image.BILINEAR)
        x = np.array(img).astype(np.float32) / 255.0
        x = (x - x.mean()) / (x.std() + 1e-5)
        return m(torch.from_numpy(x)[None, None])[0]      # (T, 32)
    img = Image.open(path).convert("L").resize((150, 40))
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None, None]
    return m(x)[0]                                        # (T, 63)


def _ctc_greedy(logits, mask=None, length=None):
    lp = logits.log_softmax(-1)
    ncls = lp.shape[-1]
    if ncls == 63:
        i2c = dict(IDX2CH)
    else:
        import crnn
        i2c = {i + 1: c for i, c in enumerate(crnn.ALPHABET)}
        i2c[0] = ""
    conf = float(lp.max(-1).values.mean())
    if mask and ncls == 63:
        # overlay the mask on the class distribution before decoding
        keep = torch.full_like(lp, float("-inf"))
        for i in range(1, ncls):
            if i2c.get(i, "") in mask:
                keep[:, i] = lp[:, i]
        keep[:, 0] = lp[:, 0]      # keep blank
        ids = keep.argmax(-1).tolist()
        conf = float(keep.max(-1).values[lp.argmax(-1) > 0].mean()) if (lp.argmax(-1) > 0).any() else conf
    else:
        ids = lp.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != 0 and i != prev:
            out.append(i2c.get(i, ""))
        prev = i
    text = "".join(out)
    if mask:                                    # local model / post-filter
        text = "".join(c for c in text if c in mask)
    if length and len(text) != length:
        return "", conf
    return text, conf


def solve_best(path, mask=None, length=None, backend="pretrained"):
    """Returns (text, mean_confidence). Empty text => low confidence, retry.
    backend: 'pretrained' (general, OOD-safe) or 'local' (trained on target)."""
    _model(backend)
    text, conf = _ctc_greedy(_logits(path, backend), mask=mask, length=length)
    # a proper CTC confidence is the product/geometric-mean of the emitted
    # (non-blank) timesteps; only fall back when the CRNN produced nothing.
    if not text:
        try:
            from solve import solve as tess_solve
            alt = tess_solve(path)
            if alt:
                return alt, 0.5
        except Exception:
            pass
    return text, conf


@torch.no_grad()
def solve_batch(paths, mask=None, bs=64, backend="pretrained"):
    """Batched inference: ~5-10x faster than one-at-a-time for large corpora."""
    kind, m = _model(backend)
    out = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        if kind == "local":
            import crnn
            xs = []
            for p in chunk:
                img = Image.open(p).convert("L")
                w = max(8, int(48 * img.width / img.height))
                img = img.resize((w, 48), Image.BILINEAR)
                a = np.array(img).astype(np.float32) / 255.0
                a = (a - a.mean()) / (a.std() + 1e-5)
                xs.append(torch.from_numpy(a)[None])
            W = max(x.shape[2] for x in xs)
            xs = [torch.nn.functional.pad(x, (0, W - x.shape[2])) for x in xs]
        else:
            xs = [torch.from_numpy(
                np.array(Image.open(p).convert("L").resize((150, 40))).astype(np.float32) / 255.0
            )[None] for p in chunk]
        logits = m(torch.stack(xs))                       # (B,T,C)
        for b in range(logits.shape[0]):
            text, conf = _ctc_greedy(logits[b], mask=mask)
            out.append((text, conf))
    return out


def solve_with_retry(fetch, mask=None, attempts=4, backend="pretrained"):
    """fetch() -> local path to a fresh CAPTCHA. Refetches on a low-confidence
    read, which is exactly how the published MCA reader turns 75% single-shot
    into ~97% within 3 fetches. `fetch` is yours; nothing leaves this process."""
    best = ("", 0.0)
    for _ in range(attempts):
        text, conf = solve_best(fetch(), mask=mask, backend=backend)
        if text and conf >= best[1]:
            best = (text, conf)
        if text and conf >= 0.9:
            return text, conf
    # fall back to refetching until a non-empty read, else return best-so-far
    for _ in range(attempts):
        text, conf = solve_best(fetch(), mask=mask, backend=backend)
        if text:
            return text, conf
    return best


if __name__ == "__main__":
    import sys, json, time
    from collections import Counter
    args = sys.argv[1:]
    backend = "pretrained"
    if args and args[0].startswith("--backend="):
        backend = args[0].split("=", 1)[1]; args = args[1:]
    mask = args[0] if args and args[0] != "-" else None
    ds = args[1:] or ["data/easy", "data/medium", "data/hard"]
    print(f"backend={backend}")
    for d in ds:
        labels = json.load(open(os.path.join(d, "labels.json")))
        ex = co = ct = 0
        t0 = time.time()
        for fn, truth in labels.items():
            got, conf = solve_best(os.path.join(d, fn), mask=mask, backend=backend)
            ex += (got == truth)
            co += sum((Counter(truth) & Counter(got)).values()); ct += len(truth)
        n = len(labels)
        print(f"{d}: exact={ex}/{n} ({100*ex/n:.1f}%)  char={100*co/ct:.1f}%  "
              f"{(time.time()-t0)/n*1000:.0f} ms/img")
