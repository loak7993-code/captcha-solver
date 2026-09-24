"""Text-CAPTCHA solver pipeline.

Preprocess an image into a clean binarised strip, then OCR it. Drop-in
per-stage so you can benchmark each defence (noise, lines, warping) against
the solver and see what actually raises the bar.

Backends:
  tess  - Tesseract OCR (ships everywhere, good on low-noise text)
  cnn   - plug in your own CRNN/CTC model (see train note at bottom)
"""
import sys, json, os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageFilter, ImageOps

ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def preprocess(path, mode="full"):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    if mode in ("full", "denoise"):
        img = img.filter(ImageFilter.MedianFilter(3))
    arr = np.array(img).astype(np.float32)

    # Otsu threshold -> binary, then remove thin interference lines with
    # a morphological opening, then close gaps inside strokes.
    if mode == "full":
        arr = otsu(arr)
        arr = remove_lines(arr)
        arr = morph_close(arr)
    else:
        arr = np.where(arr < 128, 0.0, 255.0)

    out = Image.fromarray(arr.astype(np.uint8))
    # upscale: tesseract does much better with tall glyphs
    out = out.resize((out.width * 3, out.height * 3), Image.LANCZOS)
    out = ImageOps.expand(out, border=15, fill=255)
    return out


def otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0
    best, thr = 0.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return np.where(arr < thr, 0.0, 255.0)


def remove_lines(binarr):
    """Opening with a horizontal then vertical structuring element kills
    1-2px lines while keeping thicker glyph strokes."""
    ink = (binarr < 128).astype(np.uint8)
    def erode(a, k):
        p = np.pad(a, k, constant_values=0)
        out = a.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                out &= p[k + dy:k + dy + a.shape[0], k + dx:k + dx + a.shape[1]]
        return out
    def dilate(a, k):
        p = np.pad(a, k, constant_values=0)
        out = a.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                out |= p[k + dy:k + dy + a.shape[0], k + dx:k + dx + a.shape[1]]
        return out
    ink = dilate(erode(ink, 1), 1)
    return np.where(ink > 0, 0.0, 255.0)


def morph_close(binarr):
    ink = (binarr < 128).astype(np.uint8)
    p = np.pad(ink, 1, constant_values=0)
    dil = ink.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= p[1 + dy:1 + dy + ink.shape[0], 1 + dx:1 + dx + ink.shape[1]]
    p2 = np.pad(dil, 1, constant_values=0)
    ero = dil.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ero &= p2[1 + dy:1 + dy + ink.shape[0], 1 + dx:1 + dx + ink.shape[1]]
    return np.where(ero > 0, 0.0, 255.0)


def tess_ocr(img, psm=8):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        tmp = f.name
    cfg = f"--psm {psm} -c tessedit_char_whitelist={ALPHABET}"
    try:
        r = subprocess.run(["tesseract", tmp, "stdout", *cfg.split()],
                           capture_output=True, text=True, timeout=20)
        txt = r.stdout.strip()
    finally:
        os.unlink(tmp)
    return "".join(c for c in txt.lower() if c in ALPHABET)


def solve(path, mode="full", backend="tess"):
    img = preprocess(path, mode=mode)
    if backend == "tess":
        return tess_ocr(img, 8)
    raise SystemExit("cnn backend: supply your own model.load()")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/easy"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    labels = json.load(open(os.path.join(d, "labels.json")))
    ok = exact = 0
    char_ok = char_tot = 0
    import time
    t0 = time.time()
    for fn, truth in labels.items():
        got = solve(os.path.join(d, fn), mode=mode)
        exact += (got == truth)
        # char-level
        from collections import Counter
        tc, gc = Counter(truth), Counter(got)
        char_ok += sum((tc & gc).values())
        char_tot += len(truth)
        ok += 1
    dt = time.time() - t0
    print(f"{d} mode={mode}: exact={exact}/{ok} ({100*exact/ok:.1f}%)  "
          f"char-acc={100*char_ok/char_tot:.1f}%  {dt/ok*1000:.0f} ms/img")
