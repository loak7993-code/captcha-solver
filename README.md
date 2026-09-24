<p align="center">
  <img src="assets/banner.svg" alt="CAPTCHA Solver" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CPU--only-yes-success" alt="CPU only">
  <img src="https://img.shields.io/badge/text%20CAPTCHA-99.5%25%20exact-brightgreen" alt="Text CAPTCHA 99.5% exact">
  <img src="https://img.shields.io/badge/grid%20CAPTCHA-99.1%25%20%2F%203%20tries-brightgreen" alt="Grid CAPTCHA 99.1% within 3 tries">
  <img src="https://img.shields.io/badge/third--party%20services-none-informational" alt="No third-party services">
</p>

# CAPTCHA Solver

Two CAPTCHA solvers in one repo, both **CPU-only** (no GPU needed):

| Solver | Challenge type | Result |
|---|---|---|
| **Text** — `solve_pro.py` | "type the characters you see" | **99.5%** exact on the hard test set |
| **Grid** — `solve_grid.py` | "click all the traffic lights" | **79.6%** single-shot, **99.1%** within 3 tries |

Everything runs locally. No third-party solving service, no network calls at
inference time.

---

## One entry point

Both solvers sit behind a single call that picks the right one for you:

```python
from captcha_ai import solve
solve("captcha.png")                          # auto-detects -> text
solve("grid.png", target="traffic light")     # grid
```

```bash
python3 captcha_ai.py captcha.png
python3 captcha_ai.py grid.png --target "traffic light"
```

Routing: a `target` prompt means grid; otherwise a near-square image with
straight near-white separator bands between tiles is treated as a grid, and
anything else as text. Both paths return the same `Result` object.

**What this is not:** a single neural network. There is no one model here that
both reads text and selects tiles — the text and grid specialists are different
architectures, and the local vision-language-model route is blocked (the
image processor needs torchvision, whose C extension does not load against this
torch build). This is one *interface* over two specialists.

---

## Results

<p align="center">
  <img src="assets/results.svg" alt="Measured accuracy: text CAPTCHA 99.5% on hard tier, image grid 99.1% within 3 tries, Tesseract baseline 3.0%" width="620">
</p>

Measured on held-out test sets built by the included generators (200 images per
difficulty tier, lowercase 5–6 character codes). Exact match = whole string correct.

**Text CAPTCHAs**

| Reader | easy | medium | hard | char (hard) |
|---|---|---|---|---|
| Tesseract + preprocessing (baseline) | 62.5% | 44.5% | 3.0% | 53% |
| Pretrained CRNN (zero target knowledge) | 64.5% | 57.5% | 45.0% | 88% |
| **Local CRNN + beam + TTA (this repo)** | **100%** | **100%** | **99.5%** | **99.9%** |

**Image-grid CAPTCHAs** (240 held-out 3×3 grids of real photos, 10 classes)

| Method | precision | recall | exact | within 3 tries |
|---|---|---|---|---|
| Zero-shot CLIP | 95.6% | 90.6% | 74.6% | 98.4% |
| **Trained head on CLIP features** | **96.4%** | **93.8%** | **79.6%** | **99.1%** |

The grid number is single-attempt. Real CAPTCHAs let you request a fresh grid,
so the practical success rate is the "within 3 tries" column.

---

## How it works

### Text CAPTCHAs
```
image → preprocessing → CRNN (CNN + BiLSTM) + CTC → beam search
```
- **CTC loss** aligns a variable-length string to the image without needing to
  segment individual characters. This is the architecture essentially every
  published text-CAPTCHA solver uses.
- **Beam search** decoding finds the most likely string; plain argmax throws
  away probability mass.
- **Test-time augmentation** scores a few rescaled/rotated copies and averages
  the result.
- **Model soup** averages several independently trained models.

Three lessons that mattered more than the architecture (all in `NOTES.md`):
1. **Train on the hard distribution.** The first model scored 72.5% on the hard
   tier only because it had never seen that distortion.
2. **Never cold-start.** Training CTC from scratch collapses to a blank plateau
   roughly 60% of the time. Fine-tuning from a model that already emits
   characters removes that failure mode, and chaining rounds compounds accuracy.
3. **Refetch instead of guessing.** Most remaining errors are case/homoglyph
   collisions that are not recoverable from the image. Asking for a fresh
   CAPTCHA is free, so retrying beats a better model.

### Image-grid CAPTCHAs
```
grid image → find tile boundaries → split into tiles
           → CLIP score each tile against the challenge label
           → select tiles above threshold
```
- **CLIP zero-shot** scores any challenge label with no training — just "a photo
  of a traffic light".
- **Discriminative scoring** compares the target against a candidate class list,
  so the target has to actually beat plausible competitors.
- **Trained head** (a small MLP on frozen CLIP features) beats zero-shot once
  you have a few hundred labelled tiles.

---

## Install

```bash
pip install torch numpy pillow onnxruntime tokenizers
```

Download the pretrained models used by the convenience paths (they are not
committed — too large):

```bash
python3 fetch_models.py          # ~610 MB (CLIP ONNX) + ~14 MB (CRNN)
```

You can also train everything from scratch with the included scripts.

---

## Quickstart — text CAPTCHAs

```bash
python3 gen.py                    # build a labelled testbed (3 difficulty tiers)
python3 crnn.py data/train 35     # train a CRNN+CTC
python3 solve_pro.py              # score it on all tiers
```

Programmatic use:
```python
from solve_pro import solve_pro
text, score = solve_pro("captcha.png")
```

## Quickstart — image-grid CAPTCHAs

```bash
python3 fetch_commons.py          # pull labelled photos for a testbed
python3 solve_grid.py grid.png "traffic light" --classes bicycle bus car dog
python3 eval_grid.py 240          # measure on labelled grids
python3 grid_head.py 240          # train the head and compare with zero-shot
```

Programmatic use:
```python
from solve_grid import solve_grid
selected, scores, (rows, cols) = solve_grid("grid.png", "traffic light")
```

---

## Files

| File | Purpose |
|---|---|
| `captcha_ai.py` | **unified entry point** — one `solve()` call / one CLI, auto-routes text vs grid |
| `gen.py`, `gen_mixed.py` | synthetic text-CAPTCHA generators (labelled testbeds) |
| `crnn.py` | CRNN+CTC model, training loop, greedy decode |
| `solve.py` | Tesseract baseline pipeline |
| `solve_pro.py` | text solver: beam search + TTA + model soup |
| `solve_best.py` | simpler single-model text API (`solve_best`, `solve_batch`, `solve_with_retry`) |
| `train_from.py`, `train_pool.py`, `train_seed.py`, `campaign.sh` | fine-tuning and iterative training |
| `hf_local.py`, `hfmodel/` | dependency-free loader for a pretrained CRNN |
| `clip_onnx.py` | CLIP zero-shot via ONNX Runtime |
| `solve_grid.py` | grid solver (zero-shot + trained head) |
| `grid_head.py`, `grid_probe.py` | train and evaluate the grid head |
| `eval_grid.py`, `fetch_commons.py` | grid testbed build + evaluation |
| `pool_bench.py` | score a pool of checkpoints, pick the soup |
| `NOTES.md`, `NOTES_GRID.md` | full results and the traps found along the way |

---

## What this cannot do

**reCAPTCHA v3, hCaptcha, Cloudflare Turnstile.** These do not render an image
until after they have scored your browser fingerprint, cookies, cursor
telemetry and IP reputation — there is nothing to read or click. Beating them is
a browser-automation and fingerprinting problem, not a vision-model one.

This project is for automating services you own or are authorised to test, and
for measuring the strength of a CAPTCHA you deploy.

---

## License

MIT. See `LICENSE`.
