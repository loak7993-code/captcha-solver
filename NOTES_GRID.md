# Image-selection ("click all the X") CAPTCHA solver

Text-CAPTCHA work does **not** transfer to this: there is no text, no CTC, no
OCR. This is zero-shot visual classification of grid tiles against a prompt.

## Pipeline (`solve_grid.py`)

```
grid image
  -> detect_grid(): locate near-white separator rows/cols -> (rows, cols)
  -> split_grid():  equal split with an inset to drop separator borders
  -> score_tiles(): CLIP zero-shot score per tile for the challenge label
  -> select tiles above threshold that also WIN against the candidate classes
```

Two scoring modes:
- **discriminative** (recommended): softmax over the target *plus every
  candidate class*, e.g. `["bicycle","traffic light","bus",...]`. The target has
  to actually beat plausible competitors — far stronger than comparing against
  vague "something else" prompts.
- **open**: target prompt variants vs generic negatives. Use when the label
  space is unknown; measurably worse.

**TTA**: every tile is scored at original, horizontally flipped and slightly
zoomed views, probabilities averaged. Cheap, and it steadied the decision.

## Why ONNX and not transformers

`torchvision`'s C extension fails to load against this torch build
(`libc10.so` resolve failure, then `operator torchvision::nms does not exist` —
a stable-ABI mismatch). `transformers.CLIPModel` needs torchvision, so CLIP runs
from an ONNX export instead: `vision_model.onnx` and `text_model.onnx`
(Xenova/clip-vit-base-patch32), driven by `onnxruntime`, with CLIP image
preprocessing and BPE tokenisation done by hand. No torchvision anywhere.

## Measured results

Testbed: 340 real Wikimedia photos, 10 classes, composited into labelled 3x3
grids (k = 1..4 positives, rest negatives from other classes, white separators).
Test grids use **only held-out images** (per-class 24 train / 10 test).

240 grids, zero-shot CLIP, discriminative + TTA:

| threshold | precision | recall | all-positives | exact (solve) | within 3 tries |
|---|---|---|---|---|---|
| 0.20 | 97.7% | 87.8% | 77.9% | 74.2% | **98.3%** |
| 0.30 | 98.2% | 86.6% | 76.7% | 74.2% | 98.3% |
| 0.50 | 98.6% | 84.6% | 73.3% | 71.2% | 97.6% |

**The realistic CAPTCHA criterion is multi-attempt**: a failed read just means
refetching a fresh grid. At 74% single-shot, success within 3 attempts is 98.3%.

Note the exact-match column is unforgiving by construction — all 9 tiles must be
right, so a 97% per-tile precision still caps the grid. Precision being higher
than recall means errors are mostly *missed* positives, which a retry fixes.

### Comparison of scoring approaches

| method | precision | recall | exact | within 3 tries |
|---|---|---|---|---|
| open (target vs generic negatives) | 81.3% | 91.4% | 50.8% | — |
| discriminative + TTA (zero-shot) | 97.7% | 87.8% | 74.2% | 98.3% |
| linear probe on CLIP features | 96.6% | 78.5% | 58.8% | 93.0% |

The linear probe (`grid_probe.py`, multinomial logistic regression on frozen
CLIP embeddings, flip-augmented) **underperformed zero-shot** even with 480
training embeddings. With only ~24 labelled images per class it is conservative
(very high precision, low recall). Zero-shot CLIP is the better default at this
data scale; a probe — or a fine-tuned detector like the YOLOv8 in *Breaking
reCAPTCHAv2* — wins only with many more labelled tiles.

### Trained head (final) — now the default when available

With 44 labelled tiles/class the MLP head on frozen CLIP features finally beats
zero-shot, and the gap is mostly recovered recall (missed positives):

240 held-out grids, `grid_head.py`:

| method | precision | recall | exact | within 3 tries |
|---|---|---|---|---|
| zero-shot CLIP | 95.6% | 90.6% | 74.6% | 98.4% |
| **MLP head (512→512→C)** | **96.4%** | **93.8%** | **79.6%** | **99.1%** |

`grid_head.pt` (weights + class list + feature normalisation) is saved and
`solve_grid` auto-selects the head whenever the target is in its class list,
falling back to zero-shot otherwise. Train it with `python3 grid_head.py 240`.

## Usage

```python
from solve_grid import solve_grid, solve_grid_with_attempts
sel, scores, dims = solve_grid("grid.png", "traffic light",
                               candidates=["bicycle","traffic light","bus","car", ...])
solve_grid_with_attempts(fetch, "traffic light", attempts=3, candidates=[...])
```

```bash
python3 solve_grid.py grid.png "traffic light" --classes bicycle bus car dog cat
python3 eval_grid.py 240          # rebuild the testbed and score
python3 grid_probe.py             # zero-shot vs linear probe
```

## Honest boundaries

- The candidate class list does a lot of work. With the real label space unknown,
  fall back to open mode (≈50% single-shot) or seed `candidates` with the classes
  the challenge usually uses.
- Tile accuracy is ~97% precision / ~88% recall **zero-shot**. Fine-tuning with a
  few hundred labelled tiles per class is the path to 95%+ per tile and near-100%
  single-shot grids.
- reCAPTCHA v3 / Turnstile are still not touched: no tile grid is ever rendered;
  the score comes from fingerprint, cookies and telemetry before any image.
