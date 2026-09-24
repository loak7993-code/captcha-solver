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

## Field test: adversarial confusables (open item)

A field test on a real WAF grid found the discriminative zero-shot path is exact
for clear targets but inverts on a confusable pair. Reproduced and investigated:

| target | selection | verdict |
|---|---|---|
| `clock` | tiles 4, 5 | ✅ exact |
| `hat` | tile 6 | ✅ exact |
| `suitcase` | tile 7 | ✅ exact |
| `bucket` | tiles 0, 1, 2, 3, 8 | ⚠ selects buckets **and** bucket-shaped planters |
| `pot` | (none above threshold) | ⚠ |

Two cheap fixes were tested and **both fail** — which is the useful result:

1. **Prompt disambiguation** (`prompt_override`), giving each class
   distinguishing attributes — `bucket` → "a metal bucket / a galvanised bucket /
   a red plastic bucket / a pail", `pot` → "a flower pot / a terracotta pot / a
   planter with soil". Bucket still selects the same 5 tiles; pot then selects
   *nothing*. No improvement.

2. **A bigger backbone.** Swapped CLIP ViT-B/32 for **ViT-L/14** (int8 ONNX,
   415 MB). Same selection on every target — and on `bucket` the confidences got
   *higher* (0.94–1.00 vs 0.79–0.99). More capacity did not separate the pair;
   it made the existing judgement more confident.

**Conclusion.** The pair is not separable by prompting or by model scale. These
tiles are all cylindrical metal/plastic containers — "bucket" vs "pot" is a label
distinction the image alone does not carry at this granularity. The fix is
**domain adaptation**: label a few hundred tiles of the actual target corpus and
retrain the head on their CLIP features (the head is a small MLP, so this is
minutes of work *once the labels exist*).

That is blocked on data: there is one WAF grid (9 tiles) available locally and no
ground truth, which is neither enough to train nor enough to *measure* an
off-by-one. Needed before the loop can run:

- a corpus of challenges from the target (a WAF-protected endpoint you control,
  fetched at volume), and
- a label oracle — for AWS WAF the service's own validate endpoint is the clean
  one, exactly as `cern_solver.py live` uses `captcha.web.cern.ch` to confirm
  every answer.

The machinery is straightforward once those exist: `fetch → tile → label via
oracle → CLIP features → retrain grid_head → re-measure`, and
`prompt_override` (already in `solve_grid`) covers the cases where prompting *is*
enough.

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
