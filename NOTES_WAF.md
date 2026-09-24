# The grid solver on AWS-WAF tiles — the problem, the evidence, the fix

> Converted from `NOTES_WAF.txt`. The two API problems it flagged are now fixed
> in `solve_grid.py` — see [§6](#6-api-note-fixed).

**Status:** open item resolved to a first approximation by domain-adapting the
head. On held-out WAF grids, exact-set match went **30% → 70%**. Not yet at the
VLM reference (95%), but the mechanism is proven, the labels are free, and the
remaining gap is data volume + label hygiene, not method.

---

## 1. What the problem is

The grid specialist is benchmarked on **natural photographs** (`fetch_commons.py`
pulls Wikimedia photos; `eval_grid.py` composites 3×3 grids from them). That is a
legitimate benchmark for *natural-photo* grid CAPTCHAs.

AWS WAF — the challenge Amazon actually serves — is different in kind. Its
`synthetic_grid` tiles are **ML-generated adversarial images**, chosen so that
each distractor is a near-miss of the target. On these, the natural-photo
benchmark number does not transfer, and it is not a tuning problem:

```
target = "bucket"   (real WAF tiles; the VLM's independent read in parens)
  tile:     1      2      3      4      5      6      7      8      9
          (pot) (bucket)(tub)(bucket)(clock)(clock)(hat)(suitcase)(bucket)

  zero-shot scores :  .997   .991   .998   .999   .060   .029   .897   .615  1.000
  → selected        : 1,2,3,4,7,8,9            (7 of 9 — badly over-selected)
```

The model separates **clocks** cleanly (0.03 / 0.06) but treats *every*
container-ish object as a bucket — pot .997, tub/planter .998 — and even fires on
a hat (.897) and a suitcase (.615). No threshold saves this: **the near-misses
score above the true positives.**

The same grid with a visually distinct target is flawless:

```
target = "clock"
  scores   :  .000   .001   .000   .000  1.000  1.000   .000   .000   .000
  → selected: 5, 6                     ✅ exact
```

So the solver is not "broken on WAF tiles". It is **semantically exact when the
target is distinct, and loses the fine boundary when the target has near-miss
confusables** — precisely what the adversarial generator is built to attack.

### The confusable pair, in one line

Discriminative mode (candidate classes) removes most false positives — clocks,
hat, suitcase all drop to ~.00 — but inverts the one pair that matters:

```
candidates = [bucket, pot, tub, planter, hat, suitcase, clock, bag, cup]
  scores: [ .721      .659     .320    .779    .000    .000    .012   .002   .972 ]
           (pot)    (bucket)  (tub)  (bucket)                    (bucket)
                    ↑ the real bucket scores BELOW the pot
```

### What does **not** fix it (independently confirmed)

Commit `bd7df6a` field-tested the same pair and found **better prompts and the
larger CLIP (L/14) both fail** — L/14 even raised the confidences on the wrong
tiles. That matches the mechanism above: this is a domain gap plus a fine
semantic boundary, neither of which a bigger frozen encoder or a cleverer prompt
closes. It needs a model that has seen the domain.

---

## 2. Why the existing benchmark cannot catch this

| benchmark | tiles | oracle | catches WAF transfer failure? |
|---|---|---|---|
| text (`gen.py` tiers) | synthetic | known label | n/a (different task) |
| grid (`eval_grid.py`) | **natural photos** | known label | **no — never sees adversarial tiles** |
| CERN (`cern_solver.py live`) | real service | **server 200/400** | n/a (different task) |
| **WAF grid** | **adversarial synthetic** | **server accept/reject** | **← the missing row** |

The grid benchmark is in-distribution *and* on a different image domain. There
was no measurement that could have surfaced this until the WAF oracle was used.

---

## 3. The fix that works: adapt the head on WAF tiles

The head is a small MLP on frozen CLIP features (`grid_head.py`). It only needs
labelled tiles — and the labels are free, because the WAF server grades answers.

### The oracle

```
GET  ait.<region>.captcha.awswaf.com/ait/ait/ait/problem
       → { target word, 9 tiles (base64), state, key, hmac_tag }
POST ait.<region>.captcha.awswaf.com/ait/ait/ait/verify
       → { success: true,  captcha_voucher: "<...>" }   == ACCEPTED
       → { success: false, ... }                        == REJECTED
```

An accepted answer means **the submitted tile set is the true set** — so every
tile in a WAF problem gets a server-verified label for free. No human labelling,
no paid service.

### Harvest result

```
41 problems harvested, 41 server-ACCEPTED (the proposed set was validated)
369 tiles, 20 classes:
  bucket 52  bag 40  hat 38  clock 32  belts 25  bed 22  suitcase 21
  spoon 20   fork 20  binocular 10  scissors 12  umbrella 10  seat 10
  cooking_pot 10  sofa 9  cap 6  pot 4  chair 3  tub 2  duffel 2
```

### Train + held-out evaluation

31 problems for training (279 tiles), **10 problems held out entirely**. Metric =
*exact-set match* against the server-accepted answer set — the same criterion the
WAF itself enforces. The shipped `grid_head.pt` is never touched.

```
── zero-shot CLIP (candidates = corpus classes, th 0.5) ──   exact 3/10  (30%)
     ✅ spoon, fork, clock
     ❌ bag, cooking_pot, suitcase, bed, hat, bucket, belts   (each off by 1-2 tiles)

── WAF-trained head (same held-out 10) ───────────────────   exact 7/10  (70%)
     ✅ bag  ✅ spoon  ✅ fork  ✅ clock  ✅ bed  ✅ bucket  ✅ belts
     ❌ suitcase (picked 3, missed 2)   ❌ hat (missed 1)
     ⚠  cooking_pot → class-name mismatch (now handled, see §6)
```

**30% → 70% on the exact-set metric, from a 2.3× smaller-than-ideal corpus.**
Four of the seven zero-shot failures (`bag`, `bed`, `bucket`, `belts`) are fixed
outright — all confusable-class cases. The mechanism is confirmed: **the failure
was domain gap, and the head closes it.**

Live, server-verified, single attempt: the zero-shot solver was accepted on
**3/7 grids (43%)**.

---

## 4. The remaining gap, honestly

| | exact-set match | notes |
|---|---|---|
| zero-shot CLIP | 30% (held-out), 43% (live) | shipped behaviour |
| **WAF-trained head** | **70%** (held-out) | 31 training problems |
| VLM reference | **95%** (41/43 live, server-verified) | the ceiling to chase |

Three known causes of the remaining distance, in order of expected payoff:

1. **Data volume.** 31 training problems is thin. ~150–200 problems (~1,400
   tiles) is a few hours of harvesting at the current rate.
2. **Label hygiene.** Free-form object names fragmented the vocabulary into
   **27 head classes** with synonyms split apart: `bag`/`handbag`,
   `belt`/`belts`, `chair`/`armchair`/`bench`, `sofa`/`couch`,
   `suitcase`/`briefcase`/`duffel`, `cooking pot`/`pot`. Merging these both adds
   training mass per class and removes the lookup mismatch noted above.
3. **The hardest residues** — `suitcase` and `hat` — are the genuine near-miss
   pairs (a suitcase is a box with a handle; a hat sits on a curved surface).
   Same class of error as `NOTES_CERN.md`: information that is hard to recover,
   best handled by *re-requesting* on low confidence rather than guessing.

---

## 5. Why this matters beyond one target

A VLM path is accurate (95%) but costs a network round trip per grid and a
per-call fee. A domain-adapted local head is ~600 ms on CPU, offline, and with
more data should approach the VLM number. The pattern generalises:

> For any adversarial synthetic grid CAPTCHA, the label source is the challenge's
> own **validation endpoint**. Harvest with a VLM as the proposer, keep only
> server-accepted answers, and the corpus labels itself. Then adapt the small
> head — never the frozen encoder.

Same lever as `NOTES_CERN.md` ("replicate the target's own renderer"), reached
from the other side: there, port the generator; here, let the validator label.

---

## 6. API note (fixed)

`NOTES_WAF.txt` flagged that `solve_grid.solve_grid()` had no `head_path`
argument, so a domain-adapted head could only be used by mutating module globals:

```python
# OLD (no longer needed)
solve_grid.HEAD_PATH = "/tmp/waf-head.pt"
solve_grid._HEAD = None
```

Worse, that workaround **didn't actually work**: `_head()` cached one global, so
asking for a second head silently returned the first one loaded.

Both are fixed in `solve_grid.py`:

- **`head_path=` is now a parameter** of `solve_grid()`, `score_tiles_head()`
  and `head_available()`.
- **The cache is keyed by resolved path** (`_HEADS`), so multiple heads coexist.
- **A target outside the head's vocabulary no longer raises**
  (`ValueError: 'x' is not in list` → returns an empty selection, so the caller
  can retry or fall back to zero-shot with `candidates`).

```python
from solve_grid import solve_grid
sel, scores, dims = solve_grid(img, "bucket", head_path="/tmp/waf-head.pt", threshold=0.5)
```

### Reproducing the harvest

The harvest/train/eval scripts live outside this repo (they need egress through a
clean exit IP, and the corpus is regenerated per target). At a high level:

```bash
python3 harvest_waf.py 60      # problem -> propose -> /verify grade -> labelled tiles
export CLIP_ONNX_DIR=/root/models/clip
python3 train_waf_head.py      # -> waf-head.pt   (repo head untouched)
python3 use_your_solver.py 14  # live, server-verified acceptance rate
```

`train_waf_head.py` reuses `grid_head.train_head` / `grid_head.emb`, so the
method is entirely inside this repo even though the harness is not.
