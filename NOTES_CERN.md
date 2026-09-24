# Real-CAPTCHA test: CERN captcha-api

## The target

`https://captcha.web.cern.ch/api/v1.0/captcha/` — a public, open-source
([CERN/captcha-api](https://github.com/CERN/captcha-api)) CAPTCHA service with a
validation endpoint, so answers are checked **server-side** rather than eyeballed:

```bash
GET  /api/v1.0/captcha/          -> {"id": "...", "img": "data:image/jpeg;base64,..."}
POST /api/v1.0/captcha/          -> {"id": "...", "answer": "..."}
                                 200 {"message": "Valid"} | 400 {"message": "Invalid answer"}
```

Verified by reading one image by hand: it read `PJT53J`, and `PJT53J` returned
**200 Valid**. The service is **case-sensitive** (lowercase → rejected).

## Result

| | exact | notes |
|---|---|---|
| the shipped synthetic-trained solver | **0 / 40** | structurally impossible — see below |
| **this solver, live API, server-validated** | **47 / 50 (94%)** | 1082 ms/img incl. network |
| held-out synthetic (800 images) | 91.4% | |

## Why the shipped model scored 0/40

Not a tuning problem. Its alphabet is `abcdefghjkmnpqrstuvwxyz23456789` — 31
lowercase characters, no `i l o 0 1`. The CERN service emits **mixed case** and
digits `1-9`, and validates case-sensitively, so a lowercase-only model can never
be correct. It was also genuinely misreading (`jejy447`, `4xjj2d`) because it had
only ever seen this repo's own synthetic generator.

## The fix: replicate the renderer exactly

CERN's generator is open source, so instead of guessing the distribution I ported
it (`cern_gen.py`):

- font **DejaVuSerif size 36**, canvas 250×60 on `(250,250,250)`
- 6 characters; each character drawn from ONE randomly chosen set
  (digits `1-9` | `A-Z` | `a-z`) — so case is mixed **within** an image
- each glyph rendered alone, **rotated −50…+50°**, colourised to a random RGB in
  120–200, pasted at `x = 250·0.13·(i+1)`, `y = 12`
- 15 interference lines + 16 points, same colour range
- saved as **JPEG**, so compression artefacts are part of the distribution

The only deviation: Pillow ≥ 10 removed `font.getsize`, so the per-glyph canvas is
sized with `getlength()` / `getmetrics()`, reproducing `getsize`'s
(advance, ascent+descent) semantics.

Then trained a CRNN+CTC with the correct 61-character **case-sensitive** alphabet
(`1-9A-Za-z`, + CTC blank):

```bash
python3 cern_gen.py            # 12k training images, 800 held-out
python3 cern_solver.py train 18 6000
python3 cern_solver.py live 50 # against the live API, server-validated
```

This is exactly the "a faithful renderer is the key lever" result from the
literature, reproduced end-to-end: **0/40 → 47/50** with no real images used for
training at all.

## The trap that cost a run

The first training attempt (lr 1.5e-3, 14 epochs) **collapsed to the CTC blank
plateau** — val-exact 0.000 for every epoch, loss stuck at ~3.4. Before blaming
the data I tested whether the model could overfit 16 images: at lr 1e-2 it could
(loss 0.60), proving the code path was fine and it was purely an optimisation
failure. Retraining at **lr 5e-3** converged cleanly in 18 epochs
(loss 4.7 → 0.04, val 93.5%).

Same lesson as the main text solver: cold-start CTC is unstable, and the cheap
diagnostic is always *can it overfit a tiny batch?* — if yes, it is the
optimiser, not the model.

## Where it still fails

3 of 50. The residual is the hardest interference: an opaque line crossing a
glyph erases the stroke that distinguishes it (e.g. a line through `f` making it
read as `l`). That is information destroyed at render time, not a modelling
problem — which is why real deployments should re-request a CAPTCHA on a
low-confidence read rather than submit a guess.
