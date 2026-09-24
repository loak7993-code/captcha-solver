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
| first CERN model (6 k images, cold-start) | 47 / 50 (94%) | val 91.4% |
| **final CERN model (fine-tuned on the full 12 k corpus)** | **199 / 200 (99.5%)** | two independent batches: 100/100 then 99/100 |
| held-out synthetic (800 images) | 98.2% | up from 91.4% |

Live runs are server-validated: the service answered `200 {"message":"Valid"}` for
199 of 200, at ~1.0 s/img including network.

**0 / 40 → 199 / 200.** No real images were ever used for training.

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

## The two traps that each cost a run

**1. Cold-start CTC collapses to a uniform distribution.** Running on the full
12,000-image corpus at the same lr that had worked on a 6,000 subset produced
**20 epochs of `val-exact 0.000` with loss pinned at 4.130**. That number is the
signature: it is `ln(62)`, i.e. the model emitting a uniform distribution over
the 62 classes and never leaving it.

The cheap diagnostic, before blaming the data: *can it overfit 16 images?*
At lr 1e-2 it could (loss 0.60), which proves the code path is fine and the
failure is purely optimisation. Then the fix that always works:

```bash
python3 cern_solver.py train 10 0 cern_crnn_6k.pt    # continue, don't cold-start
```

Starting from a converged checkpoint, loss went 0.146 → 0.003 and val-exact
0.800 → 0.988 in ten epochs, with no plateau at any point. Same lesson as the
main text solver's campaign: **never cold-start CTC.**

**2. The alphabet has to be able to express the answer.** Covered above — a
31-character lowercase model cannot ever be correct against a case-sensitive
mixed-case target, no matter how long you train it.

## What actually moved the number

1. **Port the renderer exactly** (0/40 → 47/50). The single biggest lever.
2. **Train on the full corpus** (47/50 → 199/200), reached by fine-tuning rather
   than cold-starting.
3. **Beam search** — +2 on a 60-image sample for the 6 k model (51→53). Once the
   final model is at 98%+ it is saturated: `greedy = beam8 = beam8+TTA`, so the
   cheap decoder is used.
4. **TTA** — no measurable gain here. The renderer already rotates each glyph
   ±50°, so extra rotation views add little. Kept as an option, off by default.

## Where it still fails

1 of 200. The residual is the hardest interference: an opaque line crossing a
glyph erases the stroke that distinguishes it (a line through `f` making it read
as `l`). That is information destroyed at render time, not a modelling problem —
which is why a real deployment should re-request a CAPTCHA on a low-confidence
read rather than submit a guess. The `retry` action in `captcha_ai.decide()`
exists for exactly this.
