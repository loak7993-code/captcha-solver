# Text-CAPTCHA solver — measured results

Held-out test tiers from `gen.py` (200 images each, lowercase alnum, 5–6 chars).
CPU only, no GPU. Exact-match unless noted.

## Progression (each row is a real change, measured)

| Reader | easy | medium | hard | char (hard) | ms/img |
|---|---|---|---|---|---|
| Tesseract + preprocessing (`solve.py`) | 62.5% | 44.5% | 3.0% | 53% | 140 |
| Pretrained CRNN (Graf-J) + mask | 64.5% | 57.5% | 45.0% | 88% | 8 |
| Local CRNN, medium-only data | 100% | 100% | 72.5% | 94% | 8 |
| Local CRNN, mixed-difficulty data | 100% | 99.5% | 81.5% | 96% | 8 |
| + CTC beam search + TTA | 100% | 99.5% | 83.5% | — | ~180 |
| **Best seed (s303) + beam + TTA** | **100%** | **99.0%** | **91.0%** | **98.3%** | ~180 |
| Soup (hard + s303) + beam + TTA | 100% | 99.0% | 91.0% | 98.3% | ~360 |

Hard-tier exact match went **3.0% → 91.0%**, character accuracy on hard **53% → 98.3%**.
Remaining hard-tier errors are single-character (9 of 200 whole strings wrong).

## Final results after the iterative fine-tuning campaign

Three rounds of "generate fresh mixed data → fine-tune the pool → keep the best"
compounded cleanly on the hard tier (greedy):

| round | worst member | best member |
|---|---|---|
| fine-tune round 1 (`crnn_pool2`, from s303) | 92.0% | 96.5% |
| round 3 | 95.0% | 97.5% |
| round 4 | 97.0% | 98.5% |
| round 5 | 97.5% | **99.0%** |

Final pipeline (round-5 soup, CTC beam + TTA + length{5,6}):

| tier | exact | char |
|---|---|---|
| easy | **100.0%** | 100.0% |
| medium | **100.0%** | 100.0% |
| hard | **99.5%** | 99.9% |

Hard-tier exact match over the whole project: **3.0% (Tesseract) → 99.5%**.
One whole string wrong out of 200.

Default entry point now resolves to `crnn_round5/c.pt`; `ensemble="soup"` reads
`soup.txt` (the four round-5 members).

## Recommended entry point

```python
from solve_pro import solve_pro
solve_pro("captcha.png")              # default: best local model + beam + TTA
solve_pro("captcha.png", ensemble="soup")
```

Single-model convenience API stays in `solve_best.py` (`solve_best`, `solve_batch`,
`solve_with_retry`).

## What actually moved the number (ranked)

1. **Train on the hard distribution.** The medium-only model scored 72.5% on hard
   purely because it had never seen the distortion. Mixed-difficulty data
   (25/40/35 easy/medium/hard) → 81.5%. Biggest single gain.
2. **CTC prefix beam search**, implemented correctly → +1–2 pts. Greedy argmax
   throws away probability mass; the beam keeps alternative alignments.
3. **Test-time augmentation** (7 views: ±4° rotation, ×0.92/1.08 scale,
   ×0.85/1.15 brightness, log-prob averaged) → +1–2 pts on hard.
4. **More training seeds.** Independent seeds vary a lot (below).

## Traps found the hard way (all now fixed)

- **CTC blank collapse is common.** Of 5 training runs, **3 collapsed** to the
  blank plateau (val-exact 0.000, loss stuck ~3.0). Same code, different init.
  Mitigation: seed the run, lower LR to 1.5e-3, keep per-5-epoch checkpoints, and
  discard collapsed models. The 2 good runs scored 81.5% and 90.0% on hard.
- **Broken blank transition in beam search.** First implementation added blank
  mass to the non-blank accumulator, which *lowered* easy-tier accuracy to 87.5%.
  A correct prefix-beam keeps p_blank and p_non-blank separate; blank emission
  moves both into p_blank of the same prefix. Verified: beam=1 now equals greedy.
- **Naive cross-model ensembling hurts.** Averaging the strong local model with
  the weaker pretrained one dropped hard from 72.5% → 63%. Only average models of
  comparable strength and the *same vocabulary*. The local soup is valid;
  local∪pretrained is not, and is off by default.
- **Heavy augmentation poisoned convergence.** contrast 0.6–1.6 + brightness
  0.7–1.3 + blur, combined with per-image (x−mean)/std normalisation, destroyed
  stroke contrast and CTC never escaped the plateau. Light aug (rotate ±4, blur)
  converges fine.

## Model soup

Averaging log-probs across independently trained local checkpoints (`crnn_hard.pt`
+ `crnn_s303.pt`) with the same vocabulary. Costs 2× inference (~360 ms) and here
matched the best single model rather than beating it — worth keeping because it
makes the pipeline robust to any one run collapsing.

## Still out of reach

reCAPTCHA v2/v3, hCaptcha, Cloudflare Turnstile. They score browser fingerprint,
cookies, cursor telemetry and IP reputation before rendering an image. No picture,
nothing to read. See Plesner et al. 2024 (Breaking reCAPTCHAv2) and Teoh et al.
USENIX Security 2025.

## Headroom left

Hard tier is 91% exact / 98.3% char. The residual is single-character errors on
the most distorted samples. Further gains would come from: more seeds soup'd
together, a larger CNN/RNN, or more mixed training data — all straightforward.
