# One brain: a single multi-task neural network

## What this is

`brain.py` is **one neural network** that solves both CAPTCHA types. Not a router
over two models — one set of weights, one checkpoint, one forward pass:

```
image ──► shared convolutional trunk ──┬──► CTC sequence head   ──► "a7Kq9"
                                       │
                                       └──► tile class head     ──► {bicycle: 0.02, ...}
```

The trunk (5 conv blocks, 2.2 M parameters total) is shared. Which head you read
out depends on the task, not on which network you loaded. `captcha_ai.solve(...,
backend="brain")` uses it for both.

## Why this is the right shape (the "System One" idea)

Worth naming the reference the request came from, because the architecture lines
up exactly:

- **Jev** (TypeSafe AI, released 2026-09-15) is a **System One decision model**.
  It is *non-autoregressive*: instead of generating text token by token it
  returns **all outputs at once**, and what it returns is **typed** — a choice
  distribution, a score, a probability with calibrated confidence — rather than
  prose. Reported 70–500 ms, orders of magnitude cheaper than autoregressive
  LLMs on classification, and hallucination-free *by construction* because it
  cannot emit free text. Built by Diogo Almeida (ex-OpenAI, RLHF/ChatGPT
  co-inventor); the name references William Stanley Jevons.
- **Laya** is an unrelated family of products (an open-source local-first
  notification command centre at laya.aay.sh, an AI-native project manager at
  laya.work, and an acquired Munich travel startup). None of them are a single
  neural network, so there is no architectural lesson to take from them.
  Different thing entirely.

The CAPTCHA brain is the same design pattern as Jev, applied to vision:

| | Jev | this brain |
|---|---|---|
| generation | non-autoregressive, one pass | non-autoregressive, one pass |
| output | typed decision + probabilities | string via CTC, or class probabilities |
| free text | none | none (greedy/beam over a fixed alphabet) |
| text task | — | **CTC is inherently non-autoregressive**: it decodes the whole string in one pass, unlike a VLM writing characters one at a time |

That is the honest reason "one brain" is a good fit here: CTC was already a
single-pass, typed-output decoder, so a multi-task trunk loses nothing
architecturally.

## The trade-off (measured)

The single brain is **a little behind the two specialists**, and that is the real
cost of one network:

- The trunk has to split its capacity between two different visual tasks.
- The tile task is **data-starved** relative to text: a few hundred real photos
  versus thousands of synthetic text CAPTCHAs.
- The specialists can each be optimal: the text specialist is a fine-tuned CRNN
  with TTA and a model soup; the grid specialist rides a CLIP backbone trained on
  400 M image–text pairs, which a 2 M-parameter trunk trained from scratch cannot
  match on tile semantics.

So the repo keeps both:

| backend | what it is | when to use it |
|---|---|---|
| `specialists` *(default)* | one model per task | best accuracy |
| `brain` | **one multi-task network** | when you want a single network |

Both are reachable through the same call, so it is still one entry point either
way.

```python
from captcha_ai import solve
solve("captcha.png", backend="brain")
solve("grid.png", target="traffic light", backend="brain")
```

```bash
python3 captcha_ai.py captcha.png --backend brain
python3 brain.py 18          # train it (~10 min, CPU only)
```

## Results

Trained on the mixed text corpus plus 40 labelled photos per tile class,
12 epochs, warm-started (`python3 brain.py 12 brain.pt crnn_best.pt 1e-3`).

| task | single brain | specialists | gap |
|---|---|---|---|
| text CAPTCHAs (200 hard) | 84.0% | 99.5% | −15.5 pts |
| image-grid CAPTCHAs (120 grids) | **6.7%** | 83.3% | **−76.6 pts** |

Per-tile classification accuracy is 51%, and 0.51 across nine tiles is why the
grid number collapses — the tile head is not close to usable.

### Why the grid number is so bad

Not a training bug. It is structural:

1. **The trunk is forced to a grayscale 48px input** because it is shared with the
   text task, which needs height 48. Object recognition wants 224px **colour** —
   a `traffic light` at 48px greyscale is a few grey pixels.
2. **The tile task is data-starved.** ~400 real photos across 10 classes trains a
   2.2 M-parameter trunk from scratch. The specialist rides **CLIP**, whose vision
   tower saw 400 M image–text pairs; nothing trained on 400 images competes with
   that on object semantics.
3. **The trunk then has to serve two incompatible objectives** at once, so it is
   optimal for neither. Text lost 15 points to the sharing too.

### The honest conclusion

The one-brain design is architecturally sound — it is the same non-autoregressive,
single-pass, typed-output pattern as Jev, and CTC was already a single-pass
decoder, so nothing was lost *in principle*. But **sharing a trunk across two
visual tasks costs far more than it saves** when the tasks want different input
resolutions and one of them is data-starved.

**Recommendation: use `backend="specialists"` (the default).** It is also one
entry point and one `solve()` call; the brain is kept because it is a real,
different design and because on text-only workloads it is competitive-ish at 84%.

If you want one network *and* good tiles, the fix is to stop sharing a from-scratch
trunk: use CLIP's vision tower as the shared trunk (frozen, 224px RGB) and attach
both heads to it. That was the original plan and it is untested — the text head
would need to read a sequence from CLIP patch tokens, which at 7×7 patches is
probably too coarse. Worth an experiment, not a claim.

