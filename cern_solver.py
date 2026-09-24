"""cern_solver.py — CRNN+CTC reader for the CERN captcha-api style.

Own 61-character alphabet (1-9, A-Z, a-z) because case is significant and the
live service validates it case-sensitively. Shares the CRNN architecture from
crnn.py; only the alphabet and the data pipeline differ.

    python3 cern_solver.py train [epochs]
    python3 cern_solver.py eval                 # held-out synthetic set
    python3 cern_solver.py live [n]             # against the live CERN API
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from crnn import CRNN, collate

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHABET = ([chr(i) for i in range(49, 58)] +
            [chr(i) for i in range(65, 91)] +
            [chr(i) for i in range(97, 123)])          # 61 chars, case-sensitive
IDX2CH = {i + 1: c for i, c in enumerate(ALPHABET)}
IDX2CH[0] = ""
CH2IDX = {c: i for i, c in IDX2CH.items()}
NCLS = len(ALPHABET) + 1
H = 48
CKPT = os.path.join(HERE, "cern_crnn.pt")
ONNX_PATH = os.path.join(HERE, "cern_crnn.onnx")


class OnnxCRNN:
    """Drop-in replacement for the torch CRNN, running under ONNX Runtime.

    Measured 2x faster than PyTorch on CPU for this model (4.0 vs 8.3 ms/img),
    with identical predictions. Exposes __call__ so logp()/predict2() are
    unchanged."""

    def __init__(self, path=ONNX_PATH, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def eval(self):
        return self

    def __call__(self, x):
        arr = x.detach().cpu().numpy().astype(np.float32)
        out = self.sess.run(None, {"x": arr})[0]
        return torch.from_numpy(out)


def export_onnx(ckpt=CKPT, out=ONNX_PATH):
    """Export the trained CRNN to ONNX (batch and width dynamic)."""
    import warnings
    m = load(ckpt, backend="torch")
    dummy = torch.randn(1, 1, H, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(m, dummy, out, input_names=["x"], output_names=["logp"],
                          dynamic_axes={"x": {0: "batch", 3: "width"},
                                        "logp": {0: "batch", 1: "T"}},
                          opset_version=17, dynamo=False)
    print(f"exported {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    return out


class CernDS(Dataset):
    def __init__(self, root, labels, augment=True):
        self.root = root
        self.items = list(labels.items())
        self.aug = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, text = self.items[i]
        img = Image.open(os.path.join(self.root, fn)).convert("L")
        if self.aug:
            # the renderer already JPEGs and rotates; keep augmentation light so
            # we stay inside the real distribution
            img = img.rotate(random.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=255)
        w = max(8, int(H * img.width / img.height))
        img = img.resize((w, H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        y = torch.tensor([CH2IDX[c] for c in text], dtype=torch.long)
        return torch.from_numpy(a)[None], y, w


def greedy(logp):
    ids = logp.argmax(-1)[0].tolist()
    out, prev = [], 0
    for i in ids:
        if i != 0 and i != prev:
            out.append(IDX2CH[i])
        prev = i
    return "".join(out)


@torch.no_grad()
def predict(model, path):
    img = Image.open(path).convert("L")
    w = max(8, int(H * img.width / img.height))
    img = img.resize((w, H), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-5)
    return greedy(model(torch.from_numpy(a)[None, None]))


# ------------------------------------------------------------------ TTA + beam
TTA_VIEWS = [(0.0, 1.0), (-4.0, 1.0), (4.0, 1.0), (0.0, 0.88), (0.0, 1.12)]


@torch.no_grad()
def logp(model, path, tta=True):
    """Log-probs over the alphabet (+blank), averaged over TTA views in one
    batched forward. Views: small rotations and brightness shifts."""
    img0 = Image.open(path).convert("L")
    views = TTA_VIEWS if tta else TTA_VIEWS[:1]
    xs = []
    for rot, bright in views:
        img = img0.rotate(rot, resample=Image.BICUBIC, fillcolor=255) if rot else img0
        if bright != 1.0:
            arr = np.asarray(img).astype(np.float32) * bright
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        w = max(8, int(H * img.width / img.height))
        img = img.resize((w, H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - a.mean()) / (a.std() + 1e-5)
        xs.append(torch.from_numpy(a)[None, None])          # (1,1,H,W)
    W = max(x.shape[-1] for x in xs)
    batch = torch.cat([torch.nn.functional.pad(x, (0, W - x.shape[-1])) for x in xs], 0)
    lp = torch.log_softmax(model(batch), dim=-1)          # (V,T,C)
    return torch.logsumexp(lp, dim=0) - np.log(len(views))


def beam_search(lp, beam=8):
    """CTC prefix beam search in log space (same correct formulation as
    solve_pro.ctc_beam_search)."""
    import math
    T, C = lp.shape
    NEG = float("-inf")

    def lse(a, b):
        if a == NEG: return b
        if b == NEG: return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    beams = {"": (0.0, NEG)}
    for t in range(T):
        topc = torch.topk(lp[t], min(C, beam * 2)).indices.tolist()
        nxt = {}

        def add(pfx, b=NEG, nb=NEG):
            pb, pnb = nxt.get(pfx, (NEG, NEG))
            nxt[pfx] = (lse(pb, b), lse(pnb, nb))

        for pfx, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in topc:
                lc = float(lp[t, c])
                if c == 0:
                    add(pfx, b=tot + lc)
                elif pfx and IDX2CH[c] == pfx[-1]:
                    add(pfx, nb=pnb + lc)
                    add(pfx + IDX2CH[c], nb=pb + lc)
                else:
                    add(pfx + IDX2CH[c], nb=tot + lc)
        beams = dict(sorted(nxt.items(), key=lambda kv: -lse(*kv[1]))[:beam])
    return max(beams, key=lambda p: lse(*beams[p]))


@torch.no_grad()
def predict2(model, path, tta=True, beam=8):
    lp = logp(model, path, tta=tta)
    return beam_search(lp, beam) if beam else greedy(lp[None] if lp.dim() == 2 else lp)


@torch.no_grad()
def predict_cascade(model, path, tta=True, beam=8, conf=0.90):
    """Greedy first, escalate only when unsure.

    A single forward pass (~4 ms on ONNX) gives the greedy read plus a per
    character confidence. If every emitted character is at least `conf`, that
    read is returned; otherwise fall back to the expensive TTA + beam path.
    Most reads are confident, so the average cost drops sharply while the hard
    cases still get the strong decoder."""
    lp = logp(model, path, tta=False)                 # (T, C)
    ids = lp.argmax(-1)
    p = lp.exp().max(-1).values
    text = greedy(lp[None])
    emitted = [float(q) for i, q in zip(ids.tolist(), p.tolist()) if i != 0]
    if text and emitted and min(emitted) >= conf:
        return text
    return predict2(model, path, tta=tta, beam=beam)


def train(epochs=20, bs=64, lr=5e-3, root="data/cern_train", out=CKPT, limit=0, init=None):
    labels = json.load(open(os.path.join(root, "labels.json")))
    items = list(labels.items()); random.seed(0); random.shuffle(items)
    if limit:
        items = items[:limit]
    n_val = max(40, len(items) // 15)
    val, tr = dict(items[:n_val]), dict(items[n_val:])
    dtr = DataLoader(CernDS(root, tr, True), bs, shuffle=True, collate_fn=collate)
    model = CRNN(ncls=NCLS)
    if init and os.path.exists(init):
        # never cold-start: CTC from scratch collapses to the uniform/blank
        # plateau often (loss pinned at ln(62)); continuing from a converged
        # checkpoint removes that failure mode entirely
        model.load_state_dict(torch.load(init, map_location="cpu"))
        print(f"init from {init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    for ep in range(1, epochs + 1):
        model.train(); tl = 0
        for x, y, yl in dtr:
            logp = model(x); T = logp.shape[1]
            loss = ctc(logp.permute(1, 0, 2), y,
                       torch.full((x.shape[0],), T, dtype=torch.long), yl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); tl += loss.item()
        sched.step()
        if True:
            acc = evaluate(model, root, val)
            torch.save(model.state_dict(), out)
            print(f"ep {ep:3d} loss {tl/len(dtr):.3f} val-exact {acc:.3f}", flush=True)
    torch.save(model.state_dict(), out)
    print("saved", out)


def evaluate(model, root, labels, tta=False, beam=0):
    model.eval(); ok = 0
    for fn, truth in labels.items():
        p = os.path.join(root, fn)
        got = predict2(model, p, tta=tta, beam=beam) if (tta or beam) else predict(model, p)
        ok += got == truth
    return ok / max(1, len(labels))


def load(path=CKPT, backend="auto", threads=4):
    """Load the reader. Prefers the ONNX Runtime session (2x faster); falls back
    to PyTorch. `backend` = "auto" | "onnx" | "torch"."""
    if backend in ("auto", "onnx") and os.path.exists(ONNX_PATH) and path == CKPT:
        try:
            return OnnxCRNN(threads=threads)
        except Exception as e:
            if backend == "onnx":
                raise
            print(f"onnx unavailable ({e}); using torch", flush=True)
    m = CRNN(ncls=NCLS)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def _live_one(i, model, tta, beam):
    """Fetch one challenge, solve it, validate it. Returns (index, answer, ok)."""
    import base64, urllib.request, urllib.error, http.cookiejar
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    with op.open("https://captcha.web.cern.ch/api/v1.0/captcha/", timeout=25) as r:
        d = json.load(r)
    p = f"/tmp/cern_live_{i}.jpg"
    with open(p, "wb") as f:
        f.write(base64.b64decode(d["img"].split(",", 1)[1]))
    ans = predict2(model, p, tta=tta, beam=beam) if (tta or beam) else predict(model, p)
    body = json.dumps({"id": d["id"], "answer": ans}).encode()
    req = urllib.request.Request("https://captcha.web.cern.ch/api/v1.0/captcha/",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with op.open(req, timeout=25) as resp:
            ok = resp.status == 200
    except urllib.error.HTTPError:
        ok = False
    except Exception:
        ok = False
    return i, ans, ok


def live(n=50, tta=True, beam=8, workers=8):
    """Solve real CAPTCHAs from the live CERN API and validate the answers.

    Defaults to the strong decoder (TTA + CTC beam search): it is worth a few
    points on the live distribution (90 samples: greedy 96.7%, strong 100%) and
    the harness is network-bound anyway (~1 s/request), so the extra ~35 ms of
    decoding is irrelevant here. Pass tta=False, beam=0 for greedy.

    The requests themselves run concurrently (`workers`), which is what actually
    sets the wall clock: ~1 s/img serial -> ~n/workers seconds.
    """
    import concurrent.futures as cf
    if workers > 1:
        torch.set_num_threads(1)          # avoid oversubscribing n workers
    model = load(backend="auto", threads=2 if workers > 1 else 4)
    ok = 0; t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_live_one, i, model, tta, beam) for i in range(n)]
        for done, fut in enumerate(cf.as_completed(futs), 1):
            i, ans, good = fut.result()
            ok += good
            print(f"  {done:3d}/{n}  read={ans:8s} {'OK' if good else '--'}", flush=True)
    dt = time.time() - t0
    print(f"live CERN API: {ok}/{n} validated correct "
          f"({dt/n*1000:.0f} ms/img wall, {workers} workers)  tta={tta} beam={beam}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 20,
              limit=int(sys.argv[3]) if len(sys.argv) > 3 else 0,
              init=sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "eval":
        m = load()
        lab = json.load(open("data/cern_val/labels.json"))
        print(f"val greedy      : {evaluate(m, 'data/cern_val', lab)*100:.1f}% ({len(lab)})")
        print(f"val beam8       : {evaluate(m, 'data/cern_val', lab, beam=8)*100:.1f}%")
        print(f"val beam8 + TTA : {evaluate(m, 'data/cern_val', lab, tta=True, beam=8)*100:.1f}%")
    elif cmd == "export":
        export_onnx()
    elif cmd == "live":
        # strong decoder is the default; `--greedy` opts into the fast path
        args = sys.argv[2:]
        use_greedy = "--greedy" in args
        workers = 8
        if "--workers" in args:
            j = args.index("--workers"); workers = int(args[j + 1])
            args = args[:j] + args[j + 2:]
        args = [a for a in args if a != "--greedy"]
        n = int(args[0]) if args else 50
        if use_greedy:
            live(n, tta=False, beam=0, workers=workers)
        else:
            tta = "greedy" not in args
            beam = int(args[1]) if len(args) > 1 else 8
            live(n, tta=tta, beam=beam, workers=workers)
