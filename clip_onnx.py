"""CLIP zero-shot image classification via ONNX Runtime (no torchvision).

torchvision's C extension is ABI-broken against this torch build, so the usual
`transformers.CLIPModel` path is unavailable. ONNX Runtime has no such problem:
we run the CLIP vision/text towers directly and do the image preprocessing and
tokenisation by hand.

  models/  -> vision_model.onnx (B,3,224,224)->(B,512)
              text_model.onnx   (B,77)      ->(B,512)
              tokenizer.json, preprocessor_config.json
"""
import os, json
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("CLIP_ONNX_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "models", "clip"))


class CLIP:
    def __init__(self, model_dir=MODEL_DIR, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.vis = ort.InferenceSession(os.path.join(model_dir, "vision_model.onnx"),
                                        so, providers=["CPUExecutionProvider"])
        self.txt = ort.InferenceSession(os.path.join(model_dir, "text_model.onnx"),
                                        so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(77)
        self.tok.enable_padding(length=77)
        pp = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
        self.mean = np.array(pp["image_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(pp["image_std"], dtype=np.float32).reshape(3, 1, 1)
        self.size = 224

    # ---- image
    def _prep(self, img):
        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)                    # shortest edge -> 224
        img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))),
                         Image.BICUBIC)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2   # center crop
        img = img.crop((left, top, left + self.size, top + self.size))
        a = np.asarray(img).astype(np.float32) / 255.0            # (H,W,3)
        a = a.transpose(2, 0, 1)                                  # (3,H,W)
        return (a - self.mean) / self.std

    def image_embeds(self, imgs):
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        x = np.stack([self._prep(i) for i in imgs]).astype(np.float32)
        e = self.vis.run(None, {"pixel_values": x})[0]
        return e / np.linalg.norm(e, axis=-1, keepdims=True)

    # ---- text
    def text_embeds(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ids = np.array([self.tok.encode(t).ids for t in texts], dtype=np.int64)
        e = self.txt.run(None, {"input_ids": ids})[0]
        return e / np.linalg.norm(e, axis=-1, keepdims=True)

    # ---- zero-shot
    def classify(self, imgs, prompts, template="a photo of a {}"):
        """prompts: list of class names (or full sentences). Returns logits
        (softmax over classes) and the per-class similarity matrix."""
        if not isinstance(imgs, list):
            imgs = [imgs]
        txt = [p if " " in p else template.format(p) for p in prompts]
        ie = self.image_embeds(imgs)                 # (N,512)
        te = self.text_embeds(txt)                   # (C,512)
        logit_scale = 100.0                          # CLIP default ln(1/0.07)
        logits = logit_scale * (ie @ te.T)           # (N,C)
        logits -= logits.max(axis=1, keepdims=True)  # stable softmax
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return prob, logits


if __name__ == "__main__":
    import sys
    c = CLIP()
    img = Image.open(sys.argv[1])
    labels = sys.argv[2:] or ["a bicycle", "a car", "a traffic light",
                              "a fire hydrant", "a dog", "a cat"]
    prob, _ = c.classify(img, labels)
    for lbl, p in sorted(zip(labels, prob[0]), key=lambda kv: -kv[1]):
        print(f"  {p*100:6.2f}%  {lbl}")
