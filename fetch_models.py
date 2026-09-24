"""Download the pretrained models used by the convenience code paths.

Nothing large is committed to the repo (weights are ~610 MB). This fetches:

  models/clip/            CLIP ViT-B/32 ONNX export (vision + text towers)
  hfmodel/model.safetensors  pretrained text-CAPTCHA CRNN (Graf-J/captcha-crnn-base)

Both are optional — you can train everything from scratch with the included
scripts instead. The grid solver works zero-shot without grid_head.pt.
"""
import os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

CLIP_REPO = "Xenova/clip-vit-base-patch32"
CLIP_FILES = ["onnx/vision_model.onnx", "onnx/vision_model_quantized.onnx",
              "onnx/text_model.onnx", "tokenizer.json",
              "preprocessor_config.json", "config.json"]
CLIP_DEST = os.path.join(HERE, "models", "clip")

CRNN_REPO = "Graf-J/captcha-crnn-base"
CRNN_FILES = ["model.safetensors", "config.json", "processor_config.json",
              "modeling_captcha.py", "configuration_captcha.py", "processing_captcha.py"]
CRNN_DEST = os.path.join(HERE, "hfmodel")


def fetch(repo, files, dest, prefix=""):
    os.makedirs(dest, exist_ok=True)
    for f in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        out = os.path.join(dest, os.path.basename(f))
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  have {prefix}{os.path.basename(f)}")
            continue
        print(f"  get  {prefix}{f} ...", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, out)
            print(f" {os.path.getsize(out)/1e6:.1f} MB")
        except Exception as e:
            print(f" FAILED ({e})")
            return False
    return True


if __name__ == "__main__":
    print("CLIP ONNX (image-grid solver):")
    ok1 = fetch(CLIP_REPO, CLIP_FILES, CLIP_DEST)
    print("\npretrained text CRNN:")
    ok2 = fetch(CRNN_REPO, CRNN_FILES, CRNN_DEST)
    print("\ndone." if (ok1 and ok2) else "\nsome downloads failed (network?).")
