"""Generate a mixed-difficulty training corpus.

The first local model trained only on 'medium' and scored 72.5% on 'hard' —
a distortion it had never seen. Training across the full difficulty range,
with heavier augmentation, is what closes that gap.
"""
import random, os, json
from gen import gen_one, ALPHABET

def make_mixed(out, n, seed=7, weights=(0.25, 0.40, 0.35)):
    random.seed(seed)
    os.makedirs(out, exist_ok=True)
    styles = ["easy", "medium", "hard"]
    labels = {}
    for i in range(n):
        style = random.choices(styles, weights=weights)[0]
        text = "".join(random.choice(ALPHABET) for _ in range(random.choice((5, 6))))
        img = gen_one(text, style=style)
        fn = f"{i:06d}.png"
        img.save(os.path.join(out, fn))
        labels[fn] = text
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(f"{out}: {n} images, mixed styles {dict(zip(styles, weights))}")


if __name__ == "__main__":
    make_mixed("data/trainmix", 5000, seed=7)
