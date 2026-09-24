"""Generate synthetic text CAPTCHAs for measuring solver strength.

This is a *testbed*: it produces labelled images so a solver can be scored
objectively. Use it to benchmark how resistant a CAPTCHA style is before you
deploy it, or to build training data for a policy you own.
"""
import random, string, os, json
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no ambiguous chars

FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]

W, H = 220, 70


def load_font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def gen_one(text, style="easy"):
    if style == "easy":
        n_lines, rotate, wave, noise = 0, 0, 0, 60
    elif style == "medium":
        n_lines, rotate, wave, noise = 2, 12, 3, 200
    else:  # hard
        n_lines, rotate, wave, noise = 4, 25, 6, 600

    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # character rendering, per-char jitter/rotation
    x = 18
    for ch in text:
        size = random.randint(34, 42)
        font = load_font(size)
        cw = d.textlength(ch, font=font)
        tile = Image.new("RGBA", (int(cw) + 12, size + 12), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text((6, 4), ch, font=font,
                fill=(random.randint(0, 90), random.randint(0, 90), random.randint(0, 90)))
        tile = tile.rotate(random.uniform(-rotate, rotate), expand=1, resample=Image.BICUBIC)
        img.paste(tile, (int(x - 6), int(H / 2 - size / 2 + random.uniform(-wave, wave))), tile)
        x += cw + random.randint(0, 3)

    # interference lines
    for _ in range(n_lines):
        d.line([(random.randint(0, W), random.randint(0, H)),
                (random.randint(0, W), random.randint(0, H))],
               fill=(random.randint(0, 120),) * 3, width=random.randint(1, 2))

    # salt-and-pepper noise
    px = img.load()
    for _ in range(noise):
        px[random.randint(0, W - 1), random.randint(0, H - 1)] = (
            random.choice([(0, 0, 0), (255, 255, 255), (random.randint(0, 255),) * 3]))

    if style != "easy":
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


def make_set(out, n, lengths=(5, 6), style="easy", seed=1):
    random.seed(seed)
    os.makedirs(out, exist_ok=True)
    labels = {}
    for i in range(n):
        text = "".join(random.choice(ALPHABET) for _ in range(random.choice(lengths)))
        img = gen_one(text, style=style)
        fn = f"{i:05d}.png"
        img.save(os.path.join(out, fn))
        labels[fn] = text
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(f"{out}: {n} images, style={style}")


if __name__ == "__main__":
    make_set("data/easy", 200, style="easy", seed=1)
    make_set("data/medium", 200, style="medium", seed=2)
    make_set("data/hard", 200, style="hard", seed=3)
