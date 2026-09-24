"""cern_gen.py — faithful port of the CERN captcha-api renderer.

Source: https://github.com/CERN/captcha-api (captcha_api/captcha_generator.py)

Replicated exactly so a model trained here transfers to the live service:
  - font DejaVuSerif at size 36, canvas 250x60 on (250,250,250)
  - 6 characters; each character is drawn from ONE randomly chosen set
    (digits 1-9 | A-Z | a-z), so case is mixed within a single image
  - each glyph rendered alone, rotated by a random -50..+50 degrees, colourised
    to a random RGB in 120..200, pasted at x = 250*0.13*(i+1), y = 60*0.2
  - 15 random interference lines + 16 random points, same colour range
  - saved as JPEG (so compression artefacts are part of the distribution)

The only change: Pillow >= 10 removed `font.getsize`, so the per-glyph canvas is
sized with getlength()/getmetrics(), which reproduces getsize's (advance, ascent+
descent) semantics.
"""
import os, json, random
from io import BytesIO
from random import randint
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
CHARSET = ([chr(i) for i in range(49, 58)] +      # 1-9
           [chr(i) for i in range(65, 91)] +      # A-Z
           [chr(i) for i in range(97, 123)])      # a-z


def _get_random_color():
    return randint(120, 200), randint(120, 200), randint(120, 200)


def _get_random_code():
    codes = [
        [chr(i) for i in range(49, 58)],
        [chr(i) for i in range(65, 91)],
        [chr(i) for i in range(97, 123)],
    ]
    codes = codes[randint(0, 2)]
    return codes[randint(0, len(codes) - 1)]


def _glyph_box(c, font):
    """Stand-in for font.getsize: (advance width, ascent + descent)."""
    return max(1, int(round(font.getlength(c)))), max(1, sum(font.getmetrics()))


def _generate_rotated_char(c, font):
    txt = Image.new("L", _glyph_box(c, font))
    ImageDraw.Draw(txt).text((0, 0), c, font=font, fill=255)
    return txt.rotate(randint(-50, 50), expand=1)


class CaptchaGenerator:
    def __init__(self, fontname=FONT_PATH, width=250, height=60):
        self.width = width
        self.height = height
        self.font = ImageFont.truetype(fontname, size=36)

    def generate_captcha(self, length=6):
        img = Image.new("RGB", (self.width, self.height), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        text = ""
        for i in range(length):
            char = _get_random_code()
            text += char
            rotated = _generate_rotated_char(char, self.font)
            colorized = ImageOps.colorize(rotated, (0, 0, 0), _get_random_color())
            img.paste(colorized,
                      (int(self.width * 0.13 * (i + 1)), int(self.height * 0.2)),
                      rotated)
        for _ in range(15):
            draw.line((randint(0, self.width), randint(0, self.height),
                       randint(0, self.width), randint(0, self.height)),
                      fill=_get_random_color())
        for _ in range(16):
            draw.point((randint(0, self.width), randint(0, self.height)),
                       fill=_get_random_color())
        buf = BytesIO()
        img.save(buf, format="jpeg")
        return buf, text


def make_set(out, n, seed=0, length=6):
    random.seed(seed)
    os.makedirs(out, exist_ok=True)
    gen = CaptchaGenerator()
    labels = {}
    for i in range(n):
        buf, text = gen.generate_captcha(length=length)
        fn = f"{i:06d}.jpg"
        with open(os.path.join(out, fn), "wb") as f:
            f.write(buf.getvalue())
        labels[fn] = text
    json.dump(labels, open(os.path.join(out, "labels.json"), "w"))
    print(f"{out}: {n} images")


if __name__ == "__main__":
    make_set("data/cern_train", 12000, seed=1)
    make_set("data/cern_val", 800, seed=2)
