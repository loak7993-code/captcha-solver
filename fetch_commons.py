"""Fetch labelled real photos from Wikimedia Commons for a grid-CAPTCHA testbed.

Uses the Commons search API (filetype:bitmap) and downloads thumbnails at a
fixed width. Categories chosen to mirror reCAPTCHA/hCaptcha prompt classes.
"""
import os, json, time, urllib.parse, urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "captcha-testbed/1.0 (research; local use)"}
CLASSES = ["bicycle", "traffic light", "bus", "dog", "cat", "fire hydrant",
           "motorcycle", "stop sign", "car", "boat"]
PER_CLASS = 70
WIDTH = 400


def search(term, limit):
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}", "gsrlimit": limit, "gsrnamespace": 6,
        "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": WIDTH,
    })
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    out = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        ii = (page.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if url and ii.get("mime", "").startswith("image/"):
            out.append(url)
    return out


def fetch(outdir):
    os.makedirs(outdir, exist_ok=True)
    for cls in CLASSES:
        urls = search(cls, PER_CLASS * 2)
        n = 0
        for u in urls:
            if n >= PER_CLASS:
                break
            dest = os.path.join(outdir, f"{cls.replace(' ', '_')}_{n:02d}.jpg")
            if os.path.exists(dest):
                n += 1; continue
            try:
                req = urllib.request.Request(u, headers=UA)
                with urllib.request.urlopen(req, timeout=30) as r:
                    b = r.read()
                if len(b) < 4000:
                    continue
                open(dest, "wb").write(b)
                n += 1
                time.sleep(0.1)
            except Exception:
                continue
        print(f"{cls}: {n} images", flush=True)


if __name__ == "__main__":
    fetch(os.environ.get("GRID_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridtest", "data")))
