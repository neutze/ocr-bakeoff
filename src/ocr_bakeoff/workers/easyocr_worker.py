"""EasyOCR as a worker: one image path per line on stdin, one JSON line per image on stdout.

Runs in its own environment (`uv run --with easyocr`), so the tool itself needs no PyTorch.
"""

import json
import os
import sys

out = os.fdopen(
    os.dup(1), "w", encoding="utf-8"
)  # the real stdout; libraries print to fd 1, which goes to stderr below
os.dup2(2, 1)

import cv2  # noqa: E402
import easyocr  # noqa: E402

reader = easyocr.Reader(["de", "en", "tr"], gpu=False, verbose=False)
print("ready", file=out, flush=True)
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        height, width = cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape
        lines = []
        for box, text, score in reader.readtext(path):
            xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
            lines.append(
                {
                    "text": text,
                    "box": [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height],
                    "confidence": float(score),
                }
            )
        lines.sort(key=lambda item: (round(item["box"][1], 2), item["box"][0]))
        print(
            json.dumps({"text": "\n".join(item["text"] for item in lines), "lines": lines}, ensure_ascii=False),
            file=out,
            flush=True,
        )
    except Exception as exc:  # one unreadable image must not end the worker
        print(json.dumps({"text": "", "error": type(exc).__name__}), file=out, flush=True)
