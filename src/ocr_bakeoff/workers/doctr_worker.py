"""docTR through OnnxTR, as a worker: one image path per line on stdin, one JSON line per image on stdout.

Runs in its own environment (`uv run --with onnxtr[cpu]`), so the tool itself needs no extra package.
"""

import json
import os
import sys

out = os.fdopen(
    os.dup(1), "w", encoding="utf-8"
)  # the real stdout; libraries print to fd 1, which goes to stderr below
os.dup2(2, 1)

from onnxtr.io import DocumentFile  # noqa: E402
from onnxtr.models import ocr_predictor  # noqa: E402

model = ocr_predictor()
print("ready", file=out, flush=True)
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        page = model(DocumentFile.from_images(path)).export()["pages"][0]
        lines, words = [], []
        for block in page["blocks"]:
            for text_line in block["lines"]:
                lines.append(" ".join(w["value"] for w in text_line["words"]))
                for w in text_line["words"]:
                    (x0, y0), (x1, y1) = w["geometry"][0], w["geometry"][-1]
                    words.append(
                        {
                            "text": w["value"],
                            "box": [float(x0), float(y0), float(x1), float(y1)],
                            "confidence": float(w["confidence"]),
                        }
                    )
        print(json.dumps({"text": "\n".join(lines), "words": words}, ensure_ascii=False), file=out, flush=True)
    except Exception as exc:  # one unreadable image must not end the worker
        print(json.dumps({"text": "", "error": type(exc).__name__}), file=out, flush=True)
