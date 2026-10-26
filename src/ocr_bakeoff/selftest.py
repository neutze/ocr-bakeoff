"""Offline checks: builds the sample folder in a temp dir and exercises everything that
decides a score. No model, no network, no Google account. Needs pdftotext (poppler)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

results: list[tuple[str, bool]] = []
skipped: list[str] = []  # live engines not installed here; printed, never counted as passed


def check(name: str, condition: object) -> None:
    results.append((name, bool(condition)))


def main() -> None:
    import pymupdf as fitz

    from .catalog import LANES
    from .corpus import discover, prepare
    from .evaluate import load_gold
    from .gdrive import DriveOCR, reference
    from .guards import degeneration, repetition_loop
    from .lanes import BANDS, join_bands, merge_blocks, run_lane
    from .models import MODELS, server_command
    from .render import blank_row_fraction, crop, decode_gray, degrade, estimate_skew, page_size
    from .report import render
    from .samples import write_samples
    from .score import aggregate, decide, score_image

    if shutil.which("pdftotext") is None:
        raise SystemExit("selftest needs pdftotext (brew install poppler)")
    scratch = Path(tempfile.mkdtemp(prefix="ocr-bakeoff-selftest-"))
    testdir, work = scratch / "testfiles", scratch / "results"
    write_samples(testdir)

    # --- Discovery: kinds from folders, tiers from the text layer, typed gold beside files.
    pages = discover(testdir)
    by_name = {Path(p.rel).name: p for p in pages}
    check("7 sample documents discovered", len(pages) == 7)
    check(
        "kinds come from subfolders",
        {p.kind for p in pages} == {"letters", "invoices", "presentations", "work-docs", "scans"},
    )
    check("born-digital letter detected", by_name["letter-text-header.pdf"].tier == "born-digital")
    check("image-only PDF is a scan", by_name["letter-scan.pdf"].tier == "scan")
    check("photo is a scan", by_name["receipt-photo.jpg"].tier == "scan")
    check("typed gold found beside its file", by_name["letter-image-header.pdf"].typed_gold is not None)
    check("results folder is not rediscovered", all(".ocr-bakeoff" not in p.rel for p in discover(testdir)))
    write_samples(scratch / "again")
    check(
        "samples are reproducible: same files, so the same page keys and simulated scans",
        sorted(p.key for p in discover(scratch / "again")) == sorted(p.key for p in pages),
    )

    prepare(pages, work)
    check(
        "born-digital pages get clean + degraded", set(by_name["invoice-table.pdf"].variants) == {"clean", "degraded"}
    )
    check("scans get one image", set(by_name["receipt-photo.jpg"].variants) == {"scan"})
    clean = work / "pages" / by_name["letter-text-header.pdf"].variants["clean"]
    check("300 DPI A4 render", abs(page_size(clean.read_bytes())[0] - 2479) <= 2)
    from .render import render_pdf_page

    offset = fitz.open()
    offset.new_page(width=612, height=792)
    offset.xref_set_key(
        offset[0].xref, "MediaBox", "[0 100 612 892]"
    )  # not at the origin: the CropBox cannot be widened
    offset.xref_set_key(offset[0].xref, "CropBox", "[10 110 600 880]")
    offset.save(scratch / "offset-mediabox.pdf")
    try:
        check(
            "a page whose MediaBox is off the origin still renders",
            page_size(render_pdf_page(scratch / "offset-mediabox.pdf", 0))[0] > 0,
        )
    except Exception as exc:
        check(f"a page whose MediaBox is off the origin still renders ({type(exc).__name__})", False)

    # --- Gold: text layer, typed additions for picture text, typed gold for scans.
    letter = load_gold(by_name["letter-text-header.pdf"], work)
    check(
        "text-layer header has the letterhead",
        "Şahin" in letter["header_text"] and "Königstraße" in letter["header_text"],
    )
    check("text-layer header excludes the body", "Einkommensteuererklärung" not in letter["header_text"])
    check("typed logo text joins the header gold", "MSB" in letter["header_text"])
    check(
        "footer gold has register and footnote", "HRB" in letter["footer_text"] and "Zahlbar" in letter["footer_text"]
    )
    image_header = load_gold(by_name["letter-image-header.pdf"], work)
    check("image-only header comes from typed gold", "Doğan" in image_header["header_text"])
    check("image-only footer comes from typed gold", "PR 1188 B" in image_header["footer_text"])
    check(
        "image header is not in the text layer",
        "Schloßstraße"
        not in json.loads((work / "gold" / f"{by_name['letter-image-header.pdf'].key}.textlayer.json").read_text())[
            "full_text"
        ],
    )
    scan = load_gold(by_name["letter-scan.pdf"], work)
    check(
        "scan gold is typed and has full text",
        scan["source"] == "typed" and scan["full_text"] and "MSB" in scan["header_text"],
    )
    receipt = load_gold(by_name["receipt-photo.jpg"], work)
    check("receipt gold carries the amounts", "1.225,00" in receipt["numbers"])

    # --- Hidden text in a real text layer is not part of the reference.
    from .corpus import text_layer_gold

    hidden_pdf = scratch / "hidden.pdf"
    doc = fitz.open()
    pg = doc.new_page(width=595, height=842)
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    kw = {"fontname": "arial", "fontfile": font} if Path(font).exists() else {"fontname": "helv"}
    pg.insert_text((60, 60), "Sichtbarer Briefkopf GmbH", fontsize=14, **kw)
    pg.insert_text((60, 90), "Weisser Geistertext", fontsize=12, color=(1, 1, 1), **kw)
    pg.insert_text((300, 90), "Unsichtbarer Modus", fontsize=12, render_mode=3, **kw)
    for i in range(12):
        pg.insert_text(
            (60, 300 + i * 20), f"Zeile {i} mit ausreichend sichtbarem Text für eine echte Seite.", fontsize=11, **kw
        )
    doc.save(hidden_pdf)
    hidden_gold = text_layer_gold(hidden_pdf, 1)
    check("reference keeps visible header text", "Briefkopf" in hidden_gold["header_text"])
    check(
        "reference drops white-on-white text",
        "Geistertext" not in hidden_gold["header_text"] and "Geistertext" not in hidden_gold["full_text"],
    )
    check("reference drops invisible-mode text", "Unsichtbarer" not in hidden_gold["full_text"])
    check("hidden words are counted", hidden_gold["hidden_words"] >= 4)

    # --- External tools can emit invalid UTF-8 (a real PDF did); reading it must not crash.
    from .text import run_text

    lenient = run_text(["printf", "\\355ok"])
    check("invalid UTF-8 from an external tool becomes U+FFFD, not a crash", lenient.stdout == "\ufffdok")

    # --- Word positions: reference boxes, Tesseract TSV, recall, and the scan transform.
    from .lanes import parse_tsv
    from .render import map_box
    from .score import word_box_recall

    layer_json = json.loads((work / "gold" / f"{by_name['letter-text-header.pdf'].key}.textlayer.json").read_text())
    check(
        "reference records visible word boxes and trusts this page's geometry",
        len(layer_json["words"]) > 50 and layer_json["geometry_ok"],
    )
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "1\t1\t0\t0\t0\t0\t0\t0\t1000\t2000\t-1\t\n"
        "5\t1\t1\t1\t1\t1\t100\t200\t50\t20\t96\tMüller\n"
        "5\t1\t1\t1\t1\t2\t160\t200\t40\t20\t95\tGmbH\n"
        "5\t1\t1\t1\t2\t1\t100\t240\t60\t20\t91\tStuttgart\n"
    )
    tsv_text, tsv_words = parse_tsv(tsv)
    check(
        "Tesseract TSV gives text by line and normalised word boxes",
        tsv_text == "Müller GmbH\nStuttgart" and tsv_words[0]["box"] == [0.1, 0.1, 0.15, 0.11],
    )
    ref = [("Müller", (0.1, 0.1, 0.15, 0.11)), ("GmbH", (0.16, 0.1, 0.2, 0.11))]
    check("word boxes: same words in the same place score 1", word_box_recall(tsv_words, ref) == 1.0)
    from .lanes import unrotate_boxes

    probe = [{"text": "x", "box": [0.4, 0.4, 0.5, 0.45]}]
    back = unrotate_boxes(unrotate_boxes(probe, 1.5, clean), -1.5, clean)[0]["box"]
    check(
        "undoing a deskew rotation is exact enough to place words",
        all(abs(a - b) < 0.01 for a, b in zip(back, probe[0]["box"])),
    )
    check(
        "word boxes: right words in the wrong place score 0",
        word_box_recall([dict(w, box=[b + 0.3 for b in w["box"]]) for w in tsv_words], ref) == 0.0,
    )
    clean_bytes = clean.read_bytes()
    width, height = page_size(clean_bytes)
    seed = int(by_name["letter-text-header.pdf"].key[:8], 16)
    degraded_gray = decode_gray(degrade(clean_bytes, seed))
    dh, dw = degraded_gray.shape

    def ink(box):
        x0, y0, x1, y1 = box
        return float(degraded_gray[int(y0 * dh) : int(y1 * dh) + 1, int(x0 * dw) : int(x1 * dw) + 1].std())

    longest = max(layer_json["words"], key=lambda w: (w[3] - w[1]) * (w[4] - w[2]))
    mapped = map_box(tuple(longest[1:5]), seed, width, height)
    check("a reference box carried onto the simulated scan lands on ink", ink(mapped) > 20)
    check("the same box in empty margin has no ink", ink((0.9, 0.45, 0.97, 0.47)) < 20)

    # --- Images: deterministic degradation, deskew, crops, blank-row cuts.
    data = clean.read_bytes()
    check("degraded variant deterministic per seed", degrade(data, 3) == degrade(data, 3))
    check("different seed, different variant", degrade(data, 3) != degrade(data, 4))
    check("degraded is ~200 DPI", abs(page_size(degrade(data, 3))[0] - round(2479 * 2 / 3)) <= 3)
    skewed = decode_gray(degrade(data, 3))
    check("skew of a simulated scan is found", 0.25 <= abs(estimate_skew(skewed)) <= 2.0)
    check("an upright page has ~no skew", abs(estimate_skew(decode_gray(data))) <= 0.25)
    cut = blank_row_fraction(data, 0.20)
    gray = decode_gray(data)
    row = gray[int(cut * gray.shape[0])]
    check("header cut lands on a blank row", (row < 160).sum() == 0)
    check("header cut stays near 20%", 0.12 <= cut <= 0.28)
    top = crop(data, (0.0, BANDS[0][0], 1.0, BANDS[0][1]))
    check("band crop keeps width", abs(page_size(top)[0] - 2479) <= 2)

    # --- Lanes that need no model: text-layer controls, merge/join helpers, guards.
    lp = by_name["letter-text-header.pdf"]
    check("pdftotext control reads the layer", "Yılmaz" in run_lane("pdftotext", clean, lp.path, 1)["text"])
    check("pymupdf control reads the layer", "Yılmaz" in run_lane("pymupdf", clean, lp.path, 1)["text"])
    sp = by_name["letter-scan.pdf"]
    check("controls come back empty on a scan", not run_lane("pdftotext", clean, sp.path, 1)["text"].strip())
    check("seam duplicate removed", join_bands(["a\nb\nSeam", "Seam\nc"]) == "a\nb\nSeam\nc")
    check(
        "blocks merged to the cap",
        len(merge_blocks([{"box": [0.1, i * 0.04, 0.9, i * 0.04 + 0.03]} for i in range(20)])) == 12,
    )
    nested = merge_blocks(
        [
            {"box": [0.1, 0.10, 0.6, 0.14]},
            {"box": [0.1, 0.10, 0.6, 0.14]},
            {"box": [0.1, 0.30, 0.9, 0.50]},
            {"box": [0.2, 0.40, 0.4, 0.45]},
            {"box": [0.6, 0.41, 0.8, 0.46]},
        ]
    )
    check("duplicate and nested Vision blocks become 2 crops", len(nested) == 2)
    check(
        "no two crops overlap substantially",
        all(
            not (
                max(a[0], b[0]) < min(a[2], b[2])
                and max(a[1], b[1]) < min(a[3], b[3])
                and (min(a[2], b[2]) - max(a[0], b[0])) * (min(a[3], b[3]) - max(a[1], b[1]))
                > 0.3 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            )
            for i, a in enumerate(nested)
            for b in nested[i + 1 :]
        ),
    )
    check("loop flagged", degeneration("\n".join(["same"] * 12), "", [])["loop"])
    check("29 distinct rows are not a loop", repetition_loop("\n".join(f"row {i}" for i in range(29))) is None)
    check(
        "native prompt echo flagged",
        degeneration(
            "Transcribe every piece of text in this document image now",
            "Transcribe every piece of text in this document image, in reading order.",
            [],
        )["echo"],
    )
    check("truncation flagged", degeneration("x", "", ["length"])["truncated"])
    from .score import judged_degeneration

    rows_text = "\n".join(["Pos 1 0,00 EUR"] * 12)
    faithful = {"degenerate": degeneration(rows_text, "", [])}
    check(
        "repetition the page itself has is not a loop",
        not judged_degeneration(faithful, {"full_text": rows_text})["loop"],
    )
    check(
        "a classic engine's repetition is the page's own, whatever the reference",
        not judged_degeneration(faithful, {"full_text": "a bag of consensus words"}, "tesseract")["loop"],
    )
    check(
        "a generative model's repetition against a bag-of-words reference is still a loop",
        judged_degeneration(faithful, {"full_text": "a bag of consensus words"}, "paddle")["loop"],
    )
    check(
        "repetition the page does not have is still a loop",
        judged_degeneration(faithful, {"full_text": "\n".join(f"Pos {i} {i},00 EUR" for i in range(12))})["loop"],
    )

    # --- Using the born-digital text where possible.
    from .lanes import garbled, layer_verdict

    v_logo = layer_verdict(by_name["letter-text-header.pdf"].path, 1)
    check(
        "layer rule: born-digital letter uses its layer and OCRs the logo",
        v_logo["use_layer"] and len(v_logo["regions"]) == 1,
    )
    v_images = layer_verdict(by_name["letter-image-header.pdf"].path, 1)
    check(
        "layer rule: image header and image footer both go to OCR",
        v_images["use_layer"] and len(v_images["regions"]) == 2,
    )
    check("layer rule: an image-only scan is OCRed", not layer_verdict(by_name["letter-scan.pdf"].path, 1)["use_layer"])
    check(
        "layer rule: a photo is OCRed",
        layer_verdict(by_name["receipt-photo.jpg"].path, 1)["reason"] == "image file, no text layer",
    )
    check(
        "layer rule: garbled layers are refused",
        garbled("Rechnung (cid:12)(cid:40)(cid:77) (cid:3)")
        and garbled("\ue001\ue002 abc")
        and not garbled("Sehr geehrte Frau Yılmaz, anbei die Rechnung Nr. 2026-0417 über 490,00 EUR."),
    )
    spec = by_name["spec-two-column.pdf"]
    auto_run = run_lane("auto-paddle", work / "pages" / spec.variants["clean"], spec.path, 1, api=None)
    check(
        "a born-digital page without pictures needs no model at all",
        auto_run["used_layer"] and auto_run["requests"] == 0 and "Belegerfassung" in auto_run["text"],
    )

    # --- Scoring saved outputs: the path a real run takes from out/ to score rows.
    from .evaluate import score_outputs

    for page in pages:
        for variant, name in page.variants.items():
            saved = run_lane("pdftotext", work / "pages" / name, page.path, page.page)
            saved |= {
                "run_id": "r1" if page.kind != "letters" else "r2",
                "provenance": {"pdftotext": page.kind == "letters"},
            }
            (work / "out" / "pdftotext").mkdir(parents=True, exist_ok=True)
            (work / "out" / "pdftotext" / f"{page.key}.{variant}.json").write_text(json.dumps(saved), encoding="utf-8")
    scored = score_outputs(pages, ["pdftotext"], work, use_consensus=False, with_cer=True)
    check(
        "saved outputs: a control runs on rendered pages and real scans, never on simulated scans",
        {r["variant"] for r in scored.rows} == {"clean", "scan"} and len(scored.rows) == len(pages),
    )
    check(
        "saved outputs: the control reads its own layer exactly",
        all(r["cer"] == 0.0 for r in scored.rows if r["variant"] == "clean" and r["gold"] == "text layer"),
    )
    check(
        "saved outputs: every page with a reference is counted once, by its reference",
        sum(scored.gold_sources.values()) == len(pages) and scored.no_gold == 0,
    )
    check(
        "saved outputs: runs are tracked and mixed provenance is warned about",
        scored.contributing_runs() == {"r1", "r2"} and len(scored.warnings()) == 1,
    )
    from .corpus import reference_path
    from .evaluate import reference_quality

    born_pages = [p for p in pages if p.tier == "born-digital"]
    for page in born_pages:  # a Drive reading that is exactly the text layer
        reference_path(work, f"{page.key}.clean").parent.mkdir(parents=True, exist_ok=True)
        reference_path(work, f"{page.key}.clean").write_text(
            json.dumps({"full_text": load_gold(page, work)["full_text"]})
        )
    drive_quality = reference_quality(pages, work, with_cer=True)
    check(
        "Drive's own accuracy is measured on born-digital pages only",
        drive_quality["pages"] == len(born_pages) and drive_quality["cer"] == 0.0 and drive_quality["drops"] == 0,
    )

    # --- Live classic engines: a wrong invocation parses fine and returns nothing. Only installed
    # engines run: the selftest never downloads (a fresh clone has no RapidOCR models).
    from .config import TESSDATA, VISION_TOOL
    from .deps import RAPIDOCR_READY

    if shutil.which("tesseract") and (TESSDATA / "deu.traineddata").exists():
        live = run_lane("tesseract", clean, lp.path, 1)
        check(
            "live: Tesseract reads the sample letter with word boxes",
            "Königstraße" in live["text"] or len(live.get("words", [])) > 40,
        )
    else:
        skipped.append("tesseract (no tesseract or no deu data)")
    for lane_id, ready in (("rapidocr", RAPIDOCR_READY.exists()), ("vision", VISION_TOOL.exists())):
        if not ready:
            skipped.append(f"{lane_id} (not installed)")
        else:
            try:
                live = run_lane(lane_id, clean, lp.path, 1)
                check(f"live: {lane_id} reads the sample letter", len(live["text"].split()) > 40)
            except Exception as exc:
                check(f"live: {lane_id} reads the sample letter ({type(exc).__name__})", False)

    # --- Models: servers can never download; every model has the same budget and one slot.
    commands = [server_command(spec, Path("w.gguf"), Path("m.gguf")) for spec in MODELS.values()]
    check("server commands never use -hf", all("-hf" not in c for c in commands))
    check("one slot per server (no silent truncation)", all(c[c.index("-np") + 1] == "1" for c in commands))
    check("same image budget for every model", len({c[c.index("--image-max-tokens") + 1] for c in commands}) == 1)
    check("ports are unique per model", len({s.port for s in MODELS.values()}) == len(MODELS))
    check("every LLM lane names a known model", all(l.model in MODELS for l in LANES.values() if l.family == "llm"))
    from . import lanes as lanes_module

    check("every lane has a runner, and every runner a lane", set(lanes_module.RUNNERS) == set(LANES))
    seen: dict[str, tuple[str, str, str]] = {}  # lane -> (how, model, prompt), recorded instead of calling a model
    originals = {name: getattr(lanes_module, name) for name in ("whole", "regions", "on_blocks")}
    try:
        for name in originals:
            setattr(
                lanes_module,
                name,
                lambda api, model, prompt, *_, how=name, **__: {
                    "text": "",
                    "calls": [],
                    "how": how,
                    "model": model,
                    "prompt": prompt,
                },
            )
        for lane in (l for l in LANES.values() if l.family == "llm"):
            out = run_lane(lane.id, clean, lp.path, 1)
            seen[lane.id] = (out["how"], out["model"], out["prompt"][:17])
    finally:
        for name, function in originals.items():
            setattr(lanes_module, name, function)
    check(
        "each LLM lane sends its own prompt to its own model, whole page, bands or blocks",
        seen
        == {
            "glm-custom": ("whole", "glm", lanes_module.glm_custom_prompt()[:17]),
            "glm": ("whole", "glm", "Text Recognition:"),
            "glm-bands": ("regions", "glm", "Text Recognition:"),
            "glm-blocks": ("on_blocks", "glm", "Text Recognition:"),
            "paddle": ("whole", "paddle", "OCR:"),
            "paddle-blocks": ("on_blocks", "paddle", "OCR:"),
            "qwen8b": ("whole", "qwen8b", "Transcribe every "),
            "qwen4b": ("whole", "qwen4b", "Transcribe every "),
            "olmocr": ("whole", "olmocr", "Below is the imag"),
            "dots": ("whole", "dots", "Extract the text "),
            "deepseek": ("whole", "deepseek", "Free OCR."),
            "lighton": ("whole", "lighton", ""),
            "chandra": ("whole", "chandra", "OCR this image to"),
            "ovis": ("whole", "ovis", lanes_module.OVIS_NATIVE[:17]),
            "lighton-tuned": ("whole", "lighton-t", ""),
            "ovis-tuned": ("whole", "ovis-t", lanes_module.OVIS_NATIVE[:17]),
            "paddle-blocks-tuned": ("on_blocks", "paddle-t", "OCR:"),
            "chandra-tuned": ("whole", "chandra-t", "OCR this image to"),
            "qwen8b-tuned": ("whole", "qwen8b-t", "Transcribe every "),
            "unlimited": ("whole", "unlimited", "document parsing."),
            "unlimited-tuned": ("whole", "unlimited-t", "document parsing."),
            "unlimited-dry": ("whole", "unlimited-t", "document parsing."),
            "deepseek-first": ("whole", "deepseek", "Free OCR."),
            "deepseek-ocr": ("whole", "deepseek", "OCR"),
            "deepseek-md": ("whole", "deepseek-t", "<|grounding|>Conv"),
            "dots-layout": ("whole", "dots-t", "Please output the"),
            "qwen8b-md": ("whole", "qwen8b", "qwenvl markdown"),
            "qwen8b-plain": ("whole", "qwen8b", "Please output onl"),
            "qwen8b-2mp": ("whole", "qwen8b", "Transcribe every "),
            "chandra-layout": ("whole", "chandra-t", "OCR this image to"),
            "chandra-192dpi": ("whole", "chandra-t", "OCR this image to"),
            "chandra-2mp": ("whole", "chandra-t", "OCR this image to"),
            "ovis-first": ("whole", "ovis", lanes_module.OVIS_NATIVE[:17]),
            "olmocr-native": ("whole", "olmocr", "Attached is one p"),
        },
    )
    check(
        "DeepSeek-OCR's block names and boxes are removed",
        lanes_module.strip_grounding(
            "<|ref|>title<|/ref|><|det|>[[37, 64, 464, 132]]<|/det|>\nRechnung 4471\ntext[[37, 194, 350, 247]]\nSumme [1] 86,20"
        )
        == "Rechnung 4471\n\nSumme [1] 86,20",
    )
    check(
        "dots.ocr's layout JSON becomes its text in order, pictures left out, a cut-off answer still read",
        lanes_module.layout_text(
            '[{"bbox": [1, 2, 3, 4], "category": "Page-header", "text": "Stand: 02.03.2026"}, '
            '{"bbox": [1, 2, 3, 4], "category": "Picture"}, {"bbox": [5, 6, 7, 8], "category": "Text", "text": "zur\\u00fcck"}]'
        )
        == "Stand: 02.03.2026\nzurück"
        and lanes_module.layout_text(
            '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "Zeile \\"eins\\""}, {"bbox": [5, 6'
        )
        == 'Zeile "eins"'
        and lanes_module.layout_text("Stand: 02.03.2026\n\n1.1 Abnahme") == "Stand: 02.03.2026\n\n1.1 Abnahme",
    )
    check(
        "Unlimited-OCR's region labels, boxes and LaTeX footnote marks are removed, text in brackets is kept",
        lanes_module.strip_regions(
            "header [113, 33, 393, 48]Stand: 02.03.2026\nimage [90, 301, 382, 752]\n"
            "text [98, 323, 456, 380]zurück. \\( ^{2} \\)\nsiehe [1, 2] oben"
        )
        == "Stand: 02.03.2026\nzurück. 2\nsiehe [1, 2] oben",
    )
    import subprocess

    from .models import Server

    stand_in = Server(MODELS["glm"], scratch / "logs")  # a harmless process stands in for llama-server
    (scratch / "logs").mkdir()
    stand_in.log = (scratch / "logs" / "glm.log").open("wb")
    stand_in.process = subprocess.Popen(["sleep", "30"], stdout=stand_in.log)
    process, log = stand_in.process, stand_in.log
    stand_in.__exit__(None, None, None)
    check("stopping a server ends its process and closes its log", process.poll() is not None and log.closed)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ocr_bakeoff.score, ocr_bakeoff.report; "
            "print(sorted(m for m in ('ocr_bakeoff.lanes', 'ocr_bakeoff.gdrive', 'ocr_bakeoff.models', 'ocr_bakeoff.corpus', 'requests', 'cv2') "
            "if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check("scoring and reporting load no engine, model server, Drive, PDF or image code", probe.stdout.strip() == "[]")

    # --- Scoring on constructed outputs.
    perfect = score_image(letter["full_text"], letter)
    check("perfect output: no drop, CER 0", not perfect["drop"] and perfect["cer"] == 0.0)
    no_header = "\n".join(
        l
        for l in letter["full_text"].splitlines()
        if "Şahin" not in l
        and "Königstraße" not in l
        and "MSB" not in l
        and "Partner" not in l
        and "Zeichen" not in l
        and "Yılmaz" not in l
    )
    check("missing letterhead is a header drop", score_image(no_header, letter)["header_drop"])
    wrong = score_image(letter["full_text"].replace("490,00", "35,00"), letter)
    check(
        "wrong amount lowers numeric recall and counts as invented",
        wrong["numeric_recall"] < 1 and wrong["invented_numbers"] >= 1,
    )
    folded = score_image(letter["full_text"].replace("ş", "s").replace("ı", "i").replace("ö", "o"), letter)
    check("folded umlauts and Turkish letters lower diacritic recall", folded["diacritic_recall"] < 1)
    from .text import number_tokens as _numbers

    check(
        "a soft hyphen in the gold equals a printed hyphen",
        _numbers("No. 2025\u00ad6720") == _numbers("No. 2025-6720") == ["2025-6720"],
    )
    check("a superscript footnote marker equals a plain digit", _numbers("days.\u00b9") == _numbers("days.1"))
    split = score_image(letter["full_text"].replace("ş", "s").replace("ı", "i"), letter)
    check(
        "losing Turkish letters shows in the Turkish column, not the German one",
        split["turkish_recall"] < 1 and split["german_recall"] == 1,
    )
    import time as _time

    from .score import character_error_rate
    from .text import levenshtein as _lev

    long_gold = " ".join(f"Wort{i} Betrag {i},00 EUR" for i in range(1200))
    started = _time.perf_counter()
    long_cer = character_error_rate(long_gold.replace("Wort600 ", "Wort6OO ", 1), long_gold)
    check(
        "CER on a 20,000-character page is exact and fast",
        abs(long_cer - 2 / len(long_gold)) < 1e-9 and _time.perf_counter() - started < 1.0,
    )
    check(
        "edit distance with a limit stops early",
        _lev("abcdef", "uvwxyz", limit=1) == 2 and _lev("Straße", "Strasse", limit=2) == 2,
    )
    tiny = dict(letter, header_text="Kurz GmbH")
    check("a header under 5 tokens cannot flag a drop", not score_image("nothing", tiny)["header_drop"])

    # --- Decision rule and report.
    def row(
        lane: str, i: int, drop: bool, seconds: float = 10.0, kind: str = "letters", degenerate: bool = False
    ) -> dict:
        return {
            "lane": lane,
            "variant": "scan",
            "kind": kind,
            "header_recall": 0.5 if drop else 1.0,
            "footer_recall": 1.0,
            "header_drop": drop,
            "footer_drop": False,
            "drop": drop,
            "numeric_recall": 1.0,
            "invented_numbers": 0,
            "diacritic_recall": 1.0,
            "cer": 0.05,
            "bow_f1": 0.9,
            "degenerate": degenerate,
            "echo": False,
            "loop": degenerate,
            "truncated": False,
            "empty": False,
            "seconds": seconds,
            "errors": 0,
        }

    kinds = ["letters", "invoices", "presentations", "work-docs", "scans"]
    rows = (
        [row("glm-custom", i, i < 9, 80, kinds[i % 5], degenerate=i < 2) for i in range(30)]
        + [row("paddle", i, i < 1, 60, kinds[i % 5]) for i in range(30)]
        + [row("vision", i, i < 2, 3, kinds[i % 5]) for i in range(30)]
        + [row("pdftotext", i, True, 0.1, kinds[i % 5]) for i in range(30)]
    )
    agg = aggregate(rows, list(LANES))
    decision = decide(agg, "glm-custom")
    check("baseline disqualified by drops and degeneration", "glm-custom" in decision["disqualified"])
    check("best survivor ranks first", decision["ranking"][:1] == ["paddle"])
    check("gap of 1 image is called a tie", any("tie" in n for n in decision["notes"]))
    check(
        "the tie note states the rule as it is applied: fewer than 2 images from the leader",
        any("fewer than 2 images from paddle" in n for n in decision["notes"]),
    )
    check("the tied top is reported as a group", decision["leading_group"] == ["paddle", "vision"])
    check(
        "a tie at the top names no single winner",
        decision["winner"] is None and "No single winner" in decision["verdict"],
    )
    check("the fastest engine in the tie is named", decision["fastest_in_group"] == "vision")
    check(
        "control lane never ranked or disqualified",
        "pdftotext" not in decision["ranking"] and "pdftotext" not in decision["disqualified"],
    )
    close = decide(
        aggregate(
            [row("glm-custom", i, i < 2, 80) for i in range(30)] + [row("glm", i, i < 1, 80) for i in range(30)],
            list(LANES),
        ),
        "glm-custom",
    )
    check(
        "narrow win over the baseline is inconclusive", close["winner"] is None and "inconclusive" in close["verdict"]
    )
    slow = decide(
        aggregate(
            [row("glm-custom", i, i < 3, 10) for i in range(30)] + [row("qwen8b", i, False, 60) for i in range(30)],
            list(LANES),
        ),
        "glm-custom",
    )
    check("cost gate flags a slow winner", "trade-off" in slow["verdict"])

    def race(**engines: tuple[int, float]) -> dict:
        """Each engine: (drop images out of 50, seconds per page)."""
        return decide(
            aggregate(
                [row(lane, i, i < drops, seconds) for lane, (drops, seconds) in engines.items() for i in range(50)],
                list(LANES),
            ),
            "glm-custom",
        )

    exact = race(**{"glm-custom": (5, 10), "paddle": (3, 10), "vision": (5, 10)})
    check(
        "a gap of exactly 2 images is a difference, not a tie (Pick the Winner: fewer than 2)",
        exact["leading_group"] == ["paddle"] and exact["winner"] == "paddle" and exact["verdict"] == "paddle wins.",
    )
    check(
        "the baseline ranking first means no switch",
        race(**{"glm-custom": (0, 10), "paddle": (4, 10)})["winner"] == "glm-custom",
    )
    unrun = race(paddle=(0, 10), vision=(3, 1))
    check(
        "without the baseline the best engine wins, rules 3 and 4 unapplied",
        unrun["winner"] == "paddle" and "not run" in unrun["verdict"],
    )
    edge = decide(
        aggregate(
            [row("glm-custom", i, i < 5, 10) for i in range(50)]
            + [row("paddle", i, i < 6, 10) for i in range(60)]
            + [row("vision", i, i < 7, 10) for i in range(60)]
            + [row("qwen8b", i, False, 10, degenerate=i < 1) for i in range(50)]
            + [row("qwen4b", i, False, 10, degenerate=i < 2) for i in range(50)],
            list(LANES),
        ),
        "glm-custom",
    )
    check(
        "drops on exactly 10% and degeneration on exactly 2% survive rule 1; more is out",
        {"glm-custom", "paddle", "qwen8b"} <= set(edge["ranking"])
        and set(edge["disqualified"]) == {"vision", "qwen4b"},
    )
    fewer_numbers = [dict(row("paddle", i, False), numeric_recall=0.9) for i in range(30)] + [
        row("vision", i, False) for i in range(30)
    ]
    check(
        "equal drops rank by number recall",
        decide(aggregate(fewer_numbers, list(LANES)), "glm-custom")["ranking"] == ["vision", "paddle"],
    )
    try:
        decide(agg, "pdftotext")
        check("a control or pipeline can never be the baseline", False)
    except ValueError:
        check("a control or pipeline can never be the baseline", True)
    from .cli import check_baseline

    def refused(baseline: str) -> str:
        try:
            check_baseline(baseline)
        except SystemExit as exc:
            return str(exc)
        return ""

    check(
        "--baseline refuses a control, a pipeline and an unknown lane, and says why",
        "never ranked" in refused("pdftotext")
        and "never ranked" in refused("auto-paddle")
        and "no such lane" in refused("glm-cutsom")
        and not refused("glm-custom")
        and not refused("tesseract"),
    )
    meta = {
        "documents": 30,
        "pages": 30,
        "images": 30,
        "kinds": {k: 6 for k in kinds},
        "gold_sources": {"typed": 30},
        "no_gold": 0,
        "skipped": {},
    }
    pipe_rows = [
        dict(
            row("auto-paddle", i, False, 5, kinds[i % 5]),
            variant="clean",
            used_layer=i % 2 == 0,
            rule="trustworthy text layer",
        )
        for i in range(10)
    ]
    pipe_table = render(
        aggregate(pipe_rows, list(LANES)), decide(aggregate(pipe_rows, list(LANES)), "glm-custom"), meta, None
    )
    check(
        "pipelines get their own section and are never ranked",
        "Using the born-digital text" in pipe_table
        and "auto-paddle" not in decide(aggregate(pipe_rows, list(LANES)), "glm-custom")["ranking"],
    )
    table = render(agg, decision, meta, None)
    check(
        "report has overall, per-kind and controls tables",
        all(s in table for s in ("Overall", "By document kind", "Controls")),
    )
    check("report names every kind", all(k in table for k in kinds))
    check("report carries no document text", "Einkommensteuererklärung" not in table)

    # --- Google Drive reference against a fake Drive: three uploads, each trashed, none deleted.
    class Call:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class Files:
        def __init__(self):
            self.created, self.trashed, self.deleted = [], [], []

        def create(self, body, media_body, ocrLanguage, fields):
            self.created.append((body["mimeType"], ocrLanguage))
            return Call({"id": f"doc{len(self.created)}"})

        def export(self, fileId, mimeType):
            return Call(("﻿Text from " + fileId).encode())

        def update(self, fileId, body):
            self.trashed.append((fileId, body))
            return Call({})

        def delete(self, fileId):
            self.deleted.append(fileId)
            return Call({})

    class Service:
        def __init__(self):
            self.f = Files()

        def files(self):
            return self.f

    service = Service()
    ref = reference(DriveOCR(service, "de"), data, "image/png", "k")
    check(
        "drive: page, header and footer each uploaded as a Google Doc",
        len(service.f.created) == 3
        and all(m == "application/vnd.google-apps.document" and lang == "de" for m, lang in service.f.created),
    )
    check(
        "drive: every temporary Doc is trashed",
        len(service.f.trashed) == 3 and all(b == {"trashed": True} for _, b in service.f.trashed),
    )
    check("drive: nothing permanently deleted", not service.f.deleted)
    check("drive: BOM stripped from exported text", not ref["full_text"].startswith("﻿"))
    check("drive: cuts recorded near 20% and 80%", 0.12 <= ref["cuts"][0] <= 0.28 and 0.72 <= ref["cuts"][1] <= 0.88)

    # --- Look-alike generator: coverage floors, exact gold, intended classification.
    from .synth import generate

    synth_dir = scratch / "synth"
    summary = generate(synth_dir, None, 20, seed=5)
    docs = summary["documents"]
    check(
        "synth: 20 documents, at least 4 of every kind",
        len(docs) == 20 and all(v >= 4 for v in summary["plan"].values()),
    )
    born = [d for d in docs if not d["file"].startswith("scans/")]
    check(
        "synth: image headers and footers meet their floors",
        sum(d["header"] in ("image", "imagetext") for d in born) >= 6
        and sum(d["footer"] in ("image", "imagetext") for d in born) >= 4,
    )
    synth_pages = discover(synth_dir)
    check(
        "synth: every scan is classified as a scan and has full typed gold",
        all(
            p.tier == "scan" and p.typed_gold and "[TEXT]" in Path(p.typed_gold).read_text()
            for p in synth_pages
            if p.kind == "scans"
        ),
    )
    check(
        "synth: every other document is born-digital",
        all(p.tier == "born-digital" for p in synth_pages if p.kind != "scans"),
    )
    from .samples import Sheet as _Sheet
    from .samples import build as _build

    sparse = scratch / "sparse" / "slides" / "short-slide.pdf"
    _build(
        _Sheet(width=960, height=540)
        .text(60, 80, "Roadmap 2027", 28)
        .text(80, 160, "• Launch in June", 18)
        .text(80, 200, "• Hire two engineers", 18)
        .image_text((40, 480, 600, 520), [("Company · Confidential", 9)]),
        sparse,
    )
    check(
        "a sparse born-digital slide is born-digital, not a scan",
        discover(scratch / "sparse")[0].tier == "born-digital",
    )
    prepare(synth_pages, scratch / "synth-results")
    image_heads = [p for p in synth_pages if p.kind != "scans" and p.typed_gold]
    check(
        "synth: picture text reaches the gold of born-digital pages",
        image_heads
        and all(load_gold(p, scratch / "synth-results")["source"] == "text layer + typed" for p in image_heads),
    )

    def repeated_lines(page) -> int:
        with fitz.open(page.path) as pdf:
            lines = [l.strip() for l in pdf[0].get_text("text").splitlines() if len(l.strip()) > 25]
        return len(lines) - len(set(lines))

    check(
        "synth: no document repeats its own lines",
        all(repeated_lines(p) == 0 for p in synth_pages if p.kind != "scans"),
    )
    junk = [
        p
        for p in synth_pages
        if p.kind == "scans"
        and p.path.suffix == ".pdf"
        and "OCRmyPDF" in (fitz.open(p.path).metadata.get("producer") or "")
    ]
    check(
        "layer rule: an old OCR layer is never trusted",
        all(layer_verdict(p.path, 1)["reason"] == "layer written by an OCR tool" for p in junk),
    )
    check(
        "synth: generated gold never ends a line mid-word on a footnote",
        all(
            not line.rstrip().endswith(("insta", "paya", "Zahl"))
            for p in synth_pages
            if p.typed_gold
            for line in Path(p.typed_gold).read_text().splitlines()
        ),
    )

    # --- Provenance and run conditions.
    from .cli import save_run
    from .env import lane_provenance, parse_llama_servers, run_conditions, snapshot
    from .report import write as write_report

    ps = (
        "  81 12000 /bin/zsh\n"
        "  97 18300000 /opt/homebrew/bin/llama-server -m /models/Qwen3.8-27B-Q4_K_M.gguf --host 127.0.0.1 --port 11435\n"
        " 105 2300000 /opt/homebrew/bin/llama-server -hf ggml-org/GLM-OCR-GGUF --port 11346\n"
        " 212 9680 /opt/homebrew/bin/llama-server -m /Users/x/Library/Application Support/llama.cpp/gemma-4-12B.gguf --host 127.0.0.1 --port 11436\n"
    )
    servers = parse_llama_servers(ps)
    check(
        "llama-server processes parsed with port, model and memory",
        servers
        == [
            {"pid": 97, "port": "11435", "model": "Qwen3.8-27B-Q4_K_M.gguf", "rss_gb": 18.3},
            {"pid": 105, "port": "11346", "model": "ggml-org/GLM-OCR-GGUF", "rss_gb": 2.3},
            {"pid": 212, "port": "11436", "model": "gemma-4-12B.gguf", "rss_gb": 0.01},
        ],
    )
    snap = snapshot("test")
    check(
        "a snapshot records time, free memory and servers", {"at", "available_memory_gb", "llama_servers"} <= set(snap)
    )
    prov = lane_provenance("paddle", {"llama_server": "version: 1 (build 2)"}, None)
    check(
        "an LLM output records its model file and llama.cpp version",
        prov["weights"] == "PaddleOCR-VL-1.6-GGUF.gguf" and prov["llama_server"] == "version: 1 (build 2)",
    )
    check(
        "a classic output records its engine version",
        "tesseract" in lane_provenance("tesseract-clean", {"tesseract": "tesseract 5.5"}, None),
    )
    check(
        "an external GLM server is recorded as such",
        lane_provenance("glm", {}, "http://x/v1")["model"] == "external GLM server",
    )
    check(
        "every lane names a model or an engine, never both",
        all((lane.model is None) != (lane.engine is None) for lane in LANES.values()),
    )
    check(
        "every engine lane records what produced its output",
        all(lane_provenance(lane.id, {}, None) for lane in LANES.values() if lane.engine),
    )
    runs_dir = scratch / "runres"
    save_run(
        runs_dir,
        {
            "run_id": "20260928T100000",
            "machine": {"model": "Mac14,6"},
            "versions": {"llama_server": "v"},
            "snapshots": [
                {
                    "available_memory_gb": 30.0,
                    "llama_servers": servers + [{"port": "11361", "model": "paddle", "rss_gb": 1.8}],
                },
                {"available_memory_gb": 22.5, "llama_servers": servers},
            ],
        },
    )
    cond = run_conditions(runs_dir, {"20260928T100000", "unrecorded"})
    check(
        "run conditions list other llama servers but not the tool's own",
        len(cond) == 1
        and len(cond[0]["other_llama_servers"]) == 3
        and not any("paddle" in s for s in cond[0]["other_llama_servers"]),
    )
    check("run conditions keep the lowest free memory seen", cond[0]["min_free_memory_gb"] == 22.5)
    write_report(runs_dir, agg, decision, dict(meta, conditions=cond, warnings=["x: mixed"]), None, [])
    saved = json.loads((runs_dir / "summary.json").read_text())
    check(
        "summary.json holds engines, decision and conditions",
        {"engines", "decision", "meta"} <= set(saved) and saved["meta"]["conditions"][0]["run_id"] == "20260928T100000",
    )
    check(
        "summary.md shows timing conditions and warnings",
        "Timing conditions" in (runs_dir / "summary.md").read_text()
        and "warning: x: mixed" in (runs_dir / "summary.md").read_text(),
    )

    # --- profile: counts only, and a folder with nothing readable is a result, not a crash.
    from .profile import summarise

    empty_profile = scratch / "empty-profile.json"
    empty_profile.write_text(json.dumps({"total_pdfs": 0, "sampled": 0, "profiles": []}))
    try:
        check("profile of a folder without readable PDFs still summarises", "0 PDFs found" in summarise(empty_profile))
    except Exception as exc:
        check(f"profile of a folder without readable PDFs still summarises ({type(exc).__name__})", False)

    # --- pick: real files by kind, opaque names, private mapping.
    from .pick import pick as pick_real

    fake_profile = scratch / "profile.json"
    born = [p for p in synth_pages if p.kind != "scans"]
    fake_profile.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "path": p.source,
                        "kind": p.kind,
                        "tier": "born-digital",
                        "producer": "word-processor",
                        "language": "de",
                        "header": "text",
                        "id": f"d{i}",
                    }
                    for i, p in enumerate(born)
                ]
                + [{"path": str(scratch / "missing.pdf"), "kind": "letters", "tier": "born-digital"}]
            }
        )
    )
    picked_dir = scratch / "picked"
    picked = pick_real(fake_profile, picked_dir, 2, seed=1)
    names = sorted(str(f.relative_to(picked_dir)) for f in picked_dir.rglob("*.pdf"))
    check(
        "pick copies per kind under opaque names",
        picked["per_kind"] == {k: 2 for k in ("letters", "invoices", "presentations", "work-docs")}
        and all(Path(n).name.startswith("r") for n in names),
    )
    check(
        "pick keeps the mapping private (mode 600)", oct((picked_dir / ".pick.json").stat().st_mode & 0o777) == "0o600"
    )
    dup = scratch / "dup-copy.pdf"
    dup.write_bytes(Path(born[0].source).read_bytes())
    dup_profile = scratch / "profile-dup.json"
    base = json.loads(fake_profile.read_text())["profiles"]
    dup_profile.write_text(json.dumps({"profiles": base + [dict(base[0], path=str(dup), id="dup")]}))
    second = scratch / "picked-second"
    second_summary = pick_real(dup_profile, second, 2, seed=1, exclude=[picked_dir])
    first_shas = {m["sha"] for m in json.loads((picked_dir / ".pick.json").read_text())["files"]}
    second_shas = [m["sha"] for m in json.loads((second / ".pick.json").read_text())["files"]]
    check("pick --exclude never repeats an earlier folder's documents", not (first_shas & set(second_shas)))
    check(
        "pick never takes the same content twice under another name",
        len(second_shas) == len(set(second_shas))
        and second_summary["skipped"].get("same content already picked or used", 0)
        + second_summary["skipped"].get("already used", 0)
        > 0,
    )
    check(
        "pick skips files that no longer exist",
        all("missing" not in m["source"] for m in json.loads((picked_dir / ".pick.json").read_text())["files"]),
    )

    scan_pages = [p for p in synth_pages if p.kind == "scans"]
    fake_scans = scratch / "profile-scans.json"
    fake_scans.write_text(
        json.dumps(
            {
                "profiles": [
                    {"path": p.source, "kind": "scans", "tier": "scan", "producer": "scanner/ocr", "id": f"s{i}"}
                    for i, p in enumerate(scan_pages)
                    if p.source.endswith(".pdf") or p.source.endswith(".jpg")
                ]
            }
        )
    )
    scans_out = scratch / "picked-scans"
    picked_scans = pick_real(fake_scans, scans_out, 3, seed=2, tier="scan")
    check(
        "pick --tier scan copies scans into one scans/ folder",
        set(picked_scans["per_kind"]) <= {"scans"} and len(list((scans_out / "scans").glob("r*"))) >= 1,
    )

    # --- Consensus reference: agreement of independent engines, leave-one-out, and its check.
    from . import consensus

    check(
        "consensus keeps what two voices agree on, as often as the second voice has it",
        sorted(consensus.agreed([["a", "b", "b"], ["a", "b"], ["c"]])) == ["a", "b"],
    )

    def voice(text, header_words):
        return {
            "text": text,
            "words": [{"text": w, "box": [0.1, 0.05, 0.2, 0.08]} for w in header_words]
            + [{"text": "Körper", "box": [0.1, 0.5, 0.2, 0.52]}],
        }

    panel = {
        "qwen8b": {"text": "Müller GmbH Rechnung 490,00 Körper"},
        "vision": voice("Müller GmbH Rechnung 490,00 Körper", ["Müller", "GmbH"]),
        "tesseract-clean": voice("Muller GmbH Rechnung 490,00 Körper", ["Muller", "GmbH"]),
        "rapidocr": voice("Müller GmbH Rechnunq 490,00 Körper", ["Müller", "GmbH"]),
    }
    for_paddle = consensus.consensus_gold(panel, exclude="paddle")
    check(
        "consensus: full text is what two engines agree on",
        "Rechnung" in for_paddle["full_text"] and "490,00" in for_paddle["numbers"],
    )
    check("consensus: header from positioned engines", sorted(for_paddle["header_text"].split()) == ["GmbH", "Müller"])
    check(
        "consensus: an engine is never part of its own reference",
        "vision" not in consensus.consensus_gold(panel, exclude="vision")["voices"],
    )
    check(
        "consensus: fewer than three other voices gives no reference",
        consensus.consensus_gold({"vision": panel["vision"], "rapidocr": panel["rapidocr"]}, exclude="x") is None,
    )
    check("consensus: no character error rate on a bag of words", score_image("Müller GmbH", for_paddle)["cer"] is None)
    same = [
        {
            "lane": l,
            "image": (i, "scan"),
            "truth": {"drop": d, "numeric_recall": n},
            "consensus": {"drop": d, "numeric_recall": n},
            "true_header": ["a", "b"],
            "consensus_header": ["a", "b"],
        }
        for i, (l, d, n) in enumerate(
            [("qwen8b", False, 1.0), ("vision", True, 0.8), ("tesseract", True, 0.6), ("glm", True, 0.4)]
        )
    ]
    agreement = consensus.check(same)
    check(
        "consensus check: identical scoring reads as full agreement",
        agreement["same_drop_verdict"] == 1.0
        and agreement["header_recall"] == 1.0
        and agreement["rank_agreement"] == 1.0,
    )

    # --- One run per results folder.
    from .cli import lock_results

    held = lock_results(scratch / "locked")
    try:
        lock_results(scratch / "locked")
        check("a second run on the same results folder is refused", False)
    except SystemExit as exc:
        check("a second run on the same results folder is refused", "already working on" in str(exc))
    held.close()
    check("the lock is released when the first run ends", lock_results(scratch / "locked") is not None)

    failed = [name for name, ok in results if not ok]
    for name, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    for name in skipped:
        print(f"SKIPPED live: {name}")
    print(f"selftest: {len(results) - len(failed)}/{len(results)} checks passed")
    shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(1 if failed else 0)
