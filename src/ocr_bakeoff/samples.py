"""A synthetic test folder: every document kind, headers and footers as text and as images.

For trying the tool without private documents, and as its end-to-end smoke test. Every
sample is invented. Text that exists only inside an image is given to the scorer in a
typed .gold.txt next to the file, exactly as a user would for their own documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from .config import FOOTER_FRACTION, HEADER_FRACTION

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _font() -> str | None:
    return next((f for f in FONT_CANDIDATES if Path(f).exists()), None)


@dataclass
class Sheet:
    width: float = 595
    height: float = 842
    font: str | None = None
    texts: list[tuple[float, float, str, float]] = field(default_factory=list)
    images: list[tuple[fitz.Rect, list[tuple[str, float]]]] = field(default_factory=list)
    rules: list[float] = field(default_factory=list)  # y of each horizontal table rule

    def text(self, x: float, y: float, value: str, size: float = 10) -> Sheet:
        self.texts.append((x, y, value, size))
        return self

    def image_text(self, rect: tuple[float, float, float, float], lines: list[tuple[str, float]]) -> Sheet:
        """Text rendered into a picture: present on the page, absent from the text layer."""
        self.images.append((fitz.Rect(*rect), lines))
        return self


def _put(page: fitz.Page, x: float, y: float, value: str, size: float, font: str | None = None) -> None:
    font = font or _font()
    if font:
        page.insert_text((x, y), value, fontname="body", fontfile=font, fontsize=size)
    else:
        page.insert_text((x, y), value.translate(str.maketrans("şŞğĞıİ", "sSgGiI")), fontname="helv", fontsize=size)


def _picture(
    rect: fitz.Rect,
    lines: list[tuple[str, float]],
    font: str | None = None,
    fill: tuple[float, float, float] = (0.93, 0.95, 0.98),
) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=rect.width, height=rect.height)
    page.draw_rect(page.rect, color=None, fill=fill)
    y = 4.0
    for value, size in lines:
        y += size * 1.15
        _put(page, 6, y, value, size, font)
    return page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).tobytes("png")


def build(sheet: Sheet, path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=sheet.width, height=sheet.height)
    for rect, lines in sheet.images:
        page.insert_image(rect, stream=_picture(rect, lines, sheet.font))
    for x, y, value, size in sheet.texts:
        _put(page, x, y, value, size, sheet.font)
    for y in sheet.rules:
        page.draw_line((50, y), (sheet.width - 50, y), color=(0.3, 0.3, 0.3), width=0.6)
    doc.set_metadata({"producer": "ocr-bakeoff samples", "creator": "ocr-bakeoff samples"})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, no_new_id=True)  # no random /ID: the same sheet gives the same file, key and scan seed


def as_scan(
    pdf: Path,
    out: Path,
    *,
    dpi: int = 150,
    degrees: float = 1.2,
    seed: int = 7,
    image: bool = False,
    noise: float = 7.0,
    quality: int = 70,
    junk_layer: str | None = None,
) -> None:
    """Rasterise, skew, grey, noise: an image-only 'scan' with no text layer at all."""
    with fitz.open(pdf) as src:
        png = src[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False).tobytes("png")
    gray = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    gray = cv2.warpAffine(gray, cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0), (w, h), borderValue=250)
    noisy = np.clip(
        gray.astype(np.float32) * 0.9 + 12 + np.random.default_rng(seed).normal(0, noise, gray.shape), 0, 255
    ).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, quality])
    out.parent.mkdir(parents=True, exist_ok=True)
    if image:
        out.write_bytes(jpg.tobytes())
        return
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=jpg.tobytes())
    producer = "ocr-bakeoff samples scanner"
    if junk_layer:
        # A prior OCR pass that got it wrong: invisible text, the right length, the wrong words.
        page.insert_textbox(fitz.Rect(40, 40, 555, 800), junk_layer, fontsize=9, render_mode=3)
        producer = "Tesseract OCRmyPDF (simulated old OCR layer)"
    doc.set_metadata({"producer": producer, "creator": producer})
    doc.save(out, no_new_id=True)


def regions(sheet: Sheet) -> dict[str, list[str]]:
    """Header / footer / all lines by the scorer's own rule: a line's top in the top 20%, its
    bottom in the bottom 20%. Picture text counts by the picture's position."""
    items = [(y - 0.9 * size, y + 0.25 * size, value) for x, y, value, size in sheet.texts]
    for rect, lines in sheet.images:
        items += [(rect.y0, rect.y1, value) for value, _ in lines]
    items.sort(key=lambda item: (item[0], item[2]))
    h = sheet.height
    return {
        "header": [v for top, _, v in items if top < HEADER_FRACTION * h],
        "footer": [v for _, bottom, v in items if bottom > (1 - FOOTER_FRACTION) * h],
        "text": [v for _, _, v in items],
    }


def gold(path: Path, **sections: list[str]) -> None:
    body = ["# Typed reference for " + path.name.replace(".gold.txt", ""), ""]
    for name, lines in sections.items():
        body += [f"[{name.upper()}]", *lines, ""]
    path.write_text("\n".join(body), encoding="utf-8")


LETTERHEAD = ["Müller & Şahin Steuerberatung GmbH", "Königstraße 12 · 70173 Stuttgart · Telefon 0711 4488 210"]
LETTER_BODY = [
    "Sehr geehrte Frau Yılmaz,",
    "anbei erhalten Sie die Rechnung Nr. 2026-0417 über 490,00 EUR für die Erstellung",
    "Ihrer Einkommensteuererklärung 2025. Die Leistungen wurden im Zeitraum vom",
    "01.02.2026 bis 28.02.2026 erbracht und sind gemäß Steuerberatervergütungsverordnung",
    "abgerechnet. Bitte überweisen Sie den Betrag fristgerecht auf das unten genannte Konto.",
    "Für Rückfragen stehen wir Ihnen gern zur Verfügung. Mit freundlichen Grüßen",
    "Ayşe Öztürk, Steuerberaterin",
]
LETTER_FOOTER = [
    "¹ Zahlbar innerhalb von 14 Tagen ohne Abzug.",
    "Amtsgericht Stuttgart HRB 723456 · USt-IdNr. DE298765432 · IBAN DE12 6005 0101 0004 1234 56",
    "Geschäftsführer: Ayşe Öztürk, Jürgen Müller · Seite 1 von 1",
]


def letter_sheet(logo: bool) -> Sheet:
    s = Sheet().text(60, 55, LETTERHEAD[0], 16).text(60, 75, LETTERHEAD[1], 8.5)
    if logo:
        s.image_text((455, 30, 560, 80), [("MSB", 20), ("Partner", 9)])
    s.text(60, 150, "Müller & Şahin GmbH · Königstraße 12 · 70173 Stuttgart", 6.5)
    s.text(60, 168, "Frau Elif Yılmaz", 10).text(60, 181, "Gartenweg 5", 10).text(60, 194, "72070 Tübingen", 10)
    s.text(400, 168, "Ihr Zeichen: EY-2025", 8.5).text(400, 180, "Unser Zeichen: MS/0417", 8.5).text(
        400, 192, "Datum: 12.03.2026", 8.5
    )
    for i, line in enumerate(LETTER_BODY):
        s.text(60, 260 + i * 20, line, 10)
    s.text(60, 735, LETTER_FOOTER[0], 7.5).text(60, 790, LETTER_FOOTER[1], 6.5).text(60, 802, LETTER_FOOTER[2], 6.5)
    return s


def write_samples(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    # 1. Letter: text letterhead, text logo inside a picture, text footer and footnote.
    p = root / "letters" / "letter-text-header.pdf"
    build(letter_sheet(logo=True), p)
    gold(p.with_name(p.stem + ".gold.txt"), header=["MSB", "Partner"])
    made.append(p)

    # 2. Letter: letterhead AND footer exist only as images; the body is text.
    p = root / "letters" / "letter-image-header.pdf"
    head = [("Kanzlei Doğan & Weiß Rechtsanwälte", 13), ("Schloßstraße 44 · 10827 Berlin · Tel. 030 8877 5500", 7.5)]
    foot = [
        ("Bankverbindung: Berliner Sparkasse · IBAN DE55 1005 0000 0190 3344 77", 6.5),
        ("Partnerschaftsregister AG Charlottenburg PR 1188 B · Seite 1 von 2", 6.5),
    ]
    s = Sheet().image_text((50, 28, 545, 88), head).image_text((50, 780, 545, 815), foot)
    s.text(60, 150, "Herrn Thomas Krüger · Hauptstraße 3 · 14482 Potsdam", 9).text(
        400, 150, "Aktenzeichen: 2026/0193", 8.5
    )
    body = [
        "Sehr geehrter Herr Krüger,",
        "in der Mietsache Krüger ./. Özdemir teilen wir Ihnen mit, dass die Gegenseite",
        "die Nebenkostenabrechnung für 2024 in Höhe von 1.284,17 EUR bis zum 30.04.2026 prüfen",
        "wird. Ein Termin zur Güteverhandlung ist für den 14.05.2026 angesetzt. Wir empfehlen,",
        "bis dahin sämtliche Belege und die Zählerstände vom 31.12.2024 bereitzuhalten.",
        "Mit freundlichen Grüßen, Selin Doğan, Rechtsanwältin",
    ]
    for i, line in enumerate(body):
        s.text(60, 200 + i * 20, line, 10)
    build(s, p)
    gold(p.with_name(p.stem + ".gold.txt"), header=[t for t, _ in head], footer=[t for t, _ in foot])
    made.append(p)

    # 3. Invoice with a line-item table.
    p = root / "invoices" / "invoice-table.pdf"
    s = invoice = (
        Sheet()
        .text(60, 55, "Fahrschule Weißenburg · Inh. Gökhan Arslan", 14)
        .text(60, 72, "Bahnhofstraße 9 · 91781 Weißenburg", 8.5)
    )
    s.text(60, 140, "RECHNUNG Nr. FS-2026-0088", 12).text(400, 140, "Datum: 03.03.2026", 9)
    rows = [
        ("Pos", "Leistung", "Menge", "Einzelpreis", "Gesamt"),
        ("1", "Fahrstunde Klasse B", "14", "35,00", "490,00"),
        ("2", "Sonderfahrt Autobahn", "4", "52,50", "210,00"),
        ("3", "Grundgebühr", "1", "380,00", "380,00"),
        ("4", "Vorstellung zur Prüfung", "1", "145,00", "145,00"),
    ]
    for r, row in enumerate(rows):
        for x, cell in zip((60, 95, 300, 360, 460), row):
            s.text(x, 190 + r * 18, cell, 9.5)
    s.text(300, 300, "Nettobetrag 1.029,41 · MwSt. 19 % 195,59 · Gesamt 1.225,00 EUR", 9)
    s.text(60, 340, "Zahlbar bis 17.03.2026 ohne Abzug. Vielen Dank für Ihr Vertrauen und allzeit gute Fahrt!", 9)
    s.text(60, 795, "Steuernummer 203/118/40221 · IBAN DE71 7655 0000 0008 8123 45 · Seite 1 von 1", 6.5)
    build(s, p)
    made.append(p)

    # 4. Presentation slide: landscape, large type, footer as an image.
    p = root / "presentations" / "slide-image-footer.pdf"
    foot = [("Muster Consulting · Vertraulich · Q1 2026 Business Review", 9)]
    s = Sheet(width=960, height=540).text(60, 70, "Umsatzentwicklung Q1 2026", 30)
    bullets = [
        "Umsatz 2,41 Mio. EUR (+18 % gegenüber Q1 2025)",
        "Neukunden: 37, davon 12 aus der Türkei und Österreich",
        "Marge stabil bei 23,5 % trotz gestiegener Lizenzkosten",
        "Nächster Meilenstein: Markteintritt Schweiz im Juni 2026",
        "Risiken: Währungsschwankungen, Fachkräftemangel in München und İzmir",
    ]
    for i, line in enumerate(bullets):
        s.text(80, 150 + i * 42, "• " + line, 18)
    s.image_text((40, 495, 700, 525), foot).text(880, 515, "3 / 12", 11)
    build(s, p)
    gold(p.with_name(p.stem + ".gold.txt"), footer=[t for t, _ in foot])
    made.append(p)

    # 5. Work document: running header, two columns, footnotes, page number.
    p = root / "work-docs" / "spec-two-column.pdf"
    s = Sheet().text(60, 40, "Spezifikation Belegerfassung v1.2 — Vertraulich", 8).text(470, 40, "Stand: 20.09.2026", 8)
    s.text(60, 90, "3.2 Anforderungen an die Texterkennung", 13)
    left = [
        "Die Erkennung muss Kopf- und Fußzeilen",
        "vollständig erfassen, auch wenn diese als",
        "Grafik vorliegen.¹ Beträge sind exakt",
        "zu übernehmen; eine Korrektur durch das",
        "Modell ist unzulässig. Umlaute (ä, ö, ü)",
        "und ß müssen erhalten bleiben.",
    ]
    right = [
        "Die Latenz je Seite darf 90 Sekunden",
        "nicht überschreiten.² Für türkische",
        "Namen wie Çelik oder Güneş gelten",
        "dieselben Regeln. Abnahme erfolgt",
        "gegen 30 Referenzdokumente mit",
        "handgeprüften Texten.",
    ]
    for i, (a, b) in enumerate(zip(left, right)):
        s.text(60, 125 + i * 15, a, 9.5).text(320, 125 + i * 15, b, 9.5)
    s.text(60, 760, "¹ Siehe Abschnitt 2.4 zur Behandlung von Briefköpfen.", 7).text(
        60, 772, "² Gemessen als Median über alle Seiten.", 7
    )
    s.text(280, 805, "Seite 7 von 24", 8)
    build(s, p)
    made.append(p)

    # 6. Real-style scans with typed gold (text never existed as a layer).
    for source, target, sheet, options in (
        (root / "letters" / "letter-text-header.pdf", root / "scans" / "letter-scan.pdf", letter_sheet(logo=True), {}),
        (
            root / "invoices" / "invoice-table.pdf",
            root / "scans" / "receipt-photo.jpg",
            invoice,
            {"dpi": 130, "degrees": -1.6, "seed": 11, "image": True},
        ),
    ):
        as_scan(source, target, **options)
        gold(target.with_name(target.stem + ".gold.txt"), **regions(sheet))
        made.append(target)
    return made
