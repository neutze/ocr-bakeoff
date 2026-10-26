"""Look-alike test documents: shaped like a profiled folder, filled with invented content.

Each generated document copies the *structure* of one real profile of its kind — paper
format, header and footer style (text, image, both, none), columns, table rules, font
sizes, language — and nothing else. Every word on it is drawn from the pools below, so
the set can be shared or published, and every file has an exact reference: born-digital
pages carry their own text layer (plus a .gold.txt for text that exists only in images),
scans get a complete .gold.txt because the generator knows every word it rasterised.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FOOTER_FRACTION, HEADER_FRACTION
from .samples import Sheet, as_scan, build, gold, regions

KINDS = ("letters", "invoices", "presentations", "work-docs", "scans")
FONTS = [
    f"/System/Library/Fonts/Supplemental/{name}.ttf"
    for name in ("Arial", "Times New Roman", "Georgia", "Verdana", "Trebuchet MS", "Courier New")
]
WIDTH_FACTOR = {
    "Arial": 0.50,
    "Times New Roman": 0.45,
    "Georgia": 0.50,
    "Verdana": 0.58,
    "Trebuchet MS": 0.49,
    "Courier New": 0.60,
}
SHAPES = {
    "a4-portrait": (595, 842),
    "letter-portrait": (612, 792),
    "portrait-other": (595, 842),
    "a4-landscape": (842, 595),
    "16:9": (960, 540),
    "4:3/landscape": (800, 600),
}

POOLS: dict[str, dict[str, Any]] = {
    "de": {
        "companies": [
            "Stadtwerke Bad Kötzting GmbH",
            "Praxis Dr. med. Jürgen Weiß",
            "Müller & Şahin Steuerberatung",
            "Versicherungskammer Süd AG",
            "Hausverwaltung Özdemir & Partner",
            "Autohaus Bräuer KG",
            "Grünwerk Energie eG",
            "Kanzlei Doğan & Weiß",
            "Bäckerei Lührs GmbH",
            "Zahnärzte am Schloßpark",
        ],
        "streets": [
            "Königstraße",
            "Hauptstraße",
            "Schloßallee",
            "Mühlenweg",
            "Bahnhofstraße",
            "Lindenstraße",
            "Am Anger",
        ],
        "cities": [
            ("70173", "Stuttgart"),
            ("37073", "Göttingen"),
            ("93444", "Bad Kötzting"),
            ("10827", "Berlin"),
            ("80331", "München"),
            ("50667", "Köln"),
            ("72070", "Tübingen"),
        ],
        "people": [
            "Elif Yılmaz",
            "Jürgen Müller",
            "Ayşe Öztürk",
            "Thomas Krüger",
            "Selin Doğan",
            "Katrin Schäfer",
            "Mehmet Çelik",
            "Björn Weiß",
        ],
        "subjects": [
            "Nebenkostenabrechnung {year}",
            "Beitragsanpassung zum 01.{mm}.{year}",
            "Ihre Rechnung Nr. {inv}",
            "Kündigungsbestätigung Vertrag {ref}",
            "Terminbestätigung für den {date}",
            "Mahnung zu Vorgang {ref}",
        ],
        "body": [
            "wir bestätigen den Eingang Ihres Schreibens vom {date} und bedanken uns für Ihre Geduld.",
            "Der offene Betrag von {amount} ist bis zum {due} auf das unten genannte Konto zu überweisen.",
            "Die Abrechnung für den Zeitraum {period} ergibt eine Nachzahlung in Höhe von {amount}.",
            "Bitte prüfen Sie die beigefügten Unterlagen und senden Sie uns die Erklärung bis zum {due} zurück.",
            "Für Rückfragen erreichen Sie Frau Öztürk montags bis freitags von 8 bis 16 Uhr unter {phone}.",
            "Gemäß § 556 BGB sind Einwendungen innerhalb von zwölf Monaten nach Zugang geltend zu machen.",
            "Ihre Vertragsnummer {ref} bleibt unverändert; der monatliche Abschlag beträgt künftig {amount}.",
            "Wir weisen darauf hin, dass für verspätete Zahlungen Verzugszinsen berechnet werden können.",
        ],
        "salutation": "Sehr geehrte Damen und Herren,",
        "closing": "Mit freundlichen Grüßen",
        "L": {
            "date": "Datum",
            "your": "Ihr Zeichen",
            "our": "Unser Zeichen",
            "invoice": "RECHNUNG",
            "no": "Nr.",
            "cols": ("Pos.", "Leistung", "Menge", "Einzelpreis", "Gesamt"),
            "net": "Nettobetrag",
            "vat": "MwSt. 19 %",
            "gross": "Gesamtbetrag",
            "page": "Seite {n} von {m}",
            "due": "Zahlbar bis {date} ohne Abzug.",
            "register": "Amtsgericht {city} HRB {hrb}",
            "ceo": "Geschäftsführer: {name}",
            "tax": "USt-IdNr. DE{vat}",
            "cert": "BESCHEINIGUNG",
            "fields": ("Name", "Geburtsdatum", "Anschrift", "Ausgestellt am"),
            "stamp": "Stempel und Unterschrift",
            "confidential": "Vertraulich",
            "status": "Stand",
        },
        "items": [
            "Wartung Heizungsanlage",
            "Fahrstunde Klasse B",
            "Beratung Steuererklärung",
            "Ersatzteil Umwälzpumpe",
            "Anfahrtspauschale",
            "Grundgebühr",
            "Prüfung nach DIN VDE 0701",
            "Reinigung Außenanlagen",
        ],
        "titles": [
            "Quartalsbericht Q{q} {year}",
            "Projektstatus Digitalisierung",
            "Umsatzentwicklung {year}",
            "Maßnahmenplan Energieeffizienz",
            "Ergebnisse der Kundenbefragung",
        ],
        "bullets": [
            "Umsatz {amount} (+{pct} % gegenüber Vorjahr)",
            "{n} Neukunden, davon {m} aus Österreich und der Türkei",
            "Marge stabil bei {pct} % trotz höherer Lizenzkosten",
            "Nächster Meilenstein: Go-live am {date}",
            "Risiken: Lieferzeiten, Fachkräftemangel in München und İzmir",
            "Budget {amount}, davon {pct} % verbraucht",
        ],
        "sections": [
            "Anforderungen an die Texterkennung",
            "Architektur und Schnittstellen",
            "Datenschutz und Löschfristen",
            "Abnahmekriterien",
            "Offene Punkte",
        ],
        "prose": [
            "Die Erkennung muss Kopf- und Fußzeilen vollständig erfassen, auch wenn diese als Grafik vorliegen.",
            "Beträge sind exakt zu übernehmen; eine Korrektur durch das Modell ist unzulässig.",
            "Umlaute und das ß müssen erhalten bleiben, ebenso türkische Namen wie Çelik oder Güneş.",
            "Die Latenz je Seite darf im Median {n} Sekunden nicht überschreiten.",
            "Personenbezogene Daten werden nach {n} Tagen automatisch gelöscht.",
            "Die Schnittstelle liefert Ergebnisse als JSON mit Seitennummer und Konfidenz.",
        ],
    },
    "en": {
        "companies": [
            "Northwind Analytics Ltd",
            "Blue Harbor Insurance Co.",
            "Oakridge Property Management",
            "Summit Dental Group",
            "Brightline Utilities Inc.",
            "Carter & Webb LLP",
            "Maple Street Clinic",
        ],
        "streets": ["Market Street", "Oak Avenue", "Riverside Drive", "King's Road", "Elm Court", "Harbour Lane"],
        "cities": [
            ("94103", "San Francisco, CA"),
            ("02139", "Cambridge, MA"),
            ("EC1V 2NX", "London"),
            ("M1 1AE", "Manchester"),
            ("10001", "New York, NY"),
            ("78701", "Austin, TX"),
        ],
        "people": ["Sarah Mitchell", "James O'Connor", "Priya Raman", "Daniel Weiss", "Emily Zhang", "Can Aydın"],
        "subjects": [
            "Your invoice {inv}",
            "Policy renewal notice {ref}",
            "Annual statement {year}",
            "Appointment confirmation for {date}",
            "Change to your plan effective {date}",
        ],
        "body": [
            "Thank you for your letter dated {date}; we have updated your account accordingly.",
            "The outstanding balance of {amount} is due by {due}. Please use the reference below.",
            "Your premium for the period {period} will be {amount}, payable in monthly instalments.",
            "Please review the enclosed documents and return the signed form by {due}.",
            "If you have any questions, call our service team at {phone}, Monday to Friday.",
            "Late payments may incur a fee of {amount} as described in section 4.2 of your agreement.",
            "Your account number {ref} remains unchanged; no action is required on your part.",
        ],
        "salutation": "Dear Customer,",
        "closing": "Kind regards,",
        "L": {
            "date": "Date",
            "your": "Your ref",
            "our": "Our ref",
            "invoice": "INVOICE",
            "no": "No.",
            "cols": ("#", "Description", "Qty", "Unit price", "Amount"),
            "net": "Subtotal",
            "vat": "Tax 8.25 %",
            "gross": "Total due",
            "page": "Page {n} of {m}",
            "due": "Payment due by {date}.",
            "register": "Registered in England No. {hrb}",
            "ceo": "Director: {name}",
            "tax": "VAT No. GB{vat}",
            "cert": "CERTIFICATE",
            "fields": ("Name", "Date of birth", "Address", "Issued on"),
            "stamp": "Stamp and signature",
            "confidential": "Confidential",
            "status": "Status",
        },
        "items": [
            "Consulting hours",
            "Annual licence",
            "On-site support",
            "Replacement part",
            "Travel expenses",
            "Setup fee",
            "Data migration",
        ],
        "titles": [
            "Q{q} {year} Business Review",
            "Project Status: Document Pipeline",
            "Revenue Outlook {year}",
            "Customer Survey Results",
            "Roadmap and Milestones",
        ],
        "bullets": [
            "Revenue {amount} (+{pct}% year over year)",
            "{n} new customers, {m} of them enterprise",
            "Gross margin steady at {pct}%",
            "Next milestone: launch on {date}",
            "Risks: supplier lead times and hiring",
            "Budget {amount}, {pct}% spent to date",
        ],
        "sections": [
            "Requirements for text recognition",
            "Architecture and interfaces",
            "Retention and privacy",
            "Acceptance criteria",
            "Open questions",
        ],
        "prose": [
            "Recognition must capture headers and footers in full, including those rendered as images.",
            "Amounts are transcribed exactly; the model may not correct them.",
            "Latency per page must not exceed {n} seconds at the median.",
            "Personal data is deleted automatically after {n} days.",
            "Results are returned as JSON with page number and confidence.",
        ],
    },
    "tr": {
        "companies": [
            "Yılmaz İnşaat A.Ş.",
            "Güneş Sigorta A.Ş.",
            "Özkan Hukuk Bürosu",
            "Ege Diş Kliniği",
            "Başkent Elektrik Dağıtım A.Ş.",
            "Çelik Muhasebe ve Danışmanlık",
        ],
        "streets": ["Atatürk Caddesi", "Cumhuriyet Bulvarı", "Gül Sokak", "İnönü Caddesi", "Kıbrıs Şehitleri Caddesi"],
        "cities": [
            ("35220", "İzmir"),
            ("26010", "Eskişehir"),
            ("48400", "Muğla"),
            ("06420", "Ankara"),
            ("34710", "İstanbul"),
        ],
        "people": ["Ayşe Yıldız", "Mehmet Öztürk", "Zeynep Çelik", "Emre Doğan", "Elif Şahin", "Barış Güneş"],
        "subjects": [
            "{inv} numaralı faturanız",
            "{year} yılı hesap özeti",
            "Poliçe yenileme bildirimi {ref}",
            "{date} tarihli randevu onayı",
        ],
        "body": [
            "{date} tarihli yazınız için teşekkür ederiz; hesabınız güncellenmiştir.",
            "{amount} tutarındaki borcunuzun {due} tarihine kadar ödenmesi gerekmektedir.",
            "{period} dönemine ait hesap özetiniz ekte bilginize sunulmuştur.",
            "Sorularınız için hafta içi {phone} numaralı hattımızdan bize ulaşabilirsiniz.",
            "Sözleşme numaranız {ref} değişmemiştir; herhangi bir işlem yapmanıza gerek yoktur.",
            "Gecikmeli ödemelerde sözleşmenin 4.2 maddesi uyarınca gecikme faizi uygulanır.",
        ],
        "salutation": "Sayın Müşterimiz,",
        "closing": "Saygılarımızla,",
        "L": {
            "date": "Tarih",
            "your": "İlgi",
            "our": "Sayı",
            "invoice": "FATURA",
            "no": "No",
            "cols": ("Sıra", "Açıklama", "Miktar", "Birim fiyat", "Tutar"),
            "net": "Ara toplam",
            "vat": "KDV %20",
            "gross": "Genel toplam",
            "page": "Sayfa {n} / {m}",
            "due": "Son ödeme tarihi: {date}.",
            "register": "Ticaret Sicil No: {hrb}",
            "ceo": "Genel Müdür: {name}",
            "tax": "Vergi No: {vat}",
            "cert": "BELGE",
            "fields": ("Adı Soyadı", "Doğum tarihi", "Adres", "Düzenlenme tarihi"),
            "stamp": "Kaşe ve imza",
            "confidential": "Gizli",
            "status": "Durum",
        },
        "items": ["Danışmanlık hizmeti", "Yıllık lisans", "Yerinde destek", "Yedek parça", "Kurulum ücreti"],
        "titles": ["{year} Yılı {q}. Çeyrek Değerlendirmesi", "Proje Durumu", "Müşteri Anketi Sonuçları"],
        "bullets": [
            "Ciro {amount} (geçen yıla göre +%{pct})",
            "{n} yeni müşteri, {m} tanesi kurumsal",
            "Brüt kâr marjı %{pct} seviyesinde",
            "Sonraki kilometre taşı: {date}",
        ],
        "sections": ["Metin tanıma gereksinimleri", "Mimari ve arayüzler", "Kabul kriterleri"],
        "prose": [
            "Tanıma, üst ve alt bilgileri görsel olsalar bile eksiksiz yakalamalıdır.",
            "Tutarlar olduğu gibi aktarılmalı; model tarafından düzeltilemez.",
            "Türkçe karakterler (ç, ğ, ı, İ, ö, ş, ü) korunmalıdır.",
            "Sayfa başına gecikme medyanda {n} saniyeyi geçmemelidir.",
        ],
    },
}


@dataclass
class Params:
    kind: str
    template: str
    lang: str
    width: float
    height: float
    header: str
    footer: str
    columns: int
    rules: bool
    body: float
    title: float
    font: str | None


class Writer:
    """Invented values, formatted the way each language formats them."""

    def __init__(self, lang: str, rng: random.Random) -> None:
        self.lang, self.rng, self.pool = lang, rng, POOLS[lang]

    def pick(self, key: str) -> Any:
        return self.rng.choice(self.pool[key])

    def date(self) -> str:
        d = dt.date(2024, 1, 1) + dt.timedelta(days=self.rng.randrange(0, 900))
        return d.strftime("%m/%d/%Y") if self.lang == "en" and self.rng.random() < 0.5 else d.strftime("%d.%m.%Y")

    def money(self, value: float) -> str:
        if self.lang == "en":
            return f"${value:,.2f}" if self.rng.random() < 0.6 else f"£{value:,.2f}"
        text = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{text} TL" if self.lang == "tr" else f"{text} EUR"

    def fill(self, template: str) -> str:
        r = self.rng
        return template.format(
            date=self.date(),
            due=self.date(),
            amount=self.money(r.uniform(12, 9800)),
            period=f"{self.date()} – {self.date()}",
            phone=f"0{r.randint(30, 999)} {r.randint(1000, 99999)} {r.randint(10, 999)}",
            ref=f"{r.choice('ABKMSV')}{r.randint(10000, 999999)}",
            inv=f"{r.randint(2024, 2026)}-{r.randint(100, 9999):04d}",
            year=r.choice((2024, 2025, 2026)),
            mm=f"{r.randint(1, 12):02d}",
            q=r.randint(1, 4),
            pct=r.randint(3, 48),
            n=r.randint(2, 180),
            m=r.randint(1, 40),
        )

    def iban(self) -> str:
        country = {"de": "DE", "tr": "TR", "en": "GB"}[self.lang]
        digits = "".join(str(self.rng.randint(0, 9)) for _ in range(20))
        return f"{country}{self.rng.randint(10, 99)} " + " ".join(digits[i : i + 4] for i in range(0, 20, 4))


def clip(text: str, limit: int) -> str:
    """Shorten at a word boundary: a word cut in half becomes a garbage token in the gold."""
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:")


def wrap(text: str, width: float, size: float, font: str | None) -> list[str]:
    factor = WIDTH_FACTOR.get(Path(font).stem if font else "Arial", 0.5)
    limit = max(10, int(width / (size * factor)))
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > limit:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


# --- Headers and footers: text, image, both, or none ---------------------------


def header(s: Sheet, w: Writer, p: Params, company: str, address: str) -> float:
    W, H = s.width, s.height
    left, top = 0.08 * W, 0.045 * H
    if p.header == "image":
        s.image_text((left, top - 8, 0.92 * W, top + 0.07 * H), [(company, min(20, p.title)), (address, 7.5)])
        return top + 0.09 * H
    if p.header == "none":
        return top
    s.text(left, top + p.title, company, min(20, p.title)).text(left, top + p.title + 14, address, 7.5)
    if p.header == "imagetext":
        initials = "".join(part[0] for part in company.replace("&", "").split()[:3])
        s.image_text((0.76 * W, top - 6, 0.92 * W, top + 0.06 * H), [(initials, 18), (w.pick("cities")[1], 7)])
    return top + p.title + 30


def footer(s: Sheet, w: Writer, p: Params, company: str, n: int = 1, m: int = 1) -> None:
    W, H = s.width, s.height
    L = w.pool["L"]
    lines = [
        f"{company} · {L['register'].format(city=w.pick('cities')[1], hrb=w.rng.randint(1000, 99999))} · "
        f"{L['tax'].format(vat=w.rng.randint(100000000, 999999999))}",
        f"IBAN {w.iban()} · {L['ceo'].format(name=w.pick('people'))}",
    ]
    if p.footer == "none":
        return
    if p.footer in ("image", "imagetext"):
        s.image_text((0.08 * W, 0.935 * H, 0.8 * W, 0.975 * H), [(line, 6.5) for line in lines])
        if p.footer == "imagetext":
            s.text(0.84 * W, 0.965 * H, L["page"].format(n=n, m=m), 7.5)
        return
    s.text(0.08 * W, 0.945 * H, lines[0], 6.5).text(0.08 * W, 0.958 * H, lines[1], 6.5)
    s.text(0.84 * W, 0.975 * H, L["page"].format(n=n, m=m), 7.5)


# --- Templates ----------------------------------------------------------------


def letter(w: Writer, p: Params) -> Sheet:
    s = Sheet(width=p.width, height=p.height, font=p.font)
    W, H = s.width, s.height
    L, left = w.pool["L"], 0.1 * W
    company, (zip_, city) = w.pick("companies"), w.pick("cities")
    address = f"{w.pick('streets')} {w.rng.randint(1, 120)} · {zip_} {city}"
    y = max(header(s, w, p, company, address), 0.17 * H)
    s.text(left, y, f"{company} · {address}", 6.5)
    rz, rc = w.pick("cities")
    for i, line in enumerate([w.pick("people"), f"{w.pick('streets')} {w.rng.randint(1, 90)}", f"{rz} {rc}"]):
        s.text(left, y + 16 + i * (p.body + 3), line, p.body)
    for i, (k, v) in enumerate(
        [
            (L["your"], f"{w.rng.choice('ABCDEF')}{w.rng.choice('KLMNP')}-{w.rng.randint(2024, 2026)}"),
            (L["our"], f"{w.rng.randint(1, 99):02d}/{w.rng.randint(100, 999)}"),
            (L["date"], w.date()),
        ]
    ):
        s.text(0.62 * W, y + 16 + i * (p.body + 1), f"{k}: {v}", p.body - 1.5)
    y += 16 + 3 * (p.body + 3) + 34
    s.text(left, y, w.fill(w.pick("subjects")), p.body + 2)
    y += 2 * p.body + 6
    s.text(left, y, w.pool["salutation"], p.body)
    y += p.body * 1.8
    limit = 0.84 * H
    for sentence in w.rng.sample(w.pool["body"], k=min(5, len(w.pool["body"]))):
        lines = wrap(w.fill(sentence), 0.8 * W, p.body, p.font)
        if y + len(lines) * p.body * 1.45 > limit - 3 * p.body:
            break
        for line in lines:
            s.text(left, y, line, p.body)
            y += p.body * 1.45
        y += p.body * 0.6
    s.text(left, y + p.body, w.pool["closing"], p.body).text(left, y + 3.2 * p.body, w.pick("people"), p.body)
    if w.rng.random() < 0.5:
        s.text(left, 0.9 * H, "¹ " + clip(w.fill(w.rng.choice(w.pool["body"])), 95), 7)
    footer(s, w, p, company)
    return s


def invoice(w: Writer, p: Params) -> Sheet:
    s = Sheet(width=p.width, height=p.height, font=p.font)
    W, H = s.width, s.height
    L, left = w.pool["L"], 0.08 * W
    company, (zip_, city) = w.pick("companies"), w.pick("cities")
    y = max(header(s, w, p, company, f"{w.pick('streets')} {w.rng.randint(1, 120)} · {zip_} {city}"), 0.17 * H)
    rz, rc = w.pick("cities")
    for i, line in enumerate([w.pick("people"), f"{w.pick('streets')} {w.rng.randint(1, 90)}", f"{rz} {rc}"]):
        s.text(left, y + i * (p.body + 3), line, p.body)
    y += 3 * (p.body + 3) + 26
    s.text(left, y, f"{L['invoice']} {L['no']} {w.fill('{inv}')}", p.body + 3).text(
        0.62 * W, y, f"{L['date']}: {w.date()}", p.body
    )
    y += 2.2 * p.body
    xs = [left, left + 0.07 * W, 0.55 * W, 0.66 * W, 0.8 * W]
    rules = [y - p.body - 2]
    for x, cell in zip(xs, L["cols"]):
        s.text(x, y, cell, p.body)
    rules.append(y + 4)
    net = 0.0
    for row in range(w.rng.randint(3, 7)):
        y += p.body * 1.7
        qty, unit = w.rng.randint(1, 16), round(w.rng.uniform(4, 480), 2)
        net += qty * unit
        for x, cell in zip(xs, [str(row + 1), w.pick("items"), str(qty), w.money(unit), w.money(qty * unit)]):
            s.text(x, y, cell, p.body - 0.5)
        if p.rules:
            rules.append(y + 4)
    vat_rate = {"de": 0.19, "en": 0.0825, "tr": 0.20}[w.lang]
    y += 2.2 * p.body
    for label, value in [(L["net"], net), (L["vat"], net * vat_rate), (L["gross"], net * (1 + vat_rate))]:
        s.text(0.55 * W, y, f"{label}: {w.money(value)}", p.body)
        y += p.body * 1.5
    s.text(left, y + p.body, L["due"].format(date=w.date()), p.body)
    s.rules = rules if p.rules else []
    footer(s, w, p, company)
    return s


def presentation(w: Writer, p: Params) -> Sheet:
    s = Sheet(width=p.width, height=p.height, font=p.font)
    W, H = s.width, s.height
    L = w.pool["L"]
    company = w.pick("companies")
    title = w.fill(w.pick("titles"))
    if p.header in ("image", "imagetext"):
        s.image_text((0.05 * W, 0.04 * H, 0.95 * W, 0.17 * H), [(title, min(26, p.title + 4))])
        if p.header == "imagetext":
            s.text(0.06 * W, 0.24 * H, company, 12)
    else:
        s.text(0.06 * W, 0.13 * H, title, min(28, p.title + 6))
    y = 0.32 * H
    for bullet in w.rng.sample(w.pool["bullets"], k=min(5, len(w.pool["bullets"]))):
        s.text(0.08 * W, y, "• " + w.fill(bullet), max(13, p.body + 5))
        y += 0.1 * H
        if y > 0.8 * H:
            break
    foot = f"{company} · {L['confidential']} · {w.date()}"
    if p.footer in ("image", "imagetext"):
        s.image_text((0.04 * W, 0.91 * H, 0.72 * W, 0.97 * H), [(foot, 9)])
    elif p.footer == "text":
        s.text(0.05 * W, 0.955 * H, foot, 9)
    s.text(0.9 * W, 0.955 * H, f"{w.rng.randint(2, 30)} / {w.rng.randint(31, 60)}", 10)
    return s


def work_doc(w: Writer, p: Params) -> Sheet:
    s = Sheet(width=p.width, height=p.height, font=p.font)
    W, H = s.width, s.height
    L, left = w.pool["L"], 0.1 * W
    running = f"{w.fill(w.pick('titles'))} — {L['confidential']}"
    if p.header in ("image", "imagetext"):
        s.image_text((left, 0.03 * H, 0.9 * W, 0.065 * H), [(running, 8)])
        if p.header == "imagetext":
            s.text(0.72 * W, 0.085 * H, f"{L['status']}: {w.date()}", 7.5)
    elif p.header == "text":
        s.text(left, 0.05 * H, running, 8).text(0.72 * W, 0.05 * H, f"{L['status']}: {w.date()}", 8)
    y = 0.12 * H
    s.text(
        left, y, f"{w.rng.randint(1, 9)}.{w.rng.randint(1, 9)} {w.pick('sections')}", p.title if p.title < 18 else 14
    )
    y += 2.5 * p.body
    # Distinct sentences only: real documents do not repeat themselves, and repeated lines
    # would brush against the loop detector for reasons that are the generator's, not the engine's.
    sentences = w.pool["prose"] + w.pool["body"]
    prose = [w.fill(t) for t in w.rng.sample(sentences, k=len(sentences))]
    cols = [(left, 0.8 * W)] if p.columns == 1 else [(left, 0.38 * W), (0.53 * W, 0.37 * W)]
    index = 0  # shared across columns: the right column continues where the left one stopped
    for x, width in cols:
        cy = y
        while index < len(prose):
            marker = "¹" if index == 1 else ("²" if index == 3 else "")
            lines = wrap(prose[index] + marker, width, p.body, p.font)
            if cy + len(lines) * p.body * 1.4 > 0.8 * H:
                break
            for line in lines:
                s.text(x, cy, line, p.body)
                cy += p.body * 1.4
            cy += p.body * 0.7
            index += 1
    s.text(left, 0.86 * H, "¹ " + clip(w.fill(w.rng.choice(w.pool["prose"])), 90), 7)
    s.text(left, 0.875 * H, "² " + clip(w.fill(w.rng.choice(w.pool["body"])), 90), 7)
    if p.footer != "none":
        page = L["page"].format(n=w.rng.randint(2, 30), m=w.rng.randint(31, 80))
        if p.footer in ("image", "imagetext"):
            s.image_text((0.4 * W, 0.94 * H, 0.6 * W, 0.97 * H), [(page, 8)])
        else:
            s.text(0.45 * W, 0.965 * H, page, 8)
    return s


def form(w: Writer, p: Params) -> Sheet:
    """Certificates and forms: the kind that arrives on paper and gets scanned."""
    s = Sheet(width=p.width, height=p.height, font=p.font)
    W, H = s.width, s.height
    L = w.pool["L"]
    company = w.pick("companies")
    y = max(
        header(s, w, p, company, f"{w.pick('streets')} {w.rng.randint(1, 120)} · {' '.join(w.pick('cities'))}"), 0.2 * H
    )
    s.text(0.35 * W, y + 20, L["cert"], p.title + 4)
    y += 70
    values = [
        w.pick("people"),
        w.date(),
        f"{w.pick('streets')} {w.rng.randint(1, 90)}, {' '.join(w.pick('cities'))}",
        w.date(),
    ]
    for label, value in zip(L["fields"], values):
        s.text(0.12 * W, y, f"{label}:", p.body).text(0.4 * W, y, value, p.body)
        y += p.body * 2.2
    for sentence in w.rng.sample(w.pool["body"], k=2):
        for line in wrap(w.fill(sentence), 0.76 * W, p.body, p.font):
            s.text(0.12 * W, y, line, p.body)
            y += p.body * 1.45
    s.image_text((0.6 * W, 0.7 * H, 0.88 * W, 0.78 * H), [(company, 8), (L["stamp"], 7)])
    s.text(0.12 * W, 0.78 * H, "______________________", 10).text(0.12 * W, 0.8 * H, L["stamp"], 7.5)
    footer(s, w, p, company)
    return s


TEMPLATES = {"letter": letter, "invoice": invoice, "presentation": presentation, "work-doc": work_doc, "form": form}
KIND_TEMPLATE = {"letters": "letter", "invoices": "invoice", "presentations": "presentation", "work-docs": "work-doc"}


# --- Planning a set shaped like the profile -----------------------------------


def quotas(counts: dict[str, int], total: int, floor: int = 4) -> dict[str, int]:
    """Proportional to the profile, at least `floor` per kind, summing to `total`."""
    known = {k: counts.get(k, 0) for k in KINDS}
    share = sum(known.values()) or 1
    q = {k: max(floor, round(total * v / share)) for k, v in known.items()}
    while sum(q.values()) > total:
        k = max((k for k in q if q[k] > floor), key=lambda k: q[k], default=None)
        if k is None:
            break
        q[k] -= 1
    while sum(q.values()) < total:
        q[max(known, key=lambda k: known[k])] += 1
    return q


def languages(profiles: list[dict[str, Any]], total: int, rng: random.Random) -> list[str]:
    known = [p["language"] for p in profiles if p.get("language") in POOLS]
    share = (
        {lang: known.count(lang) / max(1, len(known)) for lang in POOLS}
        if known
        else {"de": 0.6, "en": 0.35, "tr": 0.05}
    )
    n_tr, n_de = max(2, round(share["tr"] * total)), max(8, round(share["de"] * total))
    langs = ["tr"] * n_tr + ["de"] * n_de
    langs += ["en"] * max(0, total - len(langs))
    rng.shuffle(langs)
    return langs[:total]


def params_from(profile: dict[str, Any] | None, kind: str, template: str, lang: str, rng: random.Random) -> Params:
    profile = profile or {}
    shape = profile.get("shape", "a4-portrait")
    if template == "presentation" and shape not in ("16:9", "4:3/landscape"):
        shape = "16:9"
    if template != "presentation" and shape in ("16:9", "4:3/landscape", "a4-landscape"):
        shape = "a4-portrait"
    if lang == "en" and shape == "a4-portrait" and rng.random() < 0.6:
        shape = "letter-portrait"
    width, height = SHAPES.get(shape, SHAPES["a4-portrait"])
    header_style = profile.get("header", "text")
    footer_style = profile.get("footer", "text")
    if kind == "scans" or header_style not in ("text", "image", "imagetext", "none"):
        header_style = rng.choice(["text", "text", "imagetext", "image"])
    if footer_style not in ("text", "image", "imagetext", "none"):
        footer_style = "text"
    fonts = [f for f in FONTS if Path(f).exists()] or [None]
    return Params(
        kind=kind,
        template=template,
        lang=lang,
        width=width,
        height=height,
        header=header_style,
        footer=footer_style,
        columns=int(profile.get("columns", 1)),
        rules=profile.get("table_rules", 0) >= 4 or template == "invoice",
        body=min(11.5, max(8.5, float(profile.get("font_median") or 10))),
        title=min(24.0, max(12.0, float(profile.get("font_max") or 16))),
        font=rng.choice(fonts),
    )


# The owner's question is about headers and footers that are images as much as text, so a
# profile where they are rare must not produce a set that barely tests them.
MIN_IMAGE_HEADERS, MIN_IMAGE_FOOTERS, MIN_TWO_COLUMN = 6, 4, 2


def ensure_coverage(born_digital: list[Params], rng: random.Random) -> None:
    def top_up(count, candidates, apply) -> None:
        rng.shuffle(candidates)
        for p in candidates[: max(0, count)]:
            apply(p)

    top_up(
        MIN_IMAGE_HEADERS - sum(p.header in ("image", "imagetext") for p in born_digital),
        [p for p in born_digital if p.header not in ("image", "imagetext")],
        lambda p: setattr(p, "header", rng.choice(["image", "imagetext"])),
    )
    top_up(
        MIN_IMAGE_FOOTERS - sum(p.footer in ("image", "imagetext") for p in born_digital),
        [p for p in born_digital if p.footer not in ("image", "imagetext")],
        lambda p: setattr(p, "footer", rng.choice(["image", "imagetext"])),
    )
    docs = [p for p in born_digital if p.template == "work-doc"]
    top_up(
        MIN_TWO_COLUMN - sum(p.columns >= 2 for p in docs),
        [p for p in docs if p.columns < 2],
        lambda p: setattr(p, "columns", 2),
    )


def image_regions(sheet: Sheet) -> dict[str, list[str]]:
    """Text that exists only in pictures, by the scorer's header/footer rule: a born-digital
    page's layer already holds the rest, so only this goes into its .gold.txt."""
    h = sheet.height
    items = [(rect.y0, rect.y1, value) for rect, lines in sheet.images for value, _ in lines]
    return {
        "header": [v for top, _, v in items if top < HEADER_FRACTION * h],
        "footer": [v for _, bottom, v in items if bottom > (1 - FOOTER_FRACTION) * h],
    }


def generate(out: Path, profile_file: Path | None, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    profiles = json.loads(profile_file.read_text(encoding="utf-8"))["profiles"] if profile_file else []
    counts = {k: sum(1 for p in profiles if p.get("kind") == k) for k in KINDS} if profiles else {k: 1 for k in KINDS}
    plan = quotas(counts, count)
    langs = languages(profiles, count, rng)
    by_kind = {k: [p for p in profiles if p.get("kind") == k] for k in KINDS}
    layout_sources = by_kind["letters"] + by_kind["invoices"]
    specs, index = [], 0
    for kind in KINDS:
        for n in range(plan[kind]):
            lang = langs[index]
            index += 1
            if kind == "scans":
                template = rng.choices(["letter", "invoice", "form"], weights=[4, 3, 3])[0]
                source = rng.choice(layout_sources) if layout_sources else None
            else:
                template = KIND_TEMPLATE[kind]
                source = rng.choice(by_kind[kind]) if by_kind[kind] else None
            specs.append((kind, n, params_from(source, kind, template, lang, rng)))
    ensure_coverage([p for kind, _, p in specs if kind != "scans"], rng)
    manifest, junk_left = [], 2
    for kind, n, p in specs:
        template, lang = p.template, p.lang
        sheet = TEMPLATES[template](Writer(lang, rng), p)
        stem = f"{template}-{n + 1:02d}-{lang}"
        folder = out / kind
        folder.mkdir(parents=True, exist_ok=True)
        entry = {
            "file": f"{kind}/{stem}",
            "template": template,
            "lang": lang,
            "shape": [p.width, p.height],
            "header": p.header,
            "footer": p.footer,
            "columns": p.columns,
            "font": Path(p.font).stem if p.font else None,
        }
        if kind == "scans":
            with tempfile.TemporaryDirectory() as tmp:
                flat = Path(tmp) / "page.pdf"
                build(sheet, flat)
                photo = rng.random() < 0.3
                junk = None
                if not photo and junk_left and rng.random() < 0.4:
                    words = " ".join(regions(sheet)["text"]).split()
                    rng.shuffle(words)
                    junk = " ".join(w[::-1] if rng.random() < 0.3 else w for w in words)
                    junk_left -= 1
                target = folder / (stem + (".jpg" if photo else ".pdf"))
                options = {
                    "dpi": rng.choice([150, 200, 200, 300]),
                    "degrees": round(rng.uniform(-2.0, 2.0), 2),
                    "seed": rng.randint(1, 10**6),
                    "image": photo,
                    "noise": round(rng.uniform(3, 11), 1),
                    "quality": rng.randint(55, 88),
                    "junk_layer": junk,
                }
                as_scan(flat, target, **options)
            gold(folder / f"{stem}.gold.txt", **regions(sheet))
            entry |= {
                "file": str(target.relative_to(out)),
                "scan": {k: v for k, v in options.items() if k != "junk_layer"},
                "old_ocr_layer": bool(junk),
            }
        else:
            target = folder / f"{stem}.pdf"
            build(sheet, target)
            extra = image_regions(sheet)
            if extra["header"] or extra["footer"]:
                gold(folder / f"{stem}.gold.txt", **{k: v for k, v in extra.items() if v})
        manifest.append(entry)
    summary = {"seed": seed, "count": count, "from_profile": bool(profiles), "plan": plan, "documents": manifest}
    (out / "synth.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary
