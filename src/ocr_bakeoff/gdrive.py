"""Google Drive OCR as the reference text for scans. Opt-in: page images go to your Drive.

Drive runs OCR when an image is uploaded as a Google Doc. It returns plain text with no
positions, so the header and footer references come from separate uploads of the top
and bottom of the page, each cut at the emptiest pixel row near 20% so no line is
sliced. The full page gives the full-text reference.

Scope is drive.file: the tool can see only files it created itself. Each temporary Doc
is moved to the Drive trash after its text is read, never permanently deleted, and every
result is cached so a page is uploaded once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import FOOTER_FRACTION, GDRIVE_TOKEN, HEADER_FRACTION
from .corpus import reference_path
from .render import blank_row_fraction, crop, mime_for

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
GOOGLE_DOC = "application/vnd.google-apps.document"
ENGINE = "google-drive"


def connect(credentials: Path | None) -> Any:
    """A Drive v3 client. First use opens a browser for consent; the token is cached (0600)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(GDRIVE_TOKEN), SCOPES) if GDRIVE_TOKEN.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if credentials is None or not credentials.exists():
            raise SystemExit(
                "Google Drive reference needs an OAuth client the first time: in Google Cloud Console enable the\n"
                "Drive API, create an OAuth client ID of type 'Desktop app', download its JSON, and pass it with\n"
                "--gdrive-credentials PATH. After consent the token is cached and the flag is no longer needed."
            )
        creds = InstalledAppFlow.from_client_secrets_file(str(credentials), SCOPES).run_local_server(port=0)
        GDRIVE_TOKEN.parent.mkdir(parents=True, exist_ok=True)
        GDRIVE_TOKEN.write_text(creds.to_json(), encoding="utf-8")
        os.chmod(GDRIVE_TOKEN, 0o600)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class DriveOCR:
    def __init__(self, service: Any, language: str = "de") -> None:
        self.service, self.language = service, language
        self.uploads = 0

    def ocr(self, image: bytes, mime: str, name: str) -> str:
        from googleapiclient.http import MediaInMemoryUpload

        media = MediaInMemoryUpload(image, mimetype=mime, resumable=False)
        created = (
            self.service.files()
            .create(
                body={"name": name, "mimeType": GOOGLE_DOC},
                media_body=media,
                ocrLanguage=self.language,
                fields="id",
            )
            .execute()
        )
        self.uploads += 1
        try:
            data = self.service.files().export(fileId=created["id"], mimeType="text/plain").execute()
        finally:
            self.service.files().update(fileId=created["id"], body={"trashed": True}).execute()
        text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        return text.lstrip("\ufeff")  # Drive's export starts with a byte-order mark


def reference(ocr: DriveOCR, image: bytes, mime: str, key: str) -> dict[str, Any]:
    top = blank_row_fraction(image, HEADER_FRACTION)
    bottom = blank_row_fraction(image, 1 - FOOTER_FRACTION)
    return {
        "engine": ENGINE,
        "full_text": ocr.ocr(image, mime, f"ocr-bakeoff {key} page"),
        "header_text": ocr.ocr(crop(image, (0.0, 0.0, 1.0, top)), "image/png", f"ocr-bakeoff {key} header"),
        "footer_text": ocr.ocr(crop(image, (0.0, bottom, 1.0, 1.0)), "image/png", f"ocr-bakeoff {key} footer"),
        "cuts": [round(top, 4), round(bottom, 4)],
    }


def ensure_reference(ocr: DriveOCR, workdir: Path, image_key: str, image_path: Path) -> bool:
    """Fetch and cache one image's reference. Returns True if it uploaded anything."""
    out = reference_path(workdir, image_key)
    if out.exists():
        return False
    data = reference(ocr, image_path.read_bytes(), mime_for(image_path), image_key.split(".")[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return True
