"""Cloud storage sync — off-site redundancy via Google Drive (Module 1).

Real ``google-api-python-client`` upload, gated behind service-account
credentials + a destination folder id. When unconfigured, ``push`` records the
intent and returns ``False`` (no-op) instead of failing the pipeline.

Note on "end-to-end encrypted": Drive encrypts in transit/at rest, but for true
client-side E2EE, encrypt files with :mod:`base44.storage.backup` (Fernet)
BEFORE calling ``push`` so only ciphertext leaves the machine.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings

log = logging.getLogger("base44.storage.cloud_sync")


class CloudSync:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.gdrive_enabled

    def push(self, path: Path) -> bool:
        """Upload a file to the configured Drive folder. Returns success flag."""
        if not self.enabled:
            log.info("cloud sync disabled (no Drive credentials); skipping %s", path)
            return False
        try:  # pragma: no cover - requires network + credentials
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            creds = service_account.Credentials.from_service_account_file(
                str(self.settings.gdrive_credentials_json),
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
            service = build("drive", "v3", credentials=creds)
            metadata = {"name": path.name, "parents": [self.settings.gdrive_folder_id]}
            media = MediaFileUpload(str(path), resumable=True)
            service.files().create(body=metadata, media_body=media, fields="id").execute()
            log.info("uploaded %s to Drive folder %s", path.name, self.settings.gdrive_folder_id)
            return True
        except Exception as exc:  # pragma: no cover
            log.error("Drive upload failed for %s: %s", path, exc)
            return False
