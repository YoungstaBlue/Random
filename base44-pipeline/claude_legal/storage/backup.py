"""Secure offline backups (Module 1).

Writes encrypted copies of case data to a local backup directory for data
privacy and integrity. Uses Fernet (AES-128-CBC + HMAC) from ``cryptography``
when a key is configured; the same key enables client-side E2EE before cloud
sync. Without a key it degrades to plaintext copies with a logged warning.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings

log = logging.getLogger("claude_legal.storage.backup")


class OfflineBackup:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.backup_dir = Path(self.settings.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def encrypted(self) -> bool:
        return self.settings.backup_enabled

    def backup_bytes(self, name: str, data: bytes) -> Path:
        target = self.backup_dir / name
        if self.encrypted:
            data = self._encrypt(data)
            target = target.with_suffix(target.suffix + ".enc")
        else:
            log.warning("CLAUDE_LEGAL_BACKUP_KEY unset; writing PLAINTEXT backup %s", target)
        target.write_bytes(data)
        return target

    def backup_file(self, path: Path) -> Path:
        p = Path(path)
        if self.encrypted:
            return self.backup_bytes(p.name, p.read_bytes())
        target = self.backup_dir / p.name
        shutil.copy2(p, target)
        return target

    def restore(self, name: str) -> bytes:
        target = self.backup_dir / name
        data = target.read_bytes()
        if target.suffix == ".enc" or self.encrypted:
            data = self._decrypt(data)
        return data

    def _encrypt(self, data: bytes) -> bytes:  # pragma: no cover - optional dependency
        from cryptography.fernet import Fernet

        return Fernet(self.settings.backup_key.encode()).encrypt(data)

    def _decrypt(self, data: bytes) -> bytes:  # pragma: no cover - optional dependency
        from cryptography.fernet import Fernet

        return Fernet(self.settings.backup_key.encode()).decrypt(data)
