"""Verify the supplied release without modifying its scientific files."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ProtectedFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ResearchLock(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    lock_version: str
    protected_files: tuple[ProtectedFile, ...]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_release(root: Path) -> dict[str, str]:
    root = root.resolve()
    lock = ResearchLock.model_validate_json((root / "RESEARCH_LOCK.json").read_text())
    if lock.lock_version != "2.2-locked":
        raise ValueError(f"Unsupported research lock {lock.lock_version}")
    verified = {}
    for item in lock.protected_files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Protected path escapes release: {item.path}")
        digest = sha256_file(path)
        if digest != item.sha256:
            raise ValueError(f"Research lock mismatch: {item.path}")
        verified[item.path] = digest
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Package manifest mismatch: {item['path']}")
    return verified
