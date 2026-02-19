"""
File storage abstraction for document assets.
Default implementation uses local filesystem; interface allows cloud backends later.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol


class FileStorageProvider(Protocol):
    @property
    def root(self) -> Path:
        ...

    def ensure_ready(self):
        ...

    def save_file(self, source_path: Path, destination_name: str) -> Path:
        ...


class LocalFileStorageProvider:
    def __init__(self, root: Path):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def ensure_ready(self):
        self._root.mkdir(parents=True, exist_ok=True)

    def save_file(self, source_path: Path, destination_name: str) -> Path:
        self.ensure_ready()
        destination = self._root / str(destination_name)
        shutil.copy2(source_path, destination)
        return destination
