"""Checkpointers save the run after every step so it can be resumed later."""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from typing import Any, Dict, Optional


class MemoryCheckpointer:
    """Keeps checkpoints in a dict (tests, notebooks, one process)."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self.history: Dict[str, list] = {}

    def put(self, thread_id: str, checkpoint: Dict[str, Any]) -> None:
        cp = copy.deepcopy(checkpoint)
        self._data[thread_id] = cp
        self.history.setdefault(thread_id, []).append(cp)

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        cp = self._data.get(thread_id)
        return copy.deepcopy(cp) if cp is not None else None


class JsonFileCheckpointer:
    """One JSON file per thread. Writes are atomic (temp file + rename), so a crash in the
    middle of a save never leaves a broken checkpoint. State must be JSON-serialisable."""

    _safe = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, thread_id: str) -> str:
        if not self._safe.match(thread_id) or thread_id in (".", ".."):
            raise ValueError(f"unsafe thread_id: {thread_id!r}")
        return os.path.join(self.directory, f"{thread_id}.json")

    def put(self, thread_id: str, checkpoint: Dict[str, Any]) -> None:
        path = self._path(thread_id)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(thread_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
