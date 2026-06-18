"""I/O backend abstraction for ArionAgent workspace file operations.

Provides a transport layer between middleware and the filesystem. All
middleware file I/O routes through persistence.py, which routes through
the active IOBackend. LocalIOBackend uses direct Path operations.
RemoteIOBackend (in remote_io.py) uses HTTP to a host-side service.

This is NOT a new environment. It is infrastructure that all middleware
shares: identity, skills, heartbeat, signals, summarization, file tools,
shell, subagenting -- everything that touches the filesystem.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileStat:
    size: int
    mtime: float
    is_dir: bool


@dataclass(frozen=True)
class DirEntry:
    name: str
    is_dir: bool
    size: int
    mtime: float


class IOBackend(ABC):
    """Abstract base for workspace file I/O.

    All paths are strings relative to the workspace root. The backend
    resolves them internally. Sync interface -- file I/O is fast and
    infrequent relative to LLM calls.
    """

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def append_text(self, path: str, text: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def append_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def is_dir(self, path: str) -> bool: ...

    @abstractmethod
    def is_file(self, path: str) -> bool: ...

    @abstractmethod
    def stat(self, path: str) -> FileStat: ...

    @abstractmethod
    def list_dir(self, path: str) -> list[DirEntry]: ...

    @abstractmethod
    def mkdir(self, path: str, parents: bool = True) -> None: ...

    @abstractmethod
    def delete(self, path: str) -> None: ...

    @abstractmethod
    def delete_tree(self, path: str) -> None: ...

    @abstractmethod
    def move(self, src: str, dst: str) -> None: ...

    @abstractmethod
    def glob(self, path: str, pattern: str) -> list[str]: ...

    @abstractmethod
    def walk(self, path: str) -> Iterator[tuple[str, list[str], list[str]]]: ...

    def close(self) -> None:
        """Release resources. No-op by default."""


class LocalIOBackend(IOBackend):
    """Direct filesystem I/O. Default backend when not using remote service."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve(self, path: str) -> Path:
        return self.root / path

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._resolve(path).read_text(encoding=encoding)

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding=encoding)
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def append_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding=encoding) as f:
            f.write(text)

    def append_bytes(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "ab") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def is_dir(self, path: str) -> bool:
        return self._resolve(path).is_dir()

    def is_file(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def stat(self, path: str) -> FileStat:
        s = self._resolve(path).stat()
        return FileStat(size=s.st_size, mtime=s.st_mtime, is_dir=self._resolve(path).is_dir())

    def list_dir(self, path: str) -> list[DirEntry]:
        resolved = self._resolve(path)
        entries = []
        for item in sorted(resolved.iterdir()):
            try:
                s = item.stat()
                entries.append(DirEntry(
                    name=item.name,
                    is_dir=item.is_dir(),
                    size=s.st_size,
                    mtime=s.st_mtime,
                ))
            except OSError:
                entries.append(DirEntry(name=item.name, is_dir=item.is_dir(), size=0, mtime=0))
        return entries

    def mkdir(self, path: str, parents: bool = True) -> None:
        self._resolve(path).mkdir(parents=parents, exist_ok=True)

    def delete(self, path: str) -> None:
        self._resolve(path).unlink()

    def delete_tree(self, path: str) -> None:
        shutil.rmtree(self._resolve(path))

    def move(self, src: str, dst: str) -> None:
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._resolve(src)), str(dst_path))

    def glob(self, path: str, pattern: str) -> list[str]:
        base = self._resolve(path)
        results = []
        for match in base.rglob(pattern):
            try:
                rel = match.relative_to(self.root)
                results.append(str(rel).replace("\\", "/"))
            except ValueError:
                results.append(str(match))
        return results

    def walk(self, path: str) -> Iterator[tuple[str, list[str], list[str]]]:
        base = self._resolve(path)
        for dirpath, dirnames, filenames in os.walk(base):
            try:
                rel = str(Path(dirpath).relative_to(self.root)).replace("\\", "/")
            except ValueError:
                rel = dirpath
            yield rel, sorted(dirnames), sorted(filenames)
