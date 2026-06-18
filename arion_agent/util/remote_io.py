"""RemoteIOBackend: HTTP-based I/O backend for Docker deployments.

Routes all file operations through HTTP to a host-side service, avoiding
grpcfuse bind mount issues with long-lived file handles (Phase 17+).
The service implements the /io/* endpoint contract.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx

from arion_agent.util.io_backend import DirEntry, FileStat, IOBackend


class RemoteIOBackend(IOBackend):
    """I/O backend that delegates to a remote HTTP service."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 404:
            detail = resp.json().get("detail", "File not found") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            raise FileNotFoundError(detail)
        if resp.status_code == 400:
            detail = resp.json().get("detail", "Bad request") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            raise ValueError(detail)
        if resp.status_code >= 500:
            raise OSError(f"Remote I/O service error ({resp.status_code}): {resp.text}")

    def read_bytes(self, path: str) -> bytes:
        resp = self._client.get("/io/read", params={"path": path, "mode": "bytes"})
        self._raise_for_status(resp)
        data = resp.json()
        return base64.b64decode(data["data"])

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        resp = self._client.get("/io/read", params={"path": path, "mode": "text"})
        self._raise_for_status(resp)
        return resp.json()["text"]

    def write_bytes(self, path: str, data: bytes) -> None:
        resp = self._client.post("/io/write", json={
            "path": path,
            "data": base64.b64encode(data).decode("ascii"),
        })
        self._raise_for_status(resp)

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        resp = self._client.post("/io/write", json={"path": path, "text": text})
        self._raise_for_status(resp)

    def append_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        resp = self._client.post("/io/append", json={"path": path, "text": text})
        self._raise_for_status(resp)

    def append_bytes(self, path: str, data: bytes) -> None:
        resp = self._client.post("/io/append", json={
            "path": path,
            "data": base64.b64encode(data).decode("ascii"),
        })
        self._raise_for_status(resp)

    def exists(self, path: str) -> bool:
        resp = self._client.get("/io/exists", params={"path": path})
        self._raise_for_status(resp)
        return resp.json()["exists"]

    def is_dir(self, path: str) -> bool:
        resp = self._client.get("/io/stat", params={"path": path})
        if resp.status_code == 404:
            return False
        self._raise_for_status(resp)
        return resp.json()["is_dir"]

    def is_file(self, path: str) -> bool:
        resp = self._client.get("/io/stat", params={"path": path})
        if resp.status_code == 404:
            return False
        self._raise_for_status(resp)
        return not resp.json()["is_dir"]

    def stat(self, path: str) -> FileStat:
        resp = self._client.get("/io/stat", params={"path": path})
        self._raise_for_status(resp)
        d = resp.json()
        return FileStat(size=d["size"], mtime=d["mtime"], is_dir=d["is_dir"])

    def list_dir(self, path: str) -> list[DirEntry]:
        resp = self._client.get("/io/list", params={"path": path})
        self._raise_for_status(resp)
        entries = []
        for item in resp.json()["entries"]:
            entries.append(DirEntry(
                name=item["name"],
                is_dir=item["is_dir"],
                size=item["size"],
                mtime=item["mtime"],
            ))
        return entries

    def mkdir(self, path: str, parents: bool = True) -> None:
        resp = self._client.post("/io/mkdir", json={"path": path, "parents": parents})
        self._raise_for_status(resp)

    def delete(self, path: str) -> None:
        resp = self._client.post("/io/delete", json={"path": path})
        self._raise_for_status(resp)

    def delete_tree(self, path: str) -> None:
        resp = self._client.post("/io/delete", json={"path": path, "tree": True})
        self._raise_for_status(resp)

    def move(self, src: str, dst: str) -> None:
        resp = self._client.post("/io/move", json={"src": src, "dst": dst})
        self._raise_for_status(resp)

    def glob(self, path: str, pattern: str) -> list[str]:
        resp = self._client.get("/io/glob", params={"path": path, "pattern": pattern})
        self._raise_for_status(resp)
        return resp.json()["matches"]

    def walk(self, path: str) -> Iterator[tuple[str, list[str], list[str]]]:
        resp = self._client.get("/io/walk", params={"path": path})
        self._raise_for_status(resp)
        for entry in resp.json()["entries"]:
            yield entry["dirpath"], entry["dirnames"], entry["filenames"]

    def close(self) -> None:
        self._client.close()
