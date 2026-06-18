"""Build arion_agent/dist/arion-agent-VERSION-source-snapshot.zip for hand-off.

Includes only: pyproject.toml, README.md, the arion_agent/ package tree.
Excludes: .venv, tests, to_be_deleted, __pycache__, *.pyc, .egg-info, lockfiles.
Run from repo root: python scripts/make_source_snapshot.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise RuntimeError("Could not read version from pyproject.toml")
    return m.group(1)


def _skip_file(path: Path, root_pkg: Path) -> bool:
    if "__pycache__" in path.parts:
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    if path.name in {".DS_Store"}:
        return True
    try:
        rel = path.relative_to(root_pkg)
    except ValueError:
        return True
    if rel.parts and rel.parts[0] == "to_be_deleted":
        return True
    return False


def main() -> int:
    version = _read_version()
    out = ROOT / "dist" / f"arion-agent-{version}-source-snapshot.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    prefix = f"arion-agent-{version}"
    root_pkg = ROOT / "arion_agent"
    for path in (ROOT / "pyproject.toml", ROOT / "README.md"):
        if not path.is_file():
            print(f"Missing {path}", file=sys.stderr)
            return 1
    if not root_pkg.is_dir():
        print(f"Missing package dir {root_pkg}", file=sys.stderr)
        return 1

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (ROOT / "pyproject.toml", ROOT / "README.md"):
            zf.write(f, arcname=f"{prefix}/{f.name}")
            n += 1
        for f in sorted(root_pkg.rglob("*")):
            if f.is_file() and not _skip_file(f, root_pkg):
                rel = f.relative_to(root_pkg)
                zf.write(f, arcname=f"{prefix}/arion_agent/{rel.as_posix()}")
                n += 1

    print(f"Wrote {out} ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
