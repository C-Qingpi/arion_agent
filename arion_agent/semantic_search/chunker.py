from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from arion_agent.semantic_search.config import (
    CHUNK_OVERLAP_LINES,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
    WHOLE_FILE_MAX_CHARS,
    WHOLE_FILE_MAX_LINES,
    Chunk,
)


def chunk_file(path: Path, workspace: Path) -> list[Chunk]:
    rel = path.relative_to(workspace).as_posix()
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".py":
        chunks = _chunk_python(text, rel)
    elif suffix in {".md", ".markdown"}:
        chunks = _chunk_markdown(text, rel)
    elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
        chunks = _chunk_js_like(text, rel)
    else:
        chunks = _chunk_plain(text, rel, kind="text")

    if not chunks:
        chunks = _chunk_plain(text, rel, kind="fallback")
    return chunks


def _chunk_python(text: str, rel: str) -> list[Chunk]:
    lines = text.splitlines()
    if len(lines) <= WHOLE_FILE_MAX_LINES and len(text) <= WHOLE_FILE_MAX_CHARS:
        return [_make_chunk(rel, 1, len(lines) or 1, text, "python-whole")]

    tree = ast.parse(text)
    spans: list[tuple[int, int, str]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            kind = "python-class" if isinstance(node, ast.ClassDef) else "python-function"
            spans.append((start, end, kind))

    if not spans:
        return _chunk_plain(text, rel, kind="python")

    spans.sort()
    merged = _merge_small_spans(lines, spans)
    return [
        _make_chunk(
            rel,
            start,
            end,
            "\n".join(lines[start - 1 : end]),
            kind,
        )
        for start, end, kind in merged
    ]


def _chunk_markdown(text: str, rel: str) -> list[Chunk]:
    lines = text.splitlines()
    heading_re = re.compile(r"^(#{1,3})\s+")
    heading_lines = [i for i, line in enumerate(lines, start=1) if heading_re.match(line)]

    if len(heading_lines) >= 2:
        return _chunk_markdown_sections(lines, rel, heading_re)

    if len(lines) <= WHOLE_FILE_MAX_LINES and len(text) <= WHOLE_FILE_MAX_CHARS:
        return [_make_chunk(rel, 1, len(lines) or 1, text, "markdown-whole")]

    return _chunk_markdown_sections(lines, rel, heading_re)


def _chunk_markdown_sections(
    lines: list[str],
    rel: str,
    heading_re: re.Pattern[str],
) -> list[Chunk]:
    sections: list[tuple[int, int]] = []
    starts = [1]
    for i, line in enumerate(lines, start=1):
        if heading_re.match(line) and i != 1:
            starts.append(i)
    starts.append(len(lines) + 1)

    for idx in range(len(starts) - 1):
        start = starts[idx]
        end = starts[idx + 1] - 1
        if end >= start:
            sections.append((start, end))

    chunks: list[Chunk] = []
    buf_start = sections[0][0]
    buf_lines: list[str] = []

    for start, end in sections:
        section_lines = lines[start - 1 : end]
        section_text = "\n".join(section_lines)
        if not buf_lines:
            buf_start = start
            buf_lines = section_lines
        elif len("\n".join(buf_lines)) + len(section_text) + 1 <= TARGET_CHUNK_CHARS:
            buf_lines.extend([""] + section_lines)
        else:
            chunks.append(
                _make_chunk(
                    rel,
                    buf_start,
                    buf_start + len(buf_lines) - 1,
                    "\n".join(buf_lines),
                    "markdown-section",
                )
            )
            buf_start = start
            buf_lines = section_lines

    if buf_lines:
        chunks.append(
            _make_chunk(
                rel,
                buf_start,
                buf_start + len(buf_lines) - 1,
                "\n".join(buf_lines),
                "markdown-section",
            )
        )
    return _split_oversized(chunks)


def _chunk_js_like(text: str, rel: str) -> list[Chunk]:
    lines = text.splitlines()
    if len(lines) <= WHOLE_FILE_MAX_LINES and len(text) <= WHOLE_FILE_MAX_CHARS:
        return [_make_chunk(rel, 1, len(lines) or 1, text, "js-whole")]

    boundary = re.compile(
        r"^\s*(export\s+)?(async\s+)?(function|class|interface|type|const|let)\s+"
    )
    starts = [1]
    for i, line in enumerate(lines, start=1):
        if i > 1 and boundary.match(line):
            starts.append(i)
    starts.append(len(lines) + 1)

    spans: list[tuple[int, int, str]] = []
    for idx in range(len(starts) - 1):
        start = starts[idx]
        end = starts[idx + 1] - 1
        if end >= start:
            spans.append((start, end, "js-block"))

    merged = _merge_small_spans(lines, spans)
    chunks = [
        _make_chunk(rel, s, e, "\n".join(lines[s - 1 : e]), k)
        for s, e, k in merged
    ]
    return _split_oversized(chunks)


def _chunk_plain(text: str, rel: str, kind: str, *, allow_resplit: bool = True) -> list[Chunk]:
    lines = text.splitlines()
    if len(text) <= WHOLE_FILE_MAX_CHARS:
        return [_make_chunk(rel, 1, len(lines) or 1, text, kind)]

    chunks: list[Chunk] = []
    start = 1
    buf: list[str] = []
    buf_chars = 0

    for i, line in enumerate(lines, start=1):
        if len(line) > MAX_CHUNK_CHARS:
            if buf:
                end = i - 1
                chunks.append(_make_chunk(rel, start, end, "\n".join(buf), kind))
                buf = []
                buf_chars = 0
            chunks.extend(_hard_split_line(rel, i, line, kind))
            start = i + 1
            continue

        add = len(line) + (1 if buf else 0)
        if buf and buf_chars + add > TARGET_CHUNK_CHARS:
            end = i - 1
            chunks.append(_make_chunk(rel, start, end, "\n".join(buf), kind))
            overlap = buf[-CHUNK_OVERLAP_LINES:] if len(buf) > CHUNK_OVERLAP_LINES else buf
            start = end - len(overlap) + 1
            buf = overlap + [line]
            buf_chars = sum(len(x) + 1 for x in buf)
        else:
            buf.append(line)
            buf_chars += add

    if buf:
        chunks.append(_make_chunk(rel, start, start + len(buf) - 1, "\n".join(buf), kind))

    if allow_resplit:
        return _split_oversized(chunks)
    return chunks


def _hard_split_line(path: str, line_no: int, line: str, kind: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = TARGET_CHUNK_CHARS
    for offset in range(0, len(line), step):
        piece = line[offset : offset + step]
        chunks.append(_make_chunk(path, line_no, line_no, piece, kind + "-line"))
    return chunks


def _merge_small_spans(
    lines: list[str],
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    if not spans:
        return spans
    merged: list[tuple[int, int, str]] = []
    cur_start, cur_end, cur_kind = spans[0]
    cur_text = "\n".join(lines[cur_start - 1 : cur_end])

    for start, end, kind in spans[1:]:
        piece = "\n".join(lines[start - 1 : end])
        if len(cur_text) < MIN_CHUNK_CHARS and len(cur_text) + len(piece) + 1 <= MAX_CHUNK_CHARS:
            cur_end = end
            cur_text = cur_text + "\n\n" + piece
            if cur_kind != kind:
                cur_kind = "merged"
        else:
            merged.append((cur_start, cur_end, cur_kind))
            cur_start, cur_end, cur_kind = start, end, kind
            cur_text = piece
    merged.append((cur_start, cur_end, cur_kind))
    return merged


def _split_oversized(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= MAX_CHUNK_CHARS:
            out.append(chunk)
            continue
        out.extend(_chunk_plain(chunk.text, chunk.path, chunk.kind + "-split", allow_resplit=False))
    return out


def _make_chunk(path: str, start_line: int, end_line: int, text: str, kind: str) -> Chunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Chunk(
        path=path,
        start_line=start_line,
        end_line=end_line,
        text=text,
        search_text=text,
        kind=kind,
        content_hash=digest,
    )
