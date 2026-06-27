"""Translate Chinese text to English for embedding/search only."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from arion_agent.semantic_search.config import CJK_TRANSLATE_THRESHOLD, MT_MODEL, TRANSFORMERS_CACHE_DIR, HF_HOME_DIR

if TYPE_CHECKING:
    from transformers import MarianMTModel, MarianTokenizer

_cjk_re = re.compile(r"[\u4e00-\u9fff]")
_token_re = re.compile(
    r"`[^`]+`"
    r"|[A-Za-z_][\w.-]*\.(?:py|ts|tsx|js|jsx|md|json|jsonl|yaml|yml|toml|sqlite|sql|ps1|sh)"
    r"|\b[a-z_][a-z0-9_]{2,}\([^)]*\)"
)
_prot_re = re.compile(r"@@PROT(\d+)@@", re.IGNORECASE)
_MASK_SKIP_LEN = 120
_MT_MAX_CHARS = 420

_model: MarianMTModel | None = None
_tokenizer: MarianTokenizer | None = None


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(_cjk_re.findall(text))
    return cjk / max(len(text), 1)


def needs_translation(text: str) -> bool:
    return cjk_ratio(text) >= CJK_TRANSLATE_THRESHOLD


def _load_mt() -> tuple[MarianMTModel, MarianTokenizer]:
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        import os
        os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)
        # Env vars already set by config import; ensure dirs exist
        os.makedirs(HF_HOME_DIR, exist_ok=True)

        from transformers import MarianMTModel, MarianTokenizer

        _tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
        _model = MarianMTModel.from_pretrained(MT_MODEL)
    return _model, _tokenizer


def _mask_protected(text: str) -> tuple[str, list[str]]:
    if len(text) <= _MASK_SKIP_LEN:
        return text, []

    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"@@PROT{len(protected) - 1}@@"

    masked = _token_re.sub(repl, text)
    return masked, protected


def _unmask_protected(text: str, protected: list[str]) -> str:
    out = text

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return protected[idx]

    out = _prot_re.sub(repl, out)
    for i, original in enumerate(protected):
        out = out.replace(f"@@PROT{i}@@", original)
    return out


def _translate_one(masked: str, protected: list[str]) -> str:
    model, tokenizer = _load_mt()
    batch = tokenizer(
        [masked],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    output = model.generate(**batch, max_length=512)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    return _unmask_protected(translated, protected).strip()


def translate_zh_to_en(text: str) -> str:
    masked, protected = _mask_protected(text)
    if len(masked) <= _MT_MAX_CHARS:
        return _translate_one(masked, protected)

    # Split on paragraph boundaries (double newlines or more)
    parts = re.split(r"(\n\n+)", text)
    # Guard: if the split didn't actually break the text apart, fall back to
    # character-window chunking so we always make progress.
    if len(parts) == 1:
        out: list[str] = []
        for start in range(0, len(text), _MT_MAX_CHARS):
            chunk = text[start : start + _MT_MAX_CHARS]
            if needs_translation(chunk):
                masked_chunk, prot = _mask_protected(chunk)
                out.append(_translate_one(masked_chunk, prot))
            else:
                out.append(chunk)
        return "".join(out)

    out: list[str] = []
    for part in parts:
        if not part or re.fullmatch(r"\n+", part):
            out.append(part)
            continue
        if not needs_translation(part):
            out.append(part)
            continue
        out.append(translate_zh_to_en(part))
    return "".join(out)


def warmup_mt() -> None:
    _load_mt()


def prepare_search_text(text: str) -> str:
    if not needs_translation(text):
        return text
    return translate_zh_to_en(text)
