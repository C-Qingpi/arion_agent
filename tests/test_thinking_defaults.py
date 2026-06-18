"""Unit tests for thinking/reasoning defaults across providers.

Tests _apply_thinking_defaults (resolver) without requiring API keys or
network access.
"""

from __future__ import annotations

from arion_agent.providers.resolver import _apply_thinking_defaults


# ---------------------------------------------------------------------------
# _apply_thinking_defaults
# ---------------------------------------------------------------------------


def test_anthropic_gets_thinking_defaults():
    result = _apply_thinking_defaults("anthropic", "claude-sonnet-4-5", {})
    assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert result["max_tokens"] == 16000


def test_anthropic_preserves_caller_overrides():
    kwargs = {"thinking": {"type": "disabled"}, "max_tokens": 8000}
    result = _apply_thinking_defaults("anthropic", "claude-sonnet-4-5", kwargs)
    assert result["thinking"] == {"type": "disabled"}
    assert result["max_tokens"] == 8000


def test_anthropic_does_not_clobber_other_kwargs():
    kwargs = {"temperature": 1, "api_key": "sk-test"}
    result = _apply_thinking_defaults("anthropic", "claude-haiku-4-5", kwargs)
    assert result["temperature"] == 1
    assert result["api_key"] == "sk-test"
    assert "thinking" in result


def test_gemini_25_gets_thinking_budget():
    result = _apply_thinking_defaults("google_genai", "gemini-2.5-flash", {})
    assert result["thinking_budget"] == 8192
    assert result["include_thoughts"] is True


def test_gemini_25_pro_gets_thinking_budget():
    result = _apply_thinking_defaults("google_genai", "gemini-2.5-pro", {})
    assert result["thinking_budget"] == 8192
    assert result["include_thoughts"] is True


def test_gemini_3_uses_native_default():
    result = _apply_thinking_defaults("google_genai", "gemini-3-flash-preview", {})
    assert result["include_thoughts"] is True
    assert "thinking_level" not in result
    assert "thinking_budget" not in result


def test_gemini_31_uses_native_default():
    result = _apply_thinking_defaults("google_genai", "gemini-3.1-pro-preview", {})
    assert result["include_thoughts"] is True
    assert "thinking_level" not in result


def test_gemini_15_no_thinking():
    result = _apply_thinking_defaults("google_genai", "gemini-1.5-flash", {})
    assert "thinking_budget" not in result
    assert "thinking_level" not in result
    assert "include_thoughts" not in result


def test_gemini_20_no_thinking():
    result = _apply_thinking_defaults("google_genai", "gemini-2.0-flash", {})
    assert "thinking_budget" not in result
    assert "thinking_level" not in result


def test_gemini_preserves_caller_override():
    kwargs = {"thinking_budget": 4096}
    result = _apply_thinking_defaults("google_genai", "gemini-2.5-flash", kwargs)
    assert result["thinking_budget"] == 4096


def test_gemini_3_preserves_caller_thinking_level():
    kwargs = {"thinking_level": "high"}
    result = _apply_thinking_defaults("google_genai", "gemini-3-flash-preview", kwargs)
    assert result["thinking_level"] == "high"


def test_gemini_preserves_include_thoughts_false():
    kwargs = {"include_thoughts": False}
    result = _apply_thinking_defaults("google_genai", "gemini-3-flash-preview", kwargs)
    assert result["include_thoughts"] is False


def test_openai_unchanged():
    kwargs = {"temperature": 0.7}
    result = _apply_thinking_defaults("openai", "gpt-5-mini", kwargs)
    assert result is kwargs
    assert "thinking" not in result


def test_moonshot_unchanged():
    result = _apply_thinking_defaults("moonshot", "kimi-k2.5", {})
    assert "thinking" not in result


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
