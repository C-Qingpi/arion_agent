"""Model resolution: convert model specs to BaseChatModel instances.

ArionAgent is service-provider agnostic. The provider string determines which
LangChain adapter and wire format to use. All connection configs (base_url,
api_key, temperature, etc.) are forwarded and translated as needed.

Supported model spec formats:
  - BaseChatModel instance: used directly
  - "provider:model_id" string: routed to LangChain's init_chat_model
  - Plain model name: auto-detected by model name pattern

Proxy providers:
  Third-party proxy services are registered via register_proxy() with a
  ProxySpec. The resolver has no hardcoded knowledge of any specific proxy.
  Two styles are supported:
    - "openai": single OpenAI-compatible endpoint for all models
    - "native": separate sub-endpoints per upstream provider format

  Registration example (done in the deployment layer, not here):
    register_proxy("myproxy", ProxySpec(
        api_key_env="MYPROXY_API_KEY",
        base_url_env="MYPROXY_API_BASE",
        style="openai",
    ))

Native provider adapters:
  openai, anthropic, google_genai are handled by LangChain adapters with
  kwarg translation (base_url -> anthropic_api_url, etc.).

Thinking / reasoning defaults:
  Providers with reasoning capabilities have thinking enabled by default.
    - Anthropic (4.6+): thinking={"type": "adaptive"}
      Anthropic (older): thinking={"type": "enabled", "budget_tokens": 10000}
    - Google GenAI (Gemini 3+): include_thoughts=True (thinking_level left
      at the model's native default of "high"/dynamic)
      Google GenAI (Gemini 2.5): thinking_budget=8192, include_thoughts=True
    - OpenAI: reasoning tokens are internal
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel

_CLAUDE_PATTERN = re.compile(r"^claude-", re.IGNORECASE)
_GEMINI_PATTERN = re.compile(r"^gemini-", re.IGNORECASE)
_GPT_PATTERN = re.compile(r"^(gpt-|o[1-9]|chatgpt)", re.IGNORECASE)
_DEEPSEEK_PATTERN = re.compile(
    r"^(deepseek[-_]|deepseek$)", re.IGNORECASE,
)
_GEMINI_VERSION = re.compile(r"^gemini-(\d+)(?:\.(\d+))?", re.IGNORECASE)
_CLAUDE_VER_PAIR = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)")


# ---------------------------------------------------------------------------
# Proxy registry
# ---------------------------------------------------------------------------

ProxyResolver = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
"""Signature for a custom proxy resolver function.

Args:
    model_id: the model name after the provider prefix (e.g. "claude-opus-4-6")
    kwargs: the current kwargs dict (may contain api_key, base_url, etc.)

Returns:
    (real_provider, adapted_kwargs) where real_provider is a LangChain adapter
    name ("openai", "anthropic", "google_genai") and adapted_kwargs has the
    provider-specific params ready for init_chat_model.
"""


@dataclass
class ProxySpec:
    """Configuration for a proxy provider.

    For common patterns, use style + declarative fields:

    style="openai": all models go through a single OpenAI-compatible endpoint.
        base_url is used as-is (e.g. http://proxy:3001/v1).

    style="native": separate sub-endpoints per upstream provider format.
        base_url is the root, sub_endpoints maps native providers to suffixes.

    For anything more complex, provide a custom resolve function. When set,
    style/sub_endpoints are ignored and the function has full control over
    how (provider, kwargs) are resolved.

    resolve signature: (model_id: str, kwargs: dict) -> (real_provider, adapted_kwargs)

    model_adapter: optional chat model class (e.g. ChatMoonshot) to use instead of
        init_chat_model when this proxy is resolved. Lets deploy layer plug in
        provider-specific adapters.
    """
    api_key_env: str = ""
    base_url_env: str = ""
    default_base_url: str = ""
    style: Literal["openai", "native"] = "openai"
    sub_endpoints: dict[str, str] = field(default_factory=dict)
    resolve: ProxyResolver | None = None
    model_adapter: type[BaseChatModel] | None = None


_proxy_registry: dict[str, ProxySpec] = {}


def register_proxy(name: str, spec: ProxySpec) -> None:
    """Register a proxy provider. Call before resolve_model()."""
    _proxy_registry[name] = spec


def unregister_proxy(name: str) -> None:
    """Remove a registered proxy provider."""
    _proxy_registry.pop(name, None)


def get_registered_proxies() -> dict[str, ProxySpec]:
    """Return a copy of the proxy registry."""
    return dict(_proxy_registry)


def _ensure_builtin_providers() -> None:
    """Register built-in OpenAI-compatible providers (idempotent)."""
    if "deepseek" in _proxy_registry:
        return
    from arion_agent.providers.deepseek import ChatDeepSeek

    register_proxy("deepseek", ProxySpec(
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_API_BASE",
        default_base_url="https://api.deepseek.com",
        style="openai",
        model_adapter=ChatDeepSeek,
    ))


_ensure_builtin_providers()


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

def _infer_provider(model_id: str) -> str:
    """Infer LangChain provider from model name when no explicit provider given."""
    if _CLAUDE_PATTERN.match(model_id):
        return "anthropic"
    if _GEMINI_PATTERN.match(model_id):
        return "google_genai"
    if _DEEPSEEK_PATTERN.match(model_id):
        return "deepseek"
    if _GPT_PATTERN.match(model_id):
        return "openai"
    return "openai"


def _normalize_deepseek_model_id(model_id: str) -> str:
    """Map deploy-friendly ids (underscores) to DeepSeek API ids (hyphens)."""
    normalized = model_id.replace("_", "-")
    if normalized.lower() == "deepseek":
        return "deepseek-v4-flash"
    return normalized


# ---------------------------------------------------------------------------
# Kwarg resolution
# ---------------------------------------------------------------------------

def _resolve_proxy(
    spec: ProxySpec, model_id: str, kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve a registered proxy into the real provider + adapted kwargs."""
    if spec.resolve is not None:
        return spec.resolve(model_id, kwargs)

    base = os.environ.get(spec.base_url_env, spec.default_base_url).rstrip("/")
    key = kwargs.get("api_key") or os.environ.get(spec.api_key_env, "")
    adapted = dict(kwargs)
    adapted["api_key"] = key

    if spec.style == "openai":
        adapted["base_url"] = base
        return "openai", adapted

    # style == "native": infer sub-provider, append suffix, delegate
    native = _infer_provider(model_id)
    suffix = spec.sub_endpoints.get(native, "/v1")
    adapted["base_url"] = base + suffix
    return _resolve_native_kwargs(native, adapted)


def _resolve_native_kwargs(
    provider: str, kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Translate generic kwargs to provider-specific params for native adapters.

    anthropic: base_url -> anthropic_api_url, api_key -> anthropic_api_key
    google_genai: base_url -> client_options, api_key -> google_api_key
    """
    if provider == "anthropic" and kwargs:
        adapted = dict(kwargs)
        if "base_url" in adapted:
            adapted["anthropic_api_url"] = adapted.pop("base_url")
        if "api_key" in adapted:
            adapted["anthropic_api_key"] = adapted.pop("api_key")
        return provider, adapted

    if provider == "google_genai":
        adapted = dict(kwargs)
        if "base_url" in adapted:
            base_url = adapted.pop("base_url")
            adapted.setdefault("client_options", {"api_endpoint": base_url})
        elif "client_options" not in adapted:
            env_base = os.environ.get("GOOGLE_API_BASE", "")
            if env_base:
                adapted["client_options"] = {"api_endpoint": env_base}
        if "api_key" in adapted:
            adapted["google_api_key"] = adapted.pop("api_key")
        return provider, adapted

    return provider, kwargs


def _resolve_provider_kwargs(
    provider: str, model_id: str, kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any], ProxySpec | None]:
    """Resolve logical provider to LangChain adapter and adapt kwargs.

    Returns (real_provider, adapted_kwargs, proxy_spec_or_none).
    When proxy_spec has model_adapter, resolve_model uses it instead of init_chat_model.
    """
    # Registered proxy providers (developer-defined in deployment layer)
    spec = _proxy_registry.get(provider)
    if spec is not None:
        real_provider, adapted = _resolve_proxy(spec, model_id, kwargs)
        return real_provider, adapted, spec

    # Native providers
    real_provider, adapted = _resolve_native_kwargs(provider, kwargs)
    return real_provider, adapted, None


# ---------------------------------------------------------------------------
# Thinking defaults
# ---------------------------------------------------------------------------

def _apply_thinking_defaults(
    provider: str, model_id: str, kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Enable thinking/reasoning by default for providers that support it.

    Only sets defaults when the caller has not already provided the relevant
    kwarg, so explicit values (including disabling) always take precedence.
    """
    if provider == "anthropic":
        out = dict(kwargs)
        pairs = _CLAUDE_VER_PAIR.findall(model_id)
        if pairs:
            major, minor = int(pairs[0][0]), int(pairs[0][1])
            if (major, minor) >= (4, 6):
                out.setdefault("thinking", {"type": "adaptive"})
            else:
                out.setdefault("thinking", {"type": "enabled", "budget_tokens": 10000})
        else:
            out.setdefault("thinking", {"type": "enabled", "budget_tokens": 10000})
        out.setdefault("max_tokens", 16000)
        return out

    if provider == "google_genai":
        m = _GEMINI_VERSION.match(model_id)
        if m:
            major, minor = int(m.group(1)), int(m.group(2) or "0")
            if major > 2 or (major == 2 and minor >= 5):
                out = dict(kwargs)
                out.setdefault("include_thoughts", True)
                if major < 3:
                    out.setdefault("thinking_budget", 8192)
                # Gemini 3+ defaults to "high" (dynamic) natively;
                # we only set include_thoughts and leave thinking_level
                # unset so the model uses its own default.
                return out

    if provider == "deepseek":
        out = dict(kwargs)
        extra = dict(out.get("extra_body") or {})
        thinking = extra.get("thinking")
        if thinking is None:
            extra["thinking"] = {"type": "enabled"}
            out["extra_body"] = extra
        out.setdefault("max_tokens", 8192)
        return out

    return kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_model(
    model: str | BaseChatModel,
    **kwargs: Any,
) -> BaseChatModel:
    """Resolve a model spec into a ready-to-use BaseChatModel.

    Args:
        model: One of:
          - A BaseChatModel instance (returned as-is)
          - "provider:model_id" string (e.g. "openai:gpt-5-mini")
          - Plain model name (e.g. "gpt-5-mini", auto-detects provider)
        **kwargs: Forwarded to init_chat_model (e.g. temperature, max_tokens).
            Generic params like base_url and api_key are translated to
            provider-specific equivalents automatically.

    Returns:
        A configured BaseChatModel.

    Examples:
        resolve_model("anthropic:claude-sonnet-4-5")
        resolve_model("openai:gpt-5-mini")

        # Registered proxy (deploy layer calls register_proxy first)
        resolve_model("myproxy:claude-opus-4-6")

        # Pre-built model (full control)
        from langchain_openai import ChatOpenAI
        resolve_model(ChatOpenAI(model="gpt-5-mini", base_url="...", api_key="..."))
    """
    if isinstance(model, BaseChatModel):
        return model

    if ":" in model:
        provider, _, model_id = model.partition(":")
    else:
        provider = _infer_provider(model)
        model_id = model

    if provider == "deepseek":
        model_id = _normalize_deepseek_model_id(model_id)

    real_provider, adapted, proxy_spec = _resolve_provider_kwargs(provider, model_id, kwargs)
    effective_model_id = adapted.pop("_effective_model_id", model_id)
    adapted = _apply_thinking_defaults(provider if provider == "deepseek" else real_provider, model_id, adapted)

    if proxy_spec is not None and proxy_spec.model_adapter is not None:
        return proxy_spec.model_adapter(model=model_id, **adapted)

    from langchain.chat_models import init_chat_model
    return init_chat_model(f"{real_provider}:{effective_model_id}", **adapted)
