"""Shared test configuration.

Sets up proxy providers (CloseAI, XiaoHa) and direct providers.

CloseAI is a proxy with separate sub-endpoints per native format:
  {base}/v1 (OpenAI), {base}/anthropic, {base}/google
  Use "closeai:model-name" to route through it.

XiaoHa is a single OpenAI-compatible endpoint:
  Use "xiaoha:model-name" to route through it.

For the existing hot-switch tests that use native provider strings
(anthropic:..., openai:..., google_genai:...), we still set the legacy
per-provider env vars so those tests continue to work.

Keys are loaded from tests/.env (not tracked by git).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CLOSEAI_KEY = os.environ["CLOSEAI_KEY"]
CLOSEAI_BASE = os.environ["CLOSEAI_BASE"]
MOONSHOT_KEY = os.environ["MOONSHOT_KEY"]


def configure_closeai():
    """Set env vars and register proxy providers for tests."""
    from arion_agent.providers.resolver import ProxySpec, register_proxy

    os.environ.setdefault("CLOSEAI_API_KEY", CLOSEAI_KEY)
    os.environ.setdefault("CLOSEAI_API_BASE", CLOSEAI_BASE)

    register_proxy("closeai", ProxySpec(
        api_key_env="CLOSEAI_API_KEY",
        base_url_env="CLOSEAI_API_BASE",
        default_base_url=CLOSEAI_BASE,
        style="native",
        sub_endpoints={
            "openai": "/v1",
            "anthropic": "/anthropic",
            "google_genai": "/google",
        },
    ))

    if os.environ.get("XIAOHA_API_KEY"):
        register_proxy("xiaoha", ProxySpec(
            api_key_env="XIAOHA_API_KEY",
            base_url_env="XIAOHA_API_BASE",
            style="openai",
        ))

    from arion_agent.providers.moonshot import ChatMoonshot
    register_proxy("moonshot", ProxySpec(
        api_key_env="MOONSHOT_API_KEY",
        default_base_url="https://api.moonshot.cn/v1",
        style="openai",
        model_adapter=ChatMoonshot,
    ))

    def _moorethread_resolve(model_id: str, kwargs: dict):
        model_map = {"glm-4.7": "GLM-4.7"}
        api_id = model_map.get(model_id, model_id)
        base = kwargs.get("base_url") or os.environ.get(
            "MOORETHREAD_API_BASE", "https://coding-plan-endpoint.kuaecloud.net/v1"
        )
        key = kwargs.get("api_key") or os.environ.get("MOORETHREAD_API_KEY", "")
        adapted = dict(kwargs)
        adapted["base_url"] = base.rstrip("/") if isinstance(base, str) else base
        adapted["api_key"] = key
        adapted["_effective_model_id"] = api_id
        return "openai", adapted

    if os.environ.get("MOORETHREAD_API_KEY"):
        register_proxy("moorethread", ProxySpec(
            api_key_env="MOORETHREAD_API_KEY",
            base_url_env="MOORETHREAD_API_BASE",
            default_base_url="https://coding-plan-endpoint.kuaecloud.net/v1",
            style="openai",
            resolve=_moorethread_resolve,
        ))

    # Legacy env vars for existing tests that use native provider strings
    os.environ.setdefault("OPENAI_API_KEY", CLOSEAI_KEY)
    os.environ.setdefault("OPENAI_API_BASE", f"{CLOSEAI_BASE}/v1")
    os.environ.setdefault("ANTHROPIC_API_KEY", CLOSEAI_KEY)
    os.environ.setdefault("ANTHROPIC_API_URL", f"{CLOSEAI_BASE}/anthropic")
    os.environ.setdefault("GOOGLE_API_KEY", CLOSEAI_KEY)
    os.environ.setdefault("MOONSHOT_API_KEY", MOONSHOT_KEY)


def get_test_model(model_id: str):
    """Build a model configured for CloseAI proxy.

    OpenAI/Anthropic models work via env vars.
    Gemini models need explicit client_options for CloseAI's /google endpoint.
    """
    if model_id.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=CLOSEAI_KEY,
            client_options={"api_endpoint": f"{CLOSEAI_BASE}/google"},
        )
    return model_id


configure_closeai()
