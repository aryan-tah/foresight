"""Where Claude comes from: the Anthropic API directly, or via OpenRouter.

OpenRouter serves an Anthropic-compatible Messages endpoint, so both paths use
the same official SDK and the same agent loop; only the base URL, the model id
and a few request options differ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic

OPENROUTER_URL = "https://openrouter.ai/api"


@dataclass
class LLM:
    client: anthropic.AsyncAnthropic
    model: str
    provider: str
    extras: dict = field(default_factory=dict)  # provider-specific request options


def from_env() -> LLM | None:
    model = os.environ.get("FORESIGHT_MODEL")

    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLM(
            client=anthropic.AsyncAnthropic(),
            model=model or "claude-opus-5",
            provider="Anthropic",
            # If a safety classifier declines, re-run on Anthropic's recommended
            # fallback model inside the same request. (Not offered by OpenRouter.)
            extras={"fallbacks": "default", "betas": ["server-side-fallback-2026-07-01"]},
        )

    key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return LLM(
            client=anthropic.AsyncAnthropic(base_url=OPENROUTER_URL, auth_token=key, api_key=None),
            model=model or "anthropic/claude-opus-5",
            provider="OpenRouter",
        )
    return None
