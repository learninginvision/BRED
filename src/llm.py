"""A small async client for any OpenAI-compatible chat endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def parse_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply (fenced or bare)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in the reply")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])


class LLM:
    """Chat completions with retries. ``settings`` is the ``llm`` section of config.yaml."""

    def __init__(self, settings: Dict[str, Any], retries: int = 3):
        from openai import AsyncOpenAI

        self.model = settings["model_name"]
        self.temperature = settings.get("temperature", 0.7)
        self.max_tokens = settings.get("max_tokens", 8192)
        self.extra_body: Optional[Dict[str, Any]] = settings.get("extra_body") or None
        self.retries = retries
        self.client = AsyncOpenAI(
            base_url=settings.get("base_url"),
            api_key=settings.get("api_key") or "EMPTY",
            timeout=settings.get("timeout", 300),
        )

    async def chat(self, system: str, user: str) -> str:
        for attempt in range(1, self.retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # network errors, rate limits, 5xx
                if attempt == self.retries:
                    raise
                logger.warning("LLM call failed (%s), retry %d/%d", exc, attempt, self.retries)
                await asyncio.sleep(5 * attempt)
        return ""

    async def chat_json(self, system: str, user: str) -> Any:
        """Ask for JSON; one extra attempt if the reply does not parse."""
        reply = await self.chat(system, user)
        try:
            return parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            reply = await self.chat(system, user + "\n\nReply with valid JSON only.")
            return parse_json(reply)
