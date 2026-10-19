"""Provider abstraction + OpenAI chat-completions implementation (native tool calling)."""
from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, field
from typing import Any, Protocol

REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES_PER_TURN = 1
RETRY_BACKOFF_SECONDS = 1.0
MAX_CONCURRENT_PROVIDER_REQUESTS = 3
_TRANSIENT_NAMES = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


class ProviderError(RuntimeError):
    """Any provider failure (network, timeout, HTTP error, malformed response).
    `transient` marks errors worth one retry (timeouts, connection errors, 408/409/429/5xx)."""

    def __init__(
        self, message: str, transient: bool = False, timed_out: bool = False
    ):
        super().__init__(message)
        self.transient = transient
        self.timed_out = timed_out


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as requested by the model


@dataclass(frozen=True)
class ProviderResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    async def acomplete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> ProviderResponse: ...


class AsyncOpenAIProvider:
    def __init__(self, api_key: str, model: str, timeout: float = REQUEST_TIMEOUT_SECONDS):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model

    @classmethod
    def from_env(cls) -> "AsyncOpenAIProvider":
        from dotenv import load_dotenv

        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return cls(key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    async def acomplete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> ProviderResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice=tool_choice,
                response_format={"type": "json_object"},
                temperature=0,
            )
            msg = resp.choices[0].message
            calls = [
                ToolCall(tc.id, tc.function.name, tc.function.arguments)
                for tc in (msg.tool_calls or [])
            ]
            usage = {}
            if resp.usage is not None:
                usage = {
                    "input": resp.usage.prompt_tokens,
                    "output": resp.usage.completion_tokens,
                    "total": resp.usage.total_tokens,
                }
            return ProviderResponse(content=msg.content, tool_calls=calls, usage=usage)
        except Exception as exc:  # never echo exc text: it could contain request details
            status = getattr(exc, "status_code", None)
            transient = type(exc).__name__ in _TRANSIENT_NAMES or (
                isinstance(status, int) and (status in (408, 409, 429) or status >= 500)
            )
            raise ProviderError(
                f"{type(exc).__name__}" + (f" (HTTP {status})" if status else ""),
                transient=transient,
                timed_out=type(exc).__name__ == "APITimeoutError"
                or status in (408, 504),
            ) from exc


_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _global_request_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROVIDER_REQUESTS)
        _SEMAPHORES[loop] = semaphore
    return semaphore


class AsyncRetryingProvider:
    """Async provider wrapper with a process-wide, per-event-loop request cap."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        semaphore: asyncio.Semaphore | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        sleep=asyncio.sleep,
    ):
        self._inner = inner
        self._semaphore = semaphore
        self._timeout = timeout
        self._sleep = sleep
        self.model = getattr(inner, "model", None)

    async def acomplete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> ProviderResponse:
        for attempt in range(MAX_RETRIES_PER_TURN + 1):
            try:
                semaphore = self._semaphore or _global_request_semaphore()
                async with semaphore:
                    return await asyncio.wait_for(
                        self._inner.acomplete(messages, tools, tool_choice),
                        timeout=self._timeout,
                    )
            except asyncio.TimeoutError as exc:
                error = ProviderError("provider request timed out", transient=True, timed_out=True)
                if attempt == MAX_RETRIES_PER_TURN:
                    raise error from exc
                await self._sleep(RETRY_BACKOFF_SECONDS)
            except ProviderError as exc:
                if not exc.transient or attempt == MAX_RETRIES_PER_TURN:
                    raise
                await self._sleep(RETRY_BACKOFF_SECONDS)
        raise AssertionError("unreachable")  # pragma: no cover
