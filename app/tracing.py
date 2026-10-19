"""Privacy-conscious Langfuse tracing for the FSE I agent."""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generator

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class TraceContext:
    run_id: str
    trace_id: str | None
    run_observation_id: str | None
    base_url: str | None


_CURRENT_TRACE: ContextVar[TraceContext | None] = ContextVar(
    "kasparro_langfuse_trace", default=None
)


def tracing_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


@lru_cache(maxsize=1)
def get_langfuse_client():
    if not tracing_configured():
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
        tracing_enabled=True,
        sample_rate=1.0,
    )


@contextmanager
def observation(
    *,
    name: str,
    as_type: str,
    metadata: dict[str, Any],
    model: str | None = None,
) -> Generator[Any | None, None, None]:
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(
        as_type=as_type,
        name=name,
        metadata=metadata,
        model=model,
    ) as item:
        yield item


@contextmanager
def run_trace(
    *,
    run_id: str,
    split: str,
    agent_version: str,
    metadata: dict[str, Any],
) -> Generator[TraceContext, None, None]:
    base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    with observation(
        name="kasparro-fse-i-run",
        as_type="span",
        metadata={
            **metadata,
            "run_id": run_id,
            "split": split,
            "agent_version": agent_version,
        },
    ) as span:
        context = TraceContext(
            run_id=run_id,
            trace_id=getattr(span, "trace_id", None),
            run_observation_id=getattr(span, "id", None),
            base_url=base_url if span is not None else None,
        )
        token = _CURRENT_TRACE.set(context)
        try:
            yield context
        finally:
            _CURRENT_TRACE.reset(token)


@contextmanager
def answer_trace(
    *, answer_id: str, metadata: dict[str, Any]
) -> Generator[Any | None, None, None]:
    trace = _CURRENT_TRACE.get()
    with observation(
        name="kasparro-fse-i-answer",
        as_type="span",
        metadata={
            **metadata,
            "answer_id": answer_id,
            "run_id": trace.run_id if trace else None,
        },
    ) as span:
        yield span


def attach_development_scores(
    *,
    trace_id: str | None,
    run_scores: dict[str, float],
    answer_scores: dict[str, dict[str, float]],
    answer_observation_ids: dict[str, str | None],
) -> None:
    client = get_langfuse_client()
    if client is None or trace_id is None:
        return
    for name, value in run_scores.items():
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type="NUMERIC",
            comment="Deterministic development-set score computed after exact-span matching.",
        )
    for answer_id, scores in answer_scores.items():
        observation_id = answer_observation_ids.get(answer_id)
        if observation_id is None:
            continue
        for name, value in scores.items():
            client.create_score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                comment="Deterministic per-answer development score computed after exact-span matching.",
                metadata={"answer_id": answer_id},
            )


def flush_traces() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()
