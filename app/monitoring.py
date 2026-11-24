"""Langfuse observability and monitoring integration for AI Document Assistant.

Traces LLM requests, generation metrics (input/output tokens, latency, cost),
and metadata to Langfuse with graceful fallback when unconfigured.
"""

import logging
import os
from typing import Any

logger = logging.getLogger("app.monitoring")

_langfuse_client: Any = None
_langfuse_initialized: bool = False


def get_langfuse() -> Any | None:
    """Retrieve or initialize singleton Langfuse client if credentials exist."""
    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if public_key and secret_key:
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("Langfuse monitoring initialized successfully (host: %s)", host)
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            _langfuse_client = None
    else:
        logger.debug(
            "Langfuse keys not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Running with local metric recording."
        )
        _langfuse_client = None

    _langfuse_initialized = True
    return _langfuse_client


def generate_trace_id() -> str:
    """Generate a unique trace identifier for Langfuse tracing."""
    client = get_langfuse()
    if client and hasattr(client, "create_trace_id"):
        try:
            return str(client.create_trace_id())
        except Exception:
            pass
    import uuid

    return uuid.uuid4().hex


def trace_llm_generation(
    name: str,
    provider: str,
    model: str,
    prompt: str,
    completion: str,
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float,
    cost_usd: float,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> str | None:
    """Send execution trace and generation metric to Langfuse."""
    client = get_langfuse()
    if not client:
        return trace_id

    trace_id = trace_id or generate_trace_id()
    try:
        if hasattr(client, "start_observation"):
            obs = client.start_observation(
                name=name,
                as_type="generation",
                trace_context={"trace_id": trace_id},
                model=model,
                input=prompt,
                output=completion,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                cost_details={
                    "total": cost_usd,
                },
                metadata={
                    "provider": provider,
                    "duration_seconds": duration_seconds,
                    "cost_usd": cost_usd,
                    **(metadata or {}),
                },
            )
            obs.end()
        elif hasattr(client, "trace"):
            trace = client.trace(
                id=trace_id,
                name=name,
                metadata=metadata or {},
                tags=[provider, model],
            )
            trace.generation(
                name=f"{provider}_{model}",
                model=model,
                input=prompt,
                output=completion,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                cost_details={
                    "total": cost_usd,
                },
                metadata={
                    "provider": provider,
                    "duration_seconds": duration_seconds,
                    "cost_usd": cost_usd,
                    **(metadata or {}),
                },
            )
        if hasattr(client, "flush"):
            client.flush()
    except Exception as exc:
        logger.debug("Failed sending trace to Langfuse: %s", exc)

    return trace_id


def record_feedback(
    trace_id: str,
    score: float,
    comment: str | None = None,
) -> bool:
    """Record user feedback (e.g. 1.0 for up, 0.0 for down) against a Langfuse trace."""
    client = get_langfuse()
    if not client:
        return False

    try:
        if hasattr(client, "create_score"):
            client.create_score(
                name="user_feedback",
                value=score,
                trace_id=trace_id,
                comment=comment,
                data_type="NUMERIC",
            )
            if hasattr(client, "flush"):
                client.flush()
            return True
        elif hasattr(client, "score"):
            client.score(
                name="user_feedback",
                value=score,
                trace_id=trace_id,
                comment=comment,
            )
            if hasattr(client, "flush"):
                client.flush()
            return True
    except Exception as exc:
        logger.warning("Failed recording feedback to Langfuse: %s", exc)
        return False

    return False
