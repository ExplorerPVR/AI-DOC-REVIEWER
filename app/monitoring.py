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
) -> None:
    """Send execution trace and generation metric to Langfuse."""
    client = get_langfuse()
    if not client:
        return

    try:
        trace = client.trace(
            name=name,
            metadata=metadata or {},
            tags=[provider, model],
        )
        trace.generation(
            name=f"{provider}_{model}",
            model=model,
            input=prompt,
            output=completion,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
                "unit": "TOKENS",
            },
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
    except Exception as exc:
        logger.debug("Failed sending trace to Langfuse: %s", exc)
