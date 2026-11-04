"""LLM module for multi-provider streaming, retries, and cost tracking."""

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlmodel import Session

from app.db import engine
from app.models import LLMCall
from app.pricing import DEFAULT_MODELS, calculate_cost

load_dotenv()
logger = logging.getLogger(__name__)

# Default timeout in seconds for LLM generation calls
DEFAULT_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
DEFAULT_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, message: str = "LLM provider rate limit exceeded (429)") -> None:
        super().__init__(message=message, status_code=503)


class LLMTimeoutError(LLMError):
    """Request timeout (HTTP 504)."""

    def __init__(self, message: str = "LLM provider request timed out (504)") -> None:
        super().__init__(message=message, status_code=504)


class LLMServerError(LLMError):
    """Provider internal server error (HTTP 5xx)."""

    def __init__(self, message: str = "LLM provider internal error (5xx)") -> None:
        super().__init__(message=message, status_code=503)


# Tuple of transient errors that qualify for retry
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMServerError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


def retry(
    func_or_retries: Any = None,
    *,
    max_retries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Any:
    """Retry decorator with exponential backoff for transient failures.

    Reused from Week 1 resilience decorator with backoff support.
    """
    if isinstance(func_or_retries, int):
        effective_retries = func_or_retries
        target_func = None
    elif callable(func_or_retries):
        effective_retries = max_retries
        target_func = func_or_retries
    else:
        effective_retries = max_retries
        target_func = None

    if effective_retries < 1:
        raise ValueError(f"max_retries must be at least 1, got {effective_retries}.")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(1, effective_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "Attempt %d/%d for '%s' failed with error: %s",
                        attempt,
                        effective_retries,
                        fn.__name__,
                        exc,
                    )
                    if attempt == effective_retries:
                        raise
                    if current_delay > 0:
                        time.sleep(current_delay)
                        current_delay *= backoff

            if last_exception is not None:
                raise last_exception

        return wrapper

    if target_func is not None:
        return decorator(target_func)
    return decorator


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using standard ~4 chars/token heuristic."""
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4
    word_estimate = int(len(text.split()) * 1.33) + 1
    return max(1, max(char_estimate, word_estimate))


def _stream_gemini(
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Stream text completion chunks from Google Gemini."""
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip("'\" ")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not configured", status_code=503)

    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        err_msg = str(exc).lower()
        if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
            raise LLMRateLimitError(f"Gemini rate limit exceeded: {exc}") from exc
        if "timeout" in err_msg or "timed out" in err_msg or "deadline" in err_msg:
            raise LLMTimeoutError(f"Gemini request timed out: {exc}") from exc
        if "500" in err_msg or "503" in err_msg or "internal" in err_msg:
            raise LLMServerError(f"Gemini server error: {exc}") from exc
        if "genai_errors" in locals() and isinstance(exc, genai_errors.APIError):
            raise LLMError(f"Gemini API error: {exc}", status_code=503) from exc
        raise LLMError(f"Gemini generation error: {exc}", status_code=503) from exc


def _stream_groq(
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Stream text completion chunks from Groq Cloud."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip("'\" ")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not configured", status_code=503)

    try:
        from groq import (
            APIConnectionError,
            APITimeoutError,
            Groq,
            InternalServerError,
            RateLimitError,
        )

        client = Groq(api_key=api_key, timeout=timeout)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
    except RateLimitError as exc:
        raise LLMRateLimitError(f"Groq rate limit exceeded: {exc}") from exc
    except APITimeoutError as exc:
        raise LLMTimeoutError(f"Groq request timed out: {exc}") from exc
    except (InternalServerError, APIConnectionError) as exc:
        raise LLMServerError(f"Groq server error: {exc}") from exc
    except Exception as exc:
        err_msg = str(exc).lower()
        if "429" in err_msg:
            raise LLMRateLimitError(f"Groq rate limit: {exc}") from exc
        if "timeout" in err_msg:
            raise LLMTimeoutError(f"Groq timeout: {exc}") from exc
        if "500" in err_msg or "503" in err_msg:
            raise LLMServerError(f"Groq server error: {exc}") from exc
        raise LLMError(f"Groq generation error: {exc}", status_code=503) from exc


def _stream_ollama(
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Stream text completion chunks from local Ollama service."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/generate"

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                url,
                json={"model": model, "prompt": prompt, "stream": True},
            ) as response:
                if response.status_code == 429:
                    raise LLMRateLimitError("Ollama rate limit exceeded")
                if response.status_code >= 500:
                    raise LLMServerError(
                        f"Ollama server error ({response.status_code})"
                    )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        payload = json.loads(line)
                        chunk = payload.get("response", "")
                        if chunk:
                            yield chunk
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(f"Ollama timed out: {exc}") from exc
    except (LLMRateLimitError, LLMServerError, LLMTimeoutError):
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise LLMRateLimitError(f"Ollama rate limit: {exc}") from exc
        if exc.response.status_code >= 500:
            raise LLMServerError(f"Ollama server error: {exc}") from exc
        raise LLMError(f"Ollama HTTP error: {exc}", status_code=503) from exc
    except Exception as exc:
        raise LLMServerError(f"Ollama connection error: {exc}") from exc


@retry(
    max_retries=DEFAULT_MAX_RETRIES,
    delay=0.1,
    backoff=2.0,
    exceptions=RETRYABLE_EXCEPTIONS,
)
def _call_provider_stream(
    provider: str,
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Invoke provider-specific stream generator wrapped with automatic retries."""
    if provider == "gemini":
        return _stream_gemini(prompt=prompt, model=model, timeout=timeout)
    if provider == "groq":
        return _stream_groq(prompt=prompt, model=model, timeout=timeout)
    if provider == "ollama":
        return _stream_ollama(prompt=prompt, model=model, timeout=timeout)
    raise ValueError(f"Unsupported LLM provider '{provider}'")


def _record_llm_call(
    provider: str,
    model: str,
    prompt: str,
    input_tokens: int,
    output_tokens: int,
    time_taken_seconds: float,
    cost_usd: float,
    session: Session | None = None,
) -> None:
    """Save an LLM call metric record to PostgreSQL."""
    call_record = LLMCall(
        provider=provider,
        model=model,
        prompt=prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time_taken_seconds=round(time_taken_seconds, 4),
        cost_usd=cost_usd,
    )
    try:
        if session is not None:
            session.add(call_record)
            session.commit()
        else:
            with Session(engine) as s:
                s.add(call_record)
                s.commit()
    except Exception as exc:
        logger.error("Failed to record LLMCall metric in database: %s", exc)


def generate(
    prompt: str,
    provider: str = "gemini",
    session: Session | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Generate streaming text response from the configured LLM provider.

    Parameters
    ----------
    prompt : str
        The input question or instructions.
    provider : str
        Target LLM provider ('gemini', 'groq', 'ollama'). Default is 'gemini'.
    session : Session, optional
        Active SQLModel database session for persisting the LLMCall record.
    timeout : float
        Per-call timeout threshold in seconds. Default 30.0s.

    Yields
    ------
    str
        Streamed text chunks from the LLM.
    """
    normalized_provider = provider.lower().strip()
    if normalized_provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Allowed providers: {', '.join(sorted(DEFAULT_MODELS.keys()))}"
        )

    model = DEFAULT_MODELS[normalized_provider]
    start_time = time.perf_counter()
    input_tokens = estimate_tokens(prompt)
    output_chunks: list[str] = []

    # 1. Obtain generator wrapped with retry on 429, 5xx, or timeout
    stream = _call_provider_stream(
        provider=normalized_provider,
        prompt=prompt,
        model=model,
        timeout=timeout,
    )

    # 2. Consume and yield stream while accumulating output metrics
    try:
        for chunk in stream:
            if chunk:
                output_chunks.append(chunk)
                yield chunk
    finally:
        # 3. Always compute metrics and record to database
        duration = time.perf_counter() - start_time
        full_text = "".join(output_chunks)
        output_tokens = estimate_tokens(full_text) if full_text else 0
        cost_usd = calculate_cost(model, input_tokens, output_tokens)

        _record_llm_call(
            provider=normalized_provider,
            model=model,
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_taken_seconds=duration,
            cost_usd=cost_usd,
            session=session,
        )
