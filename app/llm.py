"""LLM module for multi-provider streaming, retries, and cost tracking."""

import inspect
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
    """Estimate token count for text using standard ~4 chars/token heuristic.

    Used as a fallback when provider usage metadata is unavailable.
    """
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4
    word_estimate = int(len(text.split()) * 1.33) + 1
    return max(1, max(char_estimate, word_estimate))


def _stream_gemini(
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    usage: dict[str, int] | None = None,
) -> Iterator[str]:
    """Stream text completion chunks from Google Gemini."""
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip("'\" ")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not configured", status_code=503)

    try:
        from google import genai
        from google.genai import errors as genai_errors

        timeout_ms = int(timeout * 1000) if timeout else None
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_ms} if timeout_ms else None,
        )
        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
        )
        for chunk in response:
            if (
                usage is not None
                and hasattr(chunk, "usage_metadata")
                and chunk.usage_metadata
            ):
                meta = chunk.usage_metadata
                prompt_tokens = getattr(meta, "prompt_token_count", None)
                if prompt_tokens is not None:
                    usage["input_tokens"] = prompt_tokens
                cand_tokens = getattr(meta, "candidates_token_count", None)
                if cand_tokens is not None:
                    usage["output_tokens"] = cand_tokens
            if chunk.text:
                yield chunk.text

        if (
            usage is not None
            and hasattr(response, "usage_metadata")
            and response.usage_metadata
        ):
            meta = response.usage_metadata
            p_tokens = getattr(meta, "prompt_token_count", None)
            c_tokens = getattr(meta, "candidates_token_count", None)
            if "input_tokens" not in usage and p_tokens is not None:
                usage["input_tokens"] = p_tokens
            if "output_tokens" not in usage and c_tokens is not None:
                usage["output_tokens"] = c_tokens
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
    usage: dict[str, int] | None = None,
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
            stream_options={"include_usage": True},
        )
        for chunk in completion:
            if usage is not None and hasattr(chunk, "usage") and chunk.usage:
                u = chunk.usage
                if getattr(u, "prompt_tokens", None) is not None:
                    usage["input_tokens"] = u.prompt_tokens
                if getattr(u, "completion_tokens", None) is not None:
                    usage["output_tokens"] = u.completion_tokens
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
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
    usage: dict[str, int] | None = None,
) -> Iterator[str]:
    """Stream text completion chunks from local Ollama service."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if os.path.exists("/.dockerenv") and (
        "localhost" in base_url or "127.0.0.1" in base_url
    ):
        base_url = (
            base_url.replace("localhost", "host.docker.internal").replace(
                "127.0.0.1", "host.docker.internal"
            )
        )
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
                        if usage is not None:
                            prompt_eval = payload.get("prompt_eval_count")
                            if prompt_eval is not None:
                                usage["input_tokens"] = int(prompt_eval)
                            eval_cnt = payload.get("eval_count")
                            if eval_cnt is not None:
                                usage["output_tokens"] = int(eval_cnt)
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
def _start_provider_stream(
    provider: str,
    prompt: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    usage: dict[str, int] | None = None,
) -> tuple[str | None, Iterator[str]]:
    """Start provider stream and pull the first chunk inside the retry loop.

    Because Python generator functions with `yield` only execute upon iteration,
    fetching the first chunk inside this function ensures initial connection
    failures, rate limits (429), timeouts, and server errors trigger retries.
    """
    func_map = {
        "gemini": _stream_gemini,
        "groq": _stream_groq,
        "ollama": _stream_ollama,
    }
    stream_func = func_map.get(provider)
    if stream_func is None:
        raise ValueError(f"Unsupported LLM provider '{provider}'")

    sig = inspect.signature(stream_func)
    if "usage" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        stream = stream_func(prompt=prompt, model=model, timeout=timeout, usage=usage)
    else:
        stream = stream_func(prompt=prompt, model=model, timeout=timeout)

    first_chunk: str | None = None
    for chunk in stream:
        if chunk:
            first_chunk = chunk
            break

    return first_chunk, stream


# Backward compatibility alias
_call_provider_stream = _start_provider_stream


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

    # Log metrics immediately before database persistence
    logger.info(
        "LLM Call: provider=%s model=%s in_tokens=%d out_tokens=%d "
        "duration=%.4fs cost=$%.8f",
        provider,
        model,
        input_tokens,
        output_tokens,
        time_taken_seconds,
        cost_usd,
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
    usage: dict[str, int] = {}
    output_chunks: list[str] = []

    # 1. Start stream and pull first chunk inside retry loop
    first_chunk, stream = _start_provider_stream(
        provider=normalized_provider,
        prompt=prompt,
        model=model,
        timeout=timeout,
        usage=usage,
    )

    # 2. Consume and yield stream while accumulating output metrics
    try:
        if first_chunk:
            output_chunks.append(first_chunk)
            yield first_chunk
        for chunk in stream:
            if chunk:
                output_chunks.append(chunk)
                yield chunk
    finally:
        # 3. Always compute metrics and record to database
        duration = time.perf_counter() - start_time
        full_text = "".join(output_chunks)

        # Use actual token counts reported by provider; fallback to estimate_tokens
        input_tokens = usage.get("input_tokens")
        if input_tokens is None or input_tokens <= 0:
            input_tokens = estimate_tokens(prompt)

        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
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
