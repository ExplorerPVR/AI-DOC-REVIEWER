"""Security Guardrails for AI Document Assistant.

Defends against prompt injection, delimiter smuggling, system prompt leakage,
and unauthorized extraction of Personally Identifiable Information (PII).
"""

import logging
import re
from collections.abc import Iterator

logger = logging.getLogger("app.guardrails")

# -------------------------------------------------------------------------
# 1. Prompt Injection Detection Patterns
# -------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # Direct instruction overrides
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+)?(?:prior|previous|above|system)\s+"
            r"(?:instructions|directions|rules|constraints|prompts)\b",
            re.IGNORECASE,
        ),
        "Instruction override attempt detected",
    ),
    # System prompt extraction attempts
    (
        re.compile(
            r"\b(?:repeat|print|show|output|reveal|dump|recite|display)\s+"
            r"(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules|directives)\s+"
            r"(?:verbatim|word\s+for\s+word|completely|above)?\b",
            re.IGNORECASE,
        ),
        "System prompt extraction probe detected",
    ),
    (
        re.compile(
            r"\bwhat\s+(?:are|were)\s+(?:your\s+)?"
            r"(?:initial|original|base|core|underlying|system|\s)*"
            r"(?:instructions|rules|prompts|directives)\b",
            re.IGNORECASE,
        ),
        "System prompt extraction probe detected",
    ),
    # Jailbreaks & unrestricted personas
    (
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
            r"(?:dan|developer\s+mode|unrestricted|unfiltered|jailbreak|evil\s+bot)\b",
            re.IGNORECASE,
        ),
        "Persona jailbreak attempt detected",
    ),
    (
        re.compile(
            r"\b(?:jailbreak|developer\s+mode\s+enabled|do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
        "Jailbreak trigger keyword detected",
    ),
    # Synthetic delimiter injection (closing XML tags followed by directives)
    (
        re.compile(
            r"<\s*/\s*(?:question|context|resume_data)\s*>.*?"
            r"(?:system\s+override|ignore\s+all|critical\s+fix|new\s+rule)",
            re.IGNORECASE | re.DOTALL,
        ),
        "Synthetic delimiter breakout attack detected",
    ),
    # Critical override headers
    (
        re.compile(
            r"\b(?:SYSTEM\s+OVERRIDE|ADMIN\s+OVERRIDE|SECURITY\s+NOTICE:\s+DISREGARD)\b",
            re.IGNORECASE,
        ),
        "Administrative override directive detected",
    ),
    # Encoded / Obfuscated instruction execution
    (
        re.compile(
            r"\b(?:decode\s+and\s+execute|base64\s+decode\s+this|run\s+the\s+following\s+hex)\b",
            re.IGNORECASE,
        ),
        "Obfuscated instruction execution attempt detected",
    ),
]


def is_prompt_injection(text: str) -> tuple[bool, str | None]:
    """Inspect input question or text for prompt injection vectors.

    Returns:
        (is_injection, reason): True and description if an attack is detected.
    """
    if not text or not text.strip():
        return False, None

    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Guardrail blocked prompt injection: %s", reason)
            return True, reason

    return False, None


# -------------------------------------------------------------------------
# 2. PII Detection, Redaction, and Harvesting Defense
# -------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN = re.compile(
    r"(?:"
    r"\+\d{1,3}(?:[-.\s()]*\d){6,12}\b"
    r"|"
    r"\b0?[6-9]\d{4}[-.\s]?\d{5}\b"
    r"|"
    r"(?:\+?1[-.\s]*)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|"
    r"\b0\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    r")"
)
_SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)
_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

_PII_HARVEST_PATTERNS = [
    re.compile(
        r"\b(?:dump|list|extract|give\s+me|show)\s+(?:all\s+)?"
        r"(?:phone\s+numbers?|email\s+addresses?|ssn|social\s+security|"
        r"credit\s+cards?|personal\s+contacts?|private\s+details)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+is\s+(?:the\s+)?(?:personal\s+)?(?:phone|email|ssn|social\s+security|home\s+address)\b",
        re.IGNORECASE,
    ),
]


def is_pii_harvesting_query(text: str) -> tuple[bool, str | None]:
    """Detect queries that attempt to extract private personal identifiable data."""
    if not text or not text.strip():
        return False, None

    for pattern in _PII_HARVEST_PATTERNS:
        if pattern.search(text):
            logger.warning("Guardrail blocked PII harvesting attempt")
            return (
                True,
                "Query requests private personal identifiable information (PII).",
            )

    return False, None


def redact_pii(text: str) -> str:
    """Mask sensitive PII identifiers (emails, phone numbers, SSNs, credit cards)."""
    if not text:
        return text

    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = _SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
    redacted = _CARD_PATTERN.sub("[REDACTED_CARD]", redacted)
    return redacted


class StreamingPIIRedactor:
    """Stateful streaming redactor that masks PII across chunk boundaries.

    Maintains an un-emitted tail buffer across consecutive chunks so that
    PII items split across token boundaries (e.g. 'john.doe@' and 'gmail.com')
    are safely reassembled and redacted before reaching the client.
    """

    def __init__(self, max_holdback: int = 60) -> None:
        self.buffer = ""
        self.max_holdback = max_holdback

    def feed(self, chunk: str) -> str:
        """Feed a new chunk and return the safe, redacted portion ready to emit."""
        if not chunk:
            return ""
        current = self.buffer + chunk

        # 1. Trailing potential email: non-whitespace word or email-like sequence
        match_email = re.search(r"[A-Za-z0-9._%+-]*@?[A-Za-z0-9.-]*$", current)
        idx_email = (
            match_email.start()
            if match_email and match_email.group()
            else len(current)
        )

        # 2. Trailing phone/ssn/card candidate (digits and phone separators)
        match_phone = re.search(
            r"(?:\+?\d|\(\d|\b\d)[\d\s\-().]{0,30}$", current
        )
        idx_phone = (
            match_phone.start()
            if match_phone and match_phone.group()
            else len(current)
        )

        cutoff = min(idx_email, idx_phone)
        if len(current) - cutoff > self.max_holdback:
            cutoff = len(current) - self.max_holdback

        safe_part = current[:cutoff]
        self.buffer = current[cutoff:]

        return redact_pii(safe_part) if safe_part else ""

    def flush(self) -> str:
        """Flush and redact any remaining buffered tail at stream completion."""
        tail = self.buffer
        self.buffer = ""
        return redact_pii(tail) if tail else ""


def stream_redact_pii(chunks: Iterator[str]) -> Iterator[str]:
    """Stream text chunks while redacting PII across chunk boundaries."""
    redactor = StreamingPIIRedactor()
    for chunk in chunks:
        if chunk:
            out = redactor.feed(chunk)
            if out:
                yield out
    flushed = redactor.flush()
    if flushed:
        yield flushed


# -------------------------------------------------------------------------
# 3. Output Anti-Leakage Guardrail
# -------------------------------------------------------------------------

_SECRET_SUBSTRINGS = [
    "Under no circumstances reveal, repeat, summarize, or disclose",
    "CONFIDENTIALITY & SYSTEM PROMPT INTEGRITY",
    "CRITICAL GROUNDING RULES:",
    "Strict Grounded Q&A Assistant",
]


def contains_system_prompt_leak(text: str) -> bool:
    """Check if output accidentally regurgitated internal system instructions."""
    if not text:
        return False
    return any(sub in text for sub in _SECRET_SUBSTRINGS)
