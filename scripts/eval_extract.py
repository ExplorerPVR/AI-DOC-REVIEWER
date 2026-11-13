"""Evaluation script for POST /extract structured resume data.

Runs benchmark resumes against /extract, compares outputs against
data/resumes/expected.json, and prints validation & accuracy scores
separately under strict and lenient evaluation criteria.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.extractor import extract_resume_data  # noqa: E402
from app.models import ResumeData  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval_extract")

DATA_DIR = PROJECT_ROOT / "data" / "resumes"
EXPECTED_FILE = DATA_DIR / "expected.json"


def normalize_phone(phone: str | None) -> str | None:
    """Extract numeric digits from phone string for comparison."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    return digits if digits else None


def compare_fields(
    predicted: ResumeData,
    expected: dict,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Compare predicted ResumeData against expected dictionary across 4 fields.

    Returns:
        (strict_checks, lenient_checks): dictionaries mapping each field
        ('name', 'email', 'phone', 'skills') to a boolean match result.
    """
    strict: dict[str, bool] = {}
    lenient: dict[str, bool] = {}

    # 1. Compare name
    pred_name = predicted.name.strip().lower() if predicted.name else None
    exp_name = expected.get("name").strip().lower() if expected.get("name") else None

    # Strict: exact case-insensitive match
    strict["name"] = pred_name == exp_name
    # Lenient: exact match OR one contains the other (e.g. titles/initials)
    if pred_name is None and exp_name is None:
        lenient["name"] = True
    elif pred_name is not None and exp_name is not None:
        lenient["name"] = (
            (pred_name == exp_name)
            or (pred_name in exp_name)
            or (exp_name in pred_name)
        )
    else:
        lenient["name"] = False

    # 2. Compare email
    pred_email = predicted.email.strip().lower() if predicted.email else None
    exp_email = expected.get("email").strip().lower() if expected.get("email") else None
    strict["email"] = pred_email == exp_email
    lenient["email"] = pred_email == exp_email

    # 3. Compare phone
    pred_phone = normalize_phone(predicted.phone)
    exp_phone = normalize_phone(expected.get("phone"))
    strict["phone"] = pred_phone == exp_phone
    if pred_phone is None and exp_phone is None:
        lenient["phone"] = True
    elif pred_phone is not None and exp_phone is not None:
        lenient["phone"] = (
            pred_phone == exp_phone
            or (
                len(pred_phone) >= 7
                and len(exp_phone) >= 7
                and (pred_phone.endswith(exp_phone) or exp_phone.endswith(pred_phone))
            )
        )
    else:
        lenient["phone"] = False

    # 4. Compare skills
    pred_skills = {s.strip().lower() for s in (predicted.skills or [])}
    exp_skills = {s.strip().lower() for s in (expected.get("skills") or [])}

    # Strict: exact set equality
    strict["skills"] = pred_skills == exp_skills

    # Lenient: Jaccard >= 0.70 OR subset overlap
    if not exp_skills and not pred_skills:
        lenient["skills"] = True
    elif exp_skills and pred_skills:
        intersection = exp_skills.intersection(pred_skills)
        union = exp_skills.union(pred_skills)
        jaccard = len(intersection) / len(union) if union else 0.0
        lenient["skills"] = (
            jaccard >= 0.70
            or pred_skills == exp_skills
            or exp_skills.issubset(pred_skills)
            or pred_skills.issubset(exp_skills)
        )
    else:
        lenient["skills"] = False

    return strict, lenient


def run_evaluation(
    url: str | None = None,
    provider: str = "gemini",
    pause_seconds: float = 2.0,
) -> dict[str, object]:
    """Evaluate extraction accuracy across all resumes in expected.json."""
    if not EXPECTED_FILE.exists():
        logger.error("Expected dataset not found at %s.", EXPECTED_FILE)
        sys.exit(1)

    expected_data: dict[str, dict] = json.loads(
        EXPECTED_FILE.read_text(encoding="utf-8")
    )
    total_resumes = len(expected_data)
    valid_resumes = 0
    total_fields = 0
    strict_correct_fields = 0
    lenient_correct_fields = 0

    strict_field_scores = {"name": 0, "email": 0, "phone": 0, "skills": 0}
    lenient_field_scores = {"name": 0, "email": 0, "phone": 0, "skills": 0}

    logger.info("=" * 70)
    logger.info(
        "Starting Extraction Evaluation (%d Resumes: 20 Base + 5 Challenging)",
        total_resumes,
    )
    logger.info("Provider: %s | Pause: %.1fs per call", provider, pause_seconds)
    logger.info("=" * 70)

    client = httpx.Client(timeout=45.0) if url else None

    try:
        for idx, (filename, expected) in enumerate(expected_data.items(), start=1):
            file_path = DATA_DIR / filename
            if not file_path.exists():
                logger.warning(
                    "[%02d/%02d] File missing: %s", idx, total_resumes, filename
                )
                continue

            resume_text = file_path.read_text(encoding="utf-8")
            start_time = time.perf_counter()
            predicted: ResumeData | None = None
            is_valid = False

            try:
                if client and url:
                    resp = client.post(
                        url,
                        json={"text": resume_text, "provider": provider},
                    )
                    if resp.status_code == 200:
                        predicted = ResumeData.model_validate(resp.json())
                        is_valid = True
                    else:
                        logger.error(
                            "[%02d/%02d] %s -> HTTP %d: %s",
                            idx,
                            total_resumes,
                            filename,
                            resp.status_code,
                            resp.text,
                        )
                else:
                    # Direct in-process extraction
                    predicted = extract_resume_data(
                        resume_text=resume_text,
                        provider=provider,
                    )
                    is_valid = True
            except Exception as err:
                logger.error(
                    "[%02d/%02d] %s -> Error: %s",
                    idx,
                    total_resumes,
                    filename,
                    err,
                )

            elapsed = time.perf_counter() - start_time
            total_fields += 4

            if is_valid and predicted is not None:
                valid_resumes += 1
                strict_chk, lenient_chk = compare_fields(predicted, expected)

                s_cnt = sum(1 for v in strict_chk.values() if v)
                l_cnt = sum(1 for v in lenient_chk.values() if v)
                strict_correct_fields += s_cnt
                lenient_correct_fields += l_cnt

                for field in strict_field_scores:
                    if strict_chk[field]:
                        strict_field_scores[field] += 1
                    if lenient_chk[field]:
                        lenient_field_scores[field] += 1

                logger.info(
                    "[%02d/%02d] %-15s: VALID | Strict: %d/4, Lenient: %d/4 [%.2fs] "
                    "| Strict: [n:%s e:%s p:%s s:%s] | Lenient: [n:%s e:%s p:%s s:%s]",
                    idx,
                    total_resumes,
                    filename,
                    s_cnt,
                    l_cnt,
                    elapsed,
                    "+" if strict_chk["name"] else "-",
                    "+" if strict_chk["email"] else "-",
                    "+" if strict_chk["phone"] else "-",
                    "+" if strict_chk["skills"] else "-",
                    "+" if lenient_chk["name"] else "-",
                    "+" if lenient_chk["email"] else "-",
                    "+" if lenient_chk["phone"] else "-",
                    "+" if lenient_chk["skills"] else "-",
                )
            else:
                logger.info(
                    "[%02d/%02d] %-15s: INVALID/FAILED [%.2fs]",
                    idx,
                    total_resumes,
                    filename,
                    elapsed,
                )

            if idx < total_resumes and pause_seconds > 0:
                time.sleep(pause_seconds)
    finally:
        if client:
            client.close()

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS (%d Resumes)", total_resumes)
    logger.info("=" * 70)
    logger.info(
        "Valid Schema Resumes:   %d/%d (%.1f%%)",
        valid_resumes,
        total_resumes,
        (valid_resumes / total_resumes * 100) if total_resumes else 0.0,
    )
    logger.info("-" * 70)
    logger.info("STRICT EVALUATION (Exact matches & exact skill set equality):")
    logger.info(
        "  Overall Fields Correct: %d/%d (%.1f%%)",
        strict_correct_fields,
        total_fields,
        (strict_correct_fields / total_fields * 100) if total_fields else 0.0,
    )
    logger.info(
        "  - Name (Exact):         %d/%d",
        strict_field_scores["name"],
        total_resumes,
    )
    logger.info(
        "  - Email (Exact):        %d/%d",
        strict_field_scores["email"],
        total_resumes,
    )
    logger.info(
        "  - Phone (Exact Digits): %d/%d",
        strict_field_scores["phone"],
        total_resumes,
    )
    logger.info(
        "  - Skills (Exact Set):   %d/%d",
        strict_field_scores["skills"],
        total_resumes,
    )
    score_line_strict = (
        f"{valid_resumes}/{total_resumes} valid, "
        f"{strict_correct_fields}/{total_fields} fields correct"
    )
    logger.info("  Score (Strict):         %s", score_line_strict)
    logger.info("-" * 70)
    logger.info("LENIENT EVALUATION (Tolerant name/phone & Jaccard >= 0.70 / subset):")
    logger.info(
        "  Overall Fields Correct: %d/%d (%.1f%%)",
        lenient_correct_fields,
        total_fields,
        (lenient_correct_fields / total_fields * 100) if total_fields else 0.0,
    )
    logger.info(
        "  - Name (Tolerant):      %d/%d",
        lenient_field_scores["name"],
        total_resumes,
    )
    logger.info(
        "  - Email:                %d/%d",
        lenient_field_scores["email"],
        total_resumes,
    )
    logger.info(
        "  - Phone (Suffix/Digits):%d/%d",
        lenient_field_scores["phone"],
        total_resumes,
    )
    logger.info(
        "  - Skills (Lenient):     %d/%d",
        lenient_field_scores["skills"],
        total_resumes,
    )
    score_line_lenient = (
        f"{valid_resumes}/{total_resumes} valid, "
        f"{lenient_correct_fields}/{total_fields} fields correct"
    )
    logger.info("  Score (Lenient):        %s", score_line_lenient)
    logger.info("=" * 70)

    return {
        "valid_resumes": valid_resumes,
        "total_resumes": total_resumes,
        "strict_correct": strict_correct_fields,
        "lenient_correct": lenient_correct_fields,
        "total_fields": total_fields,
        "strict_scores": strict_field_scores,
        "lenient_scores": lenient_field_scores,
        "score_line_strict": score_line_strict,
        "score_line_lenient": score_line_lenient,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate /extract on resumes")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional HTTP endpoint URL (e.g. http://127.0.0.1:8000/extract)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        help="LLM provider to test ('gemini', 'groq', 'ollama')",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Pause between requests in seconds for rate limits (default: 1.5s)",
    )

    args = parser.parse_args()
    run_evaluation(
        url=args.url,
        provider=args.provider,
        pause_seconds=args.pause,
    )
