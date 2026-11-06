"""Evaluation script for POST /extract structured resume data.

Runs all 20 benchmark resumes against /extract, compares outputs against
data/resumes/expected.json, and prints validation & accuracy scores.
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
) -> tuple[int, int, dict[str, bool]]:
    """Compare predicted ResumeData against expected dictionary across 4 fields."""
    checks = {}

    # 1. Compare name (case-insensitive trimmed match)
    pred_name = predicted.name.strip().lower() if predicted.name else None
    exp_name = expected.get("name").strip().lower() if expected.get("name") else None
    checks["name"] = pred_name == exp_name

    # 2. Compare email (case-insensitive trimmed match)
    pred_email = predicted.email.strip().lower() if predicted.email else None
    exp_email = expected.get("email").strip().lower() if expected.get("email") else None
    checks["email"] = pred_email == exp_email

    # 3. Compare phone (digit-based normalization)
    pred_phone = normalize_phone(predicted.phone)
    exp_phone = normalize_phone(expected.get("phone"))
    checks["phone"] = pred_phone == exp_phone

    # 4. Compare skills (case-insensitive set comparison)
    pred_skills = {s.strip().lower() for s in (predicted.skills or [])}
    exp_skills = {s.strip().lower() for s in (expected.get("skills") or [])}
    if not exp_skills and not pred_skills:
        checks["skills"] = True
    elif exp_skills and pred_skills:
        intersection = exp_skills.intersection(pred_skills)
        union = exp_skills.union(pred_skills)
        jaccard = len(intersection) / len(union) if union else 0.0
        checks["skills"] = jaccard >= 0.70 or pred_skills == exp_skills
    else:
        checks["skills"] = False

    correct_count = sum(1 for v in checks.values() if v)
    total_count = len(checks)
    return correct_count, total_count, checks


def run_evaluation(
    url: str | None = None,
    provider: str = "gemini",
    pause_seconds: float = 2.0,
) -> tuple[int, int, int, int]:
    """Evaluate extraction accuracy across all 20 resumes in expected.json."""
    if not EXPECTED_FILE.exists():
        logger.error("Expected dataset not found at %s.", EXPECTED_FILE)
        sys.exit(1)

    expected_data: dict[str, dict] = json.loads(
        EXPECTED_FILE.read_text(encoding="utf-8")
    )
    total_resumes = len(expected_data)
    valid_resumes = 0
    total_fields = 0
    correct_fields = 0

    logger.info("=" * 65)
    logger.info("Starting Structured Extraction Evaluation (%d Resumes)", total_resumes)
    logger.info("Provider: %s | Pause: %.1fs per call", provider, pause_seconds)
    logger.info("=" * 65)

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

            if is_valid and predicted is not None:
                valid_resumes += 1
                c_cnt, t_cnt, field_checks = compare_fields(predicted, expected)
                correct_fields += c_cnt
                total_fields += t_cnt
                logger.info(
                    "[%02d/%02d] %-15s: VALID (%d/%d) [%.2fs] n=%s e=%s p=%s s=%s",
                    idx,
                    total_resumes,
                    filename,
                    c_cnt,
                    t_cnt,
                    elapsed,
                    "+" if field_checks["name"] else "-",
                    "+" if field_checks["email"] else "-",
                    "+" if field_checks["phone"] else "-",
                    "+" if field_checks["skills"] else "-",
                )
            else:
                total_fields += 4
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

    logger.info("=" * 65)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info("Valid Resumes:   %d/%d", valid_resumes, total_resumes)
    logger.info("Fields Correct:  %d/%d", correct_fields, total_fields)
    score_line = (
        f"{valid_resumes}/{total_resumes} valid, "
        f"{correct_fields}/{total_fields} fields correct"
    )
    logger.info("Score:           %s", score_line)
    logger.info("=" * 65)

    return valid_resumes, total_resumes, correct_fields, total_fields


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
