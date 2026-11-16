"""Evaluation script for Assignment 10: Semantic Search retrieval accuracy.

Runs 5 targeted questions against ingested document chunks with ground-truth
expected pages, measures whether the expected page appears in the top-5 results,
and outputs a summary table and score.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlmodel import Session, select  # noqa: E402

import app.db as db_module  # noqa: E402
from app.ingestion import search_chunks  # noqa: E402
from app.models import Document, DocumentStatus  # noqa: E402

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": (
            "How does the context-free network solve jigsaw puzzles "
            "in self-supervised learning?"
        ),
        "expected_page": 2,
        "topic": "Jigsaw Pretext Tasks & Context-Free Network (Noroozi & Favaro)",
    },
    {
        "id": 2,
        "question": (
            "What are Masked Autoencoders and how does patch masking scale "
            "in vision transformers?"
        ),
        "expected_page": 3,
        "topic": "Masked Autoencoders (MAE) & Vision Transformer Patch Masking",
    },
    {
        "id": 3,
        "question": (
            "What key milestones and architectures are illustrated in the "
            "representation learning timeline?"
        ),
        "expected_page": 4,
        "topic": "Timeline of Milestones and Pretraining Framework Pipeline",
    },
    {
        "id": 4,
        "question": (
            "How do DINOv2 frozen features perform on semantic segmentation "
            "and monocular depth estimation?"
        ),
        "expected_page": 5,
        "topic": "Downstream Transfer with Frozen Features (DINOv2 Tasks)",
    },
    {
        "id": 5,
        "question": (
            "Why do latent-space predictive architectures like I-JEPA deserve "
            "further study in future directions?"
        ),
        "expected_page": 6,
        "topic": "Future Research Directions & Latent Predictive Architectures",
    },
]


def run_search_eval() -> None:
    """Execute evaluation and print formatted report."""
    print("=" * 80)
    print("Assignment 10: Semantic Search Evaluation (Top-5 Retrieval)")
    print("=" * 80)

    with Session(db_module.engine) as session:
        # Check active ready documents
        ready_docs = session.exec(
            select(Document).where(Document.status == DocumentStatus.READY)
        ).all()
        if not ready_docs:
            print("ERROR: No documents with status 'ready' found in database.")
            print("Please upload and ingest a document first.")
            sys.exit(1)

        print(f"Target Document(s): {', '.join(d.filename for d in ready_docs)}")
        print(f"Evaluation Test Questions: {len(EVAL_QUESTIONS)}\n")

        found_count = 0
        table_rows = []

        for item in EVAL_QUESTIONS:
            q_id = item["id"]
            question = item["question"]
            expected_page = item["expected_page"]

            # Run search with k=5
            results = search_chunks(query=question, k=5, session=session)
            retrieved_pages = [r.page for r in results]
            is_found = expected_page in retrieved_pages

            if is_found:
                found_count += 1
                rank = retrieved_pages.index(expected_page) + 1
                rank_str = f"Rank {rank}"
                status_icon = "PASS"
            else:
                rank_str = "Not in Top 5"
                status_icon = "FAIL"

            top_scores = [f"{r.score:.3f} (p.{r.page})" for r in results[:3]]
            scores_preview = ", ".join(top_scores)

            table_rows.append(
                (q_id, question, expected_page, retrieved_pages, rank_str, status_icon)
            )

            print(f"Q{q_id}: {question}")
            print(f"   Topic:         {item['topic']}")
            print(f"   Expected Page: {expected_page}")
            print(f"   Top-5 Pages:   {retrieved_pages}")
            print(f"   Top-3 Scores:  {scores_preview}")
            print(f"   Result:        [{status_icon}] ({rank_str})\n")

        score_fraction = f"{found_count}/{len(EVAL_QUESTIONS)}"
        accuracy_pct = (found_count / len(EVAL_QUESTIONS)) * 100

        summary_line = (
            f"Evaluation Summary: {score_fraction} found in top 5 "
            f"({accuracy_pct:.1f}% accuracy)"
        )
        print("=" * 80)
        print(summary_line)
        print("=" * 80)


if __name__ == "__main__":
    run_search_eval()
