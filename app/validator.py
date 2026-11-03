"""Document content and structure validation service."""

import io
import zipfile
from pathlib import Path

from docx import Document as DocxDocument
from fastapi import HTTPException, status
from pypdf import PdfReader


def validate_document_content(file_bytes: bytes, filename: str) -> None:
    """Validate that the file is not empty, blank, or corrupted."""
    size_bytes = len(file_bytes)
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file. Uploaded file contains no data.",
        )

    extension = Path(filename).suffix.lower()

    # 1. Plain text validation (.txt)
    if extension == ".txt":
        try:
            text = file_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted file: unable to decode text file contents.",
            )
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file: text document contains only whitespace or no text.",
            )

    # 2. PDF validation (.pdf)
    elif extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty file: PDF contains 0 pages.",
                )

            extracted_text = "".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
            has_images = any(len(page.images) > 0 for page in reader.pages)

            if not extracted_text and not has_images:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Empty file: PDF document contains no readable text or content."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted file: invalid or unreadable PDF structure.",
            )

    # 3. Word Document validation (.docx)
    elif extension == ".docx":
        try:
            doc_file = io.BytesIO(file_bytes)
            if not zipfile.is_zipfile(doc_file):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Corrupted file: invalid or unreadable DOCX structure.",
                )

            doc = DocxDocument(doc_file)
            paragraphs_text = "".join(p.text for p in doc.paragraphs).strip()
            tables_text = "".join(
                cell.text
                for table in doc.tables
                for row in table.rows
                for cell in row.cells
            ).strip()
            total_text = paragraphs_text + tables_text
            has_shapes = len(doc.inline_shapes) > 0

            if not total_text and not has_shapes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Empty file: DOCX document contains no readable text."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted file: invalid or unreadable DOCX structure.",
            )
