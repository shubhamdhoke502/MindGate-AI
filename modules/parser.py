import os
import re
import pymupdf as fitz
from datetime import datetime, timezone
from typing import List, Dict, Any, Sequence

def clean_devnagari_and_text(text: str) -> str:
    if not text:
        return ""
    import unicodedata
    normalized = unicodedata.normalize("NFC", text)
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
    return " ".join(cleaned.split())


def _chunks_from_lines(
    lines: Sequence[str],
    file_name: str,
    page_number: int,
    bbox: List[float],
    chunk_size: int = 8,
    overlap: int = 2,
    line_offset: int = 0,
) -> List[Dict[str, Any]]:
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(lines), step):
        selected = [clean_devnagari_and_text(line) for line in lines[start:start + chunk_size]]
        selected = [line for line in selected if line]
        if not selected:
            continue
        line_start = line_offset + start + 1
        line_end = line_offset + min(start + chunk_size, len(lines))
        chunk_id = f"{file_name}_p{page_number}_l{line_start}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": "\n".join(selected),
            "metadata": {
                "chunk_id": chunk_id,
                "file_name": file_name,
                "page_number": page_number,
                "line_start": line_start,
                "line_end": line_end,
                "bbox": bbox,
            },
        })
    return chunks


def _detect_language(text: str) -> str:
    devanagari = sum("\u0900" <= char <= "\u097f" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if devanagari > latin:
        return "mr" if re.search(r"[ळऱऍऑॲ]", text) else "hi"
    if latin:
        return "en"
    return "und"


def parse_pdf_production_grade(file_path: str) -> List[Dict[str, Any]]:
    """
    Robust Multi-Format Document Ingestion Engine with Path Auto-Resolution.
    """
    # PATH AUTO-RESOLUTION FIX: Agar path direct exist nahi karta, toh 'uploads/' me check karo
    resolved_path = file_path
    if not os.path.exists(resolved_path):
        alt_path = os.path.join("uploads", os.path.basename(file_path))
        if os.path.exists(alt_path):
            resolved_path = alt_path

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"File not found at resolved path: {file_path}")

    file_name = os.path.basename(resolved_path)
    ext = os.path.splitext(file_name)[1].lower()
    file_size_bytes = os.path.getsize(resolved_path)
    upload_timestamp = datetime.now(timezone.utc).isoformat()
    chunks = []
    total_pages = None

    # Case 1: PDF Files
    if ext == ".pdf":
        with fitz.open(resolved_path) as doc:
            total_pages = len(doc)
            for page_num, page in enumerate(doc, start=1):
                line_offset = 0
                for block in page.get_text("blocks", sort=True):
                    raw_lines = block[4].splitlines()
                    if not any(clean_devnagari_and_text(line) for line in raw_lines):
                        continue
                    bbox = [round(coord, 2) for coord in block[:4]]
                    chunks.extend(_chunks_from_lines(
                        raw_lines, file_name, page_num, bbox,
                        line_offset=line_offset,
                    ))
                    line_offset += len(raw_lines)

    # Case 2: Text / Markdown Files
    elif ext in [".txt", ".md"]:
        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        chunks.extend(_chunks_from_lines(lines, file_name, 1, [0, 0, 0, 0]))

    # Case 3: Word DOCX Files
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(resolved_path)
            paragraphs = [p.text for p in doc.paragraphs]
            chunks.extend(_chunks_from_lines(
                paragraphs, file_name, 1, [0, 0, 0, 0]
            ))
        except ImportError:
            raise ImportError("Please install python-docx (`pip install python-docx`) to parse .docx files.")

    else:
        raise ValueError(f"Unsupported document type: {ext or '(no extension)'}")

    for chunk in chunks:
        text = chunk["text"]
        chunk["metadata"].update({
            "file_size_bytes": file_size_bytes,
            "total_pages": total_pages,
            "upload_timestamp": upload_timestamp,
            "character_count": len(text),
            "language_detected": _detect_language(text),
        })

    return chunks