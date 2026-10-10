import os
import pymupdf as fitz
from typing import List, Dict, Any

def clean_devnagari_and_text(text: str) -> str:
    if not text:
        return ""
    import unicodedata
    normalized = unicodedata.normalize("NFC", text)
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
    return " ".join(cleaned.split())

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
    chunks = []

    # Case 1: PDF Files
    if ext == ".pdf":
        doc = fitz.open(resolved_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")

            for block_idx, b in enumerate(blocks):
                raw_text = b[4]
                cleaned = clean_devnagari_and_text(raw_text)
                if not cleaned or len(cleaned) < 5:
                    continue

                bbox = [round(coord, 2) for coord in b[:4]]
                chunk_id = f"{file_name}_p{page_num + 1}_b{block_idx}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": cleaned,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "page_number": page_num + 1,
                        "line_start": block_idx * 5 + 1,
                        "line_end": (block_idx + 1) * 5,
                        "bbox": bbox
                    }
                })
        doc.close()

    # Case 2: Text / Markdown Files
    elif ext in [".txt", ".md"]:
        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        chunk_size = 15
        for idx in range(0, len(lines), chunk_size):
            chunk_lines = lines[idx:idx + chunk_size]
            chunk_text = clean_devnagari_and_text("".join(chunk_lines))
            if not chunk_text:
                continue

            chunk_id = f"{file_name}_chunk_{idx // chunk_size}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "page_number": 1,
                    "line_start": idx + 1,
                    "line_end": idx + len(chunk_lines),
                    "bbox": [0, 0, 0, 0]
                }
            })

    # Case 3: Word DOCX Files
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(resolved_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            chunk_size = 5
            for idx in range(0, len(paragraphs), chunk_size):
                chunk_text = clean_devnagari_and_text(" ".join(paragraphs[idx:idx + chunk_size]))
                if not chunk_text:
                    continue

                chunk_id = f"{file_name}_chunk_{idx // chunk_size}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "page_number": 1,
                        "line_start": idx + 1,
                        "line_end": idx + chunk_size,
                        "bbox": [0, 0, 0, 0]
                    }
                })
        except ImportError:
            raise ImportError("Please install python-docx (`pip install python-docx`) to parse .docx files.")

    return chunks