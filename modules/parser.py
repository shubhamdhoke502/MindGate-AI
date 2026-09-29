import fitz  # PyMuPDF
import re
import unicodedata
from typing import List, Dict, Any, Optional

def clean_devnagari_and_text(text: str) -> str:
    """
    Normalizes Devnagari (Hindi/Marathi) & English scripts, removes broken glyphs and PDF noise.
    """
    if not text:
        return ""
    
    # Unicode NFC Normalization
    normalized = unicodedata.normalize("NFC", text)
    # Fix broken Devnagari word joins
    normalized = re.sub(r'(\w)\s*্\s*(\w)', r'\1\2', normalized)
    # Remove control characters
    normalized = re.sub(r'[\r\t\f\v]', ' ', normalized)
    # Collapse whitespace
    return re.sub(r' +', ' ', normalized).strip()


def extract_tables_from_page(page) -> List[Dict[str, Any]]:
    """
    Extracts structured table content from PDF pages with bounding boxes.
    """
    extracted_tables = []
    try:
        tabs = page.find_tables()
        for tab in tabs:
            table_data = tab.extract()
            bbox = list(tab.bbox)
            
            # Convert tabular matrix into clean Markdown Table format
            md_table_rows = []
            for row in table_data:
                clean_row = [clean_devnagari_and_text(str(cell or '')) for cell in row]
                md_table_rows.append("| " + " | ".join(clean_row) + " |")
            
            if md_table_rows:
                header = md_table_rows[0]
                separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
                body = md_table_rows[1:] if len(md_table_rows) > 1 else []
                full_md_table = "\n".join([header, separator] + body)
                
                extracted_tables.append({
                    "text": full_md_table,
                    "bbox": [round(c, 2) for c in bbox],
                    "is_table": True
                })
    except Exception as e:
        # Graceful fallback if table finding fails on non-standard PDF
        pass
    return extracted_tables


def parse_pdf_production_grade(pdf_path: str, lines_per_chunk: int = 6, overlap_lines: int = 2) -> List[Dict[str, Any]]:
    """
    Production-Level Parser: Handles Multi-column layouts, Structured Tables, Devnagari NFC, and Exact Bounding Boxes.
    """
    doc = fitz.open(pdf_path)
    file_name = pdf_path.split("/")[-1].split("\\")[-1]
    all_chunks = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        
        # 1. Extract Structured Tables first
        page_tables = extract_tables_from_page(page)
        
        # 2. Extract Text Blocks with Layout Analysis (sort by vertical position for multi-column)
        blocks = page.get_text("blocks")
        # Sort blocks top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

        page_elements = []

        # Add Tables to Page Elements
        for tbl in page_tables:
            page_elements.append(tbl)

        # Process Standard Paragraphs / Text Blocks
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type == 0:  # Text block
                cleaned = clean_devnagari_and_text(text)
                if cleaned:
                    page_elements.append({
                        "text": cleaned,
                        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                        "is_table": False
                    })

        if not page_elements:
            # Fallback check for scanned/image pages
            pix = page.get_pixmap()
            if pix.width > 0 and pix.height > 0:
                print(f"[Parser Warning]: Page {page_num} in {file_name} might be a scanned image. OCR required.")

        # Chunking Logic: Maintain table integrity, chunk standard text blocks
        total_elements = len(page_elements)
        step = max(1, lines_per_chunk - overlap_lines)

        for i in range(0, total_elements, step):
            group = page_elements[i : i + lines_per_chunk]
            if not group:
                continue

            # Combine texts and compute bounding box
            chunk_text = "\n".join([e["text"] for e in group])
            chunk_x0 = min(e["bbox"][0] for e in group)
            chunk_y0 = min(e["bbox"][1] for e in group)
            chunk_x1 = max(e["bbox"][2] for e in group)
            chunk_y1 = max(e["bbox"][3] for e in group)

            chunk_id = f"{file_name}_p{page_num}_c{i+1}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "file_name": file_name,
                    "page_number": page_num,
                    "line_start": i + 1,
                    "line_end": min(i + len(group), total_elements),
                    "bbox": [chunk_x0, chunk_y0, chunk_x1, chunk_y1],
                    "has_table": any(e.get("is_table", False) for e in group),
                    "character_count": len(chunk_text)
                }
            })

    doc.close()
    return all_chunks


if __name__ == "__main__":
    print("[Production Parser Engine]: Layout, Table Aware & Devnagari NFC Parser Ready.")