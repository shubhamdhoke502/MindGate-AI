import fitz  # PyMuPDF
import re
import unicodedata
from typing import List, Dict, Any

def clean_devnagari_text(text: str) -> str:
    """
    Normalizes Devnagari (Hindi/Marathi) script and cleans raw extracted text.
    Fixes split glyphs, broken diacritics, and unwanted multi-space formatting.
    """
    if not text:
        return ""
    
    # 1. Unicode NFC Normalization (Fixes Devnagari Matra/Combining Characters)
    normalized = unicodedata.normalize("NFC", text)
    
    # 2. Fix broken Hindi/Marathi word splits (e.g., 'क ् ष' -> 'क्ष')
    normalized = re.sub(r'(\w)\s*্\s*(\w)', r'\1\2', normalized)
    
    # 3. Clean control characters and weird PDF line break symbols
    normalized = re.sub(r'[\r\t\f\v]', ' ', normalized)
    
    # 4. Collapse multiple spaces into single space
    cleaned_text = re.sub(r' +', ' ', normalized).strip()
    
    return cleaned_text


def parse_pdf_with_coordinates(pdf_path: str, lines_per_chunk: int = 6, overlap_lines: int = 1) -> List[Dict[str, Any]]:
    """
    Extracts text line-by-line using PyMuPDF with exact Bounding Box (x0, y0, x1, y1) coordinates.
    Generates 5-8 line overlapping chunks with dynamic metadata schema.
    """
    doc = fitz.open(pdf_path)
    file_name = pdf_path.split("/")[-1].split("\\")[-1]
    all_chunks = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        
        # Extract text words with detailed layout bounding boxes (x0, y0, x1, y1, word, block_no, line_no, word_no)
        text_page = page.get_text("words")
        
        # Group words into line structures based on line_no and block_no
        lines_dict = {}
        for word_tuple in text_page:
            x0, y0, x1, y1, word, block_no, line_no, word_no = word_tuple
            key = (block_no, line_no)
            
            if key not in lines_dict:
                lines_dict[key] = {
                    "text": [],
                    "bbox": [x0, y0, x1, y1]
                }
            
            lines_dict[key]["text"].append(word)
            # Expand line bounding box coordinates dynamically
            lines_dict[key]["bbox"][0] = min(lines_dict[key]["bbox"][0], x0)
            lines_dict[key]["bbox"][1] = min(lines_dict[key]["bbox"][1], y0)
            lines_dict[key]["bbox"][2] = max(lines_dict[key]["bbox"][2], x1)
            lines_dict[key]["bbox"][3] = max(lines_dict[key]["bbox"][3], y1)

        # Convert dictionary to ordered line list
        page_lines = []
        for line_info in lines_dict.values():
            raw_line_str = " ".join(line_info["text"])
            clean_str = clean_devnagari_text(raw_line_str)
            if clean_str:
                page_lines.append({
                    "text": clean_str,
                    "bbox": [round(c, 2) for c in line_info["bbox"]]
                })

        total_lines = len(page_lines)
        if total_lines == 0:
            continue

        # Chunking Logic (lines_per_chunk with overlapping context)
        step = max(1, lines_per_chunk - overlap_lines)
        for i in range(0, total_lines, step):
            chunk_line_group = page_lines[i : i + lines_per_chunk]
            if not chunk_line_group:
                continue

            chunk_text = " ".join([item["text"] for item in chunk_line_group])
            start_line = i + 1
            end_line = min(i + len(chunk_line_group), total_lines)

            # Combined Bounding Box for the whole chunk paragraph
            chunk_x0 = min(item["bbox"][0] for item in chunk_line_group)
            chunk_y0 = min(item["bbox"][1] for item in chunk_line_group)
            chunk_x1 = max(item["bbox"][2] for item in chunk_line_group)
            chunk_y1 = max(item["bbox"][3] for item in chunk_line_group)

            chunk_id = f"{file_name}_p{page_num}_l{start_line}_{end_line}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "file_name": file_name,
                    "page_number": page_num,
                    "line_start": start_line,
                    "line_end": end_line,
                    "bbox": [chunk_x0, chunk_y0, chunk_x1, chunk_y1],
                    "character_count": len(chunk_text)
                }
            })

    doc.close()
    return all_chunks


if __name__ == "__main__":
    print("[Parser Module]: Devnagari & Bounding Box Parser Ready.")