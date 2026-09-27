import fitz  # PyMuPDF
import json
from datetime import datetime
from typing import List, Dict, Any

def parse_pdf_with_coordinates(pdf_path: str, chunk_size: int = 6, overlap: int = 2) -> List[Dict[str, Any]]:
    """
    Reads a PDF line-by-line, chunks text into 5-8 line blocks with overlap,
    and returns a structured metadata list for vector indexing.
    """
    doc = fitz.open(pdf_path)
    file_name = pdf_path.split("/")[-1].split("\\")[-1]
    total_pages = len(doc)
    
    all_chunks = []
    chunk_counter = 1

    for page_num in range(total_pages):
        page = doc[page_num]
        # Extract text blocks line by line
        text_lines = page.get_text("text").splitlines()
        text_lines = [line.strip() for line in text_lines if line.strip()]  # Clean empty lines
        
        if not text_lines:
            continue
            
        total_lines = len(text_lines)
        step = chunk_size - overlap
        
        for idx in range(0, total_lines, step if step > 0 else chunk_size):
            line_chunk = text_lines[idx : idx + chunk_size]
            if not line_chunk:
                continue
                
            chunk_text = " ".join(line_chunk)
            line_start = idx + 1
            line_end = min(idx + len(line_chunk), total_lines)
            
            chunk_obj = {
                "chunk_id": f"{file_name}_p{page_num + 1}_c{chunk_counter}",
                "document_metadata": {
                    "file_name": file_name,
                    "total_pages": total_pages,
                    "upload_timestamp": datetime.now().isoformat()
                },
                "chunk_coordinates": {
                    "page_number": page_num + 1,
                    "line_start": line_start,
                    "line_end": line_end,
                    "character_count": len(chunk_text)
                },
                "content_payload": {
                    "raw_text": chunk_text
                }
            }
            
            all_chunks.append(chunk_obj)
            chunk_counter += 1

    doc.close()
    return all_chunks


# Quick Test Run
if __name__ == "__main__":
    test_pdf = "sample.pdf"  # Apni kisi local PDF file ka path yahan daalein
    try:
        parsed_data = parse_pdf_with_coordinates(test_pdf)
        print(f"Successfully extracted {len(parsed_data)} chunks.\n")
        print("Sample Chunk Output:")
        print(json.dumps(parsed_data[0], indent=2))
    except Exception as e:
        print(f"Test run ke liye '{test_pdf}' file path replace kijiye. Error: {e}")