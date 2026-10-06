# CURRENT_STATE.md - MindGate AI Engine Architecture & Status

## 1. Core Goal & System Overview

MindGate AI is designed as a 100% offline, high-fidelity local document processing engine. It aims to deliver accurate, fact-based answers grounded directly in uploaded PDF documents with exact line and page citations, avoiding generative filler or dummy hardcoded responses.

---

## 2. Active Technical Issues (Observed in Runtime)

### A. Persistent LLM Hallucinations

- **Symptom:** After uploading a PDF, when asking general or broad queries (e.g., _"is me kya likha hai"_, _"summary of this document"_), the model generates random, hallucinated topics (e.g., _Home Security Systems_, _Healthy Lifestyle & Agriculture_, _Climate Change_) instead of discussing the actual PDF content[cite: 2, 3].
- **Root Cause:**
  1. **Vector Store Miss / Empty Context:** Broad or generic queries return zero similarity matches from vector search[cite: 3].
  2. **Ungrounded Model Fallback:** When context blocks are empty or mismatched, the local SLM (`phi3:mini`) falls back on its pre-trained parametric memory to generate fictional answers[cite: 3].
  3. **Legacy Vector DB Cache Contamination:** Unpurged ChromaDB cache directories retain metadata and text blocks from previous test documents[cite: 2].

### B. Prompt Echo Leakage

- **Symptom:** Output text contains repeated prompt structure tags and markers (e.g., `"Question: isme kya hai"`) directly in the rendered UI response[cite: 3].
- **Root Cause:** Structural delimiters (`<|system|>`, `<|user|>`, `<|assistant|>`) or `USER QUERY:` markers in the system prompt template are misparsed by Ollama, causing the model to mirror system wrapper text[cite: 3].

### C. Active Document Metadata Filter Bypass

- **Symptom:** Even when a specific PDF is selected in the UI sidebar, ChromaDB fails to strictly constrain retrieval to that document's metadata[cite: 2, 3].
- **Root Cause:** Inconsistency in metadata key naming (`file_name` vs `filename`) and loose `where` clause execution in the vector database query layer[cite: 2, 3].

---

## 3. Recent Code Modifications

### 1. `app.py` Updates

- **Fallback Chunk Retrieval (`retrieve_relevant_chunks`):** Added a fallback mechanism that retrieves the first 8 chunks directly from disk cache or ChromaDB collection when vector similarity search returns no results[cite: 3].
- **Prompt Isolation (`_build_prompt_and_citations`):** Updated system prompt structuring to isolate context blocks and prevent prompt marker echoing[cite: 3].
- **SSE Streaming Payload (`ask_stream`):** Configured real-time token streaming to push citation metadata as the initial SSE event payload[cite: 1].

### 2. `modules/parser.py` Updates

- **Unicode NFC Normalization (`clean_devnagari_and_text`):** Implemented text normalization to handle Devnagari script (Hindi/Marathi) and eliminate control noise[cite: 1].
- **Layout & Table Extraction (`extract_tables_from_page`):** Integrated PyMuPDF layout analysis and table detection to extract structured Markdown tables with precise bounding box (`bbox`) coordinates[cite: 1].
- **Production Class Wrapper:** Added `parse_pdf_production_grade()` and `LayoutAwareParser` wrappers to handle legacy and production import signatures seamlessly[cite: 1].

---

## 4. Pending Action Items

1. **Complete Database Cache Purge:**
   - Delete local `.mindgate/` or `storage/` ChromaDB directories to completely clear legacy vector indexes[cite: 2, 3].
2. **Enforce Strict Zero-Context Refusal Prompting:**
   - Update the system prompt so the SLM explicitly refuses to answer if context blocks are empty, preventing open-ended hallucinations[cite: 3].
3. **Standardize Metadata Key Consistency:**
   - Enforce uniform use of `file_name` across `parser.py`, `vector_db.py`, `chunk_cache.py`, and `app.py`[cite: 1, 2, 3].
