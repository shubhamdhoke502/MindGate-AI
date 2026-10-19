# MindGate AI — Current State

## Implemented

- Local FastAPI service and browser UI served from the same origin.
- PDF, TXT, Markdown, and DOCX text ingestion with bounded uploads and local caching.
- Per-workspace SQLite FTS5 persistence, chat history, and JSON memory. Source
  metadata includes file size, PDF page count, ingestion time, chunk character
  count, and a script-based language label.
- Local BM25 retrieval with a strict active-document filter and two-result context limit.
- Local Ollama answer generation, SSE responses, `@` memory recall, and page/line citations.
- Answer-and-citation persistence in the local memory container; `@` can recall saved answers.
- Restored workspace document lists and per-document chat history in the UI; PDF sources link to the cited page.
- Summary, key-points, and action-items prompt chips.
- Windows launcher script for an already installed Python environment.
- Workspace, file-name, upload-size, document-count, and total-PDF-page validation.
- No runtime CDN assets or document-index cloud dependencies.
- Loopback-only Ollama configuration and cross-origin mutation rejection.

## Verification status

Run the full test suite from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -v
```

The automated suite covers PDF coordinates and metadata, Hindi/Marathi script
handling, DOCX parsing, FTS5 ranking and workspace isolation, stale-index
replacement, cache keys, concurrent history writes, API validation/resource
limits, cross-origin rejection, Ollama error handling, and
upload/index/chat/stream/history flows. The full root-level discovery command
currently passes **21 tests**. It does not require Ollama.

Final static checks also passed: Python `compileall`, parsing the UI's inline
JavaScript with Node.js, `pip check`, and `git diff --check`.

Live Ollama verification is run separately because it depends on the local
model service. The latest recorded run uploaded a disposable policy document
over HTTP, indexed it, and queried local `phi3:mini`; the streamed answer
contained the documented 14-day period and the response cited page 1, line 1.
The synchronous HTTP path returned the same fact and citation, an unrelated
question returned the no-context response, `@14` recalled both archived
answers, and six per-document chat messages were persisted. These are
functional checks, not a general accuracy benchmark.

The Windows batch launcher was also started and its `/health` endpoint returned
HTTP 200. Browser inspection confirmed that both existing workspace documents
restore in the sidebar and all three smart-prompt chips render. Selecting a
document from the `@` suggestion list was also verified to activate that
document and retain the completed mention in the prompt.

A generated 100-page text PDF parsed and indexed into 200 chunks in **0.178
seconds** on this Windows environment, and retrieval returned two chunks. This
synthetic document is not representative of scanned, image-heavy, or unusually
complex PDFs; it is not a production performance guarantee.

## Product requirement audit

| Product requirement | Status |
|---|---|
| Local document processing and local model inference | Implemented; Ollama endpoint is restricted to loopback. |
| PDF extraction with page, line range, bounding box, and document metadata | Implemented for text-based PDFs; coordinates follow extracted text blocks. |
| 5–8 line chunks with 1–2 lines of overlap | Implemented as up to 8 extracted lines with 2-line overlap within each text block. |
| EN/HI/MR text normalization and exact-script retrieval | Implemented with Unicode normalization, script labels, and FTS5 lexical matching; cross-lingual semantic search is not implemented. |
| Top-2 local retrieval and strict active-document filtering | Implemented and covered by regression tests. Retrieval uses SQLite FTS5/BM25 instead of Chroma/BGE because Chroma native writes crashed in this Windows environment. |
| Grounded local Ollama answers and source coordinates | Implemented; local success path tested. An LLM's factual correctness is not guaranteed by code alone. |
| Clickable PDF citation focusing the exact line | Partial: links open the native PDF viewer at the cited page; exact-line highlighting/in-app PDF viewer is not implemented. |
| Summary, key-point, and action-item prompts | Implemented as UI chips; output quality depends on retrieved passages and local model. |
| Workspace limits: 10 files, 500 PDF pages | Implemented as upload checks (10 supported documents and 500 total PDF pages per workspace). |
| 100 pages indexed in 3–5 seconds; hard 2 GB operational RAM cap | A synthetic 100-page text PDF indexed in 0.178 seconds on this host; representative production PDFs were not benchmarked. A 2 GB process/model memory cap is not enforced. |
| `@` memory, cross-chat recall, answer archival | Implemented in workspace JSON; generated answer text and source coordinates are archived, not independently fact-extracted. |
| Idle `.mgdb` compression / under-100-MB target | Not implemented. |
| Tauri/React shell and preconfigured portable executable/archive | Not implemented; a Windows batch launcher for an already installed environment is provided. |
| OCR for scanned PDFs | Not implemented; empty-text documents are rejected. |

Ollama and a local model must be installed separately. Existing legacy Chroma
files are left untouched; the application currently reads/writes the SQLite
FTS5 database instead.

See [README.md](./README.md) for setup and run instructions and
[architecture.md](./architecture.md) for component boundaries.
