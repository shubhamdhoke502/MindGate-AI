# MindGate AI — Architecture

## Runtime flow

```text
Browser UI (served locally)
    ├── Upload: PDF / TXT / Markdown / DOCX
    │      └── bounded streaming upload → workspace staging → parser → chunk cache
    │                                                 └── workspace-local SQLite FTS5 index
    ├── Query → workspace + optional file filter → local BM25 ranking (top 2)
    │                                          └── local Ollama → answer + source metadata
    └── @ recall → workspace-local JSON memory
```

The application server is `ui/app_server.py`; the engine is in `modules/app.py`.
Each workspace has its own persistent SQLite database, chunk cache, chat history,
and JSON memory below `workspaces/<workspace_id>/`. Workspace names are limited
to ASCII letters, digits, underscores, and hyphens. Uploads are restricted to
50 MiB per file, 10 documents per workspace, and 500 PDF pages per workspace.

## Retrieval and citations

`modules/parser.py` extracts PDF text blocks with page, extracted-line, and
bounding-box metadata; plain text, Markdown, and DOCX paragraphs use logical
line/paragraph ranges. Chunks contain up to eight lines with two lines of
overlap. SQLite FTS5 persists local document text and metadata and ranks matches
with BM25. Metadata includes file size, PDF page count, ingestion timestamp,
chunk character count, and a script-based language label. BM25 is lexical
matching, not a multilingual semantic embedding model. A requested active document is
always a hard filter: an empty result does not trigger a search of other files.
Hindi and Marathi text is normalized and exact-script searchable, but an English
query does not automatically retrieve a Hindi/Marathi passage (and vice versa).

The engine injects only retrieved text and source coordinates into the local
Ollama prompt. The UI displays source citations as links to the cited PDF page.
Summary, key-points, and action-items chips submit grounded queries. Successful
answers and source coordinates are saved to workspace memory and associated
with the active document in persistent chat history. The archived "key fact" is
the answer text itself; no separate fact-extraction model runs. Ollama failures are
surfaced to the caller; the application does not fabricate an answer when
local inference fails.

## Privacy and storage

Runtime services bind to loopback. The UI has no remote asset dependencies.
Uploads, parsed chunks, indexes, history, and
memory remain on local disk. Model inference uses the local Ollama HTTP
endpoint; users must install/download a model separately before going offline.

## Explicitly not implemented yet

- BGE-based multilingual semantic / cross-lingual retrieval and OCR for scanned PDFs.
- In-PDF line highlighting (citations open the native viewer at the cited page).
- Automatic answer-to-memory highlight extraction and `.mgdb` idle compression.
- Tauri packaging or a React frontend; the current interface is a local HTML UI.

These are product goals, not claims about current runtime behavior. See
[current_state.md](./current_state.md) for the verified scope and limitations.
