# MindGate AI — Change Log

## [Unreleased] - 2026-09-27

### Added
- Created baseline System Architecture documentation (`architecture.md`).
- Initialized state tracking file (`current_state.md`).
- Set up project change log (`changelog.md`).
- Drafted initial starter code for `modules/parser.py` (PyMuPDF extraction with line-level metadata coordinates).

# MindGate AI — Change Log

## [v0.1.0-alpha] - 2026-09-27

### Added
- Successfully integrated `modules/parser.py` to handle PDF text extraction line-by-line using PyMuPDF (`fitz`).
- Implemented 6-line paragraph chunking with 2-line context overlap.
- Generated structured JSON metadata output containing `chunk_id`, `file_name`, `page_number`, `line_start`, and `line_end` coordinates.

### Changed
- Updated `current_state.md` to reflect the completion of Module 1.

# MindGate AI — Change Log

## [v0.2.0-alpha] - 2026-09-27

### Added
- Integrated `modules/vector_db.py` featuring `VectorStoreManager` for local ChromaDB operations and sentence-transformers embeddings.
- Implemented `StorageCompressor` with `.mgdb` (ZIP format) compression and decompression logic to optimize disk footprint.
- Added metadata filtering support for isolated multi-PDF vector querying.

### Changed
- Updated `current_state.md` to mark Module 2 as completed.

# MindGate AI — Change Log

## [v0.3.0-alpha] - 2026-09-27

### Added
- Integrated `modules/container.py` for local `.mindgate/memory.json` memory persistence.
- Implemented `register_document` and `save_highlight` methods to store grounded citations and key insights.
- Implemented `handle_at_trigger` method to handle `@` recall commands and return matching documents and highlights for UI integration.

### Changed
- Updated `current_state.md` to mark Module 3 as completed.

# MindGate AI — Change Log

## [v0.4.0-alpha] - 2026-09-27

### Added
- Created `modules/app.py` implementing `MindGateEngine` class to orchestrate full RAG workflow.
- Connected `PyMuPDF` parser, ChromaDB vector retrieval, and `Ollama` local SLM runtime.
- Implemented grounded citation generation with explicit `[Source: Doc | Page X, Lines Y-Z]` formatting.
- Integrated auto-synchronization between LLM output and `.mindgate/memory.json` (Container Box).

### Changed
- Updated `current_state.md` to reflect completion of all 4 core Python backend modules.

# MindGate AI — Change Log

## [v0.5.0-alpha] - 2026-09-28

### Added
- Created `ui/app_server.py` using FastAPI to serve `/api/ingest` and `/api/ask` REST endpoints connecting frontend to `MindGateEngine`.
- Created `ui/index.html` implementing clean dark-mode GUI with PDF ingestion, grounded citation rendering, and `@` memory recall popup interface.

### Changed
- Updated `current_state.md` to reflect completion of the UI Server and Desktop Frontend interface layer.

# MindGate AI — Change Log

## [v0.6.0-alpha] - 2026-09-28

### Added
- Created `scripts/init_storage.py` to auto-provision local `storage/` and `.mindgate/memory.json` workspace structure on first boot.

### Changed
- Updated `current_state.md` reflecting completion of storage management setup.

# MindGate AI — Change Log

## [v0.7.0-alpha] - 2026-09-28

### Added
- Created `modules/chat_history.py` for persistent user/assistant message logging in `.mindgate/chat_history.json`.
- Created `modules/chunk_cache.py` to cache extracted PDF chunks, coordinates, and page metadata in `.mindgate/chunks_cache/` using file hashing (MD5).

### Changed
- Updated `current_state.md` to reflect new chat persistence and caching architecture modules.

# MindGate AI — Change Log

## [v0.9.2-alpha] - 2026-09-28

### Refactored
- Upgraded `modules/parser.py` with Devnagari script normalization (`unicodedata.normalize('NFC')`), broken glyph regex fixing, and exact PDF Bounding Box coordinates (`bbox`: `[x0, y0, x1, y1]`) stored in chunk metadata.
- Updated `current_state.md` to reflect completion of deep parser refactoring.