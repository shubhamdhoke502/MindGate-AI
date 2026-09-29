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

# MindGate AI — Change Log

## [v0.9.3-alpha] - 2026-09-28

### Refactored
- Upgraded `modules/vector_db.py` with `StorageCompressor` class for automated background idle compression (`.mgdb` ZIP file) after 30 seconds of inactivity.
- Enforced strict dynamic `file_name` metadata filtering in ChromaDB queries to prevent multi-document context pollution.
- Updated `current_state.md` to track vector engine completion.

# MindGate AI — Change Log

## [v1.1.0-prod] - 2026-09-29

### Added
- Upgraded `modules/parser.py` with PyMuPDF Table Finder engine (`page.find_tables()`), converting structured tables directly into clean Markdown format.
- Implemented Multi-Column layout sorting (Top-to-Bottom, Left-to-Right) to preserve context in complex document formats.
- Updated `current_state.md` for Step 1 completion.

# MindGate AI — Change Log

## [v1.2.0-prod] - 2026-09-29

### Added
- Integrated BM25 Sparse Keyword Search Engine (`BM25SearchEngine`) in `modules/vector_db.py`.
- Implemented Reciprocal Rank Fusion (RRF) algorithm to merge dense vector embeddings with sparse keyword search scores, fixing exact numeric and serial number search misses.
- Updated `current_state.md` for Step 2 completion.

# MindGate AI — Change Log

## [v1.3.0-prod] - 2026-09-29

### Added
- Integrated Server-Sent Events (SSE) streaming endpoint `/api/ask_stream` in `ui/app_server.py`.
- Added generator-based token streaming (`ask_stream`) in `modules/app.py`.
- Implemented stream buffer decoder in `ui/index.html` for typewriter-style real-time token rendering.

# CHANGELOG - MindGate AI Engine

## [1.5.0] - Step 5: Multi-Tenancy Workspace & Session Isolation
### Added
- **`modules/workspace.py`**: Added `WorkspaceManager` class for session-based workspace directory creation and lifecycle management.
- Dynamic project structure initialization (`create_project_structure`) for isolated storage per user session.
- System directory cleanup and path safety using cross-platform `pathlib.Path`.

## [1.4.0] - Step 4: Resource Management & Memory Control
### Added
- Integrated `SystemResourceManager` into `MindGateEngine`.
- Configured automatic RAM garbage collection with `self.resource_mgr.flush_memory()` at the end of both `ask()` and `ask_stream()` methods.

## [1.3.0] - Step 3: SSE Streaming Engine
### Added
- Real-time token streaming (`ask_stream`) via Server-Sent Events (SSE) using Ollama local instance.

## [1.2.0] - Step 2: RRF Hybrid Search Engine
### Added
- Combined BM25 lexical search with ChromaDB dense vector search using Reciprocal Rank Fusion.

## [1.1.0] - Step 1: Layout-Aware Parser
### Added
- Production-grade PyMuPDF parser (`parser.py`) for extracting text, table structures, and bounding box coordinates.

# CHANGELOG - MindGate AI Engine

## [1.6.0] - Step 6: REST API Server Implementation
### Added
- Created `app_server.py` using FastAPI for orchestrating backend routes.
- Added `/workspace/init` endpoint for dynamic workspace partitioning.
- Added `/upload` endpoint for PDF ingestion per session.
- Added `/chat` and `/chat/stream` SSE streaming endpoints.
- Added `/system/flush` endpoint for dynamic RAM/VRAM resource management.

# CHANGELOG - MindGate AI Engine

## [1.8.2] - Dynamic Workspace Chat Logging
### Modified
- Updated `modules/chat_history.py` to dynamically accept `workspace_dir` to ensure chat logs stay isolated per user session.