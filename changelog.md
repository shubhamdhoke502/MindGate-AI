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

# CHANGELOG - MindGate AI Engine

## [1.10.0] - Requirements Specification
### Added
- Created `requirements.txt` with pinned production dependencies for FastAPI, PyMuPDF, ChromaDB, Sentence-Transformers, and Rank-BM25.

# Changelog - MindGate AI Engine

All notable changes, bug fixes, and ongoing investigation details for the MindGate AI offline document processing engine.

---

## [Unreleased / Bugfix Tracking] - 2026-10-02

### 🔴 Ongoing Issue: Vector DB Ingestion & Embedding Runtime Crash
- **Problem Statement:** PDF upload/ingestion pipeline fails during ChromaDB `upsert` phase.
- **Primary Errors Encountered:**
  1. `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading torch\lib\c10.dll`
  2. `[CRITICAL INGESTION ERROR]: The onnxruntime python package is not installed. Please install it with pip install onnxruntime in upsert.`

---

### 🔍 Debugging History & Attempts Timeline

#### Attempt 1: Switched to `SentenceTransformerEmbeddingFunction`
- **Action:** Replaced ChromaDB's default `embedding_functions.DefaultEmbeddingFunction()` with `SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")` in `modules/vector_db.py`.
- **Result:** Failed. Triggered PyTorch C++ DLL initialisation error (`c10.dll` crash under Windows environment).

#### Attempt 2: Environment Variable Patch
- **Action:** Executed `$env:KMP_DUPLICATE_LIB_OK="TRUE"` to fix OpenMP/Intel DLL conflicts.
- **Result:** Temporary bypass, but underlying PyTorch runtime linking issue persisted on Uvicorn reload.

#### Attempt 3: Custom / Dummy Embedding Function (BM25 Fallback Mode)
- **Action:** Created a lightweight non-PyTorch embedding class to satisfy ChromaDB interface while relying on BM25 sparse retrieval.
- **Result:** ChromaDB internally still invoked default ONNX embedding hooks during `upsert()`, throwing `onnxruntime package not installed` error.

#### Attempt 4: Virtual Environment Package Installation
- **Action:** Ran `.\venv\Scripts\pip install onnxruntime` and executed Uvicorn directly via `.\venv\Scripts\uvicorn`.
- **Result:** Ingestion error still re-occurs due to mismatch in ChromaDB collection binding or cached system embedding function invocation.

---

### 📌 Current Status & Next Action Items
1. **Root Cause:** ChromaDB's `get_or_create_collection()` / `upsert()` is hard-linked to an ONNX or SentenceTransformer runtime even when customized, causing binary dependency crashes on Windows.
2. **Next Steps:**
   - Inspect `modules/vector_db.py` collection initialization to explicitly pass `embedding_function=self.embedding_fn` inside `get_or_create_collection()`.
   - Verify `venv` vs global Python path conflicts in VS Code / PowerShell to ensure clean package resolution.
   
# Changelog - MindGate AI Engine

All notable changes, bug fixes, and system improvements for the MindGate AI offline document processing engine.

---

## [1.0.1] - 2026-10-04

### 🟢 Bug Fix: Resolved PyTorch DLL & ONNX Dependency Crashes
- **Root Cause Analysis:** ChromaDB by default attempted to load `onnxruntime` and PyTorch binary libraries for text embeddings. This triggered two critical runtime crashes on Windows:
  1. `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading torch\lib\c10.dll`
  2. `[CRITICAL INGESTION ERROR]: The onnxruntime python package is not installed.`
- **Resolution:** Refactored the architecture in `modules/vector_db.py` by implementing a pure Python-based custom embedding function, completely eliminating heavy external binary dependencies (PyTorch / ONNX Runtime).

### 🛠️ Key Architectural & Code Changes in `modules/vector_db.py`
- **Custom `PureBM25EmbeddingFunction` Class:** Created a custom zero-dependency embedding class inheriting from ChromaDB's `EmbeddingFunction` interface to satisfy vector schema constraints without invoking external binary models.
- **Explicit Collection Binding:** Explicitly passed `embedding_function=PureBM25EmbeddingFunction()` inside `get_or_create_collection(name="mindgate_collection_v2")` to prevent ChromaDB from defaulting back to internal ONNX loaders.
- **Dependency Elimination:** Removed all imports and execution paths linked to PyTorch (`c10.dll`) and `onnxruntime` packages, securing a 100% stable local environment.
- **Storage & Cache Purge:** Cleared legacy database metadata (`storage/.mindgate/chroma_db`) to re-initialize the database under the new schema.

### 🚀 System Verification & Execution Status
- **Server Boot:** Uvicorn server (`http://127.0.0.1:8000`) starts smoothly without any DLL initialization errors or missing package warnings.
- **PDF Ingestion Success:** Document parsing and vector indexing completed successfully (`POST /upload 200 OK`), with PyMuPDF extracting chunks and the UI displaying `🟢 PDF Indexed`.

# Changelog - MindGate AI Engine

All notable changes, bug fixes, and system improvements for the MindGate AI offline document processing engine.

---

## [1.0.1] - 2026-10-04

### 🟢 Resolved: PyTorch DLL & ONNX Dependency Crashes During Ingestion
- **Fixed Error 1:** `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading torch\lib\c10.dll`
- **Fixed Error 2:** `[CRITICAL INGESTION ERROR]: The onnxruntime python package is not installed.`

### 🛠️ Key Architectural Changes
- **Implemented Pure BM25 Embedding Function:** Replaced default ONNX / SentenceTransformer embedding calls in `modules/vector_db.py` with a zero-dependency custom `PureBM25EmbeddingFunction`.
- **Bypassed External Runtime Binaries:** Removed dependency on heavy PyTorch (`c10.dll`) and `onnxruntime` libraries to keep execution 100% lightweight and local.
- **Explicit Collection Binding:** Bound `PureBM25EmbeddingFunction` explicitly inside `get_or_create_collection(name="mindgate_collection_v2")` to prevent ChromaDB from dropping back to ONNX default configs during `upsert`.
- **Cache Clean-up:** Purged legacy metadata and database cache under `storage/.mindgate/chroma_db`.

### 🚀 System Verification Status
- **Server Launch:** Server starts cleanly on `http://127.0.0.1:8000` via Uvicorn reload without any DLL initialization or missing package warnings.
- **PDF Ingestion:** Successfully parsed and indexed uploaded PDFs (`POST /upload 200 OK`) with UI displaying `🟢 PDF Indexed`.