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