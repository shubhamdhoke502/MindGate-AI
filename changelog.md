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