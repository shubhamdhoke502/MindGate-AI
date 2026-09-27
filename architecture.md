# MindGate AI — System Architecture Blueprint

## 1. Project Overview & Core Principles
* **Goal:** 100% Offline, Privacy-First Document Engine with Grounded Page/Line Citations.
* **Core Principles:** Zero-cloud dependency, strict factual outputs (no conversational filler), multilingual (EN, HI, MR), and zero RAM/Disk footprint bloat.
* **Monetization Strategy:** Free GitHub repository for developers + Paid Portable Setup Service (₹500–₹1,000) for non-technical users.

## 2. Technical Stack
* **Language:** Python 3.10/3.11 (Backend), JS/React (Frontend), Rust (Tauri Shell)
* **Parser:** PyMuPDF (`fitz`) for line and page coordinate extraction.
* **Vector DB:** ChromaDB (Local Embedded).
* **Embeddings:** BGE-M3 / BGE-Small (Multilingual support).
* **Local LLM Execution:** Ollama / `llama.cpp` (Phi-3-Mini, Qwen-1.5B, Llama-3-8B).
* **Memory & Storage:** JSON Container Box (`@` Command Recall) + Background ZIP Compression (`.mgdb`).

## 3. Modular File Structure Strategy
```text
mindgate-engine/
├── architecture.md
├── current_state.md
├── changelog.md
├── modules/
│   ├── parser.py        # Module 1: Line/Page extraction & metadata schema
│   ├── vector_db.py     # Module 2: ChromaDB indexing & ZIP compression
│   ├── container.py     # Module 3: Local JSON Memory Box & @ command
│   └── app.py           # Module 4: SLM orchestration & citation engine
└── scripts/
    └── run_portable.bat # 1-Click launcher script for non-tech users