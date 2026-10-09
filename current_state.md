# MindGate AI — Current State

## Project Status

MindGate AI is a 100% local/offline PDF document intelligence and RAG engine.

Primary goal:

PDF → Parse → Chunk → Store → Retrieve → Grounded Local SLM Answer → Citation

The system is designed to answer questions using the uploaded document instead of relying on the model's external/pretrained knowledge.

---

# Current Architecture

```text
PDF
 ↓
PyMuPDF Parser
 ↓
Chunks + Metadata
 ↓
Chunk Cache
 ↓
ChromaDB Persistent Storage
 ↓
Local BM25 Retrieval
 ↓
Strict Active-PDF Filtering
 ↓
Grounded Context
 ↓
Local Ollama SLM
 ↓
Answer + PDF/Page/Line Citations
```
