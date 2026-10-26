<img width="1920" height="1080" alt="Screenshot 2026-10-26 222314" src="https://github.com/user-attachments/assets/e947a8ae-74d3-4479-b207-c3030faad310" />
# MindGate AI
### System Architecture & Local AI Document Research Engine

**Created by:** Shubham Dhoke  
**Project Type:** Privacy-First, Local AI Document Research Engine  
**Deployment Model:** Standalone, Local-First Application

## 1. Project Overview

MindGate AI is a privacy-focused document research and conversation engine designed to extract, index, search, and analyze complex documents using locally running AI models.

The system combines document parsing, intelligent text chunking, local information retrieval, AI-powered question answering, page-and-line citations, multilingual document search, and persistent local memory in a unified application.

Its core objective is to provide evidence-grounded answers from documents while keeping document contents and workspace data on the user's own computer, without depending on cloud AI APIs.

## 2. Core Capabilities

- **Local & Private Processing:** Documents, indexes, conversation history, and memory are stored locally.
- **Document Intelligence:** Ask questions about PDF documents and retrieve relevant information.
- **Grounded AI Responses:** Generate answers using retrieved document content instead of relying exclusively on the model's general knowledge.
- **Page & Line Citations:** Associate retrieved passages with their original document, page number, and line range.
- **Multilingual Research:** Designed for English, Hindi (हिंदी), and Marathi (मराठी), including Devanagari text processing and cross-lingual retrieval.
- **Workspace Isolation:** Maintain separate document indexes, conversation data, and memory for individual workspaces.
- **Smart Memory:** The `@ Container Box` system stores important facts, document references, dates, and extracted highlights for later recall.
- **Local Storage Optimization:** The proposed `.mgdb` archive format packages workspace data for more compact storage.
- **Local AI Inference:** Use locally running language models through Ollama or llama.cpp.

## 3. System Architecture

MindGate AI follows a modular architecture that separates document ingestion, text processing, retrieval, AI inference, memory management, and the user interface.

```text
                    MINDGATE AI
                         |
          +--------------+--------------+
          |                             |
     User Interface                Local Storage
          |                             |
          v                             v
     FastAPI Backend             Workspace Data
          |
          v
    Document Processing
          |
          v
    PyMuPDF Extraction
          |
          v
    Line & Page Mapping
          |
          v
      Smart Chunking
          |
          v
   Local Embedding Generation
          |
          v
    Local Document Index
          |
          v
      RAG Query Engine
          |
          v
    Relevant Context Retrieval
          |
          v
      Local AI Inference
          |
          v
    Answer + Source Citations
```

### Architecture Components

**Document Processing Layer**

Extracts text from PDF documents while preserving page information, line references, and text coordinates.

**Indexing and Retrieval Layer**

Divides document text into manageable chunks, indexes the extracted content, and retrieves relevant passages in response to user queries.

**RAG Engine**

Combines retrieved document passages with the user's question to provide context-grounded answers through a local language model.

**Citation Engine**

Associates responses with document names, page numbers, and line ranges, enabling source verification and citation-based navigation.

**Memory Engine**

Maintains persistent workspace memory containing important facts, references, and previously extracted information.

**Storage Engine**

Manages local documents, indexes, workspace data, chat history, and memory files.

**Application Interface**

Provides the interface for document uploads, conversations, source references, and workspace interactions.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Backend Runtime | Python 3.10 / 3.11 |
| API Framework | FastAPI |
| PDF Processing | PyMuPDF (`fitz`) |
| Local AI Runtime | Ollama / llama.cpp |
| Language Models | Phi-3 Mini, Qwen 1.5B, Llama 3 8B |
| Embedding Models | BGE-M3 / BGE-Small-EN |
| Vector Database | ChromaDB |
| Local Memory | JSON / SQLite |
| Archive Compression | ZIP / zstandard |
| Desktop Interface Design | Tauri + React |
| Workspace Storage | Local SSD / HDD |

The stack describes the system architecture specified for MindGate AI. Individual components may vary between the current implementation and the complete architecture.

## 5. Document Ingestion Pipeline

The document ingestion pipeline converts uploaded PDFs into structured, searchable local data.

1. **PDF Input:** The user imports a document through the application interface.
2. **Text Extraction:** PyMuPDF extracts document text and page information.
3. **Coordinate Mapping:** Extracted text is associated with page numbers, line ranges, and bounding-box coordinates.
4. **Smart Chunking:** Text is divided into paragraph chunks containing approximately 5–8 lines, with 1–2 overlapping lines to preserve context.
5. **Embedding Generation:** A local embedding model converts chunks into numerical vector representations.
6. **Indexing:** Vectors and associated metadata are stored in the local document index.
7. **Workspace Association:** Document records and indexed passages remain associated with their respective workspace and source document.

## 6. RAG-Based Question Answering

MindGate AI uses a Retrieval-Augmented Generation (RAG) architecture to connect user questions with relevant document evidence.

```text
User Question
     |
     v
Query Processing
     |
     v
Document Retrieval
     |
     v
Top Relevant Passages
     |
     v
Context + Source Metadata
     |
     v
Local Language Model
     |
     v
Grounded Answer
     |
     v
Page / Line Citation
```

The architecture specifies retrieval of the two most relevant chunks, followed by context injection into the local model.

Responses are designed to be direct, factual, and traceable to their supporting document passages.

**Citation format:**

`[Source: Document.pdf | Page 4, Lines 12–18]`

Citations associate an answer with its source location and are intended to support navigation to the relevant passage in the document viewer.

## 7. Document Metadata Architecture

Each indexed chunk carries metadata that connects its text to the original source.

```json
{
  "chunk_id": "pdf_doc_001_p04_c02",
  "document_metadata": {
    "file_name": "Audit_Report.pdf",
    "file_size_bytes": 2048500,
    "total_pages": 45,
    "upload_timestamp": "2026-09-27T18:03:41"
  },
  "chunk_coordinates": {
    "page_number": 4,
    "line_start": 12,
    "line_end": 18,
    "character_count": 420
  },
  "content_payload": {
    "raw_text": "Extracted document passage...",
    "language_detected": "en"
  }
}
```

The metadata structure supports:

- Source document identification.
- Page and line-level traceability.
- Document-specific retrieval filters.
- Separation of passages belonging to different PDFs.
- Synchronization of important source references with workspace memory.

## 8. @ Container Box — Persistent Memory

The `@ Container Box` is the proposed local memory subsystem for preserving useful information across conversations.

Instead of repeatedly loading every previous conversation into the active context, the system maintains structured memory on local storage.

### Memory Functions

- Store key facts, dates, entities, and extracted document highlights.
- Preserve PDF names and page-and-line references.
- Recall saved information through the `@` command.
- Associate important references with individual workspaces.
- Reuse relevant memory in new conversations.

### Memory Structure

```json
{
  "container_id": "workspace_001",
  "documents": [
    "Contract_2026.pdf",
    "Audit_Report.pdf"
  ],
  "memory_highlights": [
    {
      "pdf_name": "Contract_2026.pdf",
      "page": 12,
      "line_range": "14-22",
      "key_fact": "Termination clause duration is 30 days notice."
    }
  ]
}
```

Memory data is stored locally under the workspace's `.mindgate` directory.

## 9. Local Storage & .mgdb Architecture

The storage subsystem manages workspace documents, parsed text, indexes, conversation history, and persistent memory.

The proposed `.mgdb` format is a compressed workspace archive intended to reduce disk usage while preserving the data required to restore a workspace.

Its design includes:

- ZIP-based or zstandard-based compression.
- Background archiving during idle periods or application shutdown.
- Workspace-level data organization.
- Extraction of archived data when it is required for use.

The original architecture targets substantial storage reduction through compression; actual savings depend on the document and index data.

## 10. Modular Codebase Structure

```text
MindGate AI/
│
├── __pycache__/
│
├── .mindgate/
│
├── modules/
│   ├── __pycache__/
│   │
│   ├── app.py
│   ├── chat_history.py
│   ├── chunk_cache.py
│   ├── container.py
│   ├── parser.py
│   ├── resource_manager.py
│   ├── vector_db.py
│   └── workspace.py
│
├── scripts/
│   ├── init_storage.py
│   └── run_portable.bat
│
├── tests/
│
└── ui/
    ├── __pycache__/
    ├── app_server.py
    └── index.html
```

### Module Responsibilities

- **app.py** — Core application logic and AI orchestration.
- **chat_history.py** — Conversation history management.
- **chunk_cache.py** — Document chunk caching.
- **container.py** — Persistent memory and `@ Container Box` logic.
- **parser.py** — PDF text extraction and page/line mapping.
- **resource_manager.py** — Resource and memory management.
- **vector_db.py** — Document indexing and retrieval storage.
- **workspace.py** — Workspace management and isolation.
- **init_storage.py** — Local storage initialization.
- **run_portable.bat** — Windows portable startup script.
- **app_server.py** — FastAPI server and backend endpoints.
- **index.html** — Application user interface.
- **tests/** — Project test suite.
- **.mindgate/** — Local application data and workspace storage.

## 11. Hardware & Resource Targets

| Resource | Original Design Target |
|---|---|
| Minimum RAM | 4 GB |
| Recommended RAM for Llama 3 8B | 8 GB |
| CPU | Dual-core x86-64 / ARM64 |
| Available Disk Space | 3 GB |
| Maximum PDFs per Workspace | 10 |
| Maximum Pages per Workspace | 500 |
| Approximate Parsing Target | 100 pages in 3–5 seconds |
| Target Active RAM Buffer | Maximum 2 GB |

These are targets defined in the original architecture specification, not independently verified performance guarantees.

## 12. Running MindGate AI on Windows

Open Command Prompt in the project root directory.

### Create a virtual environment

```cmd
python -m venv venv
```

### Activate the environment

```cmd
venv\Scripts\activate
```

### Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare the local AI model

Install Ollama separately if it is not already installed, then run:

```cmd
ollama pull phi3:mini
```

Verify the installed model:

```cmd
ollama list
```

### Start the application

```cmd
python -m uvicorn ui.app_server:app --host 127.0.0.1 --port 8000
```

Open the local application in your browser at:

`http://127.0.0.1:8000`

**Note:** These commands assume that the project has the required dependencies and that `ui.app_server` exposes the FastAPI application as `app`.

## 13. Project Vision

MindGate AI represents a local-first approach to AI-powered document research: private document processing, evidence-grounded answers, source-level traceability, multilingual retrieval, and persistent context without mandatory cloud AI services.

The architecture brings together document intelligence, local language models, retrieval systems, memory management, and compact local storage in one modular research engine.

**Project:** MindGate AI  
**Author:** Shubham Dhoke  
**Repository:** Open-source project
