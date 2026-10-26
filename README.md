# MindGate AI

MindGate AI is a local document Q&A application. It parses PDF, TXT, Markdown,
and DOCX files into a workspace-local index, retrieves matching passages, and
can ask a locally running Ollama model to answer with page/line source links.
The application does not send document contents to a cloud service.

## Requirements

- Windows 10/11 and Python 3.11 (the included environment uses Python 3.11).
- Ollama and a locally installed model for generated answers. The default model
  is `phi3:mini`; choose a smaller model if the computer has limited memory.
- Disk space for the Python dependencies, local model, uploaded documents, and
  local index.

## Install

From PowerShell in the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
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
