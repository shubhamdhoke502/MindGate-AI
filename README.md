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
python -m pip install -r requirements.txt
```

Install Ollama separately, then download a model while internet access is
available:

```powershell
ollama pull phi3:mini
```

For a different local model or Ollama endpoint, set these before launching:

```powershell
$env:MINDGATE_OLLAMA_MODEL = "qwen2.5:1.5b"
$env:MINDGATE_OLLAMA_HOST = "http://127.0.0.1:11434"
```

## Run

Ensure Ollama is running, then launch the local server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ui.app_server:app --host 127.0.0.1 --port 8000
```

On Windows, after installing dependencies, you can alternatively double-click
`scripts\run_portable.bat`; it uses the existing `.venv` or `venv` environment
and binds the service to loopback.

Open <http://127.0.0.1:8000>. The UI, API, SQLite full-text index, history,
memory, and uploaded files run locally. No remote UI assets or vector-store
service are used.

The default workspace is `workspaces/default_session`. The API also accepts
other workspace IDs containing 1–64 ASCII letters, digits, underscores, or
hyphens; each workspace has independent indexes, cache, history, and memory.
Uploads are limited to 50 MiB each, 10 documents per workspace, and 500 total
PDF pages per workspace.

## Test

The parser, retrieval, workspace isolation, upload, and API boundary tests do
not need a running Ollama model:

```powershell
.\venv\Scripts\python.exe -m unittest discover -v
```

The repository-wide discovery command runs the ingestion regression and the
tests in `tests/`.

See [current_state.md](./current_state.md) for the verified feature scope and
known limitations, and [architecture.md](./architecture.md) for data flow.
