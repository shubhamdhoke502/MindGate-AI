import asyncio
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from modules.app import MindGateEngine

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = BASE_DIR / "workspaces"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_WORKSPACE_DOCUMENTS = 10
MAX_WORKSPACE_PDF_PAGES = 500
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
WORKSPACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
logger = logging.getLogger("MindGateAPI")

app = FastAPI(
    title="MindGate AI Engine API",
    description="Private, workspace-isolated local document retrieval and chat",
    version="2.1.0",
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]"],
)

_engines: dict[str, MindGateEngine] = {}
_engine_lock = Lock()
_upload_locks: dict[str, asyncio.Lock] = {}


@app.middleware("http")
async def reject_cross_origin_mutations(request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/").casefold() != str(request.base_url).rstrip("/").casefold():
            return JSONResponse(
                status_code=403,
                content={"detail": "Cross-origin requests are not allowed."},
            )
    return await call_next(request)


def _validated_workspace_id(workspace_id: str) -> str:
    if not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
        raise HTTPException(
            status_code=400,
            detail="workspace_id must contain 1-64 letters, digits, underscores, or hyphens.",
        )
    return workspace_id


def _workspace_path(workspace_id: str) -> Path:
    workspace_id = _validated_workspace_id(workspace_id)
    path = (WORKSPACES_DIR / workspace_id).resolve()
    if path.parent != WORKSPACES_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid workspace path.")
    return path


def _engine_for_workspace(workspace_id: str) -> MindGateEngine:
    workspace_id = _validated_workspace_id(workspace_id)
    with _engine_lock:
        engine = _engines.get(workspace_id)
        if engine is None:
            workspace = _workspace_path(workspace_id)
            workspace.mkdir(parents=True, exist_ok=True)
            engine = MindGateEngine(workspace_dir=str(workspace))
            upload_dir = workspace / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            supported_files = sorted(
                (
                    path for path in upload_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
                ),
                key=lambda path: path.name.casefold(),
            )
            source_names = {path.name for path in supported_files}
            for indexed_name in engine.vector_store.list_documents():
                if indexed_name not in source_names:
                    engine.vector_store.remove_document(indexed_name)
            for path in supported_files:
                engine.ingest_document(str(path))
            _engines[workspace_id] = engine
        return engine


def _safe_filename(filename: Optional[str]) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="A file name is required.")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or "\x00" in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported document types are PDF, TXT, Markdown, and DOCX.",
        )
    return name


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    workspace_id: str = Field(
        default="default_session",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    )
    active_pdf: Optional[str] = Field(default=None, max_length=255)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be empty.")
        return value


@app.get("/")
async def serve_ui():
    index_path = Path(__file__).resolve().parent / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "MindGate AI API is running. index.html not found in ui folder."},
        status_code=404,
    )


@app.post("/workspace/init")
async def init_workspace(workspace_id: str = Form("default_session")):
    workspace = _workspace_path(workspace_id)
    (workspace / "uploads").mkdir(parents=True, exist_ok=True)
    (workspace / "cache").mkdir(parents=True, exist_ok=True)
    return {"status": "success", "workspace_id": workspace_id, "path": str(workspace)}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form("default_session"),
):
    workspace_id = _validated_workspace_id(workspace_id)
    file_name = _safe_filename(file.filename)
    upload_lock = _upload_locks.setdefault(workspace_id, asyncio.Lock())
    async with upload_lock:
        return await _store_document(file, workspace_id, file_name)


async def _store_document(file: UploadFile, workspace_id: str, file_name: str):
    workspace = _workspace_path(workspace_id)
    upload_dir = workspace / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [
        path for path in upload_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    same_file = next(
        (path for path in existing_files if path.name.casefold() == file_name.casefold()),
        None,
    )
    if same_file is not None:
        file_name = same_file.name
    destination = upload_dir / file_name
    if not destination.exists() and len(existing_files) >= MAX_WORKSPACE_DOCUMENTS:
        raise HTTPException(status_code=413, detail="Workspace document limit (10) reached.")

    staging_dir = workspace / ".staging" / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=False)
    staged_file = staging_dir / file_name
    size = 0
    try:
        with staged_file.open("wb") as target:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Document exceeds the 50 MiB upload limit."
                    )
                target.write(block)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded document is empty.")

        if staged_file.suffix.lower() == ".pdf":
            try:
                with pymupdf.open(staged_file) as document:
                    incoming_pages = len(document)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc
            page_count = 0
            for path in existing_files:
                if path.suffix.lower() == ".pdf" and path.name != file_name:
                    try:
                        with pymupdf.open(path) as document:
                            page_count += len(document)
                    except Exception as exc:
                        logger.warning("Could not count existing PDF pages for %s: %s", path, exc)
            if page_count + incoming_pages > MAX_WORKSPACE_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail="Workspace PDF page limit (500) would be exceeded.",
                )

        engine = await asyncio.to_thread(_engine_for_workspace, workspace_id)
        try:
            result = await asyncio.to_thread(engine.ingest_document, str(staged_file))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Document ingestion failed for %s: %s", file_name, exc, exc_info=True)
            raise HTTPException(status_code=422, detail=f"Document ingestion failed: {exc}") from exc

        os.replace(staged_file, destination)
        return {"status": "success", "workspace_id": workspace_id, "ingest_result": result}
    except HTTPException:
        raise
    finally:
        await file.close()
        shutil.rmtree(staging_dir, ignore_errors=True)


@app.get("/documents")
async def list_documents(workspace_id: str = "default_session"):
    await asyncio.to_thread(_engine_for_workspace, workspace_id)
    upload_dir = _workspace_path(workspace_id) / "uploads"
    if not upload_dir.is_dir():
        return {"documents": []}
    documents = sorted(
        (
            {"file_name": path.name, "extension": path.suffix.lower()}
            for path in upload_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=lambda document: document["file_name"].casefold(),
    )
    return {"documents": documents}


@app.get("/documents/{file_name}")
async def open_document(file_name: str, workspace_id: str = "default_session"):
    name = _safe_filename(file_name)
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF documents can be opened.")
    path = _workspace_path(workspace_id) / "uploads" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found in this workspace.")
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@app.post("/chat")
async def chat_query(request: QueryRequest):
    try:
        engine = await asyncio.to_thread(_engine_for_workspace, request.workspace_id)
        result = await asyncio.to_thread(
            engine.ask,
            request.query.strip(),
            request.active_pdf,
        )
        return {"status": "success", "workspace_id": request.workspace_id, "result": result}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat/stream")
async def chat_stream_query(request: QueryRequest):
    engine = await asyncio.to_thread(_engine_for_workspace, request.workspace_id)
    return StreamingResponse(
        engine.ask_stream(request.query.strip(), active_doc=request.active_pdf),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history")
async def get_history(workspace_id: str = "default_session"):
    engine = await asyncio.to_thread(_engine_for_workspace, workspace_id)
    return {"history": engine.get_chat_history()}


@app.post("/flush_memory")
async def flush_system_memory(workspace_id: str = "default_session"):
    engine = await asyncio.to_thread(_engine_for_workspace, workspace_id)
    await asyncio.to_thread(engine.resource_mgr.flush_memory)
    return {"status": "success", "message": "Local application memory flushed."}


@app.get("/health")
async def health_check():
    return {"status": "online", "storage": "local", "llm": "ollama"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.app_server:app", host="127.0.0.1", port=8000)
