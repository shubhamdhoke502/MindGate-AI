import os
import sys
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.app import MindGateEngine
from modules.workspace import WorkspaceManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

app = FastAPI(
    title="MindGate AI Engine API",
    description="Production-Grade Local RAG System with Multi-Tenancy & Resource Control",
    version="1.0.0"
)

# CORS configuration for Web UI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Manager Instances
workspace_mgr = WorkspaceManager(base_dir="workspaces")
engine = MindGateEngine()


class QueryRequest(BaseModel):
    query: str
    workspace_id: str = "default_session"
    active_pdf: Optional[str] = None


@app.post("/workspace/init")
async def init_workspace(workspace_id: str = Form("default_session")):
    """Session-based isolated workspace create aur initialize karta hai."""
    try:
        ws_path = workspace_mgr.create_project_structure(
            project_name=workspace_id, 
            subfolders=["uploads", "cache"]
        )
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "path": str(ws_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workspace initialization failed: {str(e)}")


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form("default_session")
):
    """PDF document ko user workspace me save karta hai aur Vector DB me index karta hai."""
    try:
        ws_path = workspace_mgr.base_dir / workspace_id / "uploads"
        ws_path.mkdir(parents=True, exist_ok=True)
        
        file_location = ws_path / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # MindGate Engine me document ingest karna
        ingest_res = engine.ingest_document(str(file_location))
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "ingest_result": ingest_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@app.post("/chat")
async def chat_query(req: QueryRequest):
    """Standard non-streaming RAG search aur SLM response generation."""
    try:
        response = engine.ask(query=req.query, active_pdf=req.active_pdf)
        return {
            "status": "success",
            "workspace_id": req.workspace_id,
            "result": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream_query(req: QueryRequest):
    """Real-time token streaming via Server-Sent Events (SSE)."""
    try:
        return StreamingResponse(
            engine.ask_stream(query=req.query, active_pdf=req.active_pdf),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.post("/system/flush")
async def flush_system_memory():
    """Manual system RAM/VRAM resource flush hook."""
    try:
        engine.resource_mgr.flush_memory()
        return {"status": "success", "message": "System RAM/VRAM flushed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flush failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Server and system health check status."""
    return {
        "status": "online",
        "engine_model": engine.ollama_model,
        "ollama_host": engine.ollama_host
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_server:app", host="127.0.0.1", port=8000, reload=True)