import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Add parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.app import MindGateEngine

app = FastAPI(title="MindGate AI API Server")
engine = MindGateEngine()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PDF_DIR = os.path.join(STORAGE_DIR, "documents")
os.makedirs(PDF_DIR, exist_ok=True)

# Mount raw PDF storage directory for PDF.js viewing
app.mount("/pdf_files", StaticFiles(directory=PDF_DIR), name="pdf_files")


class QueryRequest(BaseModel):
    query: str
    active_pdf: Optional[str] = None


@app.get("/")
def read_root():
    """Serves the main desktop web UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF upload, saves to disk, and ingests into vector DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(PDF_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    result = engine.ingest_document(file_path)
    return JSONResponse(content=result)


@app.post("/api/ask")
def ask_question(request: QueryRequest):
    """Executes RAG query and returns factual response with bounding box citations."""
    response = engine.ask(query=request.query, active_pdf=request.active_pdf)
    return JSONResponse(content=response)


@app.get("/api/history")
def get_chat_history():
    """Fetches persistent chat history log."""
    history = engine.get_chat_history()
    return JSONResponse(content={"history": history})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)