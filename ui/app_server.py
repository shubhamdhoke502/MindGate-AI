import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Parent directory access for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.app import MindGateEngine

app = FastAPI(title="MindGate AI Local API Engine")

# Enable CORS for local UI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engine
engine = MindGateEngine()

class IngestRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    query: str
    active_pdf: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "MindGate Engine Online", "mode": "100% Offline"}

@app.post("/api/ingest")
def ingest_pdf(req: IngestRequest):
    if not os.path.exists(req.pdf_path):
        raise HTTPException(status_code=404, detail="File path invalid or file does not exist.")
    try:
        result = engine.ingest_document(req.pdf_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
def ask_query(req: QueryRequest):
    try:
        response = engine.ask(query=req.query, active_pdf=req.active_pdf)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)