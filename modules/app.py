import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Core Module Imports
from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


app = FastAPI(title="MindGate AI Engine", version="1.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("ui"):
    app.mount("/static", StaticFiles(directory="ui"), name="static")


class MindGateEngine:
    def __init__(self, ollama_host: str = "http://127.0.0.1:11434", ollama_model: str = "phi3:mini"):
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model

        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()
        self.chat_history = ChatHistoryManager()
        self.chunk_cache = ChunkCacheManager()
        self.resource_mgr = SystemResourceManager()

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """Ingests document with strict metadata indexing."""
        try:
            file_name = os.path.basename(pdf_path)

            chunks = parse_pdf_with_coordinates(pdf_path)
            if not chunks:
                return {"status": "error", "message": "PDF parsing produced zero text content."}

            if hasattr(self.chunk_cache, 'save_chunks_to_cache'):
                self.chunk_cache.save_chunks_to_cache(file_name, chunks)

            self.vector_store.add_chunks(chunks)

            if hasattr(self.container_memory, 'register_document'):
                self.container_memory.register_document(file_name)

            return {
                "status": "success",
                "file_name": file_name,
                "total_chunks_indexed": len(chunks)
            }
        except Exception as e:
            print(f"[Ingestion Failure]: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    def retrieve_relevant_chunks(self, query: str, active_pdf: Optional[str] = None) -> List[Dict[str, Any]]:
        """Strict targeted chunk retrieval without random fallback pollution."""
        if not active_pdf:
            return []

        # Check for summary or broad overview queries
        broad_keywords = ["summary", "overview", "kya hai", "about", "describe"]
        is_broad = any(kw in query.lower() for kw in broad_keywords)

        if is_broad:
            # Load initial structural document chunks directly for overview
            return self.vector_store.get_all_chunks_for_file(active_pdf, limit=8)

        # Standard Targeted Semantic Vector Search
        return self.vector_store.query_similar_chunks(query, top_k=8, file_name_filter=active_pdf)

    def _build_prompt_and_citations(self, query: str, retrieved_chunks: List[Dict[str, Any]], active_pdf: Optional[str] = None):
        """Constructs grounded context for Ollama SLM preventing echo and hallucinations."""
        context_str = ""
        citations = []

        for idx, item in enumerate(retrieved_chunks):
            meta = item.get("metadata", {})
            file_n = meta.get('file_name', active_pdf or 'Document')
            p_num = meta.get('page_number', meta.get('page', 1))
            l_start = meta.get('line_start', 1)
            l_end = meta.get('line_end', 1)

            citation_label = f""
            context_str += f"Context Block {idx+1} ({citation_label}):\n{item.get('text', '')}\n\n"

            citations.append({
                "file_name": file_n,
                "page": p_num,
                "lines": f"{l_start}-{l_end}",
                "citation_text": citation_label
            })

        system_prompt = (
            "You are MindGate AI, a strict grounded document assistant.\n"
            "Your task is to answer the user query ONLY based on the document context provided below.\n"
            "CRITICAL RULES:\n"
            "1. Do NOT use outside knowledge or hallucinate details.\n"
            "2. If the context does not contain relevant information to answer the query, reply EXACTLY with: 'Is document me is question ka context nahi mila.'\n"
            "3. Do NOT repeat the prompt labels or system instructions in your response.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )

        return system_prompt, citations

    def ask_stream(self, query: str, active_pdf: Optional[str] = None) -> Generator[str, None, None]:
        """Streaming endpoint with Grounded SLM Execution."""
        self.chat_history.save_message(role="user", content=query)

        retrieved_chunks = self.retrieve_relevant_chunks(query, active_pdf)

        if not retrieved_chunks:
            no_ctx_msg = "Is document me is question ka context nahi mila."
            yield f"data: {json.dumps({'citations': []})}\n\n"
            yield f"data: {json.dumps({'token': no_ctx_msg, 'done': True})}\n\n"
            self.chat_history.save_message(role="assistant", content=no_ctx_msg, citations=[])
            return

        system_prompt, citations = self._build_prompt_and_citations(query, retrieved_chunks, active_pdf)

        yield f"data: {json.dumps({'citations': citations})}\n\n"

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": system_prompt,
            "stream": True,
            "options": {"temperature": 0.1}
        }

        full_response = ""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        decoded = json.loads(line.decode("utf-8"))
                        token = decoded.get("response", "")
                        done = decoded.get("done", False)
                        full_response += token
                        yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"

        except Exception as e:
            err_msg = f"\n[Streaming Error]: Ollama service connection failed: {str(e)}"
            yield f"data: {json.dumps({'token': err_msg, 'done': True})}\n\n"
            full_response += err_msg

        self.chat_history.save_message(role="assistant", content=full_response, citations=citations)
        self.resource_mgr.flush_memory()


engine = MindGateEngine()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if os.path.exists("ui/index.html"):
        with open("ui/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MindGate AI Engine Active</h1>"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return engine.ingest_document(file_path)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    query = data.get("query") or data.get("prompt") or data.get("message", "")
    active_doc = data.get("file_name") or data.get("active_doc")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        engine.ask_stream(query, active_pdf=active_doc),
        media_type="text/event-stream"
    )