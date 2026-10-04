import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Production-grade module imports
from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


app = FastAPI(title="MindGate AI Engine", version="1.0.1")

# CORS Setup
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

        # Core Managers
        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()
        self.chat_history = ChatHistoryManager()
        self.chunk_cache = ChunkCacheManager()
        self.resource_mgr = SystemResourceManager()

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """In-memory or disk-cached PDF ingestion with exact coordinate mapping."""
        try:
            file_name = os.path.basename(pdf_path)

            if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
                self.vector_store.compressor.register_activity()

            cached_chunks = None
            if hasattr(self.chunk_cache, 'get_cached_chunks'):
                cached_chunks = self.chunk_cache.get_cached_chunks(pdf_path)

            if cached_chunks:
                chunks = cached_chunks
            else:
                chunks = parse_pdf_with_coordinates(pdf_path)
                if chunks and hasattr(self.chunk_cache, 'save_chunks_to_cache'):
                    self.chunk_cache.save_chunks_to_cache(pdf_path, chunks)

            if not chunks:
                return {"status": "error", "message": "PDF parsing produced zero text content."}

            if hasattr(self.vector_store, 'add_chunks'):
                self.vector_store.add_chunks(chunks)
            elif hasattr(self.vector_store, 'add_documents'):
                self.vector_store.add_documents(chunks)

            if hasattr(self.container_memory, 'register_document'):
                self.container_memory.register_document(file_name)

            return {
                "status": "success",
                "file_name": file_name,
                "total_chunks_indexed": len(chunks),
                "from_cache": cached_chunks is not None
            }
        except Exception as e:
            print(f"\n[CRITICAL INGESTION ERROR]: {str(e)}\n")
            raise Exception(f"Ingestion failed: {str(e)}")

    def retrieve_relevant_chunks(self, query: str, active_pdf: Optional[str] = None) -> List[Dict[str, Any]]:
        """100% Dynamic retrieval with guaranteed fallback to primary document chunks."""
        retrieved_chunks = []

        # 1. Similarity Search (Vector / Hybrid)
        if hasattr(self.vector_store, 'query_similar_chunks'):
            try:
                retrieved_chunks = self.vector_store.query_similar_chunks(query, top_k=8, file_name_filter=active_pdf)
            except Exception:
                retrieved_chunks = []

        # 2. Direct Fallback: Pick initial document chunks directly from local store
        if not retrieved_chunks:
            if hasattr(self.vector_store, 'get_all_chunks_for_file') and active_pdf:
                retrieved_chunks = self.vector_store.get_all_chunks_for_file(active_pdf, limit=8)
            elif hasattr(self.vector_store, 'collection'):
                try:
                    all_data = self.vector_store.collection.get()
                    docs = all_data.get("documents", [])
                    metas = all_data.get("metadatas", [])
                    for i, doc in enumerate(docs):
                        meta = metas[i] if metas else {}
                        fname = meta.get("file_name") or meta.get("filename")
                        if not active_pdf or fname == active_pdf:
                            retrieved_chunks.append({"text": doc, "metadata": meta})
                        if len(retrieved_chunks) >= 8:
                            break
                except Exception:
                    pass

        return retrieved_chunks

    def _build_prompt_and_citations(self, query: str, retrieved_chunks: List[Dict[str, Any]], active_pdf: Optional[str] = None):
        """Constructs grounded context for Ollama SLM."""
        context_str = ""
        citations = []

        for idx, item in enumerate(retrieved_chunks):
            meta = item.get("metadata", {})
            file_n = meta.get('file_name', active_pdf or 'Document')
            p_num = meta.get('page_number', meta.get('page', 1))
            l_start = meta.get('line_start', 1)
            l_end = meta.get('line_end', 1)

            citation_label = f"[Source: {file_n} | Page {p_num}]"
            context_str += f"Context Block {idx+1} ({citation_label}):\n{item.get('text', '')}\n\n"

            citations.append({
                "file_name": file_n,
                "page": p_num,
                "lines": f"{l_start}-{l_end}",
                "citation_text": citation_label
            })

        system_prompt = (
            "You are MindGate AI, an intelligent local document assistant.\n"
            "Analyze the document context provided below and answer the user query accurately and naturally.\n"
            "If the user asks for a summary or general overview ('isme kya hai', 'overview'), synthesize the key information from all context blocks.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUERY: {query}\n\n"
            "ANSWER:"
        )

        return system_prompt, citations

    def ask(self, query: str, active_pdf: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous Query API."""
        if query.startswith("@"):
            search_term = query[1:]
            return self.container_memory.handle_at_trigger(search_term)

        if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
            self.vector_store.compressor.register_activity()

        self.chat_history.save_message(role="user", content=query)

        # Retrieve chunks guaranteed
        retrieved_chunks = self.retrieve_relevant_chunks(query, active_pdf)

        system_prompt, citations = self._build_prompt_and_citations(query, retrieved_chunks, active_pdf)

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": system_prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode("utf-8"))
                slm_response = res.get("response", "").strip()
        except Exception as e:
            slm_response = f"[Local Runtime Error]: Ollama service connection failed. Details: {e}"

        self.chat_history.save_message(role="assistant", content=slm_response, citations=citations)

        if citations and hasattr(self.container_memory, 'save_highlight'):
            top_meta = citations[0]
            self.container_memory.save_highlight(
                pdf_name=top_meta["file_name"],
                page=top_meta["page"],
                line_range=top_meta["lines"],
                key_fact=slm_response[:120] + "..."
            )

        self.resource_mgr.flush_memory()

        return {
            "answer": slm_response,
            "citations": citations
        }

    def ask_stream(self, query: str, active_pdf: Optional[str] = None) -> Generator[str, None, None]:
        """True SSE Streaming Endpoint without hardcoded strings."""
        if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
            self.vector_store.compressor.register_activity()

        self.chat_history.save_message(role="user", content=query)

        retrieved_chunks = self.retrieve_relevant_chunks(query, active_pdf)

        system_prompt, citations = self._build_prompt_and_citations(query, retrieved_chunks, active_pdf)

        # First Event: Citations Metadata
        yield f"data: {json.dumps({'citations': citations})}\n\n"

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": system_prompt,
            "stream": True,
            "options": {"temperature": 0.2}
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
            err_msg = f"\n[Streaming Error]: Local Ollama execution failed: {str(e)}"
            yield f"data: {json.dumps({'token': err_msg, 'done': True})}\n\n"
            full_response += err_msg

        self.chat_history.save_message(role="assistant", content=full_response, citations=citations)

        if citations and hasattr(self.container_memory, 'save_highlight'):
            top_meta = citations[0]
            self.container_memory.save_highlight(
                pdf_name=top_meta["file_name"],
                page=top_meta["page"],
                line_range=top_meta["lines"],
                key_fact=full_response[:120] + "..."
            )

        self.resource_mgr.flush_memory()

    def get_chat_history(self) -> List[Dict]:
        return self.chat_history.get_all_messages()


engine = MindGateEngine()


# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if os.path.exists("ui/index.html"):
        with open("ui/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MindGate AI Engine Active</h1>"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        result = engine.ingest_document(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/chat")
async def chat_sync(request: Request):
    data = await request.json()
    query = data.get("query") or data.get("prompt") or data.get("message", "")
    active_doc = data.get("file_name") or data.get("active_doc")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return engine.ask(query, active_pdf=active_doc)


@app.get("/history")
async def get_history():
    return {"history": engine.get_chat_history()}


@app.post("/flush_memory")
async def flush_ram():
    engine.resource_mgr.flush_memory()
    return {"status": "success", "message": "RAM / VRAM Flushed Successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)