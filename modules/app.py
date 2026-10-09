import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Core Production Modules
from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


app = FastAPI(title="MindGate AI Engine", version="1.0.4")

# CORS Configuration
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

        # Core Managers Setup
        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()
        self.chat_history = ChatHistoryManager()
        self.chunk_cache = ChunkCacheManager()
        self.resource_mgr = SystemResourceManager()

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Supports Multi-Format Document Ingestion & Metadata Indexing."""
        try:
            file_name = os.path.basename(file_path)

            if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
                self.vector_store.compressor.register_activity()

            # 1. Parse File and Extract Chunks with Metadata Coordinates
            chunks = parse_pdf_with_coordinates(file_path)

            if not chunks:
                return {"status": "error", "message": "Document parsing produced zero text content."}

            # Inject file_name into all chunk metadatas explicitly
            for idx, chk in enumerate(chunks):
                if "metadata" not in chk:
                    chk["metadata"] = {}
                chk["metadata"]["file_name"] = file_name
                chk["chunk_id"] = f"{file_name}_chunk_{idx}"

            # 2. Save Chunks to Disk Cache
            if hasattr(self.chunk_cache, 'save_chunks_to_cache'):
                self.chunk_cache.save_chunks_to_cache(file_name, chunks)

            # 3. Add to ChromaDB Vector DB using upsert_chunks
            if hasattr(self.vector_store, 'upsert_chunks'):
                self.vector_store.upsert_chunks(chunks)
            elif hasattr(self.vector_store, 'add_chunks'):
                self.vector_store.add_chunks(chunks)

            if hasattr(self.container_memory, 'register_document'):
                self.container_memory.register_document(file_name)

            return {
                "status": "success",
                "file_name": file_name,
                "total_chunks_indexed": len(chunks)
            }
        except Exception as e:
            print(f"\n[CRITICAL INGESTION ERROR]: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    def retrieve_relevant_chunks(self, query: str, active_doc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Strict Targeted Vector Retrieval with Document Isolation."""
        if not active_doc:
            return []

        # Check for broad summary or overview questions
        broad_keywords = ["summary", "overview", "kya hai", "about", "describe", "detail"]
        is_broad = any(kw in query.lower() for kw in broad_keywords)

        if is_broad and hasattr(self.vector_store, 'get_all_chunks_for_file'):
            # Directly retrieve top structural document chunks for summary
            return self.vector_store.get_all_chunks_for_file(active_doc, limit=8)

        # Standard Targeted Cosine Semantic Vector Search
        retrieved = []
        if hasattr(self.vector_store, 'query_similar_chunks'):
            retrieved = self.vector_store.query_similar_chunks(query, top_k=8, file_name_filter=active_doc)
        elif hasattr(self.vector_store, 'query'):
            retrieved = self.vector_store.query(query_text=query, top_k=8, file_filter=active_doc)

        return retrieved

    def _build_prompt_and_citations(self, query: str, retrieved_chunks: List[Dict[str, Any]], active_doc: Optional[str] = None):
        """Builds clean, grounded prompt preventing prompt leaking and hallucinations."""
        context_str = ""
        citations = []

        for idx, item in enumerate(retrieved_chunks):
            meta = item.get("metadata", {})
            file_n = meta.get('file_name', active_doc or 'Document')
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
            "You are MindGate AI, an intelligent offline document assistant.\n"
            "Analyze the document context provided below and answer the user question directly, accurately, and naturally.\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY based on the provided document context blocks.\n"
            "2. Do NOT hallucinate or use external knowledge.\n"
            "3. If the context does not contain enough info, state clearly: 'Is document me is question ka context nahi mila.'\n"
            "4. Do NOT output prompt tags, system labels, or repeat system rules.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )

        return system_prompt, citations

    def ask(self, query: str, active_doc: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous Query Execution."""
        if query.startswith("@"):
            search_term = query[1:]
            return self.container_memory.handle_at_trigger(search_term)

        if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
            self.vector_store.compressor.register_activity()

        self.chat_history.save_message(role="user", content=query)

        retrieved_chunks = self.retrieve_relevant_chunks(query, active_doc)

        if not retrieved_chunks:
            no_ctx_msg = "Is document me is question ka context nahi mila."
            self.chat_history.save_message(role="assistant", content=no_ctx_msg, citations=[])
            return {"answer": no_ctx_msg, "citations": []}

        system_prompt, citations = self._build_prompt_and_citations(query, retrieved_chunks, active_doc)

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": system_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode("utf-8"))
                slm_response = res.get("response", "").strip()
        except Exception as e:
            slm_response = f"[Local SLM Error]: Ollama execution failed. Details: {e}"

        self.chat_history.save_message(role="assistant", content=slm_response, citations=citations)
        self.resource_mgr.flush_memory()

        return {
            "answer": slm_response,
            "citations": citations
        }

    def ask_stream(self, query: str, active_doc: Optional[str] = None) -> Generator[str, None, None]:
        """SSE Streaming Endpoint."""
        if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
            self.vector_store.compressor.register_activity()

        self.chat_history.save_message(role="user", content=query)

        retrieved_chunks = self.retrieve_relevant_chunks(query, active_doc)

        if not retrieved_chunks:
            no_ctx_msg = "Is document me is question ka context nahi mila."
            yield f"data: {json.dumps({'citations': []})}\n\n"
            yield f"data: {json.dumps({'token': no_ctx_msg, 'done': True})}\n\n"
            self.chat_history.save_message(role="assistant", content=no_ctx_msg, citations=[])
            return

        system_prompt, citations = self._build_prompt_and_citations(query, retrieved_chunks, active_doc)

        # 1. First event: Send Citations payload to UI
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
            err_msg = f"\n[Streaming Error]: Ollama execution failed: {str(e)}"
            yield f"data: {json.dumps({'token': err_msg, 'done': True})}\n\n"
            full_response += err_msg

        self.chat_history.save_message(role="assistant", content=full_response, citations=citations)
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

        # Full path pass handling
        result = engine.ingest_document(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"[Upload Server Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    query = data.get("query") or data.get("prompt") or data.get("message", "")
    active_doc = data.get("file_name") or data.get("active_doc")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        engine.ask_stream(query, active_doc=active_doc),
        media_type="text/event-stream"
    )


@app.post("/chat")
async def chat_sync(request: Request):
    data = await request.json()
    query = data.get("query") or data.get("prompt") or data.get("message", "")
    active_doc = data.get("file_name") or data.get("active_doc")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return engine.ask(query, active_doc=active_doc)


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