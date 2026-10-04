import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Production-grade imports
from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


app = FastAPI(title="MindGate AI Engine")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount UI static directory if present
if os.path.exists("ui"):
    app.mount("/static", StaticFiles(directory="ui"), name="static")


class MindGateEngine:
    def __init__(self, ollama_host: str = "http://127.0.0.1:11434", ollama_model: str = "phi3:mini"):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        
        # Initialize Core Managers
        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()
        self.chat_history = ChatHistoryManager()
        self.chunk_cache = ChunkCacheManager()
        self.resource_mgr = SystemResourceManager()

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """Uncompress DB -> Check Cache/Parse PDF -> Index Chunks -> Register Container."""
        try:
            file_name = os.path.basename(pdf_path)
            
            # Safety Lock Release
            if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
                self.vector_store.compressor.register_activity()

            # Check Disk Cache first
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
                return {"status": "error", "message": "No text extracted from PDF."}

            # Add Chunks to Local Vector DB safely
            if hasattr(self.vector_store, 'add_chunks'):
                self.vector_store.add_chunks(chunks)
            elif hasattr(self.vector_store, 'add_documents'):
                self.vector_store.add_documents(chunks)
            
            # Register in Memory Container
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

    def _query_local_slm(self, prompt: str) -> str:
        """Sends grounded RAG prompt to local Ollama instance (100% Offline)."""
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("response", "").strip()
        except Exception as e:
            return f"[Local SLM Error]: Ollama local instance run nahi ho raha hai. Details: {e}"

    def ask(self, query: str, active_pdf: Optional[str] = None) -> Dict[str, Any]:
        """Performs Search, SLM Answer, Citation Generation, Container Sync, and Chat Saving."""
        if query.startswith("@"):
            search_term = query[1:]
            return self.container_memory.handle_at_trigger(search_term)

        if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
            self.vector_store.compressor.register_activity()

        self.chat_history.save_message(role="user", content=query)

        # 1. Fetch search chunks (top_k=8 for comprehensive context)
        retrieved_chunks = []
        if hasattr(self.vector_store, 'query_similar_chunks'):
            retrieved_chunks = self.vector_store.query_similar_chunks(query, top_k=8, file_name_filter=active_pdf)

        # 2. Fallback Mechanism: Generic prompts ("isme kya likha hai", "summarize") ke liye direct PDF chunks fetch karo
        if not retrieved_chunks and active_pdf:
            if hasattr(self.vector_store, 'get_all_chunks_for_file'):
                retrieved_chunks = self.vector_store.get_all_chunks_for_file(active_pdf, limit=8)
            elif hasattr(self.vector_store, 'collection'):
                try:
                    all_data = self.vector_store.collection.get()
                    docs = all_data.get("documents", [])
                    metas = all_data.get("metadatas", [])
                    for i, doc in enumerate(docs):
                        meta = metas[i] if metas else {}
                        fname = meta.get("file_name") or meta.get("filename")
                        if fname == active_pdf or not fname:
                            retrieved_chunks.append({"text": doc, "metadata": meta})
                        if len(retrieved_chunks) >= 8:
                            break
                except Exception:
                    pass

        # 3. Final Check
        if not retrieved_chunks:
            fallback_ans = "Document me is topic par koi info nahi mili ya PDF selected nahi hai."
            self.chat_history.save_message(role="assistant", content=fallback_ans)
            self.resource_mgr.flush_memory()
            return {"answer": fallback_ans, "citations": []}

        # 4. Context & Citation Preparation
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

        # 5. Smart Prompt Construct
        system_prompt = (
            "You are MindGate AI, an expert document assistant.\n"
            "Carefully analyze the document context provided below and answer the user query in a clear, well-structured manner.\n"
            "If the user asks 'isme kya likha hai' or requests an overview/summary, summarize the core details from the context blocks.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUERY: {query}\n\n"
            "ANSWER:"
        )

        slm_response = self._query_local_slm(system_prompt)

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
        res = self.ask(query, active_pdf)
        ans = res.get("answer", "")
        words = ans.split(" ")
        for word in words:
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"

    def get_chat_history(self) -> List[Dict]:
        return self.chat_history.get_all_messages()


# Global Engine Instance
engine = MindGateEngine()


# --- FastApi Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if os.path.exists("ui/index.html"):
        with open("ui/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MindGate AI Server Running</h1>"


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


@app.post("/flush_memory")
async def flush_ram():
    engine.resource_mgr.flush_memory()
    return {"status": "success", "message": "RAM / VRAM Flushed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)