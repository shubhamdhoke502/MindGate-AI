import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

# Production-grade imports
from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


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
        """
        Step 1: Uncompress DB -> Step 2: Check Cache/Parse PDF -> Step 3: Index Chunks -> Step 4: Register Container.
        """
        try:
            file_name = os.path.basename(pdf_path)
            
            # 0. Safety Lock Release
            if hasattr(self.vector_store, 'compressor') and self.vector_store.compressor:
                self.vector_store.compressor.register_activity()

            # 1. Check Disk Cache first
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

            # 2. Add Chunks to Local Vector DB safely
            if hasattr(self.vector_store, 'add_chunks'):
                self.vector_store.add_chunks(chunks)
            elif hasattr(self.vector_store, 'add_documents'):
                self.vector_store.add_documents(chunks)
            
            # 3. Register in Memory Container
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
            "options": {"temperature": 0.1}
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

        retrieved_chunks = self.vector_store.query_similar_chunks(query, top_k=2, file_name_filter=active_pdf)
        
        if not retrieved_chunks:
            fallback_ans = "Document me is topic par koi reliable info nahi mili."
            self.chat_history.save_message(role="assistant", content=fallback_ans)
            self.resource_mgr.flush_memory()
            return {"answer": fallback_ans, "citations": []}

        context_str = ""
        citations = []
        for idx, item in enumerate(retrieved_chunks):
            meta = item.get("metadata", {})
            file_n = meta.get('file_name', 'Document')
            p_num = meta.get('page_number', 1)
            l_start = meta.get('line_start', 1)
            l_end = meta.get('line_end', 1)
            
            citation_label = f"[Source: {file_n} | Page {p_num}, Lines {l_start}-{l_end}]"
            context_str += f"Context Block {idx+1} ({citation_label}):\n{item.get('text', '')}\n\n"
            
            citations.append({
                "file_name": file_n,
                "page": p_num,
                "lines": f"{l_start}-{l_end}",
                "citation_text": citation_label
            })

        system_prompt = (
            "You are MindGate AI, a strict factual document assistant. "
            "Answer the user query ONLY using the provided context blocks. "
            "Do NOT add conversational filler or unverified assumptions.\n\n"
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
        """Server-Sent Events streaming generator."""
        res = self.ask(query, active_pdf)
        ans = res.get("answer", "")
        for word in ans.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"

    def get_chat_history(self) -> List[Dict]:
        return self.chat_history.get_all_messages()