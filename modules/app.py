import json
import urllib.request
from typing import Dict, Any, List, Optional

from modules.parser import parse_pdf_production_grade as parse_pdf_with_coordinates
from modules.vector_db import VectorStoreManager, StorageCompressor
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager

class MindGateEngine:
    def __init__(self, ollama_model: str = "phi3:mini", ollama_host: str = "http://localhost:11434"):
        """
        Main Engine Orchestrator for MindGate AI.
        Connects Parser, Vector DB, Container Box, Chat History, Chunk Cache, and Local SLM.
        """
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()
        self.chat_history = ChatHistoryManager()
        self.chunk_cache = ChunkCacheManager()
        self.resource_mgr = SystemResourceManager()

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Step 1: Check Cache -> Step 2: Parse PDF if missing -> Step 3: Store Vector DB -> Step 4: Register Container.
        """
        file_name = pdf_path.split("/")[-1].split("\\")[-1]
        
        # 1. Check Disk Cache first
        cached_chunks = self.chunk_cache.get_cached_chunks(pdf_path)
        if cached_chunks:
            chunks = cached_chunks
        else:
            # Parse fresh PDF and save to cache
            chunks = parse_pdf_with_coordinates(pdf_path)
            self.chunk_cache.save_chunks_to_cache(pdf_path, chunks)
        
        # 2. Add Chunks to Local Vector DB
        self.vector_store.add_chunks(chunks)
        
        # 3. Register in Memory Container
        self.container_memory.register_document(file_name)
        
        return {
            "status": "success",
            "file_name": file_name,
            "total_chunks_indexed": len(chunks),
            "from_cache": cached_chunks is not None
        }

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
            return f"[Local SLM Error]: Ollama local instance run nahi ho raha hai. Please run 'ollama run {self.ollama_model}'. Details: {e}"

    def ask(self, query: str, active_pdf: Optional[str] = None) -> Dict[str, Any]:
        """Performs Search, SLM Answer, Citation Generation, Container Sync, and Chat Saving."""
        # Handle '@' Trigger Command
        if query.startswith("@"):
            search_term = query[1:]
            return self.container_memory.handle_at_trigger(search_term)

        # 1. Save User Query in Chat History
        self.chat_history.save_message(role="user", content=query)

        # 2. Retrieve Top-2 relevant chunks from Vector DB
        retrieved_chunks = self.vector_store.query_similar_chunks(query, top_k=2, file_name_filter=active_pdf)
        
        if not retrieved_chunks:
            fallback_ans = "Document me is topic par koi reliable info nahi mili."
            self.chat_history.save_message(role="assistant", content=fallback_ans)
            self.resource_mgr.flush_memory()
            return {"answer": fallback_ans, "citations": []}

        # 3. Prepare Context & Citations Metadata
        context_str = ""
        citations = []
        for idx, item in enumerate(retrieved_chunks):
            meta = item["metadata"]
            citation_label = f"[Source: {meta['file_name']} | Page {meta['page_number']}, Lines {meta['line_start']}-{meta['line_end']}]"
            context_str += f"Context Block {idx+1} ({citation_label}):\n{item['text']}\n\n"
            
            citations.append({
                "file_name": meta["file_name"],
                "page": meta["page_number"],
                "lines": f"{meta['line_start']}-{meta['line_end']}",
                "citation_text": citation_label
            })

        # 4. Construct Grounded Prompt
        system_prompt = (
            "You are MindGate AI, a strict factual document assistant. "
            "Answer the user query ONLY using the provided context blocks. "
            "Do NOT add conversational filler or unverified assumptions. "
            "At the end of your answer, explicitly cite the source page and lines.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUERY: {query}\n\n"
            "ANSWER:"
        )

        # 5. Generate Answer via Local SLM
        slm_response = self._query_local_slm(system_prompt)

        # 6. Save Assistant Response with Citations in Chat History
        self.chat_history.save_message(role="assistant", content=slm_response, citations=citations)

        # 7. Background Sync into Container Memory Boxx
        if citations:
            top_meta = citations[0]
            self.container_memory.save_highlight(
                pdf_name=top_meta["file_name"],
                page=top_meta["page"],
                line_range=top_meta["lines"],
                key_fact=slm_response[:120] + "..."
            )

        # 8. Flush RAM Memory
        self.resource_mgr.flush_memory()

        return {
            "answer": slm_response,
            "citations": citations
        }

    def get_chat_history(self) -> List[Dict]:
        """Returns all persistent chat messages."""
        return self.chat_history.get_all_messages()


if __name__ == "__main__":
    engine = MindGateEngine()
    print("MindGate Core Engine with Persistent History & Cache Initialized Successfully.")