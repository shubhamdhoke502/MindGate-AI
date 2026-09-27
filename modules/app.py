import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List

from parser import parse_pdf_with_coordinates
from vector_db import VectorStoreManager, StorageCompressor
from container import ContainerMemoryManager

class MindGateEngine:
    def __init__(self, ollama_model: str = "phi3:mini", ollama_host: str = "http://localhost:11434"):
        """
        Main Engine Orchestrator for MindGate AI.
        Connects Parser, Vector DB, Container Box, and Local SLM.
        """
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.vector_store = VectorStoreManager()
        self.container_memory = ContainerMemoryManager()

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Step 1: Parses PDF coordinates -> Step 2: Stores in ChromaDB -> Step 3: Registers in Container Box.
        """
        file_name = pdf_path.split("/")[-1].split("\\")[-1]
        
        # 1. Parse PDF
        chunks = parse_pdf_with_coordinates(pdf_path)
        
        # 2. Add to Local Vector DB
        self.vector_store.add_chunks(chunks)
        
        # 3. Register in Memory Container
        self.container_memory.register_document(file_name)
        
        return {
            "status": "success",
            "file_name": file_name,
            "total_chunks_indexed": len(chunks)
        }

    def _query_local_slm(self, prompt: str) -> str:
        """
        Sends grounded RAG prompt to local Ollama instance (100% Offline).
        """
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # Low temperature for strict factual outputs
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("response", "").strip()
        except Exception as e:
            return f"[Local SLM Error]: Ollama local instance run nahi ho raha hai. Please run 'ollama run {self.ollama_model}'. Details: {e}"

    def ask(self, query: str, active_pdf: str = None) -> Dict[str, Any]:
        """
        Performs Vector Search, RAG Injection, Citation Generation, and Container Memory Sync.
        """
        # Handle '@' Trigger Command
        if query.startswith("@"):
            search_term = query[1:]
            return self.container_memory.handle_at_trigger(search_term)

        # 1. Retrieve Top-2 relevant chunks from Vector DB
        retrieved_chunks = self.vector_store.query_similar_chunks(query, top_k=2, file_name_filter=active_pdf)
        
        if not retrieved_chunks:
            return {
                "answer": "Document me is topic par koi reliable info nahi mili.",
                "citations": []
            }

        # 2. Prepare Context & Citations Metadata
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

        # 3. Construct Grounded RAG Prompt
        system_prompt = (
            "You are MindGate AI, a strict factual document assistant. "
            "Answer the user query ONLY using the provided context blocks. "
            "Do NOT add conversational filler or unverified assumptions. "
            "At the end of your answer, explicitly cite the source page and lines.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}\n"
            f"USER QUERY: {query}\n\n"
            "ANSWER:"
        )

        # 4. Generate Answer via Local SLM
        slm_response = self._query_local_slm(system_prompt)

        # 5. Background Sync into Container Memory Box
        if citations:
            top_meta = citations[0]
            self.container_memory.save_highlight(
                pdf_name=top_meta["file_name"],
                page=top_meta["page"],
                line_range=top_meta["lines"],
                key_fact=slm_response[:120] + "..."  # Save short snippet
            )

        return {
            "answer": slm_response,
            "citations": citations
        }


# Quick Test Pipeline
if __name__ == "__main__":
    engine = MindGateEngine()
    print("MindGate Core Engine Initialized Successfully.")