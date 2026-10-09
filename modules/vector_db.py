import os
import chromadb
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    def __init__(self, persist_directory: str = ".mindgate/chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        # Persistent Chroma Client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Multi-format universal document collection
        self.collection = self.client.get_or_create_collection(
            name="mindgate_doc_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Adds multi-format document chunks (PDF, DOCX, TXT, MD) into ChromaDB with explicit metadata."""
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id")
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            if text and chunk_id:
                documents.append(text)
                metadatas.append(meta)
                ids.append(chunk_id)

        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def query_similar_chunks(self, query: str, top_k: int = 8, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        True Cosine Semantic Vector Search with Strict Document Isolation (Supports PDF, DOCX, TXT, etc.).
        """
        where_clause = None
        if file_name_filter:
            where_clause = {"file_name": file_name_filter}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
        except Exception as e:
            print(f"[VectorDB Query Error]: {e}")
            return []

        formatted_chunks = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []

            for i in range(len(docs)):
                formatted_chunks.append({
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {}
                })

        return formatted_chunks

    def get_all_chunks_for_file(self, file_name: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Retrieves exact chunks belonging strictly to the selected active document."""
        try:
            results = self.collection.get(where={"file_name": file_name})
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            chunks = []
            for i in range(min(len(docs), limit)):
                chunks.append({
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {}
                })
            return chunks
        except Exception as e:
            print(f"[VectorDB Document Get Error]: {e}")
            return []