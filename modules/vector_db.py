import os
import chromadb
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    def __init__(self, storage_dir: str = "storage/.mindgate"):
        self.storage_dir = storage_dir
        self.chroma_path = os.path.join(self.storage_dir, "chroma_db")
        os.makedirs(self.chroma_path, exist_ok=True)

        # Persistent Chroma DB Client
        self.client = chromadb.PersistentClient(path=self.chroma_path)

        # Collection with default semantic embeddings (Cosine Metric)
        self.collection = self.client.get_or_create_collection(
            name="mindgate_doc_store_v4",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Upserts document chunks (PDF, DOCX, TXT) with strict file_name metadata binding."""
        if not chunks:
            return

        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id")
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            if text and chunk_id:
                ids.append(str(chunk_id))
                documents.append(text)
                metadatas.append(meta)

        if documents:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.upsert_chunks(chunks)

    def query_similar_chunks(self, query: str, top_k: int = 8, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        True Cosine Semantic Vector Search with Strict Metadata Isolation.
        """
        where_clause = {"file_name": file_name_filter} if file_name_filter else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
        except Exception as e:
            print(f"[VectorDB Query Exception]: {e}")
            return []

        formatted = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            ids = results["ids"][0] if results.get("ids") else []

            for i in range(len(docs)):
                formatted.append({
                    "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {}
                })

        return formatted

    def get_all_chunks_for_file(self, file_name: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Retrieves structural top chunks directly for overview/summary queries."""
        try:
            results = self.collection.get(where={"file_name": file_name})
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            ids = results.get("ids", [])

            chunks = []
            for i in range(min(len(docs), limit)):
                chunks.append({
                    "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {}
                })
            return chunks
        except Exception as e:
            print(f"[VectorDB Document Get Exception]: {e}")
            return []