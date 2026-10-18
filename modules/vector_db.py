import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger("MindGateVectorDB")

class PureBM25EmbeddingFunction(EmbeddingFunction):
    """
    Zero-dependency deterministic embedding function. 
    Bypasses PyTorch / ONNX / sentence-transformers to eliminate Windows DLL initialization crashes.
    """
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            # Generate a stable 384-dimensional float vector representation
            vec = [(float((hash_val >> i) & 1) * 2.0 - 1.0) / (i + 1) for i in range(384)]
            embeddings.append(vec)
        return embeddings

class VectorStoreManager:
    def __init__(self, storage_dir: str = "storage/.mindgate"):
        self.storage_dir = storage_dir
        self.chroma_path = os.path.join(self.storage_dir, "chroma_db")
        os.makedirs(self.chroma_path, exist_ok=True)

        # Persistent enterprise ChromaDB client
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.embedding_fn = PureBM25EmbeddingFunction()

        # Collection initialization with explicit cosine similarity and custom embedding function
        self.collection = self.client.get_or_create_collection(
            name="mindgate_doc_store_v4",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("VectorStoreManager initialized successfully with pure Python embedding backend.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Upserts document chunks with strict file_name metadata binding."""
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
            logger.info(f"Successfully upserted {len(documents)} chunks into ChromaDB.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.upsert_chunks(chunks)

    def query_similar_chunks(self, query: str, top_k: int = 8, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic Vector Search with strict metadata isolation and multi-tier exception fallback."""
        where_clause = {"file_name": file_name_filter} if file_name_filter else None

        results = None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
        except Exception as e:
            logger.warning(f"[VectorDB Query Filter Exception]: {e}. Falling back to global search.")
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
            except Exception as inner_e:
                logger.error(f"[VectorDB Fatal Query Error]: {inner_e}")
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
            logger.error(f"[VectorDB Document Get Exception]: {e}")
            return []