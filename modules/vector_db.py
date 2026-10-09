import os
import zipfile
import shutil
import json
import chromadb
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    def __init__(self, storage_dir: str = "storage/.mindgate"):
        self.storage_dir = storage_dir
        self.chroma_path = os.path.join(self.storage_dir, "chroma_db")
        self.chunks_cache_dir = os.path.join(self.storage_dir, "chunks_cache")
        self.zip_db_path = os.path.join(self.storage_dir, "chroma_db.mgdb")

        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.chunks_cache_dir, exist_ok=True)

        # Persistent Chroma Client
        self.client = chromadb.PersistentClient(path=self.chroma_path)

        # Get or create collection with default Chroma embeddings (Cosine space)
        self.collection = self.client.get_or_create_collection(
            name="mindgate_doc_chunks_v3",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Upserts document chunks (PDF, DOCX, TXT, etc.) with strict metadata binding.
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id")
            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})

            if text and chunk_id:
                ids.append(str(chunk_id))
                documents.append(text)
                metadatas.append(metadata)

        if documents:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"[VectorStore] Successfully upserted {len(documents)} chunks.")
            except Exception as e:
                print(f"[VectorStore Error] Ingestion failed: {str(e)}")
                raise e

    def query_similar_chunks(self, query: str, top_k: int = 5, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        True Cosine Semantic Vector Search using ChromaDB's query engine.
        Applies strict metadata filter to isolate the active document.
        """
        retrieved_chunks = []
        
        # Strict document level filtering
        where_clause = None
        if file_name_filter:
            where_clause = {"file_name": file_name_filter}

        try:
            # Native ChromaDB Semantic Similarity Search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )

            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else []
                ids = results["ids"][0] if results.get("ids") else []

                for i in range(len(docs)):
                    retrieved_chunks.append({
                        "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                        "text": docs[i],
                        "metadata": metas[i] if i < len(metas) else {}
                    })

        except Exception as e:
            print(f"[VectorStore Query Error]: {str(e)}")

        return retrieved_chunks

    def query(self, query_text: str, top_k: int = 5, file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.query_similar_chunks(query=query_text, top_k=top_k, file_name_filter=file_filter)

    def get_all_chunks_for_file(self, file_name: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Retrieves initial document chunks for overview queries (e.g. 'summary', 'isme kya hai')."""
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
            print(f"[VectorStore Document Get Error]: {str(e)}")
            return []

    def compress_db(self) -> None:
        try:
            with zipfile.ZipFile(self.zip_db_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.chroma_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.chroma_path)
                        zipf.write(file_path, arcname)
        except Exception as e:
            print(f"[VectorStore Compression Error]: {str(e)}")

    def decompress_db(self) -> None:
        if os.path.exists(self.zip_db_path):
            try:
                with zipfile.ZipFile(self.zip_db_path, 'r') as zipf:
                    zipf.extractall(self.chroma_path)
            except Exception as e:
                print(f"[VectorStore Decompression Error]: {str(e)}")