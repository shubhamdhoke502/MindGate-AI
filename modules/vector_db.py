import os
import zipfile
import shutil
import json
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# =====================================================================
# 1. Zero-Dependency Embedding Function for Pure BM25 / Custom RAG
# =====================================================================
class PureBM25EmbeddingFunction(EmbeddingFunction):
    """
    Zero-dependency custom embedding function for BM25-only / Hybrid RAG.
    Bypasses PyTorch (torch/c10.dll) and ONNX runtime completely.
    """
    def __call__(self, input: Documents) -> Embeddings:
        return [[0.0] * 384 for _ in input]


# =====================================================================
# 2. Main VectorStoreManager Class
# =====================================================================
class VectorStoreManager:
    def __init__(self, storage_dir="storage/.mindgate"):
        self.storage_dir = storage_dir
        self.chroma_path = os.path.join(self.storage_dir, "chroma_db")
        self.chunks_cache_dir = os.path.join(self.storage_dir, "chunks_cache")
        self.zip_db_path = os.path.join(self.storage_dir, "chroma_db.mgdb")

        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.chunks_cache_dir, exist_ok=True)

        self.embedding_fn = PureBM25EmbeddingFunction()

        self.client = chromadb.PersistentClient(path=self.chroma_path)

        self.collection = self.client.get_or_create_collection(
            name="mindgate_collection_v2",
            embedding_function=self.embedding_fn
        )

    def upsert_chunks(self, chunks: list):
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(metadata)

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"[VectorStore] Successfully upserted {len(chunks)} chunks.")
        except Exception as e:
            print(f"[VectorStore Error] Ingestion failed: {str(e)}")
            raise e

    # -----------------------------------------------------------------
    # Query Retrieval Function (Matched with app.py expectation)
    # -----------------------------------------------------------------
    def query_similar_chunks(self, query: str, top_k: int = 2, file_name_filter: str = None):
        """
        Retrieves top_k most relevant chunks.
        Accepts parameter name `file_name_filter` matching app.py call.
        """
        where_clause = None
        if file_name_filter:
            where_clause = {"file_name": file_name_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for i in range(len(docs)):
                retrieved_chunks.append({
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i]
                })

        return retrieved_chunks

    # Backward compatibility ke liye query alias
    def query(self, query_text: str, top_k: int = 2, file_filter: str = None):
        return self.query_similar_chunks(query=query_text, top_k=top_k, file_name_filter=file_filter)

    def compress_db(self):
        try:
            with zipfile.ZipFile(self.zip_db_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.chroma_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.chroma_path)
                        zipf.write(file_path, arcname)
            print(f"[VectorStore] Compressed DB successfully saved to {self.zip_db_path}")
        except Exception as e:
            print(f"[VectorStore Compression Error]: {str(e)}")

    def decompress_db(self):
        if os.path.exists(self.zip_db_path):
            try:
                with zipfile.ZipFile(self.zip_db_path, 'r') as zipf:
                    zipf.extractall(self.chroma_path)
                print(f"[VectorStore] Decompressed DB successfully into {self.chroma_path}")
            except Exception as e:
                print(f"[VectorStore Decompression Error]: {str(e)}")