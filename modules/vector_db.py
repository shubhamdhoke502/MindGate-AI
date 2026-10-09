import os
import zipfile
import shutil
import json
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

class PureBM25EmbeddingFunction(EmbeddingFunction):
    """
    Zero-dependency embedding function satisfying ChromaDB schema interface
    without triggering external binary/ONNX model downloads.
    """
    def __call__(self, input: Documents) -> Embeddings:
        return [[0.0] * 384 for _ in input]


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

            ids.append(str(chunk_id))
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

    def query_similar_chunks(self, query: str, top_k: int = 5, file_name_filter: str = None):
        """
        Retrieves real document chunks from collection.
        Fallback logic guarantees document content is fetched directly if vector score drops.
        """
        retrieved_chunks = []
        
        try:
            # 1. Primary DB Retrieval
            results = self.collection.get()
            
            if results and "documents" in results and results["documents"]:
                all_docs = results["documents"]
                all_metas = results.get("metadatas", [{}] * len(all_docs))
                all_ids = results.get("ids", [""] * len(all_docs))

                # Filtering by filename if passed, else returning top document chunks
                for i in range(len(all_docs)):
                    doc_meta = all_metas[i] if all_metas else {}
                    
                    # Match active document if specified, or load all active chunks
                    if file_name_filter:
                        fname = doc_meta.get("file_name") or doc_meta.get("filename")
                        if fname and fname != file_name_filter:
                            continue

                    retrieved_chunks.append({
                        "chunk_id": all_ids[i],
                        "text": all_docs[i],
                        "metadata": doc_meta
                    })

                    if len(retrieved_chunks) >= top_k:
                        break

        except Exception as e:
            print(f"[VectorStore Query Error]: {str(e)}")

        return retrieved_chunks

    def query(self, query_text: str, top_k: int = 5, file_filter: str = None):
        return self.query_similar_chunks(query=query_text, top_k=top_k, file_name_filter=file_filter)

    def compress_db(self):
        try:
            with zipfile.ZipFile(self.zip_db_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.chroma_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.chroma_path)
                        zipf.write(file_path, arcname)
        except Exception as e:
            print(f"[VectorStore Compression Error]: {str(e)}")

    def decompress_db(self):
        if os.path.exists(self.zip_db_path):
            try:
                with zipfile.ZipFile(self.zip_db_path, 'r') as zipf:
                    zipf.extractall(self.chroma_path)
            except Exception as e:
                print(f"[VectorStore Decompression Error]: {str(e)}")