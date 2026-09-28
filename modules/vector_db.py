import os
import shutil
import zipfile
import threading
import time
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class StorageCompressor:
    """
    Manages background idle compression of ChromaDB vector files
    to save up to 80% local disk storage footprint (.mgdb ZIP format).
    """
    def __init__(self, db_dir: str, archive_path: str, idle_threshold_sec: int = 30):
        self.db_dir = db_dir
        self.archive_path = archive_path
        self.idle_threshold_sec = idle_threshold_sec
        self.last_activity_time = time.time()
        self.is_compressed = False
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._idle_monitor_loop, daemon=True)
        self._worker_thread.start()

    def register_activity(self):
        """Resets the idle timer on active user interaction."""
        self.last_activity_time = time.time()
        if self.is_compressed:
            self.decompress_db()

    def _idle_monitor_loop(self):
        """Monitors system activity and compresses DB when idle."""
        while not self._stop_event.is_set():
            time.sleep(5)
            idle_duration = time.time() - self.last_activity_time
            if idle_duration >= self.idle_threshold_sec and not self.is_compressed:
                if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
                    self.compress_db()

    def compress_db(self):
        """Compresses ChromaDB folder into .mgdb ZIP archive."""
        if not os.path.exists(self.db_dir):
            return
        
        try:
            with zipfile.ZipFile(self.archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.db_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.db_dir)
                        zipf.write(full_path, rel_path)
            
            shutil.rmtree(self.db_dir)
            os.makedirs(self.db_dir, exist_ok=True)
            self.is_compressed = True
            print("[StorageCompressor]: Vector DB successfully compressed to .mgdb archive.")
        except Exception as e:
            print(f"[StorageCompressor Error]: Compression failed - {e}")

    def decompress_db(self):
        """Restores ChromaDB folder from .mgdb ZIP archive."""
        if not os.path.exists(self.archive_path):
            self.is_compressed = False
            return
        
        try:
            with zipfile.ZipFile(self.archive_path, 'r') as zipf:
                zipf.extractall(self.db_dir)
            
            os.remove(self.archive_path)
            self.is_compressed = False
            print("[StorageCompressor]: Vector DB restored from .mgdb archive.")
        except Exception as e:
            print(f"[StorageCompressor Error]: Decompression failed - {e}")

    def stop(self):
        self._stop_event.set()


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector storage with strict dynamic metadata filters.
    Supports isolation per document name and background ZIP compression.
    """
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "storage", ".mindgate")
        
        self.db_dir = os.path.join(storage_dir, "chroma_db")
        self.archive_path = os.path.join(storage_dir, "chroma_db.mgdb")
        os.makedirs(self.db_dir, exist_ok=True)

        self.compressor = StorageCompressor(self.db_dir, self.archive_path)
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Ensures DB is uncompressed and returns active Chroma collection."""
        self.compressor.register_activity()
        
        if self._client is None or self.compressor.is_compressed:
            if chromadb:
                self._client = chromadb.PersistentClient(path=self.db_dir)
                self._collection = self._client.get_or_create_collection(
                    name="mindgate_pdf_chunks",
                    metadata={"hnsw:space": "cosine"}
                )
        return self._collection

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds text chunks with metadata (page, lines, bbox) into ChromaDB."""
        collection = self._get_collection()
        if not collection or not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk["text"])
            # Flat metadata for ChromaDB compatibility
            meta = chunk["metadata"].copy()
            meta["bbox"] = str(meta["bbox"])  # Store bbox list as string representation
            metadatas.append(meta)
            ids.append(chunk["chunk_id"])

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_similar_chunks(self, query: str, top_k: int = 2, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks with strict file_name namespace filter.
        Prevents cross-contamination between different uploaded PDFs.
        """
        collection = self._get_collection()
        if not collection:
            return []

        where_clause = None
        if file_name_filter:
            where_clause = {"file_name": file_name_filter}

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                meta = metas[i].copy()
                # Restore bbox back to array
                if isinstance(meta.get("bbox"), str):
                    try:
                        meta["bbox"] = eval(meta["bbox"])
                    except Exception:
                        pass

                formatted_results.append({
                    "text": docs[i],
                    "metadata": meta,
                    "score": distances[i] if i < len(distances) else 0.0
                })

        return formatted_results


if __name__ == "__main__":
    print("[VectorDB Module]: Persistent Vector Store with Idle Compression & Strict Filters Ready.")