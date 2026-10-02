import os
import shutil
import zipfile
import threading
import gc
import time
import math
from typing import List, Dict, Any, Optional
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions

try:
    import chromadb
except ImportError:
    chromadb = None


class BM25SearchEngine:
    """
    In-Memory BM25 (Sparse Keyword Search) Engine.
    Used alongside Vector DB to ensure exact keyword, serial number, and numeric accuracy.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.corpus: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def tokenize(self, text: str) -> List[str]:
        """Simple lowercase tokenization."""
        return text.lower().split()

    def index_documents(self, chunks: List[Dict[str, Any]]):
        """Indexes chunks for keyword-based retrieval."""
        self.doc_ids = []
        self.doc_metadatas = []
        self.corpus = []
        self.doc_len = []
        self.doc_freqs = []
        df_counter = Counter()

        for chunk in chunks:
            tokens = self.tokenize(chunk["text"])
            self.corpus.append(tokens)
            self.doc_ids.append(chunk["chunk_id"])
            self.doc_metadatas.append(chunk["metadata"])
            self.doc_len.append(len(tokens))

            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for word in freqs.keys():
                df_counter[word] += 1

        self.total_docs = len(self.corpus)
        self.avgdl = sum(self.doc_len) / self.total_docs if self.total_docs > 0 else 0.0

        # Calculate IDF values
        self.idf = {}
        for word, freq in df_counter.items():
            self.idf[word] = math.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Executes BM25 keyword matching with file metadata filtering."""
        if self.total_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.corpus):
            # Apply file isolation filter
            if file_name_filter and self.doc_metadatas[idx].get("file_name") != file_name_filter:
                continue

            score = 0.0
            doc_len = self.doc_len[idx]
            freqs = self.doc_freqs[idx]

            for token in query_tokens:
                if token in freqs:
                    freq = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = idf * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += numerator / denominator

            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "chunk_id": self.doc_ids[idx],
                "text": " ".join(self.corpus[idx]),
                "metadata": self.doc_metadatas[idx],
                "score": round(score, 4),
                "source": "bm25"
            })
        return results


class StorageCompressor:
    """Manages background idle compression of ChromaDB vector files (.mgdb ZIP format)."""
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
        self.last_activity_time = time.time()
        if self.is_compressed:
            self.decompress_db()

    def _idle_monitor_loop(self):
        while not self._stop_event.is_set():
            time.sleep(5)
            idle_duration = time.time() - self.last_activity_time
            if idle_duration >= self.idle_threshold_sec and not self.is_compressed:
                if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
                    self.compress_db()

    def compress_db(self):
        if not os.path.exists(self.db_dir):
            return
        
        # 1. Python garbage collection clear garbage/open file handles
        gc.collect()

        try:
            # 2. Archive creation inside try block
            with zipfile.ZipFile(self.archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.db_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.db_dir)
                        # Skip if attempting to zip archive into itself
                        if os.path.abspath(full_path) == os.path.abspath(self.archive_path):
                            continue
                        zipf.write(full_path, rel_path)
            
            # 3. Graceful deletion with lock exception handling
            for root, dirs, files in os.walk(self.db_dir, topdown=False):
                for file in files:
                    file_p = os.path.join(root, file)
                    if os.path.abspath(file_p) != os.path.abspath(self.archive_path):
                        try:
                            os.remove(file_p)
                        except Exception:
                            pass
            
            self.is_compressed = True
            print("[StorageCompressor]: Vector DB auto-compressed to .mgdb archive.")
        except Exception as e:
            # Fallback handling for Windows File Lock
            print(f"[StorageCompressor Warning]: Compression skipped due to active lock - {e}")

    def decompress_db(self):
        if not os.path.exists(self.archive_path):
            self.is_compressed = False
            return
        try:
            with zipfile.ZipFile(self.archive_path, 'r') as zipf:
                zipf.extractall(self.db_dir)
            try:
                os.remove(self.archive_path)
            except Exception:
                pass
            self.is_compressed = False
            print("[StorageCompressor]: Vector DB auto-restored from archive.")
        except Exception as e:
            print(f"[StorageCompressor Error]: Decompression failed - {e}")

    def stop(self):
        self._stop_event.set()

class VectorStoreManager:
    """
    Production Hybrid Vector Engine: Combines Dense Vector DB (ChromaDB)
    with Sparse Keyword Search (BM25) using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "storage", ".mindgate")
        
        self.db_dir = os.path.join(storage_dir, "chroma_db")
        self.archive_path = os.path.join(storage_dir, "chroma_db.mgdb")
        os.makedirs(self.db_dir, exist_ok=True)

        self.compressor = StorageCompressor(self.db_dir, self.archive_path)
        self.bm25_engine = BM25SearchEngine()
        self._client = None
        self._collection = None
        self._indexed_chunks = []

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
)
    def _get_collection(self):
        self.compressor.register_activity()
        if self._client is None or self.compressor.is_compressed:
            if chromadb:
                self._client = chromadb.PersistentClient(path=self.db_dir)
                self._collection = self._client.get_or_create_collection(
                    name="mindgate_pdf_chunks",
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
        return self._collection

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes chunks into both ChromaDB (Vector) and BM25 (Keyword Engine)."""
        collection = self._get_collection()
        if not collection or not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk["text"])
            meta = chunk["metadata"].copy()
            meta["bbox"] = str(meta.get("bbox", ""))
            metadatas.append(meta)
            ids.append(chunk["chunk_id"])

        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        
        # Index in BM25 Sparse Search Engine
        self._indexed_chunks.extend(chunks)
        self.bm25_engine.index_documents(self._indexed_chunks)

    def query_similar_chunks(self, query: str, top_k: int = 2, file_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Retrieval (Vector + BM25) fused via Reciprocal Rank Fusion (RRF).
        """
        collection = self._get_collection()
        if not collection:
            return []

        # 1. Vector Search Results
        where_clause = {"file_name": file_name_filter} if file_name_filter else None
        vector_res = collection.query(query_texts=[query], n_results=top_k * 2, where=where_clause)
        
        vector_chunks = []
        if vector_res and "documents" in vector_res and vector_res["documents"]:
            docs = vector_res["documents"][0]
            metas = vector_res["metadatas"][0]
            ids = vector_res["ids"][0]
            for i in range(len(docs)):
                meta = metas[i].copy()
                if isinstance(meta.get("bbox"), str):
                    try:
                        meta["bbox"] = eval(meta["bbox"])
                    except Exception:
                        pass
                vector_chunks.append({
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "metadata": meta,
                    "source": "vector"
                })

        # 2. BM25 Keyword Search Results
        bm25_chunks = self.bm25_engine.search(query, top_k=top_k * 2, file_name_filter=file_name_filter)

        # 3. Reciprocal Rank Fusion (RRF) Logic
        rrf_scores = {}
        chunk_map = {}
        rrf_k = 60

        for rank, chunk in enumerate(vector_chunks):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(bm25_chunks):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            chunk_map[cid] = chunk

        # Sort combined results by fusion score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_results = [chunk_map[cid] for cid in sorted_cids[:top_k]]

        return final_results


if __name__ == "__main__":
    print("[Production Hybrid VectorDB Engine]: Vector + BM25 Search Engine Ready.")