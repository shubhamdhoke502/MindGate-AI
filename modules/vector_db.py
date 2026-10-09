
import os
import re
import math
import hashlib
import zipfile
from collections import Counter
from typing import List, Dict, Optional, Any

import chromadb


class VectorStoreManager:
    """
    MindGate AI local document store.

    Storage: ChromaDB
    Retrieval: Pure Python BM25
    Embeddings: Explicit local placeholder vectors only.
    No ONNX, PyTorch, or sentence-transformers required by this module.
    """

    COLLECTION_NAME = "mindgate_collection_v4"
    EMBEDDING_DIMENSION = 384

    def __init__(self, storage_dir: str = "storage/.mindgate"):
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        if not os.path.isabs(storage_dir):
            storage_dir = os.path.join(project_root, storage_dir)

        self.storage_dir = os.path.abspath(storage_dir)
        self.chroma_path = os.path.join(
            self.storage_dir, "chroma_db"
        )
        self.chunks_cache_dir = os.path.join(
            self.storage_dir, "chunks_cache"
        )
        self.zip_db_path = os.path.join(
            self.storage_dir, "chroma_db.mgdb"
        )

        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.chunks_cache_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.chroma_path
        )

        # Explicitly disable Chroma's default embedding function.
        # upsert_chunks() always supplies embeddings explicitly.
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------
    # TEXT / METADATA HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Unicode tokenization for English, Hindi and Marathi."""
        if not text:
            return []

        return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)

    @staticmethod
    def _clean_metadata_value(value: Any) -> Any:
        """Chroma metadata supports scalar values, not None or objects."""
        if value is None:
            return ""

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    @staticmethod
    def _get_text(chunk: Dict) -> str:
        return str(
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("document")
            or ""
        ).strip()

    def _make_id(self, chunk: Dict, file_name: str, index: int) -> str:
        """
        Generate a stable ID that avoids collisions between PDFs
        whose chunk IDs both start at zero.
        """
        identity = "|".join([
            file_name,
            str(chunk.get("chunk_id", index)),
            str(chunk.get("page_number", "")),
            str(chunk.get("line_start", "")),
            str(chunk.get("line_end", "")),
            self._get_text(chunk)
        ])

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    def _placeholder_embedding(self, text: str) -> List[float]:
        """
        Deterministic hash-based vector.

        This is NOT a semantic embedding and is NOT used for retrieval.
        Its only purpose is to satisfy Chroma's storage requirements
        without invoking ONNX or downloading an embedding model.
        BM25 performs the actual retrieval.
        """
        vector = [0.0] * self.EMBEDDING_DIMENSION

        for token in self._tokenize(text):
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8
            ).digest()

            number = int.from_bytes(digest, "little")
            index = number % self.EMBEDDING_DIMENSION
            sign = 1.0 if (number >> 8) & 1 else -1.0
            vector[index] += sign

        # Keep a valid, non-zero vector even for empty text.
        if not any(vector):
            vector[0] = 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    # ---------------------------------------------------------
    # DOCUMENT INGESTION
    # ---------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict]) -> int:
        """
        Save parsed chunks to ChromaDB.

        Every upsert supplies explicit embeddings, so Chroma's default
        ONNX embedding function is not called.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            text = self._get_text(chunk)
            if not text:
                continue

            raw_metadata = chunk.get("metadata")
            metadata = (
                dict(raw_metadata)
                if isinstance(raw_metadata, dict)
                else {}
            )

            # Support both metadata-nested and top-level parser output.
            for key, value in chunk.items():
                if key not in {
                    "text", "content", "document",
                    "metadata", "embedding"
                }:
                    metadata.setdefault(key, value)

            file_name = str(
                metadata.get("file_name")
                or metadata.get("filename")
                or chunk.get("file_name")
                or chunk.get("filename")
                or "unknown_document"
            )

            # Standardize the document filename field.
            metadata.pop("filename", None)
            metadata["file_name"] = file_name

            metadata.setdefault("page_number", 0)
            metadata.setdefault("line_start", 0)
            metadata.setdefault("line_end", 0)
            metadata.setdefault("chunk_id", index)

            safe_metadata = {
                key: self._clean_metadata_value(value)
                for key, value in metadata.items()
                if key != "embedding"
            }

            ids.append(self._make_id(chunk, file_name, index))
            documents.append(text)
            metadatas.append(safe_metadata)
            embeddings.append(self._placeholder_embedding(text))

        if not documents:
            return 0

        # Batch writes to avoid oversized requests.
        batch_size = 128

        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end]
            )

        print(
            f"[VectorStore] Stored {len(documents)} chunks "
            f"in {self.COLLECTION_NAME}"
        )
        return len(documents)

    # ---------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # ---------------------------------------------------------

    def delete_document(self, file_name: str) -> int:
        """Delete chunks belonging to exactly one filename."""
        if not file_name:
            return 0

        result = self.collection.get(
            where={"file_name": os.path.basename(file_name)},
            include=["metadatas"]
        )

        ids = result.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

        print(
            f"[VectorStore] Deleted {len(ids)} chunks "
            f"for {file_name}"
        )
        return len(ids)

    def get_document_chunks(
        self,
        file_name: Optional[str] = None
    ) -> List[Dict]:
        """Return stored chunks, optionally for one exact filename."""
        if file_name:
            result = self.collection.get(
                where={"file_name": os.path.basename(file_name)},
                include=["documents", "metadatas"]
            )
        else:
            result = self.collection.get(
                include=["documents", "metadatas"]
            )

        ids = result.get("ids", [])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks = []

        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )

            chunks.append({
                "id": chunk_id,
                "text": document or "",
                "document": document or "",
                "metadata": metadata,
                **metadata
            })

        return chunks

    # ---------------------------------------------------------
    # BM25 RETRIEVAL
    # ---------------------------------------------------------

    @staticmethod
    def _bm25_score(
        query_tokens: List[str],
        document_tokens: List[str],
        document_frequency: Counter,
        total_documents: int,
        average_length: float,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """Compute a standard BM25 lexical relevance score."""
        if not query_tokens or not document_tokens:
            return 0.0

        frequencies = Counter(document_tokens)
        doc_length = len(document_tokens)
        score = 0.0

        for term in set(query_tokens):
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            df = document_frequency.get(term, 0)

            idf = math.log(
                1.0 + (
                    total_documents - df + 0.5
                ) / (df + 0.5)
            )

            denominator = term_frequency + k1 * (
                1.0 - b + b * doc_length / max(average_length, 1.0)
            )

            score += idf * (
                term_frequency * (k1 + 1.0)
            ) / denominator

        return score

    def query_similar_chunks(
        self,
        query: str,
        top_k: int = 5,
        file_name_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Rank documents with BM25.

        The query is actually used for ranking; results are not simply
        the first stored chunks. A file filter is exact and strict.
        """
        query_tokens = self._tokenize(query)

        if not query_tokens or top_k <= 0:
            return []

        # Fetch only the selected document when a filename is supplied.
        chunks = self.get_document_chunks(
            os.path.basename(file_name_filter)
            if file_name_filter else None
        )

        if not chunks:
            return []

        tokenized_documents = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        total_documents = len(tokenized_documents)
        document_frequency = Counter()

        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        average_length = (
            sum(len(tokens) for tokens in tokenized_documents)
            / max(total_documents, 1)
        )

        ranked = []

        for chunk, tokens in zip(chunks, tokenized_documents):
            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=tokens,
                document_frequency=document_frequency,
                total_documents=total_documents,
                average_length=average_length
            )

            if score > 0:
                chunk["score"] = score
                ranked.append(chunk)

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return ranked[:top_k]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        file_filter: Optional[str] = None
    ) -> List[Dict]:
        """Backward-compatible alias used by existing code."""
        return self.query_similar_chunks(
            query=query_text,
            top_k=top_k,
            file_name_filter=file_filter
        )

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def compress_db(self) -> str:
        """Create a local .mgdb ZIP snapshot of the Chroma directory."""
        if os.path.exists(self.zip_db_path):
            os.remove(self.zip_db_path)

        with zipfile.ZipFile(
            self.zip_db_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for root, _, files in os.walk(self.chroma_path):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(
                        full_path, self.chroma_path
                    )
                    archive.write(full_path, relative_path)

        return self.zip_db_path

    def decompress_db(self) -> bool:
        """
        Restore a .mgdb snapshot into the Chroma directory.

        Stop the running server before restoring a snapshot.
        """
        if not os.path.isfile(self.zip_db_path):
            return False

        with zipfile.ZipFile(self.zip_db_path, "r") as archive:
            base = os.path.abspath(self.chroma_path)

            # Prevent archive paths escaping the target directory.
            for member in archive.infolist():
                destination = os.path.abspath(
                    os.path.join(base, member.filename)
                )
                if os.path.commonpath([base, destination]) != base:
                    raise ValueError(
                        "Unsafe path found in database archive"
                    )

            archive.extractall(base)

        return True
