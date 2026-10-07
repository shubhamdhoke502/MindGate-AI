import os
import re
import math
import zipfile
from collections import Counter
from typing import List, Dict, Optional, Any

import chromadb


class VectorStoreManager:
    """
    MindGate AI local document store.

    Responsibilities:
    - Persist PDF chunks + metadata in ChromaDB.
    - Perform real local BM25 retrieval.
    - Strictly filter results by active PDF.
    - No PyTorch / ONNX / external embedding model required.
    """

    COLLECTION_NAME = "mindgate_collection_v3"

    def __init__(self, storage_dir: str = "storage/.mindgate"):
        self.storage_dir = os.path.abspath(storage_dir)

        self.chroma_path = os.path.join(
            self.storage_dir,
            "chroma_db"
        )

        self.zip_db_path = os.path.join(
            self.storage_dir,
            "chroma_db.mgdb"
        )

        self.chunks_cache_dir = os.path.join(
            self.storage_dir,
            "chunks_cache"
        )

        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.chunks_cache_dir, exist_ok=True)

        # Persistent local Chroma database.
        self.client = chromadb.PersistentClient(
            path=self.chroma_path
        )

        # IMPORTANT:
        # No embedding_function here.
        # BM25 retrieval is implemented locally below.
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Unicode-aware tokenization.

        Supports English, Hindi, Marathi and other Unicode text.
        """
        if not text:
            return []

        text = text.lower()

        tokens = re.findall(
            r"\w+",
            text,
            flags=re.UNICODE
        )

        return tokens

    # =========================================================
    # INGESTION
    # =========================================================

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Inserts or updates parsed PDF chunks.
        """

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            chunk_id = str(
                chunk.get("chunk_id", "")
            ).strip()

            text = str(
                chunk.get("text", "")
            ).strip()

            metadata = chunk.get(
                "metadata",
                {}
            )

            if not chunk_id or not text:
                continue

            # Standardize filename metadata.
            file_name = (
                metadata.get("file_name")
                or metadata.get("filename")
            )

            if file_name:
                metadata["file_name"] = str(file_name)

            # Ensure numeric metadata is Chroma-compatible.
            if "page_number" in metadata:
                metadata["page_number"] = int(
                    metadata["page_number"]
                )

            if "line_start" in metadata:
                metadata["line_start"] = int(
                    metadata["line_start"]
                )

            if "line_end" in metadata:
                metadata["line_end"] = int(
                    metadata["line_end"]
                )

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f"[VectorStore] Upserted {len(ids)} chunks."
        )

    # Backward-compatible aliases.
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.upsert_chunks(chunks)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        self.upsert_chunks(chunks)

    # =========================================================
    # DOCUMENT DELETION
    # =========================================================

    def delete_document(self, file_name: str) -> int:
        """
        Deletes every indexed chunk belonging to one PDF.
        """

        if not file_name:
            return 0

        try:
            result = self.collection.get(
                where={
                    "file_name": file_name
                }
            )

            ids = result.get("ids", [])

            if ids:
                self.collection.delete(
                    ids=ids
                )

            print(
                f"[VectorStore] Deleted {len(ids)} chunks "
                f"for {file_name}"
            )

            return len(ids)

        except Exception as e:
            print(
                f"[VectorStore Delete Error]: {e}"
            )
            return 0

    # =========================================================
    # GET DOCUMENT CHUNKS
    # =========================================================

    def get_document_chunks(
        self,
        file_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns persisted chunks.

        If file_name is supplied, ONLY that PDF is returned.
        """

        try:

            if file_name:

                result = self.collection.get(
                    where={
                        "file_name": file_name
                    }
                )

            else:

                result = self.collection.get()

            documents = result.get(
                "documents",
                []
            )

            metadatas = result.get(
                "metadatas",
                []
            )

            ids = result.get(
                "ids",
                []
            )

            chunks = []

            for i, document in enumerate(documents):

                metadata = (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                )

                # Defensive strict filter.
                if file_name:
                    stored_file = metadata.get(
                        "file_name"
                    )

                    if stored_file != file_name:
                        continue

                chunks.append({
                    "chunk_id": (
                        ids[i]
                        if i < len(ids)
                        else ""
                    ),
                    "text": document or "",
                    "metadata": metadata
                })

            return chunks

        except Exception as e:

            print(
                f"[VectorStore Get Error]: {e}"
            )

            return []

    # =========================================================
    # BM25
    # =========================================================

    @staticmethod
    def _bm25_score(
        query_tokens: List[str],
        document_tokens: List[str],
        document_frequency: Counter,
        total_documents: int,
        avg_doc_length: float,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:

        if not document_tokens:
            return 0.0

        doc_length = len(
            document_tokens
        )

        term_frequency = Counter(
            document_tokens
        )

        score = 0.0

        for term in query_tokens:

            if term not in term_frequency:
                continue

            df = document_frequency.get(
                term,
                0
            )

            if df == 0:
                continue

            # BM25 IDF.
            idf = math.log(
                1.0
                +
                (
                    total_documents
                    - df
                    + 0.5
                )
                /
                (
                    df
                    + 0.5
                )
            )

            tf = term_frequency[term]

            denominator = (
                tf
                +
                k1
                *
                (
                    1
                    -
                    b
                    +
                    b
                    *
                    (
                        doc_length
                        /
                        max(avg_doc_length, 1.0)
                    )
                )
            )

            score += (
                idf
                *
                (
                    tf
                    *
                    (k1 + 1)
                )
                /
                denominator
            )

        return score

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def query_similar_chunks(
        self,
        query: str,
        top_k: int = 8,
        file_name_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Real local BM25 retrieval.

        IMPORTANT:
        The query is actually used here.

        file_name_filter is STRICT.
        """

        query = (query or "").strip()

        if not query:
            return []

        all_chunks = self.get_document_chunks(
            file_name=file_name_filter
        )

        if not all_chunks:
            return []

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        # -----------------------------------------------------
        # Document frequency
        # -----------------------------------------------------

        document_frequency = Counter()

        tokenized_documents = []

        for chunk in all_chunks:

            tokens = self._tokenize(
                chunk.get("text", "")
            )

            tokenized_documents.append(
                tokens
            )

            for term in set(tokens):
                document_frequency[term] += 1

        total_documents = len(
            all_chunks
        )

        avg_doc_length = (
            sum(
                len(tokens)
                for tokens in tokenized_documents
            )
            /
            max(total_documents, 1)
        )

        # -----------------------------------------------------
        # Score documents
        # -----------------------------------------------------

        scored = []

        for chunk, tokens in zip(
            all_chunks,
            tokenized_documents
        ):

            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=tokens,
                document_frequency=document_frequency,
                total_documents=total_documents,
                avg_doc_length=avg_doc_length
            )

            if score > 0:
                scored.append(
                    (
                        score,
                        chunk
                    )
                )

        # Highest score first.
        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        results = []

        for score, chunk in scored[:top_k]:

            result = dict(chunk)

            result["score"] = round(
                score,
                6
            )

            results.append(
                result
            )

        print(
            f"[BM25] query={query!r} "
            f"pdf={file_name_filter!r} "
            f"matches={len(results)}"
        )

        return results

    # Backward-compatible method.
    def query(
        self,
        query_text: str,
        top_k: int = 8,
        file_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        return self.query_similar_chunks(
            query=query_text,
            top_k=top_k,
            file_name_filter=file_filter
        )

    # =========================================================
    # DATABASE INFO
    # =========================================================

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    # =========================================================
    # COMPRESSION
    # =========================================================

    def compress_db(self) -> None:
        """
        Compresses Chroma storage into .mgdb.
        """

        try:

            if not os.path.exists(
                self.chroma_path
            ):
                return

            with zipfile.ZipFile(
                self.zip_db_path,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zipf:

                for root, _, files in os.walk(
                    self.chroma_path
                ):

                    for file in files:

                        file_path = os.path.join(
                            root,
                            file
                        )

                        arcname = os.path.relpath(
                            file_path,
                            self.chroma_path
                        )

                        zipf.write(
                            file_path,
                            arcname
                        )

            print(
                f"[VectorStore] Database compressed: "
                f"{self.zip_db_path}"
            )

        except Exception as e:

            print(
                f"[VectorStore Compression Error]: {e}"
            )

    def decompress_db(self) -> None:

        if not os.path.exists(
            self.zip_db_path
        ):
            return

        try:

            with zipfile.ZipFile(
                self.zip_db_path,
                "r"
            ) as zipf:

                zipf.extractall(
                    self.chroma_path
                )

            print(
                "[VectorStore] Database decompressed."
            )

        except Exception as e:

            print(
                f"[VectorStore Decompression Error]: {e}"
            )


if __name__ == "__main__":

    store = VectorStoreManager()

    print(
        "[MindGate Vector Store]"
    )

    print(
        "Collection:",
        store.COLLECTION_NAME
    )

    print(
        "Chunks:",
        store.count()
    )