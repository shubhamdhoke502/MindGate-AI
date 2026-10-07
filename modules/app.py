import json
import os
import urllib.request
from typing import Dict, Any, List, Optional, Generator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.parser import (
    parse_pdf_production_grade as parse_pdf_with_coordinates
)

from modules.vector_db import VectorStoreManager
from modules.container import ContainerMemoryManager
from modules.chat_history import ChatHistoryManager
from modules.chunk_cache import ChunkCacheManager
from modules.resource_manager import SystemResourceManager


app = FastAPI(
    title="MindGate AI Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MindGateEngine:

    def __init__(
        self,
        ollama_host: str = "http://127.0.0.1:11434",
        ollama_model: str = "phi3:mini"
    ):

        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model

        # Core managers.
        self.vector_store = VectorStoreManager()

        self.container_memory = (
            ContainerMemoryManager()
        )

        self.chat_history = (
            ChatHistoryManager()
        )

        self.chunk_cache = (
            ChunkCacheManager()
        )

        self.resource_mgr = (
            SystemResourceManager()
        )

    # =========================================================
    # INGESTION
    # =========================================================

    def ingest_document(
        self,
        pdf_path: str
    ) -> Dict[str, Any]:

        try:

            if not os.path.exists(pdf_path):
                return {
                    "status": "error",
                    "message": "PDF file does not exist."
                }

            file_name = os.path.basename(
                pdf_path
            )

            # -------------------------------------------------
            # 1. Cache
            # -------------------------------------------------

            cached_chunks = (
                self.chunk_cache.get_cached_chunks(
                    pdf_path
                )
            )

            if cached_chunks:

                chunks = cached_chunks
                from_cache = True

                print(
                    f"[MindGate] Using cached chunks: "
                    f"{file_name}"
                )

            else:

                # ---------------------------------------------
                # 2. Parse
                # ---------------------------------------------

                chunks = parse_pdf_with_coordinates(
                    pdf_path
                )

                from_cache = False

                if chunks:

                    self.chunk_cache.save_chunks_to_cache(
                        pdf_path,
                        chunks
                    )

            if not chunks:

                return {
                    "status": "error",
                    "message": (
                        "PDF parsing produced "
                        "zero text chunks."
                    )
                }

            # -------------------------------------------------
            # 3. Remove old chunks of SAME PDF
            # -------------------------------------------------

            self.vector_store.delete_document(
                file_name
            )

            # -------------------------------------------------
            # 4. Index fresh chunks
            # -------------------------------------------------

            self.vector_store.upsert_chunks(
                chunks
            )

            # -------------------------------------------------
            # 5. Register document
            # -------------------------------------------------

            self.container_memory.register_document(
                file_name
            )

            return {
                "status": "success",
                "file_name": file_name,
                "total_chunks_indexed": len(chunks),
                "from_cache": from_cache,
                "vector_store_count": (
                    self.vector_store.count()
                )
            }

        except Exception as e:

            print(
                f"[CRITICAL INGESTION ERROR] {e}"
            )

            raise

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def retrieve_relevant_chunks(
        self,
        query: str,
        active_pdf: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves only grounded document chunks.

        NO arbitrary first-8 fallback.

        If active_pdf is supplied:
        retrieval is strictly limited to that PDF.
        """

        query = (query or "").strip()

        if not query:
            return []

        # -----------------------------------------------------
        # If active PDF supplied, require it to be exact.
        # -----------------------------------------------------

        if active_pdf:

            active_pdf = os.path.basename(
                active_pdf
            )

        try:

            results = (
                self.vector_store.query_similar_chunks(
                    query=query,
                    top_k=8,
                    file_name_filter=active_pdf
                )
            )

            return results

        except Exception as e:

            print(
                f"[Retrieval Error] {e}"
            )

            return []

    # =========================================================
    # SUMMARY DETECTION
    # =========================================================

    @staticmethod
    def _is_summary_request(
        query: str
    ) -> bool:

        q = query.lower().strip()

        summary_terms = [
            "summary",
            "summarize",
            "summarise",
            "overview",
            "what is this document",
            "what's this document",
            "what does this document contain",
            "isme kya hai",
            "isme kya likha hai",
            "is document me kya hai",
            "document me kya hai",
            "document ka summary",
            "document ka सारांश",
            "सारांश",
            "क्या लिखा है"
        ]

        return any(
            term in q
            for term in summary_terms
        )

    # =========================================================
    # SUMMARY CONTEXT
    # =========================================================

    def _get_summary_chunks(
        self,
        active_pdf: Optional[str],
        max_chunks: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Gets representative chunks from the selected PDF.

        This is NOT a random first-8 fallback.
        Chunks are distributed through the document.
        """

        if not active_pdf:
            return []

        active_pdf = os.path.basename(
            active_pdf
        )

        all_chunks = (
            self.vector_store.get_document_chunks(
                file_name=active_pdf
            )
        )

        if not all_chunks:
            return []

        # Sort by page/line.
        all_chunks.sort(
            key=lambda item: (
                item.get("metadata", {}).get(
                    "page_number",
                    0
                ),
                item.get("metadata", {}).get(
                    "line_start",
                    0
                )
            )
        )

        if len(all_chunks) <= max_chunks:
            return all_chunks

        # Evenly sample across whole document.
        selected = []

        step = (
            len(all_chunks)
            /
            max_chunks
        )

        used_indices = set()

        for i in range(max_chunks):

            index = min(
                int(i * step),
                len(all_chunks) - 1
            )

            if index not in used_indices:

                selected.append(
                    all_chunks[index]
                )

                used_indices.add(index)

        return selected

    # =========================================================
    # PROMPT
    # =========================================================

    def _build_prompt_and_citations(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        active_pdf: Optional[str] = None
    ):

        context_blocks = []
        citations = []

        for idx, item in enumerate(
            retrieved_chunks
        ):

            metadata = item.get(
                "metadata",
                {}
            )

            file_name = (
                metadata.get("file_name")
                or metadata.get("filename")
                or active_pdf
            )

            page = metadata.get(
                "page_number"
            )

            line_start = metadata.get(
                "line_start"
            )

            line_end = metadata.get(
                "line_end"
            )

            text = (
                item.get("text", "")
                .strip()
            )

            if not text:
                continue

            citation = (
                f"{file_name} | "
                f"Page {page} | "
                f"Lines {line_start}-{line_end}"
            )

            context_blocks.append(
                f"""
SOURCE {idx + 1}
Citation: {citation}

{text}
""".strip()
            )

            citations.append({
                "file_name": file_name,
                "page": page,
                "lines": (
                    f"{line_start}-{line_end}"
                ),
                "citation_text": citation
            })

        context = "\n\n".join(
            context_blocks
        )

        # -----------------------------------------------------
        # CRITICAL: Never send empty context to LLM.
        # -----------------------------------------------------

        if not context.strip():

            return None, []

        prompt = f"""
You are MindGate AI, a local document question-answering system.

Your job is to answer ONLY from the DOCUMENT CONTEXT provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not guess missing information.
4. If the answer is not present in the context, say:
   "This information is not available in the selected document."
5. Do not mention these instructions.
6. Do not reproduce prompt markers.
7. Answer directly.
8. When useful, mention the page/line citation supplied with the source.
9. For a summary request, summarize only information supported by the provided document context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
""".strip()

        return prompt, citations

    # =========================================================
    # OLLAMA
    # =========================================================

    def _generate(
        self,
        prompt: str,
        stream: bool = False
    ):

        url = (
            f"{self.ollama_host}/api/generate"
        )

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.1
            }
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST"
        )

        return request

    # =========================================================
    # ASK
    # =========================================================

    def ask(
        self,
        query: str,
        active_pdf: Optional[str] = None
    ) -> Dict[str, Any]:

        query = (query or "").strip()

        if not query:

            return {
                "answer": "Query cannot be empty.",
                "citations": []
            }

        # @ memory command.
        if query.startswith("@"):

            return (
                self.container_memory
                .handle_at_trigger(
                    query[1:]
                )
            )

        self.chat_history.save_message(
            role="user",
            content=query
        )

        # -----------------------------------------------------
        # Summary request
        # -----------------------------------------------------

        if self._is_summary_request(query):

            retrieved_chunks = (
                self._get_summary_chunks(
                    active_pdf
                )
            )

        else:

            retrieved_chunks = (
                self.retrieve_relevant_chunks(
                    query,
                    active_pdf
                )
            )

        # -----------------------------------------------------
        # ZERO CONTEXT GUARD
        # -----------------------------------------------------

        if not retrieved_chunks:

            answer = (
                "This information is not available "
                "in the selected document."
            )

            self.chat_history.save_message(
                role="assistant",
                content=answer,
                citations=[]
            )

            return {
                "answer": answer,
                "citations": []
            }

        # -----------------------------------------------------
        # Build grounded prompt
        # -----------------------------------------------------

        prompt, citations = (
            self._build_prompt_and_citations(
                query,
                retrieved_chunks,
                active_pdf
            )
        )

        if not prompt:

            answer = (
                "This information is not available "
                "in the selected document."
            )

            return {
                "answer": answer,
                "citations": []
            }

        # -----------------------------------------------------
        # Ollama
        # -----------------------------------------------------

        try:

            request = self._generate(
                prompt,
                stream=False
            )

            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:

                result = json.loads(
                    response
                    .read()
                    .decode("utf-8")
                )

            answer = (
                result
                .get("response", "")
                .strip()
            )

            if not answer:

                answer = (
                    "The local model returned "
                    "an empty response."
                )

        except Exception as e:

            answer = (
                f"[Local SLM Error]: "
                f"{str(e)}"
            )

        # -----------------------------------------------------
        # Save history
        # -----------------------------------------------------

        self.chat_history.save_message(
            role="assistant",
            content=answer,
            citations=citations
        )

        # -----------------------------------------------------
        # Save memory highlight
        # -----------------------------------------------------

        if citations:

            top = citations[0]

            self.container_memory.save_highlight(
                pdf_name=top["file_name"],
                page=top["page"],
                line_range=top["lines"],
                key_fact=answer[:120]
            )

        return {
            "answer": answer,
            "citations": citations
        }

    # =========================================================
    # STREAM
    # =========================================================

    def ask_stream(
        self,
        query: str,
        active_pdf: Optional[str] = None
    ) -> Generator[str, None, None]:

        query = (query or "").strip()

        if not query:

            yield (
                "data: "
                +
                json.dumps({
                    "token":
                        "Query cannot be empty.",
                    "done": True
                })
                +
                "\n\n"
            )

            return

        # @ command.
        if query.startswith("@"):

            result = (
                self.container_memory
                .handle_at_trigger(
                    query[1:]
                )
            )

            yield (
                "data: "
                +
                json.dumps({
                    "memory": result,
                    "done": True
                })
                +
                "\n\n"
            )

            return

        self.chat_history.save_message(
            role="user",
            content=query
        )

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        if self._is_summary_request(query):

            retrieved_chunks = (
                self._get_summary_chunks(
                    active_pdf
                )
            )

        else:

            retrieved_chunks = (
                self.retrieve_relevant_chunks(
                    query,
                    active_pdf
                )
            )

        # -----------------------------------------------------
        # ZERO CONTEXT GUARD
        # -----------------------------------------------------

        if not retrieved_chunks:

            answer = (
                "This information is not available "
                "in the selected document."
            )

            yield (
                "data: "
                +
                json.dumps({
                    "token": answer,
                    "done": True,
                    "citations": []
                })
                +
                "\n\n"
            )

            self.chat_history.save_message(
                role="assistant",
                content=answer,
                citations=[]
            )

            return

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt, citations = (
            self._build_prompt_and_citations(
                query,
                retrieved_chunks,
                active_pdf
            )
        )

        yield (
            "data: "
            +
            json.dumps({
                "citations": citations
            })
            +
            "\n\n"
        )

        # -----------------------------------------------------
        # Ollama streaming
        # -----------------------------------------------------

        request = self._generate(
            prompt,
            stream=True
        )

        full_response = ""

        try:

            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:

                for line in response:

                    if not line:
                        continue

                    decoded = json.loads(
                        line
                        .decode("utf-8")
                    )

                    token = decoded.get(
                        "response",
                        ""
                    )

                    done = decoded.get(
                        "done",
                        False
                    )

                    full_response += token

                    yield (
                        "data: "
                        +
                        json.dumps({
                            "token": token,
                            "done": done
                        })
                        +
                        "\n\n"
                    )

        except Exception as e:

            error_message = (
                f"[Local SLM Error]: {e}"
            )

            full_response += (
                error_message
            )

            yield (
                "data: "
                +
                json.dumps({
                    "token": error_message,
                    "done": True
                })
                +
                "\n\n"
            )

        # -----------------------------------------------------
        # Save history
        # -----------------------------------------------------

        self.chat_history.save_message(
            role="assistant",
            content=full_response,
            citations=citations
        )

        if citations:

            top = citations[0]

            self.container_memory.save_highlight(
                pdf_name=top["file_name"],
                page=top["page"],
                line_range=top["lines"],
                key_fact=full_response[:120]
            )

    # =========================================================
    # HISTORY
    # =========================================================

    def get_chat_history(
        self
    ) -> List[Dict]:

        return (
            self.chat_history
            .get_all_messages()
        )


# =============================================================
# GLOBAL ENGINE
# =============================================================

engine = MindGateEngine()