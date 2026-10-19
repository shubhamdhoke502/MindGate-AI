import os
import json
import hashlib
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ChunkCacheManager:
    """
    Caches parsed PDF metadata and chunks to disk (.mindgate/chunks_cache/).
    Prevents re-parsing already processed PDFs.
    """
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "storage", ".mindgate", "chunks_cache")
        
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """Hashes content and name because cached chunks contain the source file name."""
        hasher = hashlib.sha256()
        hasher.update(os.path.basename(file_path).encode("utf-8"))
        hasher.update(b"\0")
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def get_cached_chunks(self, pdf_path: str) -> Optional[List[Dict]]:
        """Retrieves cached chunks if file was previously parsed."""
        file_hash = self._get_file_hash(pdf_path)
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if not isinstance(chunks, list):
                    raise ValueError("Cached chunks must be a JSON array.")
                return chunks
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Ignoring invalid chunk cache %s: %s", cache_file, exc)
                return None
        return None

    def save_chunks_to_cache(self, pdf_path: str, chunks: List[Dict]):
        """Saves extracted chunks and metadata to disk cache."""
        file_hash = self._get_file_hash(pdf_path)
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
        temporary_file = f"{cache_file}.tmp"
        with open(temporary_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        os.replace(temporary_file, cache_file)