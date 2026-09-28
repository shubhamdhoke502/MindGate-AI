import os
import json
import hashlib
from typing import List, Dict, Optional

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
        """Generates MD5 hash of PDF file to detect changes."""
        hasher = hashlib.md5()
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
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_chunks_to_cache(self, pdf_path: str, chunks: List[Dict]):
        """Saves extracted chunks and metadata to disk cache."""
        file_hash = self._get_file_hash(pdf_path)
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)