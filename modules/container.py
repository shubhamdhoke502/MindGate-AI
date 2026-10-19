import os
import json
import logging
from threading import RLock
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContainerMemoryManager:
    def __init__(self, storage_dir: str = "./storage/.mindgate", workspace_id: str = "default_workspace"):
        """
        Initializes local JSON Container Box for persisting conversation highlights & metadata.
        """
        self.workspace_dir = os.path.join(storage_dir, workspace_id)
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.memory_file = os.path.join(self.workspace_dir, "memory.json")
        self._lock = RLock()
        self._ensure_memory_file()

    def _ensure_memory_file(self) -> None:
        """Creates an empty memory structure if it does not exist."""
        if not os.path.exists(self.memory_file):
            initial_data = {
                "workspace_id": os.path.basename(self.workspace_dir),
                "created_at": datetime.now().isoformat(),
                "documents": [],
                "memory_highlights": []
            }
            self._save_file(initial_data)

    def _load_file(self) -> Dict[str, Any]:
        """Loads raw JSON memory file."""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("documents"), list) or not isinstance(data.get("memory_highlights"), list):
            raise ValueError(f"Invalid MindGate memory file: {self.memory_file}")
        return data

    def _save_file(self, data: Dict[str, Any]) -> None:
        """Saves JSON memory data to disk."""
        temporary_file = f"{self.memory_file}.tmp"
        with open(temporary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temporary_file, self.memory_file)

    def register_document(self, file_name: str) -> None:
        """Tracks active documents inside the container."""
        with self._lock:
            data = self._load_file()
            if file_name not in data["documents"]:
                data["documents"].append(file_name)
                self._save_file(data)

    def save_highlight(self, pdf_name: str, page: int, line_range: str, key_fact: str) -> None:
        """
        Appends a key factual insight and citation coordinates into the container box.
        """
        if not key_fact.strip():
            return
        with self._lock:
            data = self._load_file()
            highlight_entry = {
                "id": f"mem_{len(data['memory_highlights']) + 1:04d}",
                "pdf_name": pdf_name,
                "page": page,
                "line_range": line_range,
                "key_fact": key_fact.strip()[:2000],
                "timestamp": datetime.now().isoformat()
            }
            data["memory_highlights"].append(highlight_entry)
            self._save_file(data)

    def handle_at_trigger(self, search_query: str = "") -> Dict[str, Any]:
        """
        Handles the '@' recall command. Returns matching active PDFs and saved memory highlights.
        """
        with self._lock:
            data = self._load_file()
        query = search_query.strip().casefold()

        # Filter active documents
        matched_docs = [
            doc for doc in data["documents"]
            if query in doc.casefold()
        ]

        # Filter memory highlights
        matched_highlights = [
            item for item in data["memory_highlights"]
            if query in item["pdf_name"].casefold() or query in item["key_fact"].casefold()
        ]

        return {
            "trigger_type": "@_recall",
            "matching_documents": matched_docs,
            "matching_highlights": matched_highlights
        }


# Quick Test Run
if __name__ == "__main__":
    mem = ContainerMemoryManager()
    mem.register_document("Audit_Report_2026.pdf")
    mem.save_highlight("Audit_Report_2026.pdf", page=4, line_range="12-18", key_fact="Royalty fee is 5% on gross revenue.")
    
    # Simulate typing '@' command in search UI
    recall_result = mem.handle_at_trigger("Audit")
    print("Recall Test Output (@ Command):")
    print(json.dumps(recall_result, indent=2))