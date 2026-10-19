import os
import json
import logging
from threading import RLock
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Manages local persistent chat logs per workspace session.
    Saves and loads chat sessions dynamically.
    """
    def __init__(self, workspace_dir: Optional[str] = None):
        if workspace_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            workspace_dir = os.path.join(base_dir, "workspaces", "default_session")
        
        os.makedirs(workspace_dir, exist_ok=True)
        self.history_file = os.path.join(workspace_dir, "chat_history.json")
        self._lock = RLock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        with self._lock:
            if not os.path.exists(self.history_file):
                self._write_history([])

    def _write_history(self, history: List[Dict]) -> None:
        temporary_file = f"{self.history_file}.tmp"
        with open(temporary_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        os.replace(temporary_file, self.history_file)

    def save_message(
        self,
        role: str,
        content: str,
        citations: List[Dict] = None,
        file_name: Optional[str] = None,
    ):
        """Appends a new chat message to persistent storage."""
        with self._lock:
            history = self.get_all_messages()
            history.append({
                "role": role,
                "content": content,
                "citations": citations or [],
                "file_name": file_name,
            })
            self._write_history(history)

    def get_all_messages(self) -> List[Dict]:
        """Loads all historical messages for the active workspace."""
        with self._lock:
            with open(self.history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                raise ValueError("Chat history must contain a JSON array.")
            return history

    def clear_history(self):
        """Clears the chat history log for the active workspace."""
        with self._lock:
            self._write_history([])
        