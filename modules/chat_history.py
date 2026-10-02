import os
import json
from typing import List, Dict, Optional


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
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def save_message(self, role: str, content: str, citations: List[Dict] = None):
        """Appends a new chat message to persistent storage."""
        history = self.get_all_messages()
        message_data = {
            "role": role,        # 'user' or 'assistant'
            "content": content,
            "citations": citations or []
        }
        history.append(message_data)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_all_messages(self) -> List[Dict]:
        """Loads all historical messages for the active workspace."""
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def clear_history(self):
        """Clears the chat history log for the active workspace."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        